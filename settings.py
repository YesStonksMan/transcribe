"""Catalog of transcription models (built-in + user-added) + external
dependencies, plus persistence of the user's default-model choice in ~/Library/Application Support/Transcribe/settings.json.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# --------- Model catalog (built-in + user-added from Réglages) ---------

@dataclass(frozen=True)
class ModelSpec:
    id: str                 # short stable identifier used in URLs + settings.json
    repo: str               # huggingface repo
    label: str              # display name
    description: str        # one-liner
    size_bytes: int         # approximate, for disk-free check + UI display
    custom: bool = False    # added by the user from Réglages (removable)

    @property
    def engine(self) -> str:
        return engine_for_repo(self.repo)

    @property
    def repos(self) -> tuple[str, ...]:
        """Every repo this model needs on disk (its own + any aligner)."""
        aligner = aligner_for_repo(self.repo)
        return (self.repo, aligner) if aligner else (self.repo,)


# Word-level timestamps for Qwen3-ASR, which on its own only times whole chunks.
QWEN3_ALIGNER_REPO = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"


def aligner_for_repo(repo: str) -> Optional[str]:
    """Forced-aligner repo to run after this ASR model, or None."""
    return QWEN3_ALIGNER_REPO if "qwen3-asr" in repo.lower() else None


def engine_for_repo(repo: str) -> str:
    """Which backend runs a repo: the in-process mlx-whisper path, or mlx-audio
    (Qwen3-ASR, Parakeet, Voxtral…) in its own venv.

    ponytail: name heuristic — an mlx-audio-format Whisper repo would be routed
    to mlx-whisper and fail to load. Store the engine per model if that happens.
    """
    return "mlx-whisper" if "whisper" in repo.lower() else "mlx-audio"


# Both entries run the same pipeline: MLX on the Apple GPU, silero-VAD chunking,
# identical post-cleanup. Only the weights differ — see the A/B in
# DEFAULT_MODEL_FALLBACK below before reordering or re-defaulting these.
MODELS: dict[str, ModelSpec] = {
    "large-v3-turbo": ModelSpec(
        id="large-v3-turbo",
        repo="mlx-community/whisper-large-v3-turbo",
        label="Large-v3 Turbo",
        description="GPU Metal, VAD-chunked. Le plus léger (1,9 Go RAM) et le plus rapide. Sur notre unique test FR de 50 s il sortait aussi le meilleur WER.",
        size_bytes=1_600_000_000,
    ),
    "large-v3": ModelSpec(
        id="large-v3",
        repo="mlx-community/whisper-large-v3-mlx",
        label="Large-v3 (complet)",
        description="Le modèle Whisper complet. Défaut. Plus lourd (3,6 Go RAM) et plus lent que Turbo ; le vocabulaire personnalisé répare les noms propres du premier chunk.",
        size_bytes=3_100_000_000,
    ),
    # Different engine (mlx-audio, own venv — see MLX_AUDIO_PYTHON in
    # transcribe.py). Timestamps are per VAD chunk (≤18s), not per sentence.
    "qwen3-asr-1.7b": ModelSpec(
        id="qwen3-asr-1.7b",
        repo="mlx-community/Qwen3-ASR-1.7B-8bit",
        label="Qwen3-ASR 1.7B",
        description="Autre moteur (mlx-audio). Alibaba, 8-bit, + Qwen3-ForcedAligner pour l'horodatage au mot. Non mesuré sur ton français : à comparer avant d'en faire le défaut.",
        size_bytes=3_750_000_000,
    ),
}


def get_models() -> dict[str, ModelSpec]:
    """Built-in catalog followed by the models the user added in Réglages."""
    out = dict(MODELS)
    for raw in _read_settings().get("custom_models") or []:
        try:
            spec = ModelSpec(**raw, custom=True)
        except TypeError:
            continue   # hand-edited / stale entry: skip rather than crash the UI
        out.setdefault(spec.id, spec)
    return out


def list_models_static() -> list[ModelSpec]:
    """All known models in catalog order (regardless of install state)."""
    return list(get_models().values())


def model_repo(model_id: str) -> str:
    """Return the HuggingFace repo for a model id. Raises KeyError if unknown."""
    return get_models()[model_id].repo


_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*$")


def validate_custom_repo(repo: str) -> str:
    """Stripped repo, or ValueError if malformed or already listed. Offline —
    the server checks the repo actually exists on HuggingFace."""
    repo = (repo or "").strip()
    if not _REPO_RE.match(repo):
        raise ValueError("Format attendu : organisation/modèle (ex. mlx-community/whisper-medium-mlx).")
    if any(m.repo.lower() == repo.lower() for m in get_models().values()):
        raise ValueError("Ce modèle est déjà dans la liste.")
    return repo


def add_custom_model(repo: str, size_bytes: int) -> str:
    """Add a HuggingFace repo to the catalog. Returns its model id."""
    repo = validate_custom_repo(repo)
    models = get_models()
    model_id = repo.split("/", 1)[1].lower()
    if model_id in models:
        model_id = repo.replace("/", "--").lower()
    name = repo.split("/", 1)[1]
    engine = "Whisper (MLX)" if engine_for_repo(repo) == "mlx-whisper" else "mlx-audio"
    entry = {
        "id": model_id, "repo": repo, "label": name,
        "description": f"Ajouté par toi · moteur {engine} · {repo}",
        "size_bytes": int(size_bytes),
    }
    data = _read_settings()
    data["custom_models"] = [*(data.get("custom_models") or []), entry]
    _write_settings(data)
    return model_id


def remove_custom_model(model_id: str) -> None:
    """Drop a user-added model from the catalog (cache deletion is separate).
    Raises ValueError for built-ins."""
    if model_id in MODELS:
        raise ValueError("Les modèles intégrés ne se retirent pas.")
    data = _read_settings()
    data["custom_models"] = [
        m for m in (data.get("custom_models") or []) if m.get("id") != model_id
    ]
    _write_settings(data)


# --------- Persistence ---------

# Qwen3-ASR + ForcedAligner, set on explicit request 01-10 after the bench in
# docs/bench/2026-10-01-sequence76.md (64s face-cam clip, hand-made subtitles as
# reference, VAD-chunked, vocabulary on):
#   qwen3 + aligner  WER 4.6% · start-time p90 0.05s · 6.1s · 4.0 GB
#   large-v3         WER 6.5% · start-time p90 0.27s · 8.7s · 3.6 GB
#   turbo            WER 6.9% · start-time p90 0.25s · 4.2s · 1.9 GB
# Known weakness: one meaning-changing substitution ("du cash" → "du capital").
# One scripted clip only — not yet measured on conversational audio (calls).
# This is only the fallback when the user has never chosen; MLX_WHISPER_REPO in
# transcribe.py stays the Whisper engine's own fallback.
DEFAULT_MODEL_FALLBACK = "qwen3-asr-1.7b"
# Fallback when that engine's venv is absent (any Mac but the dev one): Whisper
# runs in-process, so a fresh install transcribes out of the box.
WHISPER_FALLBACK = "large-v3"

# mlx-audio interpreter (own venv, see transcribe.py). Lives here so the default
# model can depend on whether it exists.
MLX_AUDIO_PYTHON = Path(os.environ.get(
    "TRANSCRIBE_MLX_AUDIO_PYTHON",
    Path.home() / "Claude code/video-pipeline/bin/mlx-audio-env/bin/python",
))


def _settings_dir() -> Path:
    """User data directory (created on first use). Matches the convention
    set in app.py for TRANSCRIBE_DATA_DIR."""
    d = Path.home() / "Library" / "Application Support" / "Transcribe"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _settings_file() -> Path:
    return _settings_dir() / "settings.json"


def _read_settings() -> dict:
    f = _settings_file()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_settings(data: dict) -> None:
    f = _settings_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_default_model() -> str:
    """Currently-selected model id, or DEFAULT_MODEL_FALLBACK if unset / unknown."""
    raw = _read_settings().get("default_model")
    if raw in get_models():
        return raw
    if engine_for_repo(MODELS[DEFAULT_MODEL_FALLBACK].repo) == "mlx-audio" and not MLX_AUDIO_PYTHON.exists():
        return WHISPER_FALLBACK
    return DEFAULT_MODEL_FALLBACK


def set_default_model(model_id: str) -> None:
    """Persist the chosen default model. Raises ValueError on unknown id."""
    if model_id not in get_models():
        raise ValueError(f"Unknown model id: {model_id}")
    data = _read_settings()
    data["default_model"] = model_id
    _write_settings(data)


# Whisper truncates the decoder prompt to the LAST (n_ctx // 2 - 1) = 223 tokens,
# silently dropping the head. 600 chars is comfortably under that for French
# even at a pessimistic ~2.5 chars/token, so nothing the user types is lost.
VOCABULARY_MAX_CHARS = 600


def get_vocabulary() -> str:
    """User's custom vocabulary, as free text. Empty string if unset."""
    raw = _read_settings().get("vocabulary")
    return raw.strip() if isinstance(raw, str) else ""


def set_vocabulary(text: str) -> None:
    """Persist the custom vocabulary. Raises ValueError if over the cap."""
    text = (text or "").strip()
    if len(text) > VOCABULARY_MAX_CHARS:
        raise ValueError(
            f"Vocabulaire trop long : {len(text)} caractères, maximum {VOCABULARY_MAX_CHARS}."
        )
    data = _read_settings()
    data["vocabulary"] = text
    _write_settings(data)


# How many seconds of audio one second of computing gets through, per model:
# what the time estimates are built on. These are only starting points, from
# whole jobs on this Mac (15-11-2026, 179 s of French speech, VAD-chunked:
# large-v3 4.3x cold, 5.3x warm; turbo and Qwen3 from the 01-10 bench, 15x and
# 10x on 64 s) — record_speed() then replaces them
# with what the user's own jobs measure. The old fixed 25x came from a 50 s turbo
# clip and announced ~2 min for an hour that large-v3 takes ~10 min to do.
SPEED_DEFAULTS = {"large-v3": 5.0, "large-v3-turbo": 14.0, "qwen3-asr-1.7b": 10.0}
SPEED_FALLBACK = 5.0


def get_speed(model_id: Optional[str]) -> float:
    """Measured real-time factor for a model (audio seconds per computing second)."""
    stored = (_read_settings().get("speed") or {}).get(model_id)
    if isinstance(stored, (int, float)) and stored > 0:
        return float(stored)
    return SPEED_DEFAULTS.get(model_id or "", SPEED_FALLBACK)


def record_speed(model_id: Optional[str], audio_s: float, elapsed_s: float) -> None:
    """Fold one finished job into the model's speed. Smoothed (70/30) so a cold
    start or a busy Mac nudges the estimate instead of rewriting it; clips under
    20 s are skipped, their time is mostly fixed overhead."""
    if not model_id or audio_s < 20 or elapsed_s <= 0:
        return
    speed = round(0.7 * get_speed(model_id) + 0.3 * (audio_s / elapsed_s), 2)
    data = _read_settings()
    data.setdefault("speed", {})[model_id] = speed
    _write_settings(data)


def get_output_dir() -> Path:
    """Folder where finished transcripts are saved automatically."""
    raw = _read_settings().get("output_dir")
    if isinstance(raw, str) and raw.strip():
        return Path(raw).expanduser()
    return Path.home() / "Downloads" / "Transcriptions"


def set_output_dir(path: str) -> None:
    """Persist the auto-save folder. Raises ValueError unless it is an absolute path."""
    p = Path((path or "").strip()).expanduser()
    if not p.is_absolute():
        raise ValueError("Chemin absolu attendu pour le dossier d'enregistrement.")
    data = _read_settings()
    data["output_dir"] = str(p)
    _write_settings(data)


# --------- HuggingFace cache detection ---------

def _hf_cache_root() -> Path:
    """Root of the HF Hub cache. Overridable via HF_HOME env."""
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _repo_to_cache_name(repo: str) -> str:
    """HF cache layout: 'org/name' becomes 'models--org--name'."""
    return "models--" + repo.replace("/", "--")


def _hf_cache_dir(model_id: str) -> Path:
    """Directory in HF cache for a given model id. May not exist."""
    spec = get_models()[model_id]
    return _hf_cache_root() / _repo_to_cache_name(spec.repo)


def is_model_installed(model_id: str) -> bool:
    """True if the model has been downloaded and has at least one snapshot
    with content."""
    spec = get_models().get(model_id)
    if spec is None or not spec.repo:
        return False
    return all(_repo_cached(r) for r in spec.repos)


def _repo_cached(repo: str) -> bool:
    snapshots = _hf_cache_root() / _repo_to_cache_name(repo) / "snapshots"
    if not snapshots.exists():
        return False
    return any(s.is_dir() and any(s.iterdir()) for s in snapshots.iterdir())


def uninstall_model(model_id: str) -> None:
    """Remove the model's cache directory. Raises KeyError on unknown id."""
    if model_id not in get_models():
        raise KeyError(model_id)
    d = _hf_cache_dir(model_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def disk_free_bytes(path: Path) -> int:
    """Free space at `path` in bytes."""
    return shutil.disk_usage(path).free


# --------- External dependencies ---------

@dataclass(frozen=True)
class DepSpec:
    id: str
    label: str
    brew_installable: bool
    version_arg: Optional[str]


DEPS: dict[str, DepSpec] = {
    "ffmpeg":  DepSpec("ffmpeg",  "ffmpeg",     True,  "-version"),
    "yt-dlp":  DepSpec("yt-dlp",  "yt-dlp",     True,  "--version"),
}


@dataclass(frozen=True)
class DepState:
    id: str
    label: str
    installed: bool
    version: Optional[str]
    brew_installable: bool


def list_deps_static() -> list[DepSpec]:
    return list(DEPS.values())


def _dep_version_probe(dep_id: str, path: str) -> Optional[str]:
    spec = DEPS.get(dep_id)
    if spec is None or spec.version_arg is None:
        return None
    # Lazy import: transcribe imports settings, so a top-level import would be circular.
    from transcribe import _external_env
    try:
        r = subprocess.run(
            [path, spec.version_arg],
            capture_output=True, text=True, timeout=5,
            env=_external_env(),
        )
        out = (r.stdout or r.stderr or "").strip()
        return out.split("\n", 1)[0] if out else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def dep_state(dep_id: str) -> DepState:
    spec = DEPS[dep_id]
    path = shutil.which(dep_id)
    if path is None:
        return DepState(
            id=spec.id, label=spec.label, installed=False, version=None,
            brew_installable=spec.brew_installable,
        )
    return DepState(
        id=spec.id, label=spec.label, installed=True,
        version=_dep_version_probe(dep_id, path),
        brew_installable=spec.brew_installable,
    )


def brew_available() -> bool:
    return shutil.which("brew") is not None


def list_models() -> list[dict]:
    default = get_default_model()
    out: list[dict] = []
    for spec in get_models().values():
        out.append({
            "id": spec.id,
            "repo": spec.repo,
            "label": spec.label,
            "description": spec.description,
            "size_bytes": spec.size_bytes,
            "installed": is_model_installed(spec.id),
            "is_default": spec.id == default,
            "custom": spec.custom,
            "engine": spec.engine,
        })
    return out


def list_deps() -> list[dict]:
    return [asdict(dep_state(spec.id)) for spec in DEPS.values()]
