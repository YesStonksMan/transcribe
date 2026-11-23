"""Core transcription pipeline.

Runs MLX-Whisper (large-v3) on Apple Silicon and writes raw Markdown,
SRT and JSON artifacts.

Designed to be called from server.py via a generator that yields progress
events. Also usable as a CLI: `python transcribe.py path/to/video.mp4`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse
from typing import Callable, Generator, Optional

import mlx_whisper
import settings as _settings

YTDLP_CLI = "yt-dlp"
# Full large-v3, set on explicit request 19-08. Same weights as the /watch skill,
# so the two engines are back in parity. Counter-evidence on record: a 29-07 FR
# bench (single 50.5s sample) had turbo ahead on both axes — 1.98s vs 3.45s and
# 12.2% vs 22.6% WER — and large-v3 mangled "Kakashi" across the first chunk.
# The custom-vocabulary prompt repairs that first chunk (docs/2026-08-03 §4), but
# only on French audio since the 04-08 language gate. Flip this back to
# "mlx-community/whisper-large-v3-turbo" — and DEFAULT_MODEL_FALLBACK in
# settings.py, which is what actually drives the app — to restore turbo.
MLX_WHISPER_REPO = "mlx-community/whisper-large-v3-mlx"

# silero-VAD interpreter. VAD needs torch, which would add ~4 GB to the py2app
# bundle, so it stays in an external virtualenv we subprocess out to: the same
# whisperx-env that run.sh and build.sh already depend on (torch 2.1.2 is there;
# silero-vad was added with `uv pip install --no-deps silero-vad` — with deps it
# swaps torch and breaks whisperx). It used to live in canary-env, deleted in
# the 28-10 disk cleanup: the app then ran whole-file for 18 days without saying
# so. If it is missing, _vad_chunks returns None, MLX falls back to whole-file,
# and _vad_note carries the reason to the progress line and the log.
VAD_PYTHON = Path(os.environ.get(
    "TRANSCRIBE_VAD_PYTHON",
    Path.home() / "Claude code/video-pipeline/bin/whisperx-env/bin/python",
))

log = logging.getLogger("transcribe")

# Why the last _vad_chunks call fell back to whole-file, or None if it chunked.
_vad_note: Optional[str] = None


def vad_available() -> bool:
    return VAD_PYTHON.exists()

# mlx-audio interpreter (Qwen3-ASR, Parakeet, Voxtral… — any non-Whisper repo,
# see settings.engine_for_repo). Own venv because mlx-audio pulls transformers 5,
# which would break whisperx in the app's shared venv, and the py2app bundle
# stays untouched. Created with:
#   uv venv --python 3.11 video-pipeline/bin/mlx-audio-env
#   VIRTUAL_ENV=…/mlx-audio-env uv pip install mlx-audio
MLX_AUDIO_PYTHON = _settings.MLX_AUDIO_PYTHON

# Python-launcher env vars that py2app injects into the process environment so
# the *bundle's* embedded interpreter boots correctly. They MUST NOT leak into
# external tools that are themselves Python scripts (yt-dlp ships a
# `#!/.../python` shebang). If they do, the child interpreter tries to load its
# stdlib from the bundle and dies with:
#   Fatal Python error: Failed to import encodings module
#   ModuleNotFoundError: No module named 'encodings'
_PYTHON_LAUNCHER_VARS = (
    "PYTHONHOME",
    "PYTHONPATH",
    "PYTHONEXECUTABLE",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONNOUSERSITE",
    "PYTHONOPTIMIZE",
    "PYTHONSTARTUP",
    "PYTHONUNBUFFERED",
    "__PYVENV_LAUNCHER__",
)


def _external_env() -> dict:
    """Environment for spawning external CLI tools (ffmpeg, ffprobe, yt-dlp).

    Returns a copy of the current environment with every Python-launcher
    variable stripped, so a child tool that runs its own Python interpreter
    (yt-dlp) isn't redirected at the bundle's stdlib. PATH — already augmented
    for GUI launch by app.py — is preserved as-is.
    """
    env = os.environ.copy()
    for var in _PYTHON_LAUNCHER_VARS:
        env.pop(var, None)
    return env


def sanitize_title(s: str, max_len: int = 120) -> str:
    """Make a string safe to use as a filename on macOS/Linux while keeping
    accents, spaces and unicode. Strips only chars that break filesystems.
    """
    # Remove control chars and chars that confuse macOS/the URL path
    s = re.sub(r"[\x00-\x1f/\\<>:|*?\"]", "", s)
    # Collapse runs of whitespace
    s = re.sub(r"\s+", " ", s).strip(" .")
    # Trim
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s or "transcript"

# Matches `[download]  12.3% of    4.55MiB at 1.50MiB/s ETA 00:02`
_YTDLP_PROGRESS = re.compile(
    r"\[download\]\s+(?P<pct>\d+(?:\.\d+)?)%"
    r"\s+of\s+~?\s*(?P<size>\S+)"
    r"(?:\s+at\s+(?P<rate>\S+))?"
    r"(?:\s+ETA\s+(?P<eta>\S+))?",
)


@dataclass
class TranscribeResult:
    language: str
    duration: float
    segments: list[dict] = field(default_factory=list)
    raw_text: str = ""
    srt: str = ""
    md_raw: str = ""
    elapsed: float = 0.0


def extract_audio(video_path: Path, out_dir: Path) -> Path:
    """Extract mono 16kHz WAV from any video/audio container.
    Always re-extracts — no caching. Job isolation (output dir per job)
    means we never need to dedupe at this level.
    """
    out = out_dir / f"{video_path.stem}.wav"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=_external_env())
    if r.returncode != 0:
        # ffmpeg's last lines name the real cause (no audio track, corrupt file,
        # unknown format); a bare CalledProcessError only showed the command.
        tail = "\n".join((r.stderr or "").strip().splitlines()[-4:])
        raise RuntimeError(f"ffmpeg n'a pas pu lire ce fichier.\n{tail}")
    return out


# Whisper's encoder window is 30s, so VAD chunks have headroom at 18s. Changing
# it changes the speed the estimates are built on (settings.get_speed), which
# then re-learns from the next jobs.
MLX_MAX_CHUNK_S = 18.0

# Hallucination cleanup for chunked output.
_REPEAT_TOKEN_RE = re.compile(r'\b(\w+)(?:[\s,.!?]+\1){3,}\b', re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _norm_sentence(s: str) -> str:
    return re.sub(r'[^\w\s]', '', s.lower()).strip()


def _dedupe_sentences(text: str) -> str:
    """Collapse consecutive near-duplicate sentences."""
    parts = _SENT_SPLIT_RE.split(text)
    if len(parts) < 2:
        return text
    out = [parts[0]]
    for p in parts[1:]:
        prev_n, cur_n = _norm_sentence(out[-1]), _norm_sentence(p)
        if cur_n and prev_n and (cur_n == prev_n or cur_n in prev_n or prev_n in cur_n):
            if len(cur_n) > len(prev_n):
                out[-1] = p
            continue
        out.append(p)
    return ' '.join(out).strip()


def _clean_text(text: str) -> str:
    """Collapse token-run loops (`quand quand quand quand`) and duplicate sentences."""
    return _dedupe_sentences(_REPEAT_TOKEN_RE.sub(r'\1', text).strip())


def _probe_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds via ffprobe. 0.0 if probing fails."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            check=True, capture_output=True, text=True, timeout=10,
            encoding="utf-8", errors="replace",
            env=_external_env(),
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return 0.0


def _format_duration(seconds: float) -> str:
    """Human-readable duration: 1h23, 12min04, 45s, or '?' if unknown."""
    if seconds <= 0:
        return "?"
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h{m:02d}"
    if m > 0:
        return f"{m}min{s:02d}"
    return f"{s}s"


# Heartbeat interval during MLX transcription. MLX doesn't stream segments —
# it returns everything at once — so without these heartbeats the user sees
# "0%" for several minutes on long videos and assumes the server is frozen.
# 1s, not 5s: heartbeats are the only thing driving the progress bar (MLX hands
# back every segment at once, at the very end), so the tick rate *is* the bar's
# frame rate. One SSE frame per second is negligible on localhost.
HEARTBEAT_INTERVAL_SEC = 1.0


def _run_whisper_with_heartbeats(
    audio_path: Path,
    language: Optional[str],
    interval: float = HEARTBEAT_INTERVAL_SEC,
    model_repo: Optional[str] = None,
    vocabulary: Optional[str] = None,
) -> Generator[dict, None, tuple[list[dict], str, float]]:
    """Run `_run_whisper` in a background thread and yield heartbeat events
    every `interval` seconds until completion. The final `(segments, language,
    duration)` triple is returned via `StopIteration.value`, so callers use
    `result = yield from _run_whisper_with_heartbeats(...)`.

    Heartbeats carry real VAD-chunk progress once the chunking pass has run, so
    the front-end shows a bar that actually advances instead of a static ETA.

    If the background thread raises, the exception is re-raised in the caller.
    """
    result_box: dict = {}
    # Written by the worker thread, read by this one. Plain stores under the GIL
    # — a lock would only serialise a value the reader is happy to see one tick
    # stale. `t1` is when the first chunk landed; see the rate maths below.
    progress = {"done": 0, "total": 0, "t1": 0.0}

    def on_chunk(done: int, total: int) -> None:
        if done == 1 and not progress["t1"]:
            progress["t1"] = time.time()
        progress["done"], progress["total"] = done, total

    def worker() -> None:
        try:
            result_box["value"] = _run_whisper(
                audio_path, language=language, model_repo=model_repo,
                on_chunk=on_chunk, vocabulary=vocabulary,
            )
        except BaseException as e:  # noqa: BLE001 — propagated below
            result_box["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    start = time.time()

    while True:
        t.join(timeout=interval)
        if not t.is_alive():
            break
        now = time.time()
        elapsed = now - start
        done, total, t1 = progress["done"], progress["total"], progress["t1"]
        ev: dict = {
            "type": "heartbeat",
            "stage": "transcribe",
            "elapsed": elapsed,
            "compute": "GPU (Metal)",   # MLX is Metal-only by construction
            "chunks_done": done,
            "chunks_total": total,
        }
        if total > 0:
            ev["progress"] = done / total
            ev["msg"] = f"Transcription · passage {min(done + 1, total)}/{total}"
            if done >= 2 and t1:
                # Rate measured from the *second* chunk onward. Chunk 1 carries
                # the model load — on a cold run it took 6s where the next two
                # took ~1s each, so `elapsed / done` predicted 15s remaining on
                # a job that had 4s left. Excluding it is the difference between
                # a countdown and a guess.
                ev["remaining"] = (now - t1) / (done - 1) * (total - done)
        elif _vad_note:
            # Whole-file fallback: there will never be a chunk count. Say so,
            # rather than "Analyse…" for the entire job.
            ev["msg"] = f"Transcription d'un bloc ({_vad_note}) · {_format_duration(elapsed)} écoulé"
        else:
            # Still in silero-VAD / model load — no chunk count to report yet.
            ev["msg"] = f"Analyse de l'audio · {_format_duration(elapsed)} écoulé"
        yield ev

    if "error" in result_box:
        raise result_box["error"]
    return result_box["value"]


def _run_whisper(
    audio_path: Path,
    language: Optional[str] = None,
    model_repo: Optional[str] = None,
    on_chunk: Optional[Callable[[int, int], None]] = None,
    vocabulary: Optional[str] = None,
) -> tuple[list[dict], str, float]:
    """Transcribe and return (segments, detected_language, duration).

    Routes to the engine that can load `model_repo`: in-process mlx-whisper, or
    mlx-audio in its own venv.
    """
    if model_repo and _settings.engine_for_repo(model_repo) == "mlx-audio":
        return _run_mlx_audio(
            audio_path, language=language, repo=model_repo, on_chunk=on_chunk,
            vocabulary=vocabulary,
        )
    return _run_mlx_whisper(
        audio_path, language=language, model_repo=model_repo, on_chunk=on_chunk,
        vocabulary=vocabulary,
    )


def _vocabulary_prompt(
    vocabulary: Optional[str], language: Optional[str] = None
) -> Optional[str]:
    """Turn the user's custom vocabulary into a Whisper decoder prompt.

    Whisper has no hotword list — the only lever is `initial_prompt`, a piece of
    text the decoder is conditioned on as if it had just transcribed it. Names
    spelled there become far likelier to come out spelled the same way.

    The words are wrapped in a sentence rather than passed as a bare CSV: the
    prompt is scored as ordinary speech, so text that reads like a transcript
    biases the model better than a list does.

    That carrier sentence is French, which is exactly why this is gated on
    French audio. Because the decoder treats the prompt as text it just
    produced, it will happily *continue* it: on an English video (04-08) it
    echoed the prompt's own opening as "Voici un peu de…" five times, closed on
    Whisper's canonical French silence hallucination ("Sous-titrage Société
    Radio-Canada"), and mangled "Trading 212" into "3D212" — all while the
    detected language was correctly reported as `en`. An unknown language
    returns None too: no prompt is safer than one in the wrong language.
    """
    words = (vocabulary or "").strip()
    if not words:
        return None
    if (language or "").strip().lower()[:2] != "fr":
        return None
    return f"Voici quelques termes qui apparaissent dans cet enregistrement : {words}."


def _mlx_transcribe_file(
    wav: Path, repo: str, language: Optional[str], prompt: Optional[str] = None
) -> tuple[list[dict], str]:
    """One MLX-Whisper pass over a WAV. Returns (segments, detected_language).

    `prompt` is re-sent for every chunk on purpose. Whisper only conditions its
    *first* 30s window on initial_prompt, and condition_on_previous_text=False
    resets the prompt after it — but each VAD chunk is its own transcribe() call
    of ≤18s, so one window each, so every chunk gets the vocabulary.

    A prompted window that needed temperature fallback is redone without the
    prompt. Measured 12-10 on `say` French speech: on chunks the 18s split cuts
    mid-word, the prompt made the greedy decode trip Whisper's own
    compression/logprob gate on 9 of 33 fragments (1 of 33 unprompted), and the
    fallback sampling then produced "Votre Votre…", "Lyse- Lyse-…" or "Sous-titrage
    Société Radio-Canada" where the unprompted decode read "plus forte". Only
    seek 0 counts: that is the one window the prompt conditions.
    """
    def decode(initial_prompt: Optional[str]) -> dict:
        return mlx_whisper.transcribe(
            str(wav),
            path_or_hf_repo=repo,
            language=language,
            word_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
        )

    result = decode(prompt)
    if prompt and any(
        s.get("seek", 0) == 0 and s.get("temperature", 0.0) > 0
        for s in (result.get("segments") or [])
    ):
        result = decode(None)
    segments = [
        {"start": float(s["start"]), "end": float(s["end"]), "text": s["text"]}
        for s in (result.get("segments") or [])
    ]
    return segments, result.get("language") or (language or "")


def _detect_language_probe(audio_path: Path, repo: str) -> Optional[str]:
    """One-shot language ID on a dedicated 30s window, not a VAD fragment.

    The first VAD speech chunk can be a 1-3s fragment (a name, "hey guys", a
    sting) that Whisper's language ID reads unreliably — and that guess then
    gets pinned for the whole file. 30s of real audio from the start of the
    track is a far more reliable window to detect on. Returns None (silent
    fallback to the old first-chunk pin) on any failure.
    """
    with tempfile.TemporaryDirectory(prefix="mlxprobe_") as td:
        probe = Path(td) / "probe.wav"
        duration = _probe_audio_duration(audio_path) or 30.0
        if not _slice_wav(audio_path, 0.0, min(30.0, duration), probe):
            return None
        try:
            result = mlx_whisper.transcribe(
                str(probe), path_or_hf_repo=repo, no_speech_threshold=0.6,
            )
        except Exception:
            return None
        return result.get("language") or None


def _run_mlx_whisper(
    audio_path: Path,
    language: Optional[str] = None,
    model_repo: Optional[str] = None,
    on_chunk: Optional[Callable[[int, int], None]] = None,
    vocabulary: Optional[str] = None,
) -> tuple[list[dict], str, float]:
    """Run MLX-Whisper over silero-VAD speech regions, or whole-file as fallback.

    Chunking is not just a speed win: on a recording with long pauses, Whisper
    handed the full file loops on the silence (see _VAD_CHUNKS_SCRIPT).

    `on_chunk(done, total)` is called after every chunk so the caller can render
    real progress. It is never called on the whole-file fallback path, where
    there is no intermediate state to report.
    """
    repo = model_repo or MLX_WHISPER_REPO
    chunks = _vad_chunks(audio_path, MLX_MAX_CHUNK_S)

    # Resolve the language BEFORE building the prompt — _vocabulary_prompt only
    # emits on French audio, so it cannot decide anything until this is known.
    # The probe costs one 30s decode, so only pay for it when something
    # downstream actually reads the answer: the per-chunk pin below, or the
    # prompt decision when a vocabulary is set.
    if language:
        resolved = language
    elif chunks or vocabulary:
        resolved = _detect_language_probe(audio_path, repo) or ""
    else:
        resolved = ""
    prompt = _vocabulary_prompt(vocabulary, resolved)

    if not chunks:
        # One pass over the whole file. It is its own — and better — detection
        # window than a 30s probe, so leave it unpinned when the caller gave us
        # no language; `resolved` only informed the prompt.
        segments, detected = _mlx_transcribe_file(audio_path, repo, language, prompt)
        # Same loop cleanup as the chunked path: this fallback is where Whisper
        # loops most, since silence is not cut out.
        segments = [dict(s, text=t) for s in segments if (t := _clean_text(s["text"].strip()))]
        return segments, detected, (segments[-1]["end"] if segments else 0.0)

    total = len(chunks)
    if on_chunk:
        on_chunk(0, total)   # publish the denominator before the first decode

    segments: list[dict] = []
    detected = resolved
    with tempfile.TemporaryDirectory(prefix="mlxvad_") as td:
        tmp = Path(td)
        for i, ch in enumerate(chunks, 1):
            start_s = float(ch["start"])
            dur = float(ch["end"]) - start_s
            if dur < 0.3:
                if on_chunk:
                    on_chunk(i, total)
                continue
            piece = tmp / f"chunk_{i:04d}.wav"
            if not _slice_wav(audio_path, start_s, dur, piece):
                if on_chunk:
                    on_chunk(i, total)
                continue
            # `detected` was resolved up front by _detect_language_probe (a 30s
            # window, not this chunk) and is pinned for every chunk. Per-chunk
            # detection lets a single anglicism flip a later chunk to English
            # mid-file; the only time this still detects live is when the probe
            # itself failed, in which case chunk 1's own guess becomes the pin.
            segs, lang = _mlx_transcribe_file(piece, repo, language or detected or None, prompt)
            if not detected:
                detected = lang
            for s in segs:
                text = _clean_text(s["text"].strip())
                if text:
                    segments.append({
                        "start": round(s["start"] + start_s, 3),
                        "end": round(s["end"] + start_s, 3),
                        "text": text,
                    })
            if on_chunk:
                on_chunk(i, total)

    duration = segments[-1]["end"] if segments else 0.0
    return segments, detected, duration


# Source shipped to the external VAD interpreter (it cannot import from this
# module — different process, different venv). Loads no ASR model: it returns
# speech-region boundaries and exits. The MLX-Whisper path transcribes
# in-process and only needs to know where the speech is.
#
# Whisper hallucinates hard on silence — handed the whole file it looped
# "et c'est bon et c'est bon et c'est bon…" through 40s of a quiet recording,
# with condition_on_previous_text already False. Feeding it speech-only chunks
# removes the failure mode and is ~2.5× faster, since silence is never decoded.
#
# A speech region longer than max_chunk_s (continuous speech: jump-cut Reels
# never pause ≥250 ms) is split by split_region(), kept torch-free so tests can
# exec it. It used to cut at fixed 18 s offsets — mid-word, with a remainder of
# any length (37.9 s → 18 + 18 + 1.9 s); a prompted greedy decode of such a
# fragment invents words ("plus" → the custom vocabulary list read back). Now: the
# fewest pieces ≤ max_chunk_s, each cut near an even split (so the smallest
# piece is ~max/2 − SPLIT_SEARCH_S, never a sliver), on the quietest frame of
# 50 ms-smoothed energy within ±SPLIT_SEARCH_S. Measured 12-10-2026 on `say -v
# Thomas` continuous French, cut ±4 s decoded split vs whole: 79 → 43 word
# edits over 44 cuts. Not a word aligner: the quietest frame is often a stop
# closure inside a word ("fort|ement" → "fort.") — silero's frame probability
# scored the same (42 edits) for a second model pass, so energy it is.
_VAD_SPLIT_SRC = r"""
import math

SPLIT_FRAME_S = 0.01    # one energy value per 10 ms frame
SPLIT_SEARCH_S = 2.0    # look this far either side of the even-split point

def split_region(start, end, max_s, energy):
    pieces, t = [], start
    while end - t > max_s:
        n = math.ceil((end - t) / max_s)
        target = t + (end - t) / n
        # lo keeps the rest coverable by n-1 pieces; hi is the hard ceiling.
        lo = max(target - SPLIT_SEARCH_S, end - (n - 1) * max_s)
        hi = min(target + SPLIT_SEARCH_S, t + max_s)
        frames = range(math.ceil((lo - start) / SPLIT_FRAME_S),
                       min(math.floor((hi - start) / SPLIT_FRAME_S), len(energy) - 1) + 1)
        cut = start + min(frames, key=energy.__getitem__) * SPLIT_FRAME_S if frames else hi
        cut = min(max(cut, lo), hi)    # frame→seconds drift under lo costs an extra piece
        pieces.append({'start': t, 'end': cut})
        t = cut
    pieces.append({'start': t, 'end': end})
    return pieces
"""

_VAD_CHUNKS_SCRIPT = _VAD_SPLIT_SRC + r"""
import sys, json, warnings
warnings.filterwarnings('ignore')

audio_path  = sys.argv[1]
max_chunk_s = float(sys.argv[2])

from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
vad = load_silero_vad()
wav = read_audio(audio_path, sampling_rate=16000)
speech = get_speech_timestamps(
    wav, vad, sampling_rate=16000, return_seconds=True,
    speech_pad_ms=200,
    min_silence_duration_ms=250,
)

if not speech:
    print('__VAD_JSON__' + json.dumps([]))
    sys.exit(0)

chunks = []
cur = {'start': speech[0]['start'], 'end': speech[0]['end']}
for s in speech[1:]:
    if s['end'] - cur['start'] <= max_chunk_s:
        cur['end'] = s['end']
    else:
        chunks.append(cur)
        cur = {'start': s['start'], 'end': s['end']}
chunks.append(cur)

import torch

def region_energy(ch):
    hop = round(16000 * SPLIT_FRAME_S)
    x = wav[int(ch['start'] * 16000):int(ch['end'] * 16000)]
    n = x.numel() // hop
    e = x[:n * hop].reshape(n, hop).pow(2).mean(1)
    return torch.nn.functional.avg_pool1d(
        e[None, None], 5, stride=1, padding=2, count_include_pad=False,
    )[0, 0].tolist()

out = []
for ch in chunks:
    if ch['end'] - ch['start'] <= max_chunk_s:
        out.append(ch)
    else:
        out.extend(split_region(ch['start'], ch['end'], max_chunk_s, region_energy(ch)))

print('__VAD_JSON__' + json.dumps(out))
"""


def _vad_chunks(audio_path: Path, max_chunk_s: float) -> Optional[list[dict]]:
    """Speech-region boundaries via silero-VAD, or None if unavailable.

    Returning None rather than raising is deliberate: silero lives in an
    external venv (it needs torch, which the app bundle does not ship), so the
    MLX path must still work when that venv is missing — it just falls back to
    whole-file transcription.
    """
    global _vad_note
    _vad_note = None

    def failed(note: str, detail: str = "") -> None:
        global _vad_note
        _vad_note = note
        log.warning("VAD indisponible (%s) — transcription du fichier entier. %s", note, detail)

    if not VAD_PYTHON.exists():
        return failed("découpage non installé", str(VAD_PYTHON))
    try:
        proc = subprocess.run(
            [str(VAD_PYTHON), "-c", _VAD_CHUNKS_SCRIPT,
             str(audio_path), f"{max_chunk_s}"],
            capture_output=True, text=True, timeout=600,
            encoding="utf-8", errors="replace", env=_external_env(),
        )
    except (subprocess.SubprocessError, OSError) as e:
        return failed("découpage en échec", repr(e))
    if proc.returncode != 0:
        return failed("découpage en échec", (proc.stderr or "").strip()[-400:])
    for line in (proc.stdout or "").splitlines():
        if line.startswith("__VAD_JSON__"):
            try:
                chunks = json.loads(line[len("__VAD_JSON__"):])
            except json.JSONDecodeError:
                return failed("découpage en échec", "JSON illisible")
            return chunks or None   # no speech found: a real answer, not a failure
    return failed("découpage en échec", "aucune réponse")


# mlx-audio models take a language *name*; the app speaks ISO codes.
_ISO_TO_LANGUAGE = {
    "fr": "French", "en": "English", "es": "Spanish",
    "de": "German", "it": "Italian", "pt": "Portuguese",
}

# Source shipped to the mlx-audio interpreter. Loads the model once, decodes
# every chunk, prints one __SEG__ line per chunk so the parent can drive the
# progress bar. Only passes the kwargs the model's generate() accepts — Qwen3
# takes language + hotwords, Parakeet takes neither — which is what lets any
# mlx-audio STT repo run without per-model code.
#
# Result shapes differ per model family: Qwen3 returns dict `segments`,
# Parakeet returns `sentences` objects. to_segments() normalises both; text with
# no timing comes back as 0→0 and the parent stretches it over its chunk.
#
# Qwen3-ASR only times whole chunks, so its text goes through Qwen3-ForcedAligner
# (word timings). The aligner strips punctuation and hyphens ("peut-être" →
# "peutêtre"), so align_segments() walks the original tokens, consumes the
# aligned words that spell each one, and regroups them into sentences: break on
# .!?…, on a pause ≥ PAUSE_S, or at MAX_WORDS. Any mismatch → None, and the
# caller keeps the chunk-level segment instead of shipping wrong timings.
_MLX_AUDIO_SEGMENTS_SRC = r"""
import re as _re
PAUSE_S, MAX_WORDS = 0.8, 25

def _norm(w):
    return _re.sub(r"[^\w']", "", w).lower()

def align_segments(text, words):
    words = [(_norm(t), float(a), float(b)) for t, a, b in words if _norm(t)]
    timed, k = [], 0
    for tok in text.split():
        target = _norm(tok)
        if not target:
            if timed:
                timed[-1] = (timed[-1][0] + ' ' + tok, timed[-1][1], timed[-1][2])
            continue
        spelled, start, end = '', None, None
        while k < len(words) and len(spelled) < len(target):
            spelled += words[k][0]
            start = words[k][1] if start is None else start
            end = words[k][2]
            k += 1
        if spelled != target:
            return None
        timed.append((tok, start, end))
    if not timed or k != len(words):
        return None
    segs, cur = [], []
    for i, (tok, a, b) in enumerate(timed):
        cur.append((tok, a, b))
        nxt = timed[i + 1][1] if i + 1 < len(timed) else None
        if tok[-1] in '.!?…' or nxt is None or nxt - b >= PAUSE_S or len(cur) >= MAX_WORDS:
            segs.append({'start': cur[0][1], 'end': cur[-1][2], 'text': ' '.join(t for t, _, _ in cur)})
            cur = []
    return segs

def to_segments(r):
    out = []
    for s in (getattr(r, 'segments', None) or getattr(r, 'sentences', None) or []):
        get = s.get if isinstance(s, dict) else (lambda k, _s=s: getattr(_s, k, None))
        start, end, text = get('start'), get('end'), get('text')
        if start is not None and end is not None and text:
            out.append({'start': float(start), 'end': float(end), 'text': text})
    if not out and (getattr(r, 'text', '') or '').strip():
        out = [{'start': 0.0, 'end': 0.0, 'text': r.text}]
    return out
"""

_MLX_AUDIO_SCRIPT = _MLX_AUDIO_SEGMENTS_SRC + r"""
import sys, json, inspect, warnings
warnings.filterwarnings('ignore')
job = json.load(open(sys.argv[1], encoding='utf-8'))
from mlx_audio.stt import load
model = load(job['repo'])
aligner = load(job['aligner']) if job.get('aligner') else None
accepted = inspect.signature(model.generate).parameters
kw = {}
if job.get('language') and 'language' in accepted:
    kw['language'] = job['language']
if job.get('hotwords') and 'hotwords' in accepted:
    kw['hotwords'] = job['hotwords']
for i, path in enumerate(job['chunks']):
    r = model.generate(path, **kw)
    segs = to_segments(r)
    lang = getattr(r, 'language', None)
    lang = next((l for l in lang if l), None) if isinstance(lang, list) else lang
    if aligner is not None and (r.text or '').strip() and (job.get('language') or lang):
        try:
            res = aligner.generate(audio=path, text=r.text, language=job.get('language') or lang)
            segs = align_segments(r.text, [(w.text, w.start_time, w.end_time) for w in res.items]) or segs
        except Exception as e:
            print('aligner failed: %s' % e, flush=True)   # keep chunk-level timing
    print('__SEG__' + json.dumps({'i': i, 'language': lang, 'segments': segs}, ensure_ascii=False), flush=True)
"""


def _iso_language(name) -> str:
    """mlx-audio's detected language ('French', ['French'], 'fr') → ISO code."""
    if isinstance(name, list):
        name = next((n for n in name if n), None)
    if not name:
        return ""
    name = str(name).strip()
    for iso, full in _ISO_TO_LANGUAGE.items():
        if name.lower() in (iso, full.lower()):
            return iso
    return name.lower()[:2]


def _run_mlx_audio(
    audio_path: Path,
    language: Optional[str] = None,
    repo: str = "",
    on_chunk: Optional[Callable[[int, int], None]] = None,
    vocabulary: Optional[str] = None,
) -> tuple[list[dict], str, float]:
    """Run an mlx-audio STT model over silero-VAD chunks, in MLX_AUDIO_PYTHON.

    Same contract as _run_mlx_whisper. Vocabulary goes in as hotwords — a list,
    not a carrier sentence, so no French-only gate is needed here.
    """
    if not MLX_AUDIO_PYTHON.exists():
        raise RuntimeError(
            f"Moteur mlx-audio absent ({MLX_AUDIO_PYTHON}). Ce modèle en a besoin — "
            "voir MLX_AUDIO_PYTHON dans transcribe.py pour l'installer."
        )
    chunks = _vad_chunks(audio_path, MLX_MAX_CHUNK_S)
    hotwords = [w.strip() for w in re.split(r"[,;\n]", vocabulary or "") if w.strip()]

    with tempfile.TemporaryDirectory(prefix="mlxaudio_") as td:
        tmp = Path(td)
        offsets: list[float] = []
        lengths: list[float] = []
        paths: list[str] = []
        if not chunks:
            offsets, lengths, paths = [0.0], [_probe_audio_duration(audio_path)], [str(audio_path)]
        for i, ch in enumerate(chunks or [], 1):
            start_s = float(ch["start"])
            dur = float(ch["end"]) - start_s
            piece = tmp / f"chunk_{i:04d}.wav"
            if dur >= 0.3 and _slice_wav(audio_path, start_s, dur, piece):
                offsets.append(start_s)
                lengths.append(dur)
                paths.append(str(piece))
        job_file = tmp / "job.json"
        job_file.write_text(json.dumps({
            "repo": repo,
            "aligner": _settings.aligner_for_repo(repo),
            "language": _ISO_TO_LANGUAGE.get((language or "")[:2]) if language else None,
            "hotwords": hotwords,
            "chunks": paths,
        }), encoding="utf-8")

        total = len(paths)
        if on_chunk:
            on_chunk(0, total)
        proc = subprocess.Popen(
            [str(MLX_AUDIO_PYTHON), "-c", _MLX_AUDIO_SCRIPT, str(job_file)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,   # merged: no full-pipe deadlock
            text=True, encoding="utf-8", errors="replace", env=_external_env(),
        )
        segments: list[dict] = []
        detected = language or ""
        tail: list[str] = []
        for line in proc.stdout:
            if not line.startswith("__SEG__"):
                tail = (tail + [line.rstrip()])[-15:]
                continue
            ev = json.loads(line[len("__SEG__"):])
            offset, length = offsets[ev["i"]], lengths[ev["i"]]
            detected = detected or _iso_language(ev.get("language"))
            for s in ev["segments"]:
                text = _clean_text(s["text"].strip())
                if text:
                    end = s["end"] if s["end"] > s["start"] else max(length, s["start"])
                    segments.append({
                        "start": round(s["start"] + offset, 3),
                        "end": round(end + offset, 3),
                        "text": text,
                    })
            if on_chunk:
                on_chunk(ev["i"] + 1, total)
        if proc.wait() != 0:
            raise RuntimeError("mlx-audio a échoué :\n" + "\n".join(tail))

    duration = segments[-1]["end"] if segments else 0.0
    return segments, detected, duration


def _slice_wav(audio_path: Path, start_s: float, dur_s: float, out_path: Path) -> bool:
    """Cut [start_s, start_s+dur_s] into a 16kHz mono WAV. False if ffmpeg failed."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", str(audio_path),
         "-vn", "-c:a", "pcm_s16le", "-ar", "16000", "-ac", "1", str(out_path)],
        capture_output=True, text=True, env=_external_env(),
    )
    return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        lines.append(seg["text"].strip())
        lines.append("")
    return "\n".join(lines)


# Silence between two segments long enough to start a new paragraph in the
# timestamp-free export (otherwise the text reads as one wall).
PARAGRAPH_GAP_SEC = 2.0


def build_md_raw(segments: list[dict]) -> str:
    """Timestamped export: one `[hh:mm:ss]` line per segment, no header."""
    parts = []
    for seg in segments:
        text = seg["text"].strip()
        if text:
            ts = format_timestamp(seg["start"]).split(",")[0]
            parts.append(f"`[{ts}]` {text}")
    return "\n\n".join(parts) + "\n"


def build_md_plain(segments: list[dict]) -> str:
    """Timestamp-free export: flowing text, paragraph break on long pauses."""
    paragraphs: list[list[str]] = []
    prev_end = None
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        if prev_end is None or seg["start"] - prev_end >= PARAGRAPH_GAP_SEC:
            paragraphs.append([])
        paragraphs[-1].append(text)
        prev_end = seg["end"]
    return "\n\n".join(" ".join(p) for p in paragraphs) + "\n"


def ytdlp_available() -> bool:
    from shutil import which
    return which(YTDLP_CLI) is not None


# Browsers yt-dlp can pull login cookies from, mapped to their macOS app bundle.
# Login-walled sites (Instagram, private/age-gated) only work when the user is
# signed in in one of these — we try each installed one in turn.
# Brave first: it's the browser most often signed in to Instagram here, and
# Instagram rejects Chrome's cookies ("Failed to parse JSON", same as /watch).
_BROWSER_APPS = [
    ("brave",   "/Applications/Brave Browser.app"),
    ("chrome",  "/Applications/Google Chrome.app"),
    ("edge",    "/Applications/Microsoft Edge.app"),
    ("firefox", "/Applications/Firefox.app"),
    ("vivaldi", "/Applications/Vivaldi.app"),
    ("opera",   "/Applications/Opera.app"),
    ("safari",  "/Applications/Safari.app"),
]

_SAFARI_COOKIES = Path.home() / "Library/Containers/com.apple.Safari/Data/Library/Cookies/Cookies.binarycookies"

# Hosts that always need a signed-in session: skip the anonymous attempt, which
# only burns a request and makes Instagram rate-limit the cookie retries after it.
_AUTH_WALLED_HOSTS = ("instagram.com", "facebook.com", "threads.net")


def _needs_login(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return any(host == h or host.endswith("." + h) for h in _AUTH_WALLED_HOSTS)


def _safari_cookies_readable() -> bool:
    # Without Full Disk Access, macOS refuses the read ("Operation not permitted").
    try:
        with open(_SAFARI_COOKIES, "rb"):
            return True
    except OSError:
        return False


def _browsers_to_try() -> list[str]:
    """yt-dlp browser names for the browsers actually installed on this Mac,
    in priority order. Safari is skipped when its cookie file is unreadable.
    Falls back to chrome if none are detected."""
    found = [
        name for name, path in _BROWSER_APPS
        if os.path.isdir(path) and (name != "safari" or _safari_cookies_readable())
    ]
    return found or ["chrome"]


def _probe_url_meta(url: str, browser: Optional[str] = None) -> Optional[dict]:
    """Lightweight metadata probe before downloading. Returns dict with title,
    uploader, id, extractor — or None if probe fails. Each field may be empty.
    """
    cmd = [
        YTDLP_CLI,
        "--no-warnings",
        "--simulate",
        "--no-playlist",
        "--print", "%(title|)s|%(uploader|)s|%(id|)s|%(extractor|)s",
    ]
    if browser:
        cmd += ["--cookies-from-browser", browser]
    cmd.append(url)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=45,
                           encoding="utf-8", errors="replace", env=_external_env())
    except Exception:
        return None
    if r.returncode != 0:
        return None
    line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
    parts = line.split("|")
    if len(parts) < 4:
        return None
    return {
        "title":     parts[0].strip(),
        "uploader":  parts[1].strip(),
        "id":        parts[2].strip(),
        "extractor": parts[3].strip(),
    }


# Patterns that yt-dlp synthesizes when there's no real title.
# Matched case-insensitively against the start of the title.
_GENERIC_TITLE_PATTERNS = re.compile(
    r"^(?:video by\b|post by\b|story by\b|reel\b|short\b|untitled\b|video$|reel$|short$)",
    re.IGNORECASE,
)


def _friendly_name_from_meta(meta: dict) -> str:
    """Compose a clean, descriptive filename stem from probe metadata.
    Falls back gracefully when the title is missing, generic ('Video by X',
    'Reel', 'Untitled', …), too short, or identical to the id.
    Order: meaningful title → site + uploader + id → 'video'.
    """
    title = (meta.get("title") or "").strip()
    vid_id = (meta.get("id") or "").strip()
    uploader = (meta.get("uploader") or "").strip()
    extractor = (meta.get("extractor") or "").strip()

    title_is_meaningful = (
        title
        and title.upper() != "NA"
        and title != vid_id
        and len(title) > 3
        and not _GENERIC_TITLE_PATTERNS.match(title)
    )
    if title_is_meaningful:
        return sanitize_title(title)

    pieces = [p for p in (extractor, uploader, vid_id) if p]
    composed = " - ".join(pieces) if pieces else "video"
    return sanitize_title(composed)


def _build_ytdlp_cmd(url: str, dest_dir: Path, browser: Optional[str] = None, friendly_stem: Optional[str] = None) -> list[str]:
    # If we know a friendly name from the probe, use it. Otherwise fall back to title.
    if friendly_stem:
        # Friendly name baked in literally — yt-dlp still sanitizes any leftover chars.
        out_tpl = str(dest_dir / f"{friendly_stem}.%(ext)s")
    else:
        out_tpl = str(dest_dir / "%(title).100B.%(ext)s")
    cmd = [
        YTDLP_CLI,
        "-f", "bestaudio/best",
        "--no-playlist",
        "--no-warnings",
        "--newline",
        "--progress",
        "-o", out_tpl,
        "--print", "after_move:%(filepath)s",
        "--print", "before_dl:title=%(title)s",
    ]
    if browser:
        cmd += ["--cookies-from-browser", browser]
    cmd.append(url)
    return cmd


def _run_ytdlp(cmd: list[str], dest_dir: Path):
    """Generator that yields progress events; returns (final_path, all_lines)."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace", env=_external_env())
    final_path: Optional[str] = None
    last_pct = -1.0
    all_lines: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        all_lines.append(line)
        if line.startswith("title="):
            title = line[len("title="):]
            yield {"type": "status", "stage": "download", "msg": f"Téléchargement : {title}"}
            continue
        m = _YTDLP_PROGRESS.search(line)
        if m:
            pct = float(m.group("pct"))
            if pct - last_pct >= 0.5 or pct >= 100:
                last_pct = pct
                eta = (m.group("eta") or "").strip()
                rate = (m.group("rate") or "").strip()
                msg = f"Téléchargement : {pct:.1f}%"
                if rate and "Unknown" not in rate and not rate.startswith("-"):
                    msg += f" · {rate}"
                if eta and "Unknown" not in eta and not eta.startswith("-"):
                    msg += f" · ETA {eta}"
                yield {"type": "status", "stage": "download", "msg": msg, "download_pct": pct}
            continue
        if line and (line.startswith("/") or line.startswith(str(dest_dir))) and Path(line).exists():
            final_path = line
    proc.wait()
    yield {"_done": True, "rc": proc.returncode, "path": final_path, "lines": all_lines}


def download_from_url_stream(
    url: str,
    dest_dir: Path,
) -> Generator[dict, None, Path]:
    """Yield progress events while yt-dlp downloads. Instagram/Facebook/Threads go
    straight to browser cookies (Brave first); other hosts try anonymously first,
    then each installed browser's cookies if the failure looks login-walled.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    browsers = _browsers_to_try()

    # 1) Probe metadata first to compose a clean filename.
    yield {"type": "status", "stage": "download", "msg": "Recherche du titre…"}
    login_first = _needs_login(url)
    if login_first:
        meta = _probe_url_meta(url, browser=browsers[0])
    else:
        meta = _probe_url_meta(url) or _probe_url_meta(url, browser=browsers[0])
    friendly_stem = _friendly_name_from_meta(meta) if meta else None
    if friendly_stem:
        yield {"type": "status", "stage": "download", "msg": f"Titre : {friendly_stem}"}

    attempts: list[Optional[str]] = (browsers if login_first else [None] + browsers)
    tried: list[str] = []
    last_tail = "(no output)"
    for browser in attempts:
        cmd = _build_ytdlp_cmd(url, dest_dir, browser=browser, friendly_stem=friendly_stem)
        if browser:
            yield {"type": "status", "stage": "download",
                   "msg": f"Connexion requise — essai avec les cookies de {browser.capitalize()}…"}
        result = None
        for ev in _run_ytdlp(cmd, dest_dir):
            if ev.get("_done"):
                result = ev
                break
            yield ev
        assert result is not None
        if result["rc"] == 0 and result["path"]:
            final_path = result["path"]
            yield {"type": "status", "stage": "download", "msg": f"Téléchargé : {Path(final_path).name}"}
            yield {"type": "downloaded", "path": final_path}
            return Path(final_path)

        last_tail = "\n".join(result["lines"][-15:]) if result["lines"] else "(no output)"
        if browser:
            errors = [l for l in result["lines"] if l.startswith("ERROR")]
            tried.append(f"{browser} → {(errors[-1] if errors else last_tail.splitlines()[-1])[:200]}")
        login_walled = any(
            kw in last_tail.lower() for kw in
            ["login", "sign in", "cookies", "rate-limit", "private", "age-restricted",
             "unavailable", "empty media response", "403", "401", "not logged"]
        )
        # Anonymous attempt failed for a non-login reason (bad URL, network, geo) —
        # cookies won't help, so fail now with the raw detail.
        if browser is None and not login_walled:
            raise RuntimeError(f"yt-dlp exit {result['rc']}\n— dernières lignes —\n{last_tail}")
        # otherwise fall through and try the next browser's cookies

    # All attempts exhausted on a login-walled link — give a human, actionable error.
    raise RuntimeError(
        "Ce lien nécessite d'être connecté (Instagram, compte privé, vidéo avec âge requis…).\n"
        "yt-dlp lit les cookies de connexion depuis ton navigateur : connecte-toi au site "
        "dans Brave (ou Chrome, Edge, Firefox), puis relance.\n"
        "— navigateurs essayés —\n" + ("\n".join(tried) if tried else "aucun")
    )


def transcribe_url_stream(
    url: str,
    language: Optional[str] = None,
    work_dir: Optional[Path] = None,
    dl_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
) -> Generator[dict, None, None]:
    """Download a URL via yt-dlp then run the standard transcription pipeline.

    `dl_dir` — where yt-dlp downloads the media (defaults to the project's
    `uploads/` folder for back-compat). Server callers pass a job-scoped folder.
    `work_dir` — passed through to transcribe_stream for output isolation.
    """
    download_into = dl_dir if dl_dir is not None else (Path(__file__).parent / "uploads")
    try:
        download_gen = download_from_url_stream(url, download_into)
        media_path: Optional[Path] = None
        while True:
            try:
                ev = next(download_gen)
                yield ev
            except StopIteration as stop:
                media_path = stop.value
                break
        if media_path is None:
            yield {"type": "error", "msg": "Téléchargement échoué (aucun fichier)"}
            return
        yield from transcribe_stream(
            media_path,
            language=language,
            display_title=media_path.stem,
            work_dir=work_dir,
            model_name=model_name,
        )
    except Exception as e:
        log.exception("transcription en échec")   # the UI only gets one line; keep the traceback
        yield {"type": "error", "msg": f"{type(e).__name__}: {e}"}


def transcribe_stream(
    video_path: Path,
    language: Optional[str] = None,
    display_title: Optional[str] = None,
    work_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
) -> Generator[dict, None, None]:
    """Yields progress events while transcribing. Final event carries the result.

    `work_dir` — destination folder for all artifacts. When omitted (CLI usage),
    defaults to `output/<title>/` for back-compat. The server always passes a
    job-scoped folder so jobs are fully isolated.

    Event shapes:
      {"type": "status",  "stage": "audio|model|transcribe|align", "msg": str}
      {"type": "segment", "start": float, "end": float, "text": str, "progress": float, "eta": float}
      {"type": "done",    "result": {...files written, summary...}}
      {"type": "error",   "msg": str}
    """
    t0 = time.time()
    title = sanitize_title(display_title or video_path.stem)
    out_dir = work_dir if work_dir is not None else (Path(__file__).parent / "output" / title)
    out_dir.mkdir(parents=True, exist_ok=True)

    # File names — title-prefixed so they're meaningful when downloaded
    md_raw_name   = f"{title} — Transcript.md"
    md_plain_name = f"{title} — Texte.md"
    srt_name      = f"{title} — Transcript.srt"
    json_name     = f"{title} — Transcript.json"

    try:
        yield {"type": "status", "stage": "audio", "msg": "Extraction de l'audio…"}
        audio_path = extract_audio(video_path, out_dir)

        try:
            repo = _settings.model_repo(model_name) if model_name else None
        except KeyError:
            repo = None   # unknown id → fall back to the default MLX repo

        # Probe duration up front: heartbeats only start reporting real chunk
        # progress once silero has run, so this is what fills the gap between
        # "Extraction" and the first chunk.
        audio_duration = _probe_audio_duration(audio_path)
        if audio_duration > 0:
            # Speed measured on this Mac for this model, not a fixed factor.
            eta_seconds = audio_duration / _settings.get_speed(
                model_name or _settings.get_default_model())
            yield {
                "type": "status",
                "stage": "audio",
                "msg": f"Audio : {_format_duration(audio_duration)} · Transcription estimée : ~{_format_duration(eta_seconds)}",
                "audio_duration": audio_duration,
                "eta_estimate": eta_seconds,
            }

        yield {
            "type": "status", "stage": "model",
            "msg": "Chargement du modèle MLX (GPU Metal) + silero-VAD…",
        }
        # First call to _run_whisper also lazily downloads the model from HuggingFace if missing.

        yield {
            "type": "status", "stage": "transcribe",
            "msg": "Découpage de la parole (silero-VAD)…",
        }
        segments, detected_language, speech_end = yield from _run_whisper_with_heartbeats(
            audio_path, language=language, model_repo=repo,
            vocabulary=_settings.get_vocabulary(),
        )

        # The transcriber can only report where the *speech* ended. On a
        # recording that finishes with music, an outro or plain silence that is
        # nowhere near the file's length: a 1h video whose talking stops at
        # 2min was announced as a 2min video, and the SRT/JSON carried it too.
        # ffprobe knows the real length, so prefer it and keep the speech end
        # only as a fallback for when probing failed.
        total_duration = audio_duration if audio_duration > 0 else speech_end

        yield {
            "type": "status",
            "stage": "transcribe",
            "msg": f"Langue : {detected_language} · Durée audio : {total_duration:.0f}s",
            "duration": total_duration,
            "language": detected_language,
        }

        # MLX returns all segments at once. Fan them out as SSE events so the
        # front-end's segment renderer keeps its existing contract.
        for seg in segments:
            progress = min(seg["end"] / total_duration, 1.0) if total_duration > 0 else 1.0
            yield {
                "type": "segment",
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "progress": progress,
                "eta": 0.0,
            }

        srt = build_srt(segments)
        md_raw = build_md_raw(segments)
        md_plain = build_md_plain(segments)

        (out_dir / srt_name).write_text(srt, encoding="utf-8")
        (out_dir / md_raw_name).write_text(md_raw, encoding="utf-8")
        (out_dir / md_plain_name).write_text(md_plain, encoding="utf-8")
        (out_dir / json_name).write_text(
            json.dumps({"title": title, "language": detected_language, "duration": total_duration, "segments": segments}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        elapsed_total = time.time() - t0
        yield {
            "type": "done",
            "result": {
                "title": title,
                "language": detected_language,
                "duration": total_duration,
                "elapsed": elapsed_total,
                "out_dir": str(out_dir),
                "files": {
                    "srt": srt_name,
                    "md_raw": md_raw_name,
                    "md_plain": md_plain_name,
                    "json": json_name,
                },
                "preview": md_raw[:2000],
            },
        }
    except Exception as e:
        log.exception("transcription en échec")   # the UI only gets one line; keep the traceback
        yield {"type": "error", "msg": f"{type(e).__name__}: {e}"}


def cli():
    if len(sys.argv) < 2:
        print("usage: transcribe.py <video_or_audio> [--lang fr|en|es|de] [--model <id>]")
        print("  known model ids: " + ", ".join(_settings.get_models().keys()))
        sys.exit(1)
    path = Path(sys.argv[1]).expanduser().resolve()
    lang = None
    model = None
    args = sys.argv[2:]
    while args:
        a = args.pop(0)
        if a == "--lang":
            lang = args.pop(0)
        elif a == "--model":
            model = args.pop(0)
    for ev in transcribe_stream(path, language=lang, model_name=model):
        if ev["type"] == "segment":
            sys.stdout.write(f"\r[{ev['progress']*100:5.1f}%] ETA {ev['eta']:6.0f}s  ")
            sys.stdout.flush()
        elif ev["type"] == "status":
            print(f"\n→ {ev['msg']}")
        elif ev["type"] == "done":
            print(f"\n✓ Fait en {ev['result']['elapsed']:.0f}s — sortie : {ev['result']['out_dir']}")
        elif ev["type"] == "error":
            print(f"\n✗ {ev['msg']}", file=sys.stderr)
            sys.exit(2)


if __name__ == "__main__":
    cli()
