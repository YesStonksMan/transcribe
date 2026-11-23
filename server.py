"""FastAPI server — uploads a video, runs transcribe_stream in a thread,
streams progress events to the browser via Server-Sent Events.

Run:  python server.py    (defaults to http://localhost:7878)
"""

from __future__ import annotations

import asyncio
import json
import logging
import logging.handlers
import os
import queue
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from transcribe import (
    transcribe_stream,
    transcribe_url_stream,
    ytdlp_available,
    sanitize_title,
    vad_available,
    _probe_audio_duration,
    _external_env,
)

import os as _os
import settings as _settings_mod
import subprocess as _subprocess
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import RepositoryNotFoundError

ROOT = Path(__file__).parent
# When bundled (py2app), __file__ lives inside python311.zip — not a writable
# location. Honour env overrides so the native app can point to writable
# user-data + bundle-resource paths.
STATIC_ROOT = Path(_os.environ.get("TRANSCRIBE_STATIC_DIR", str(ROOT / "static")))
DATA_ROOT   = Path(_os.environ.get("TRANSCRIBE_DATA_DIR",   str(ROOT)))
UPLOAD_DIR = DATA_ROOT / "uploads"
OUTPUT_DIR = DATA_ROOT / "output"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("transcribe")


def _setup_logging() -> None:
    """Rotating file log in ~/Library/Logs/Transcribe/. The bundled app has no
    console: without this, a failed job left no trace anywhere."""
    log_dir = Path.home() / "Library" / "Logs" / "Transcribe"
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    if any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        return
    handler = logging.handlers.RotatingFileHandler(
        log_dir / "transcribe.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)      # warnings from uvicorn and libraries land here too
    log.setLevel(logging.INFO)


# ---------- Job workspace helpers ----------

def job_upload_dir(job_id: str) -> Path:
    """Per-job sub-folder under uploads/ that holds the source media."""
    return UPLOAD_DIR / job_id


def job_output_dir(job_id: str) -> Path:
    """Per-job sub-folder under output/ that holds transcription artifacts."""
    return OUTPUT_DIR / job_id


_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")


def is_valid_job_id(s: str) -> bool:
    """Validate a job_id so it can never be used for path traversal."""
    return bool(_JOB_ID_RE.match(s or ""))


def cleanup_job(job_id: str) -> None:
    """Remove a job from memory and wipe its on-disk artifacts.

    Idempotent and safe — silently no-ops on invalid or unknown job_ids.
    """
    if not is_valid_job_id(job_id):
        return
    JOBS.pop(job_id, None)
    for d in (job_upload_dir(job_id), job_output_dir(job_id)):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


def cleanup_orphans() -> None:
    """Remove any uploads/<job_id>/ or output/<job_id>/ folder whose job_id
    isn't currently in the JOBS dict. Called at server startup so that
    'tout supprimer après transcription' is honored across restarts.
    Only touches folders matching the job_id pattern — leaves legacy /
    unrelated files alone.
    """
    for parent in (UPLOAD_DIR, OUTPUT_DIR):
        if not parent.exists():
            continue
        for entry in parent.iterdir():
            if entry.is_dir() and is_valid_job_id(entry.name) and entry.name not in JOBS:
                shutil.rmtree(entry, ignore_errors=True)


def _drop_job_audio(job_id: str) -> None:
    """Delete the job's audio once transcription ends: the downloaded/uploaded
    copy (uploads/<job_id>/) and the extracted .wav. Keeps only the transcript
    files listed in the result. Never touches a local file read in place."""
    shutil.rmtree(job_upload_dir(job_id), ignore_errors=True)
    out_dir = job_output_dir(job_id)
    if not out_dir.is_dir():
        return
    keep = set(((JOBS.get(job_id) or {}).get("result") or {}).get("files", {}).values())
    for f in out_dir.iterdir():
        if f.is_file() and f.name not in keep:
            f.unlink(missing_ok=True)


# One decode at a time: every job waits in this FIFO for the single runner
# thread. Two decodes at once share one GPU (no gain) and double the memory —
# and a second job used to delete the first one's files under its feet.
_RUN_QUEUE: queue.Queue = queue.Queue()
_JOB_LOCK = threading.Lock()
_LAST_MCP_JOB: Optional[str] = None


def _open_job(origin: str, title: str, kind: str, timestamps: bool = True,
              model: Optional[str] = None, source: Optional[Path] = None) -> str:
    """Create a queued job. The window's jobs stay listed until the user removes
    them. An MCP job replaces the previous *finished* MCP job (its caller has
    read it); the two origins never erase each other's results."""
    global _LAST_MCP_JOB
    job_id = _new_job()
    JOBS[job_id].update(origin=origin, title=title, kind=kind, timestamps=timestamps, model=model)
    if source is not None:
        # Known length → a waiting item can already say how long it will take.
        JOBS[job_id]["audio_s"] = _probe_audio_duration(source) or None
    if origin == "mcp":
        with _JOB_LOCK:
            previous, _LAST_MCP_JOB = _LAST_MCP_JOB, job_id
        if previous and (JOBS.get(previous) or {}).get("done"):
            cleanup_job(previous)
    return job_id


def _autosave(job: dict) -> None:
    """Copy a finished window job's text into the user's folder: timestamped
    or flowing .md, as chosen when it was queued. Never overwrites."""
    res = job.get("result") or {}
    name = (res.get("files") or {}).get("md_raw" if job.get("timestamps", True) else "md_plain")
    if not name:
        return
    out_dir = _settings_mod.get_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    dest, n = out_dir / f"{res['title']}.md", 2
    while dest.exists():
        dest, n = out_dir / f"{res['title']} ({n}).md", n + 1
    shutil.copy2(Path(res["out_dir"]) / name, dest)
    job["saved"] = str(dest)


def _eta(job: dict) -> Optional[float]:
    """Seconds of computing left for a job, or None if its length is unknown
    (a link not downloaded yet). Once chunks are flowing this is the measured
    chunk rate; before that, audio length / the model's measured speed."""
    if job.get("done"):
        return None
    if job.get("remaining") is not None:
        return max(0.0, job["remaining"] - (time.time() - job["remaining_at"]))
    if not job.get("audio_s"):
        return None
    spent = time.time() - job["t_audio"] if job.get("t_audio") else 0.0
    return max(0.0, job["audio_s"] / _settings_mod.get_speed(job.get("model")) - spent)


def _job_view(job_id: str, job: dict) -> dict:
    res = job.get("result") or {}
    state = ("error" if job.get("error") else "done" if job.get("done")
             else "running" if job.get("started") else "queued")
    return {
        "id": job_id, "title": res.get("title") or job.get("title") or "",
        "kind": job.get("kind", "file"), "state": state, "msg": job.get("msg"),
        "progress": job.get("progress"), "error": job.get("error"),
        "duration": res.get("duration"), "saved": job.get("saved"), "eta": _eta(job),
    }


app = FastAPI(title="Transcribe")


# The app's WKWebView keeps a persistent HTTP cache (private_mode=False), and
# without Cache-Control it heuristically reuses the old index.html after an
# app rebuild — the UI looked unchanged until the cache was wiped. no-cache
# forces a (cheap, local) revalidation on every load.
@app.middleware("http")
async def _no_cache(request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.on_event("startup")
def _on_startup():
    _setup_logging()
    cleanup_orphans()
    if not vad_available():
        log.warning("VAD absent au démarrage : les transcriptions tourneront en fichier entier.")

# Job state: id -> queue of events, plus a "done" marker.
JOBS: dict[str, dict] = {}


# Download / install jobs use a parallel registry. Same queue-based pattern as JOBS,
# but isolated so cleanup_orphans doesn't touch them.
DOWNLOAD_JOBS: dict[str, dict] = {}


def _new_download_job() -> str:
    job_id = uuid.uuid4().hex[:12]
    DOWNLOAD_JOBS[job_id] = {"queue": queue.Queue(), "done": False, "error": None}
    return job_id


@app.get("/")
def root():
    return FileResponse(STATIC_ROOT / "index.html")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "url_enabled": ytdlp_available(),
        "vad": vad_available(),
    }


@app.get("/api/settings")
def get_settings():
    return {
        "default_model": _settings_mod.get_default_model(),
        "models": _settings_mod.list_models(),
        # The speech splitter is not a brew tool, but its absence degrades every
        # job — list it with the others so a missing one is visible.
        "deps": _settings_mod.list_deps() + [{
            "id": "vad", "label": "Découpage de la parole", "installed": vad_available(),
            "version": None, "brew_installable": False,
        }],
        "brew_available": _settings_mod.brew_available(),
        "vocabulary": _settings_mod.get_vocabulary(),
        "vocabulary_max_chars": _settings_mod.VOCABULARY_MAX_CHARS,
        "output_dir": str(_settings_mod.get_output_dir()),
    }


@app.post("/api/settings/output-dir")
def set_output_dir(payload: dict = Body(...)):
    try:
        _settings_mod.set_output_dir(payload.get("path") or "")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(status_code=204)


@app.post("/api/open-output-dir")
def open_output_dir():
    """Reveal the auto-save folder in Finder (created if it does not exist yet)."""
    d = _settings_mod.get_output_dir()
    d.mkdir(parents=True, exist_ok=True)
    _subprocess.run(["open", str(d)], check=False, timeout=10)
    return Response(status_code=204)


@app.post("/api/settings/model")
def set_default_model(payload: dict = Body(...)):
    model_id = payload.get("id")
    try:
        _settings_mod.set_default_model(model_id)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(status_code=204)


@app.post("/api/settings/vocabulary")
def set_vocabulary(payload: dict = Body(...)):
    try:
        _settings_mod.set_vocabulary(payload.get("text") or "")
    except ValueError as e:
        raise HTTPException(400, str(e))
    return Response(status_code=204)


def _hf_repo_size(repo: str) -> int:
    """Total size of a HuggingFace repo's files, in bytes. Doubles as the
    existence check. Raises LookupError with a user-facing message."""
    try:
        info = HfApi().model_info(repo, files_metadata=True)
    except RepositoryNotFoundError:
        raise LookupError("Repo introuvable sur HuggingFace (ou privé).")
    except Exception as e:
        raise LookupError(f"Impossible de joindre HuggingFace : {type(e).__name__}.")
    return sum(f.size or 0 for f in (info.siblings or []))


@app.post("/api/models")
def add_model(payload: dict = Body(...)):
    repo = (payload.get("repo") or "").strip()
    try:
        repo = _settings_mod.validate_custom_repo(repo)
        size = _hf_repo_size(repo)
        model_id = _settings_mod.add_custom_model(repo, size)
    except (ValueError, LookupError) as e:
        raise HTTPException(400, str(e))
    return {"id": model_id}


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    spec = _settings_mod.get_models().get(model_id)
    if spec is None:
        raise HTTPException(404, f"Unknown model: {model_id}")
    if model_id == _settings_mod.get_default_model():
        raise HTTPException(409, "Modèle actuellement sélectionné — change le default d'abord.")
    _settings_mod.uninstall_model(model_id)
    if spec.custom:
        _settings_mod.remove_custom_model(model_id)
    return Response(status_code=204)


@app.post("/api/models/{model_id}/download")
def post_download_model(model_id: str):
    spec = _settings_mod.get_models().get(model_id)
    if spec is None:
        raise HTTPException(404, f"Unknown model: {model_id}")
    needed = int(spec.size_bytes * 1.2)  # 20% margin for temp files
    if _settings_mod.disk_free_bytes(Path.home()) < needed:
        raise HTTPException(507, "Espace disque insuffisant pour ce téléchargement.")

    job_id = _new_download_job()

    def worker():
        q: queue.Queue = DOWNLOAD_JOBS[job_id]["queue"]
        try:
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
            q.put({"type": "progress", "pct": 0.0, "msg": "Démarrage du téléchargement…"})
            for repo in spec.repos:   # + the forced aligner for Qwen3-ASR
                snapshot_download(repo_id=repo, local_files_only=False)
            q.put({"type": "progress", "pct": 100.0, "msg": "Téléchargement terminé."})
            q.put({"type": "done"})
        except Exception as e:
            DOWNLOAD_JOBS[job_id]["error"] = str(e)
            q.put({"type": "error", "msg": f"{type(e).__name__}: {e}"})
        finally:
            DOWNLOAD_JOBS[job_id]["done"] = True
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/models/{model_id}/download/{job_id}/stream")
async def stream_download(model_id: str, job_id: str, request: Request):
    if job_id not in DOWNLOAD_JOBS:
        raise HTTPException(404, "Unknown download job")
    return _sse(DOWNLOAD_JOBS[job_id], request)


@app.post("/api/deps/{dep_id}/install")
def post_install_dep(dep_id: str):
    if dep_id not in _settings_mod.DEPS:
        raise HTTPException(404, f"Unknown dep: {dep_id}")
    spec = _settings_mod.DEPS[dep_id]
    if not spec.brew_installable:
        raise HTTPException(400, f"{spec.label} ne s'installe pas via brew — install manuel requis.")
    if not _settings_mod.brew_available():
        raise HTTPException(424, "Homebrew non détecté sur cette machine.")

    job_id = _new_download_job()

    def worker():
        q: queue.Queue = DOWNLOAD_JOBS[job_id]["queue"]
        try:
            q.put({"type": "progress", "pct": 0.0, "msg": f"brew install {dep_id}…"})
            proc = _subprocess.Popen(
                ["brew", "install", dep_id],
                stdout=_subprocess.PIPE, stderr=_subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace",
                env=_external_env(),
            )
            for line in proc.stdout:
                q.put({"type": "progress", "msg": line.rstrip()})
            proc.wait()
            if proc.returncode == 0 and shutil.which(dep_id):
                q.put({"type": "done"})
            else:
                q.put({"type": "error", "msg": f"brew exit {proc.returncode}"})
        except Exception as e:
            q.put({"type": "error", "msg": f"{type(e).__name__}: {e}"})
        finally:
            DOWNLOAD_JOBS[job_id]["done"] = True
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/deps/{dep_id}/install/{job_id}/stream")
async def stream_dep_install(dep_id: str, job_id: str, request: Request):
    if job_id not in DOWNLOAD_JOBS:
        raise HTTPException(404, "Unknown install job")
    return _sse(DOWNLOAD_JOBS[job_id], request)


def _sse(job: dict, request: Request) -> StreamingResponse:
    """Stream a job's event queue as Server-Sent Events.

    The read times out every second so the reader can leave: when the client
    is gone (a blocking q.get used to pin an executor thread forever and then
    swallow the next event), or when the job is done and drained (a second
    reader, after the sentinel was consumed, used to hang for good). `done` is
    set after the last real event is queued, so done + empty loses nothing."""
    q: queue.Queue = job["queue"]

    async def gen():
        loop = asyncio.get_running_loop()
        while True:
            try:
                ev = await loop.run_in_executor(None, q.get, True, 1.0)
            except queue.Empty:
                if (job["done"] and q.empty()) or await request.is_disconnected():
                    break
                continue
            if ev is None:
                break
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        gen(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                 "Connection": "keep-alive"},
    )


def _start_worker(job_id: str, stream_factory, what: str = ""):
    """Queue the job; the single runner thread takes them in order."""
    # Pass the dict itself: a job removed while it waits is no longer in JOBS.
    _RUN_QUEUE.put((job_id, JOBS[job_id], stream_factory, what))


def _runner():
    while True:
        job_id, job, stream_factory, what = _RUN_QUEUE.get()
        if JOBS.get(job_id) is job:      # else: removed from the list while waiting
            _run_job(job_id, job, stream_factory, what)


threading.Thread(target=_runner, daemon=True).start()


def _run_job(job_id: str, job: dict, stream_factory, what: str) -> None:
    """Consume a transcription generator and push its events to the job queue."""
    q: queue.Queue = job["queue"]
    started = time.time()
    job["started"] = True
    log.info("job %s démarre : %s", job_id, what)

    try:
        for ev in stream_factory():
            # Capture downloaded path so the download endpoint can resolve outputs.
            if ev.get("type") == "downloaded":
                job["file"] = ev["path"]
                continue  # internal event, don't forward to client
            q.put(ev)
            if ev.get("audio_duration"):
                job["audio_s"], job["t_audio"] = ev["audio_duration"], time.time()
            if ev.get("remaining") is not None:
                job["remaining"], job["remaining_at"] = ev["remaining"], time.time()
            if ev["type"] == "done":
                job["result"] = ev["result"]
                _settings_mod.record_speed(job.get("model"), ev["result"].get("duration") or 0,
                                           ev["result"].get("elapsed") or 0)
                log.info("job %s terminé en %.1fs (audio %.0fs)", job_id,
                         time.time() - started, ev["result"].get("duration") or 0)
                if job.get("origin") == "ui":
                    try:
                        _autosave(job)
                    except Exception as e:
                        log.exception("job %s : enregistrement automatique en échec", job_id)
                        job["save_error"] = f"{type(e).__name__}: {e}"
            elif ev["type"] == "error":
                job["error"] = ev.get("msg")
                log.error("job %s en échec après %.1fs : %s", job_id,
                          time.time() - started, ev.get("msg"))
            elif ev.get("msg"):
                # Last progress line, for GET /api/jobs/{id} (the MCP polls
                # it; the SSE queue can only be consumed once).
                job["msg"] = ev["msg"]
                if ev.get("progress") is not None:
                    job["progress"] = ev["progress"]
    except Exception as e:
        log.exception("job %s a planté", job_id)
        job["error"] = f"{type(e).__name__}: {e}"
        q.put({"type": "error", "msg": job["error"]})
    finally:
        _drop_job_audio(job_id)
        job["done"] = True
        q.put(None)


def _new_job(initial_file: Optional[str] = None) -> str:
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "queue": queue.Queue(),
        "done": False,
        "result": None,
        "error": None,
        "msg": None,
        "progress": None,
        "file": initial_file,
        "origin": "ui", "title": "", "kind": "file", "timestamps": True,
        "started": False, "saved": None,
    }
    return job_id


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    timestamps: Optional[str] = Form(None),
):
    original = Path(file.filename or "upload")
    safe_disk_name = "".join(c for c in original.name if c.isalnum() or c in "._- ") or "upload"
    model_id = model if model in _settings_mod.get_models() else _settings_mod.get_default_model()
    job_id = _open_job("ui", sanitize_title(original.stem), "file", timestamps != "false", model_id)

    upload_dir = job_upload_dir(job_id)
    dest = upload_dir / safe_disk_name
    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            while chunk := await file.read(1 << 20):
                f.write(chunk)
    except Exception:
        # A job that never starts would stay "running" and 409 every later one.
        log.exception("job %s : copie du fichier déposé en échec", job_id)
        cleanup_job(job_id)
        raise
    JOBS[job_id]["file"] = str(dest)
    JOBS[job_id]["audio_s"] = _probe_audio_duration(dest) or None

    display_title = sanitize_title(original.stem)
    lang = language if language not in (None, "", "auto") else None
    _start_worker(
        job_id,
        lambda: transcribe_stream(
            dest, language=lang,
            display_title=display_title, work_dir=job_output_dir(job_id),
            model_name=model_id,
        ),
        f"interface, fichier « {original.name} », modèle {model_id}, langue {lang or 'auto'}",
    )
    return {"job_id": job_id}


class UrlJobRequest(BaseModel):
    url: str
    language: Optional[str] = None
    model: Optional[str] = None
    require_idle: bool = False   # sent by the MCP: marks the job as its own, not the window's
    timestamps: bool = True      # which .md the window's auto-save keeps


def _refuse_if_busy() -> None:
    if any(not j["done"] for j in JOBS.values()):
        raise HTTPException(409, "Une transcription est déjà en cours — réessaie quand elle est finie.")


@app.post("/api/jobs/url")
def create_url_job(req: UrlJobRequest):
    if not ytdlp_available():
        raise HTTPException(503, "yt-dlp introuvable sur le PATH")
    if not (req.url.startswith("http://") or req.url.startswith("https://")):
        raise HTTPException(400, "URL invalide (doit commencer par http(s)://)")
    origin = "mcp" if req.require_idle else "ui"
    model_id = req.model if req.model in _settings_mod.get_models() else _settings_mod.get_default_model()
    job_id = _open_job(origin, req.url, "url", req.timestamps, model_id)
    lang = req.language if req.language not in (None, "", "auto") else None
    _start_worker(
        job_id,
        lambda: transcribe_url_stream(
            req.url,
            language=lang,
            dl_dir=job_upload_dir(job_id),
            work_dir=job_output_dir(job_id),
            model_name=model_id,
        ),
        f"{origin}, lien {req.url}, modèle {model_id}, langue {lang or 'auto'}",
    )
    return {"job_id": job_id}


class LocalJobRequest(BaseModel):
    path: str
    language: Optional[str] = None
    model: Optional[str] = None
    origin: str = "mcp"          # "ui": a file picked or dropped in the window, read in place
    timestamps: bool = True


@app.post("/api/jobs/local")
def create_local_job(req: LocalJobRequest):
    """Transcribe a file already on this Mac, read in place (no upload copy).
    Used by the MCP server."""
    src = Path(req.path).expanduser()
    if not src.is_absolute():
        raise HTTPException(400, "Chemin absolu attendu.")
    if not src.is_file():
        raise HTTPException(404, f"Fichier introuvable : {src}")
    origin = "ui" if req.origin == "ui" else "mcp"
    model_id = req.model if req.model in _settings_mod.get_models() else _settings_mod.get_default_model()
    job_id = _open_job(origin, sanitize_title(src.stem), "file", req.timestamps, model_id, src)
    lang = req.language if req.language not in (None, "", "auto") else None
    _start_worker(
        job_id,
        lambda: transcribe_stream(
            src, language=lang,
            display_title=sanitize_title(src.stem), work_dir=job_output_dir(job_id),
            model_name=model_id,
        ),
        f"{origin}, fichier {src}, modèle {model_id}, langue {lang or 'auto'}",
    )
    return {"job_id": job_id, "model": model_id}


# ponytail: fixed model, read get_default_model() if dictation ever needs another engine.
DICTATE_MODEL = "mlx-community/whisper-large-v3-turbo"


def _dictate_decode(wav: Path, lang: Optional[str]) -> list[dict]:
    """One Whisper pass over a dictation clip.
    No word timestamps: 45 % of the time on a 5 s clip (median 532 -> 296 ms,
    12-10) and a dictation never needs them. No initial_prompt: on a few-second
    clip the vocabulary sentence made Whisper loop ("...", "정말 정말…", 12-10),
    so the vocabulary is returned for the formatter to apply instead."""
    import mlx_whisper
    return mlx_whisper.transcribe(
        str(wav), path_or_hf_repo=DICTATE_MODEL, language=lang,
        word_timestamps=False, condition_on_previous_text=False,
    )["segments"]


@app.post("/api/dictate")
async def dictate(request: Request, lang: str = "fr"):
    """Push-to-talk clip from Mini-Flow: raw WAV body in, text out.
    Synchronous and job-free, so it never wipes a running video job — but it
    refuses while one runs rather than share the GPU with it.
    # ponytail: 409 during a video job, a dedicated process if that gets in the way."""
    import tempfile, time
    wav = await request.body()
    if not wav:
        raise HTTPException(400, "Corps WAV vide.")
    _refuse_if_busy()
    lang = None if lang in ("", "auto") else lang
    started = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav)
    try:
        segments = await asyncio.to_thread(_dictate_decode, Path(f.name), lang)
    finally:
        Path(f.name).unlink(missing_ok=True)
    text = " ".join(s["text"].strip() for s in segments if s["text"].strip())
    return {
        "text": text,
        "vocabulary": _settings_mod.get_vocabulary(),
        "ms": int((time.perf_counter() - started) * 1000),
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return {**{k: job[k] for k in ("done", "result", "error", "msg", "progress")},
            "state": _job_view(job_id, job)["state"], "saved": job.get("saved")}


@app.get("/api/jobs")
def list_jobs():
    """The window's queue, in the order it was filled. MCP jobs are not listed
    (they belong to a Claude session) but `busy_elsewhere` says one is decoding,
    so waiting items do not look stuck for no reason."""
    return {
        "jobs": [_job_view(i, j) for i, j in list(JOBS.items()) if j.get("origin", "ui") == "ui"],
        "busy_elsewhere": any(j.get("origin") == "mcp" and not j.get("done") for j in list(JOBS.values())),
        "output_dir": str(_settings_mod.get_output_dir()),
    }


@app.get("/api/jobs/{job_id}/stream")
async def stream(job_id: str, request: Request):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return _sse(JOBS[job_id], request)


@app.get("/api/jobs/{job_id}/file/{filename}")
def download(job_id: str, filename: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    result = JOBS[job_id].get("result")
    if not result:
        raise HTTPException(409, "not ready")
    # Allowlist built from the result itself — only files we actually generated.
    allowed = {v for v in result.get("files", {}).values() if v}
    if filename not in allowed:
        raise HTTPException(400, "invalid filename")
    out_dir = Path(result["out_dir"]).resolve()
    target = (out_dir / filename).resolve()
    # Defensive: confirm target stays within OUTPUT_DIR after path resolution.
    if not str(target).startswith(str(OUTPUT_DIR.resolve())):
        raise HTTPException(400, "invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(target, filename=filename, media_type="application/octet-stream")


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str):
    job = JOBS.get(job_id)
    if job and job.get("started") and not job.get("done"):
        # No way to stop a decode mid-flight; wiping its files would only make it fail.
        raise HTTPException(409, "Cet élément est en cours de transcription.")
    cleanup_job(job_id)
    return None  # 204 No Content


# Static assets (served at /static/*)
app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7878))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
