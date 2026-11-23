#!/usr/bin/env python3
"""MCP server for Transcribe, over stdio. Lets Claude transcribe local files and
URLs and manage the default model, by driving the running app's HTTP API
(127.0.0.1:7878) — same engines, same default model, same vocabulary as the UI.

Stdlib only, so it runs with any python3. Registered in Claude Code with:
    claude mcp add --scope user transcribe -- python3 "<path>/mcp_transcribe.py"
TRANSCRIBE_URL overrides the target (tests). stdout carries JSON-RPC only.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

URL = os.environ.get("TRANSCRIBE_URL", "http://127.0.0.1:7878")
APP_START_TIMEOUT_S = 60
POLL_S = 2.0
# Hand control back before a client-side tool timeout can kill the call; the
# job keeps running in the app and suivre_transcription picks it up.
WAIT_MAX_S = 540
TEXT_MAX_CHARS = 200_000
LANGUAGES = ["auto", "fr", "en", "es", "de", "it", "pt"]


class ToolError(Exception):
    """A failure the model should read and act on (shown as isError)."""


# ---------- HTTP ----------

class Api:
    def __init__(self, base: str = URL):
        self.base = base.rstrip("/")

    def call(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail")
            except Exception:  # noqa: BLE001
                detail = None
            raise ToolError(detail or f"Transcribe a répondu HTTP {e.code}.")
        except urllib.error.URLError as e:
            raise ToolError(f"Transcribe injoignable sur {self.base} ({e.reason}).")
        return json.loads(raw) if raw else {}

    def ensure_running(self) -> None:
        """Open Transcribe.app if its server doesn't answer, then wait for it."""
        if self._healthy():
            return
        subprocess.run(["open", "-a", "Transcribe"], capture_output=True)
        deadline = time.time() + APP_START_TIMEOUT_S
        while time.time() < deadline:
            time.sleep(1)
            if self._healthy():
                return
        raise ToolError("Transcribe ne répond pas sur %s, même après l'avoir lancée." % self.base)

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(self.base + "/api/health", timeout=2):
                return True
        except Exception:  # noqa: BLE001
            return False


# ---------- tools ----------

_JOB_INPUTS = {
    "langue": {"type": "string", "enum": LANGUAGES, "default": "auto",
               "description": "Langue parlée. 'auto' = détection."},
    "modele": {"type": "string",
               "description": "Id de modèle (voir lister_modeles). Omis = modèle par défaut de l'app."},
    "dossier_sortie": {"type": "string",
                       "description": "Dossier absolu où copier .md/.srt/.json. Sans lui, les fichiers "
                                      "de l'app sont effacés au prochain job."},
}

TOOLS = [
    {
        "name": "transcrire_fichier",
        "description": "Transcrit un fichier audio/vidéo local (lu sur place, jamais modifié). "
                       "Attend la fin jusqu'à ~9 min ; au-delà rend un job_id pour suivre_transcription. "
                       "Si l'app transcrit déjà, le fichier attend son tour dans la file.",
        "inputSchema": {"type": "object", "required": ["chemin"], "properties": {
            "chemin": {"type": "string", "description": "Chemin absolu du fichier, ex. /Users/…/video.mp4"},
            **_JOB_INPUTS}},
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False},
    },
    {
        "name": "transcrire_url",
        "description": "Télécharge (yt-dlp) puis transcrit une URL : YouTube, Instagram, TikTok… "
                       "Même attente et même retour que transcrire_fichier.",
        "inputSchema": {"type": "object", "required": ["url"], "properties": {
            "url": {"type": "string", "description": "URL http(s) de la vidéo."},
            **_JOB_INPUTS}},
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
    },
    {
        "name": "suivre_transcription",
        "description": "Reprend l'attente d'un job rendu par transcrire_fichier/transcrire_url, "
                       "et rend le texte une fois fini.",
        "inputSchema": {"type": "object", "required": ["job_id"], "properties": {
            "job_id": {"type": "string"},
            "dossier_sortie": _JOB_INPUTS["dossier_sortie"]}},
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "lister_modeles",
        "description": "Modèles de transcription connus : id, moteur, installé ou non, lequel est le défaut.",
        "inputSchema": {"type": "object", "properties": {}},
        "annotations": {"readOnlyHint": True, "openWorldHint": False},
    },
    {
        "name": "choisir_modele_defaut",
        "description": "Change le modèle par défaut de l'app (doit être installé).",
        "inputSchema": {"type": "object", "required": ["id"], "properties": {
            "id": {"type": "string", "description": "Id de modèle, voir lister_modeles."}}},
        "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
    },
]


def _job_body(args: dict) -> dict:
    langue = args.get("langue") or "auto"
    if langue not in LANGUAGES:
        raise ToolError(f"langue inconnue : {langue!r}. Valeurs : {', '.join(LANGUAGES)}.")
    return {"language": langue, "model": args.get("modele") or None}


def _out_dir(args: dict) -> Path | None:
    raw = args.get("dossier_sortie")
    if not raw:
        return None
    d = Path(raw).expanduser()
    if not d.is_absolute():
        raise ToolError("dossier_sortie doit être un chemin absolu.")
    return d


def wait_for(api: Api, job_id: str, out_dir: Path | None) -> dict:
    deadline = time.time() + WAIT_MAX_S
    while True:
        st = api.call("GET", f"/api/jobs/{job_id}")
        if st.get("done"):
            break
        if time.time() >= deadline:
            return {"job_id": job_id, "termine": False, "etape": st.get("msg"),
                    "progression": st.get("progress"),
                    "suite": "Toujours en cours : rappelle suivre_transcription avec ce job_id."}
        time.sleep(POLL_S)
    if st.get("error"):
        raise ToolError(f"La transcription a échoué : {st['error']}")
    return _result(job_id, st["result"], out_dir)


def _result(job_id: str, res: dict, out_dir: Path | None) -> dict:
    src = Path(res["out_dir"])
    files = {k: src / v for k, v in (res.get("files") or {}).items() if v}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {k: Path(shutil.copy2(p, out_dir / p.name)) for k, p in files.items()}
    text = files["md_plain"].read_text(encoding="utf-8") if "md_plain" in files else ""
    truncated = len(text) > TEXT_MAX_CHARS
    return {
        "job_id": job_id, "termine": True,
        "titre": res.get("title"), "langue": res.get("language"),
        "duree_s": res.get("duration"), "calcul_s": round(res.get("elapsed") or 0, 1),
        "fichiers": {k: str(p) for k, p in files.items()},
        "texte": text[:TEXT_MAX_CHARS] + ("\n[… tronqué, texte complet dans fichiers.md_plain]" if truncated else ""),
    }


def call_tool(api: Api, name: str, args: dict) -> dict:
    if name in ("transcrire_fichier", "transcrire_url", "suivre_transcription"):
        api.ensure_running()
    if name == "transcrire_fichier":
        chemin = str(Path(args.get("chemin") or "").expanduser())
        job = api.call("POST", "/api/jobs/local", {"path": chemin, **_job_body(args)})
        return wait_for(api, job["job_id"], _out_dir(args))
    if name == "transcrire_url":
        job = api.call("POST", "/api/jobs/url", {"url": args.get("url") or "", "require_idle": True, **_job_body(args)})
        return wait_for(api, job["job_id"], _out_dir(args))
    if name == "suivre_transcription":
        return wait_for(api, str(args.get("job_id") or ""), _out_dir(args))
    if name == "lister_modeles":
        api.ensure_running()
        s = api.call("GET", "/api/settings")
        return {"defaut": s["default_model"], "modeles": [
            {k: m[k] for k in ("id", "label", "engine", "installed", "is_default", "custom", "size_bytes")}
            for m in s["models"]]}
    if name == "choisir_modele_defaut":
        api.ensure_running()
        models = {m["id"]: m for m in api.call("GET", "/api/settings")["models"]}
        model = models.get(args.get("id"))
        if model is None:
            raise ToolError(f"Modèle inconnu : {args.get('id')!r}. Ids : {', '.join(models)}.")
        if not model["installed"]:
            raise ToolError(f"{model['label']} n'est pas téléchargé : télécharge-le dans Réglages de l'app.")
        api.call("POST", "/api/settings/model", {"id": model["id"]})
        return {"defaut": model["id"]}
    raise ToolError(f"Outil inconnu : {name}")


# ---------- JSON-RPC ----------

def handle(msg: dict, api: Api) -> dict | None:
    mid, method = msg.get("id"), msg.get("method")
    if mid is None:
        return None   # notification (e.g. notifications/initialized)
    if method == "initialize":
        result = {"protocolVersion": (msg.get("params") or {}).get("protocolVersion", "2025-06-18"),
                  "capabilities": {"tools": {}},
                  "serverInfo": {"name": "transcribe", "version": "1.0.0"}}
    elif method == "ping":
        result = {}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        p = msg.get("params") or {}
        try:
            data = call_tool(api, p.get("name"), p.get("arguments") or {})
            result = {"content": [{"type": "text", "text": json.dumps(data, ensure_ascii=False, indent=1)}],
                      "structuredContent": data, "isError": False}
        except ToolError as e:
            result = {"content": [{"type": "text", "text": str(e)}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"Méthode inconnue : {method}"}}
    return {"jsonrpc": "2.0", "id": mid, "result": result}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    api = Api()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            msg, rep = None, {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "JSON illisible."}}
        try:
            if msg is not None:
                rep = handle(msg, api)
        except Exception:  # noqa: BLE001 — one bad message must not kill the session
            traceback.print_exc(file=sys.stderr)
            rep = {"jsonrpc": "2.0", "id": msg.get("id") if isinstance(msg, dict) else None,
                   "error": {"code": -32603, "message": "Erreur interne."}}
        if rep is not None:
            sys.stdout.write(json.dumps(rep, ensure_ascii=False) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
