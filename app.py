"""Transcribe — native Mac app entry point.

Boots the FastAPI server in a daemon thread on localhost:7878, then opens
a native WKWebView window pointed at it. When the window closes, the daemon
thread dies with the process and the server stops.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib import error as url_error
from urllib import request as url_request
from urllib.parse import quote


def _augment_path_for_gui_launch() -> None:
    """macOS GUI apps (launched from Dock / Launchpad / .app double-click) get
    a minimal PATH (`/usr/bin:/bin:/usr/sbin:/sbin`) that omits Homebrew and
    user-local bins. Without this, ffmpeg and yt-dlp — both installed
    via brew at /opt/homebrew/bin — are not found and the pipeline crashes
    with `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`.

    Two layers of fallback:
      1. Prepend the standard Homebrew + user-local locations unconditionally.
      2. Ask the user's interactive shell what its PATH is — captures any
         custom rc-file additions (pyenv, asdf, mise, pipx, etc.). Best-effort,
         silently skipped if it fails.
    """
    # Standard locations to always try, in priority order. Apple Silicon brew
    # lives at /opt/homebrew, Intel brew at /usr/local.
    standard = [
        "/opt/homebrew/bin",
        "/opt/homebrew/sbin",
        "/usr/local/bin",
        "/usr/local/sbin",
        os.path.expanduser("~/.local/bin"),
    ]

    current = os.environ.get("PATH", "")
    current_parts = current.split(":") if current else []

    # Try the user's interactive shell PATH (best source of truth).
    shell = os.environ.get("SHELL") or "/bin/zsh"
    try:
        r = subprocess.run(
            [shell, "-l", "-i", "-c", "printf %s \"$PATH\""],
            capture_output=True, text=True, timeout=3,
        )
        shell_path = (r.stdout or "").strip()
        if shell_path:
            for p in shell_path.split(":"):
                if p and p not in current_parts:
                    current_parts.append(p)
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        pass  # Fall through to the hardcoded list only.

    # Always also prepend the standard locations (in case the shell hack failed
    # or the user has an unusual login shell config that omits them).
    for p in reversed(standard):
        if p not in current_parts:
            current_parts.insert(0, p)

    os.environ["PATH"] = ":".join(current_parts)


_augment_path_for_gui_launch()

# Configure data + static paths BEFORE importing server, which reads these on
# module load. When bundled inside Transcribe.app, server's __file__ lives in
# python311.zip — not writable, no static/. We point it at the bundle's
# Resources/static/ (read-only assets) and ~/Library/Application Support/
# Transcribe/ (writable user data).
_RESOURCE_PATH = os.environ.get("RESOURCEPATH")  # set by py2app __boot__
if _RESOURCE_PATH:
    os.environ.setdefault(
        "TRANSCRIBE_STATIC_DIR", os.path.join(_RESOURCE_PATH, "static")
    )
_USER_DATA = os.path.expanduser("~/Library/Application Support/Transcribe")
Path(_USER_DATA).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TRANSCRIBE_DATA_DIR", _USER_DATA)

import uvicorn
import webview

# WKWebView cancels any navigation whose response is an attachment / non-
# renderable MIME type (every "Télécharger" link is application/octet-stream
# with Content-Disposition: attachment). pywebview gates downloads behind
# ALLOW_DOWNLOADS, which defaults to False — so without this line, clicking a
# download link inside the bundled app silently does nothing. Enabling it makes
# WKWebView show a native save panel (defaulting to ~/Downloads).
webview.settings["ALLOW_DOWNLOADS"] = True

from server import app

HOST = "127.0.0.1"
PORT = 7878
HEALTH_URL = f"http://{HOST}:{PORT}/api/health"
STARTUP_TIMEOUT_SEC = 10.0
POLL_INTERVAL_SEC = 0.1


class Api:
    """JS bridge exposed to the page as ``window.pywebview.api``.

    WKWebView's native download path (an ``<a download>`` click → WKDownload →
    save panel, gated behind ALLOW_DOWNLOADS) silently fails on this build:
    clicking a download link cancels the navigation and writes nothing. The
    server is local and the generated file already exists on disk, so we
    sidestep WKWebView's download subsystem entirely.

    The frontend calls ``save_file``; Python fetches the bytes from the
    *already validated* download endpoint (allowlist + path-traversal checks
    live there — we don't re-touch the filesystem), lets the user pick a
    destination via a native save panel, writes the bytes there, and reveals
    the result in Finder. The panel is forced to open in ~/Downloads: by
    default NSSavePanel reuses its last-used folder, which once silently
    buried saved files in an opaque internal job directory the user never
    checks — the whole reason downloads seemed to "do nothing".
    """

    def __init__(self) -> None:
        self._window = None  # set after create_window returns

    def save_file(self, job_id: str, filename: str) -> dict:
        """Fetch a generated job file and save it where the user picks.

        Returns a small dict the frontend can react to:
          {"ok": True,  "path": "/Users/.../chosen/Café.md"}
          {"ok": False, "reason": "cancelled" | "fetch: ..." | "write: ..."}
        """
        url = (
            f"http://{HOST}:{PORT}/api/jobs/"
            f"{quote(job_id)}/file/{quote(filename)}"
        )
        try:
            with url_request.urlopen(url, timeout=30) as r:
                data = r.read()
        except (url_error.URLError, OSError) as e:
            return {"ok": False, "reason": f"fetch: {e}"}

        if self._window is None:  # defensive — should never happen at runtime
            return {"ok": False, "reason": "no window"}

        # Force the panel to open in ~/Downloads. Without an explicit directory
        # NSSavePanel reuses its last-used folder — that's how files ended up
        # saved into an internal job directory the user never thinks to open.
        dest = self._window.create_file_dialog(
            webview.FileDialog.SAVE,
            directory=str(Path.home() / "Downloads"),
            save_filename=filename,
        )
        if not dest:  # user cancelled the panel
            return {"ok": False, "reason": "cancelled"}
        if isinstance(dest, (list, tuple)):  # some backends return a sequence
            dest = dest[0]

        try:
            Path(dest).write_bytes(data)
        except OSError as e:
            return {"ok": False, "reason": f"write: {e}"}

        # Reveal in Finder so the saved file is immediately visible.
        try:
            subprocess.run(["open", "-R", str(dest)], check=False, timeout=10)
        except (OSError, subprocess.SubprocessError):
            pass  # the file is already saved; revealing is a nicety

        return {"ok": True, "path": str(dest)}

    def pick_files(self) -> list:
        """Native open panel, several files at once. Returns real paths: the
        server then reads each file in place — a browser file input only hands
        over the bytes, which meant copying every multi-GB video before starting."""
        if self._window is None:
            return []
        return list(self._window.create_file_dialog(webview.FileDialog.OPEN, allow_multiple=True) or [])

    def choose_folder(self) -> str:
        """Native folder panel for the auto-save folder. '' if cancelled."""
        if self._window is None:
            return ""
        picked = self._window.create_file_dialog(webview.FileDialog.FOLDER)
        if isinstance(picked, (list, tuple)):
            picked = picked[0] if picked else ""
        return picked or ""


def _bind_native_drop(window) -> None:
    """Hand the real paths of dropped files to the page (window.nativeDrop).

    pywebview only collects dropped paths while a Python-side drop listener is
    registered. The page falls back to uploading the File objects if this
    never calls back, so a failure here costs speed, not the feature."""
    try:
        from webview.dom import DOMEventHandler

        def on_drop(event):
            files = (event.get("dataTransfer") or {}).get("files") or []
            paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
            if paths:
                window.evaluate_js(f"window.nativeDrop && window.nativeDrop({json.dumps(paths)})")

        window.dom.document.events.drop += DOMEventHandler(on_drop, True, False)
    except Exception:
        import logging
        logging.getLogger("transcribe").exception("dépôt natif indisponible — repli sur l'envoi des fichiers")


def _port_in_use(host: str, port: int) -> bool:
    """Return True if something is already bound to (host, port)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _run_server() -> None:
    """Block forever serving the FastAPI app via uvicorn."""
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning", access_log=False)


def _wait_for_server(timeout: float = STARTUP_TIMEOUT_SEC) -> bool:
    """Poll /api/health until it responds 200 or `timeout` seconds elapse."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with url_request.urlopen(HEALTH_URL, timeout=0.5) as r:
                if r.status == 200:
                    return True
        except (url_error.URLError, OSError):
            pass
        time.sleep(POLL_INTERVAL_SEC)
    return False


def _error_window(title: str, heading: str, body: str) -> None:
    """Show a small native window with an error message and block."""
    webview.create_window(
        title,
        html=(
            "<html><body style='font-family:-apple-system;padding:40px;"
            "background:#0c1219;color:#e9f1f6;line-height:1.55;'>"
            f"<h2 style='color:#56d3c9;margin:0 0 14px;'>{heading}</h2>"
            f"<p style='margin:0;'>{body}</p>"
            "</body></html>"
        ),
        width=520, height=260,
    )
    webview.start()


def _set_dock_icon() -> None:
    """Force the Dock tile to the bundled icon at runtime.

    macOS iconservices caches app icons aggressively (keyed on bundle id);
    after an icon redesign the Dock can keep serving the stale artwork until
    a reboot. Setting NSApplication's icon image explicitly bypasses the
    cache entirely. No-op outside the bundle or if AppKit is unavailable.
    """
    try:
        from AppKit import NSApplication, NSImage
        res = Path(__file__).resolve().parent
        for name in ("AppIcon.icns", "icon.icns"):
            p = res / name
            if p.exists():
                img = NSImage.alloc().initWithContentsOfFile_(str(p))
                if img:
                    NSApplication.sharedApplication().setApplicationIconImage_(img)
                return
    except Exception:
        pass


def main() -> int:
    if _port_in_use(HOST, PORT):
        _error_window(
            "Transcribe — port occupé",
            f"Port {PORT} déjà utilisé",
            "Une autre instance de Transcribe (ou un autre serveur) "
            f"utilise le port {PORT}. Ferme-la avant de relancer.",
        )
        return 1

    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()

    if not _wait_for_server():
        _error_window(
            "Transcribe — démarrage échoué",
            "Le serveur n'a pas répondu",
            f"Aucune réponse sur {HEALTH_URL} après "
            f"{STARTUP_TIMEOUT_SEC:.0f}s. Vérifie les logs.",
        )
        return 2

    api = Api()
    window = webview.create_window(
        "Transcribe",
        f"http://{HOST}:{PORT}",
        width=860, height=680,
        min_size=(640, 560),
        js_api=api,
    )
    api._window = window  # save_file() needs the window to open the save panel
    window.events.loaded += lambda: _bind_native_drop(window)
    _set_dock_icon()
    # private_mode defaults to True, which makes the WKWebView cocoa backend wipe
    # ALL website data (localStorage included) on every launch — so the user's
    # saved theme and any client-side preferences never survive a quit. Persist
    # them by using the default (non-ephemeral) data store.
    webview.start(private_mode=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
