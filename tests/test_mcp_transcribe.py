"""MCP server: JSON-RPC handling and tool logic against a fake app API."""
import json
from pathlib import Path

import pytest

import mcp_transcribe as m


class FakeApi:
    """Records calls; answers like the app's HTTP API."""
    def __init__(self, out_dir: Path, statuses=None, settings=None):
        self.calls, self.out_dir = [], out_dir
        self.statuses = list(statuses or [])
        self.settings = settings or {"default_model": "qwen3-asr-1.7b", "models": [
            {"id": "qwen3-asr-1.7b", "label": "Qwen3", "engine": "mlx-audio", "installed": True,
             "is_default": True, "custom": False, "size_bytes": 1, "repo": "x", "description": "d"},
            {"id": "large-v3", "label": "Large-v3", "engine": "mlx-whisper", "installed": False,
             "is_default": False, "custom": False, "size_bytes": 1, "repo": "y", "description": "d"}]}

    def ensure_running(self):
        self.calls.append(("ensure_running",))

    def call(self, method, path, body=None):
        self.calls.append((method, path, body))
        if path in ("/api/jobs/local", "/api/jobs/url"):
            return {"job_id": "abcdef012345"}
        if path == "/api/settings":
            return self.settings
        if path.startswith("/api/jobs/"):
            return self.statuses.pop(0)
        return {}


def _done(out_dir: Path) -> dict:
    (out_dir / "v — Texte.md").write_text("Voilà 4 ETF.\n", encoding="utf-8")
    (out_dir / "v — Transcript.srt").write_text("1\n", encoding="utf-8")
    return {"done": True, "error": None, "result": {
        "title": "v", "language": "fr", "duration": 63.7, "elapsed": 6.12, "out_dir": str(out_dir),
        "files": {"md_plain": "v — Texte.md", "srt": "v — Transcript.srt", "json": None}}}


@pytest.fixture(autouse=True)
def fast_poll(monkeypatch):
    monkeypatch.setattr(m, "POLL_S", 0)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)


def _call(api, name, args):
    rep = m.handle({"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                    "params": {"name": name, "arguments": args}}, api)
    return rep["result"]


def test_initialize_and_tools_list():
    api = FakeApi(Path("/tmp"))
    init = m.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                     "params": {"protocolVersion": "2025-06-18"}}, api)
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert "tools" in init["result"]["capabilities"]
    names = [t["name"] for t in m.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, api)["result"]["tools"]]
    assert names == ["transcrire_fichier", "transcrire_url", "suivre_transcription",
                     "lister_modeles", "choisir_modele_defaut"]
    assert m.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}, api) is None


def test_transcrire_fichier_waits_and_returns_text_and_files(tmp_path):
    api = FakeApi(tmp_path, statuses=[{"done": False, "msg": "passage 1/2"}, _done(tmp_path)])
    res = _call(api, "transcrire_fichier", {"chemin": "/Users/a/v.mp4", "langue": "fr"})
    assert res["isError"] is False
    data = res["structuredContent"]
    assert data["texte"] == "Voilà 4 ETF.\n" and data["langue"] == "fr" and data["calcul_s"] == 6.1
    assert set(data["fichiers"]) == {"md_plain", "srt"}
    assert ("POST", "/api/jobs/local", {"path": "/Users/a/v.mp4", "language": "fr", "model": None}) in api.calls


def test_dossier_sortie_copies_the_files(tmp_path):
    app_out, dest = tmp_path / "app", tmp_path / "keep"
    app_out.mkdir()
    api = FakeApi(app_out, statuses=[_done(app_out)])
    data = _call(api, "transcrire_url", {"url": "https://youtu.be/x", "dossier_sortie": str(dest)})["structuredContent"]
    assert Path(data["fichiers"]["md_plain"]).parent == dest
    assert (dest / "v — Transcript.srt").exists()
    assert api.calls[1][2]["require_idle"] is True     # never wipes a running UI job


def test_long_job_hands_back_a_job_id(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "WAIT_MAX_S", -1)
    api = FakeApi(tmp_path, statuses=[{"done": False, "msg": "passage 3/40", "progress": 0.07}])
    data = _call(api, "transcrire_fichier", {"chemin": "/a.mp4"})["structuredContent"]
    assert data["termine"] is False and data["job_id"] == "abcdef012345" and data["progression"] == 0.07


def test_failed_job_and_bad_inputs_are_tool_errors(tmp_path):
    api = FakeApi(tmp_path, statuses=[{"done": True, "error": "ffmpeg absent", "result": None}])
    res = _call(api, "suivre_transcription", {"job_id": "abcdef012345"})
    assert res["isError"] is True and "ffmpeg absent" in res["content"][0]["text"]
    assert _call(api, "transcrire_fichier", {"chemin": "/a.mp4", "langue": "klingon"})["isError"] is True
    assert _call(api, "transcrire_fichier", {"chemin": "/a.mp4", "dossier_sortie": "rel"})["isError"] is True


def test_models_list_and_default_switch(tmp_path):
    api = FakeApi(tmp_path)
    data = _call(api, "lister_modeles", {})["structuredContent"]
    assert data["defaut"] == "qwen3-asr-1.7b" and len(data["modeles"]) == 2
    assert _call(api, "choisir_modele_defaut", {"id": "large-v3"})["isError"] is True     # not installed
    assert _call(api, "choisir_modele_defaut", {"id": "nope"})["isError"] is True
    assert _call(api, "choisir_modele_defaut", {"id": "qwen3-asr-1.7b"})["structuredContent"] == {"defaut": "qwen3-asr-1.7b"}


def test_http_errors_surface_the_app_detail(monkeypatch):
    import io, urllib.error
    def boom(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 409, "x", {}, io.BytesIO(
            json.dumps({"detail": "Une transcription est déjà en cours"}).encode()))
    monkeypatch.setattr(m.urllib.request, "urlopen", boom)
    with pytest.raises(m.ToolError, match="déjà en cours"):
        m.Api("http://x").call("POST", "/api/jobs/local", {})
