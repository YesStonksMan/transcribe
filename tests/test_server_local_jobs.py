"""Local-path jobs + job status: the surface the MCP server drives."""
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    for mod in ["server", "transcribe", "settings"]:
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import server
    return TestClient(server.app), server


def _fake_stream(events, gate=None):
    def stream(path, **kw):
        if gate:
            gate.wait(5)
        yield from events
    return stream


def _wait_done(server, job_id):
    for _ in range(200):
        if server.JOBS[job_id]["done"]:
            return
        threading.Event().wait(0.01)
    raise AssertionError("job never finished")


def test_local_job_transcribes_the_file_in_place(client, tmp_path, monkeypatch):
    c, server = client
    src = tmp_path / "rush.mp4"
    src.write_bytes(b"x")
    seen = {}
    def stream(path, **kw):
        seen.update(path=path, **kw)
        yield {"type": "done", "result": {"title": "rush"}}
    monkeypatch.setattr(server, "transcribe_stream", stream)
    r = c.post("/api/jobs/local", json={"path": str(src), "language": "fr", "model": "large-v3"})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    _wait_done(server, job_id)
    assert seen["path"] == src and seen["language"] == "fr" and seen["model_name"] == "large-v3"
    assert src.exists()                      # never moved, copied or deleted


def test_local_job_rejects_missing_or_relative_paths(client, tmp_path):
    c, _ = client
    assert c.post("/api/jobs/local", json={"path": str(tmp_path / "nope.mp4")}).status_code == 404
    assert c.post("/api/jobs/local", json={"path": "relative/file.mp4"}).status_code == 400
    assert c.post("/api/jobs/local", json={"path": str(tmp_path)}).status_code == 404


def test_local_job_waits_while_another_job_runs(client, tmp_path, monkeypatch):
    c, server = client
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    gate = threading.Event()
    monkeypatch.setattr(server, "transcribe_stream", _fake_stream([{"type": "done", "result": {}}], gate))
    first = c.post("/api/jobs/local", json={"path": str(src)}).json()["job_id"]
    r = c.post("/api/jobs/local", json={"path": str(src)})
    assert r.status_code == 200              # queued behind the first, not refused
    second = r.json()["job_id"]
    assert first in server.JOBS              # the running job was not wiped
    assert c.get(f"/api/jobs/{second}").json()["state"] == "queued"
    gate.set()
    _wait_done(server, first)
    _wait_done(server, second)


def test_job_status_reports_progress_then_result(client, tmp_path, monkeypatch):
    c, server = client
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    gate = threading.Event()
    def stream(path, **kw):
        yield {"type": "heartbeat", "stage": "transcribe", "msg": "passage 1/3", "progress": 0.33}
        gate.wait(5)
        yield {"type": "done", "result": {"title": "a", "language": "fr"}}
    monkeypatch.setattr(server, "transcribe_stream", stream)
    job_id = c.post("/api/jobs/local", json={"path": str(src)}).json()["job_id"]
    for _ in range(200):
        st = c.get(f"/api/jobs/{job_id}").json()
        if st.get("progress"):
            break
        threading.Event().wait(0.01)
    assert st["done"] is False and st["msg"] == "passage 1/3" and st["progress"] == 0.33
    gate.set()
    _wait_done(server, job_id)
    st = c.get(f"/api/jobs/{job_id}").json()
    assert st["done"] is True and st["result"]["language"] == "fr" and st["error"] is None


def test_job_status_carries_the_error(client, tmp_path, monkeypatch):
    c, server = client
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    monkeypatch.setattr(server, "transcribe_stream", _fake_stream([{"type": "error", "msg": "ffmpeg absent"}]))
    job_id = c.post("/api/jobs/local", json={"path": str(src)}).json()["job_id"]
    _wait_done(server, job_id)
    st = c.get(f"/api/jobs/{job_id}").json()
    assert st["done"] is True and st["error"] == "ffmpeg absent" and st["result"] is None


def test_job_status_unknown_job_is_404(client):
    c, _ = client
    assert c.get("/api/jobs/abcdef012345").status_code == 404


def test_mcp_url_job_queues_behind_a_running_job(client, tmp_path, monkeypatch):
    c, server = client
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    gate = threading.Event()
    monkeypatch.setattr(server, "transcribe_stream", _fake_stream([{"type": "done", "result": {}}], gate))
    monkeypatch.setattr(server, "transcribe_url_stream", _fake_stream([{"type": "done", "result": {}}]))
    monkeypatch.setattr(server, "ytdlp_available", lambda: True)
    first = c.post("/api/jobs/local", json={"path": str(src)}).json()["job_id"]
    r = c.post("/api/jobs/url", json={"url": "https://youtu.be/x", "require_idle": True})
    assert r.status_code == 200
    second = r.json()["job_id"]
    assert server.JOBS[second]["origin"] == "mcp" and not server.JOBS[second]["started"]
    gate.set()
    _wait_done(server, first)
    _wait_done(server, second)
