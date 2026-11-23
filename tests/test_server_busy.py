"""One job at a time, results kept per origin, streams that end, readable errors."""
import queue
import subprocess
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
    server.UPLOAD_DIR = tmp_path / "uploads"
    server.OUTPUT_DIR = tmp_path / "output"
    monkeypatch.setattr(server, "ytdlp_available", lambda: True)
    return TestClient(server.app), server


def _stream(gate=None, result=None):
    def stream(source, **kw):
        if gate:
            gate.wait(5)
        yield {"type": "done", "result": result or {}}
    return stream


def _wait(job):
    for _ in range(300):
        if job["done"]:
            return
        threading.Event().wait(0.01)
    raise AssertionError("job never finished")


def _result(tmp_path, title):
    out = tmp_path / "work" / title
    out.mkdir(parents=True, exist_ok=True)
    (out / "ts.md").write_text("`[00:00:01]` bonjour\n", encoding="utf-8")
    (out / "plain.md").write_text("bonjour\n", encoding="utf-8")
    return {"title": title, "duration": 12.0, "out_dir": str(out),
            "files": {"md_raw": "ts.md", "md_plain": "plain.md"}}


def test_jobs_wait_their_turn_and_stay_listed(client, monkeypatch):
    c, server = client
    gate = threading.Event()
    monkeypatch.setattr(server, "transcribe_url_stream", _stream(gate))
    first = c.post("/api/jobs/url", json={"url": "https://youtu.be/x"}).json()["job_id"]
    second = c.post("/api/jobs/url", json={"url": "https://youtu.be/y"}).json()["job_id"]
    for _ in range(200):                      # the runner picks up the first
        if server.JOBS[first]["started"]:
            break
        threading.Event().wait(0.01)
    listed = {j["id"]: j["state"] for j in c.get("/api/jobs").json()["jobs"]}
    assert listed == {first: "running", second: "queued"}   # one decode at a time
    gate.set()
    _wait(server.JOBS[first]); _wait(server.JOBS[second])
    assert [j["state"] for j in c.get("/api/jobs").json()["jobs"]] == ["done", "done"]   # nothing wiped


def test_mcp_and_window_never_erase_each_other(client, tmp_path, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "transcribe_url_stream", _stream())
    monkeypatch.setattr(server, "transcribe_stream", _stream())
    src = tmp_path / "a.mp4"
    src.write_bytes(b"x")
    ui = c.post("/api/jobs/url", json={"url": "https://youtu.be/x"}).json()["job_id"]
    mcp = c.post("/api/jobs/local", json={"path": str(src)}).json()["job_id"]
    _wait(server.JOBS[ui]); _wait(server.JOBS[mcp])
    mcp2 = c.post("/api/jobs/local", json={"path": str(src)}).json()["job_id"]
    _wait(server.JOBS[mcp2])
    assert ui in server.JOBS and mcp not in server.JOBS      # MCP only replaces its own finished job
    listed = c.get("/api/jobs").json()
    assert [j["id"] for j in listed["jobs"]] == [ui]        # the window lists only its own jobs


def test_running_job_cannot_be_removed_but_a_waiting_one_can(client, monkeypatch):
    c, server = client
    gate, ran = threading.Event(), []
    def stream(url, **kw):
        ran.append(url); gate.wait(5)
        yield {"type": "done", "result": {}}
    monkeypatch.setattr(server, "transcribe_url_stream", stream)
    first = c.post("/api/jobs/url", json={"url": "https://youtu.be/x"}).json()["job_id"]
    second = c.post("/api/jobs/url", json={"url": "https://youtu.be/y"}).json()["job_id"]
    third = c.post("/api/jobs/url", json={"url": "https://youtu.be/z"}).json()["job_id"]
    for _ in range(200):
        if server.JOBS[first]["started"]:
            break
        threading.Event().wait(0.01)
    assert c.delete(f"/api/jobs/{first}").status_code == 409
    assert c.delete(f"/api/jobs/{second}").status_code == 204
    gate.set()
    _wait(server.JOBS[first]); _wait(server.JOBS[third])
    assert ran == ["https://youtu.be/x", "https://youtu.be/z"]   # the removed one never ran


def test_window_job_is_saved_to_the_output_folder(client, tmp_path, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "transcribe_url_stream", lambda url, **kw: iter(
        [{"type": "done", "result": _result(tmp_path, url[-1])}]))
    out = tmp_path / "Sorties"
    assert c.post("/api/settings/output-dir", json={"path": str(out)}).status_code == 204
    assert c.post("/api/settings/output-dir", json={"path": "relatif"}).status_code == 400
    a = c.post("/api/jobs/url", json={"url": "https://x/a", "timestamps": False}).json()["job_id"]
    _wait(server.JOBS[a])
    b = c.post("/api/jobs/url", json={"url": "https://x/a"}).json()["job_id"]   # same title, with timestamps
    _wait(server.JOBS[b])
    assert (out / "a.md").read_text(encoding="utf-8") == "bonjour\n"
    assert (out / "a (2).md").read_text(encoding="utf-8").startswith("`[00:00:01]`")   # never overwrites
    assert c.get(f"/api/jobs/{b}").json()["saved"] == str(out / "a (2).md")


def test_stream_ends_when_the_job_is_done_and_drained(client):
    c, server = client
    server.JOBS["a" * 12] = {"queue": queue.Queue(), "done": True, "result": {}, "error": None}
    r = c.get(f"/api/jobs/{'a' * 12}/stream")   # sentinel already consumed by a first reader
    assert r.status_code == 200 and r.text == ""


def test_extract_audio_reports_what_ffmpeg_said(tmp_path, monkeypatch):
    import transcribe as T
    monkeypatch.setattr(T.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 1, stdout="", stderr="banner\nInvalid data found when processing input"))
    with pytest.raises(RuntimeError, match="Invalid data found"):
        T.extract_audio(tmp_path / "broken.mp4", tmp_path)


def test_whole_file_fallback_cleans_loops_and_says_why(monkeypatch):
    import transcribe as T
    monkeypatch.setattr(T, "VAD_PYTHON", Path("/nonexistent/python"))
    monkeypatch.setattr(T, "_mlx_transcribe_file", lambda *a, **k: (
        [{"start": 0.0, "end": 1.0, "text": " oui oui oui oui oui "}], "fr"))
    segments, _, _ = T._run_mlx_whisper(Path("x.wav"), language="fr")
    assert [s["text"] for s in segments] == ["oui"]
    assert T._vad_note and not T.vad_available()


def test_time_left_uses_measured_speed_then_chunk_rate(client, tmp_path, monkeypatch):
    c, server = client
    import settings
    assert settings.get_speed("large-v3") == 5.0            # starting point, not the old 25x
    settings.record_speed("large-v3", 600, 60)              # one real job at 10x
    assert settings.get_speed("large-v3") == 6.5            # nudged (70/30), not overwritten
    settings.record_speed("large-v3", 10, 5)                # too short to mean anything
    assert settings.get_speed("large-v3") == 6.5

    gate, seen = threading.Event(), threading.Event()
    def stream(url, **kw):
        yield {"type": "status", "stage": "audio", "msg": "Audio", "audio_duration": 720.0}
        seen.set(); gate.wait(5)
        yield {"type": "heartbeat", "stage": "transcribe", "msg": "passage 3/9", "remaining": 30.0}
        yield {"type": "done", "result": {"duration": 720.0, "elapsed": 72.0}}
    monkeypatch.setattr(server, "transcribe_url_stream", stream)
    job_id = c.post("/api/jobs/url", json={"url": "https://youtu.be/x", "model": "large-v3"}).json()["job_id"]
    assert seen.wait(5)
    eta = c.get("/api/jobs").json()["jobs"][0]["eta"]
    assert 105 < eta <= 111                                  # 720 s of audio / 6.5x
    gate.set()
    _wait(server.JOBS[job_id])
    assert server.JOBS[job_id]["remaining"] == 30.0          # then the measured chunk rate takes over
    assert settings.get_speed("large-v3") == 7.55            # and the finished job recalibrates: 0.7*6.5 + 0.3*10
