"""Tests for job cleanup endpoint and helpers."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh FastAPI app with isolated UPLOAD_DIR / OUTPUT_DIR per test."""
    upload = tmp_path / "uploads"; upload.mkdir()
    output = tmp_path / "output"; output.mkdir()
    for mod in ["server", "transcribe"]:
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import server
    server.UPLOAD_DIR = upload
    server.OUTPUT_DIR = output
    server.JOBS.clear()
    server._LAST_JOB_ID = None
    return TestClient(server.app), server


def test_delete_existing_job_removes_dirs_and_memory(client):
    c, server = client
    job_id = "a" * 12
    server.JOBS[job_id] = {"queue": None, "done": True, "result": {}, "file": None}
    (server.UPLOAD_DIR / job_id).mkdir(parents=True)
    (server.UPLOAD_DIR / job_id / "video.mp4").write_bytes(b"x")
    (server.OUTPUT_DIR / job_id).mkdir(parents=True)
    (server.OUTPUT_DIR / job_id / "transcript.md").write_text("hello")

    r = c.delete(f"/api/jobs/{job_id}")
    assert r.status_code == 204
    assert job_id not in server.JOBS
    assert not (server.UPLOAD_DIR / job_id).exists()
    assert not (server.OUTPUT_DIR / job_id).exists()


def test_delete_unknown_job_is_idempotent(client):
    c, _ = client
    r = c.delete("/api/jobs/" + "f" * 12)
    assert r.status_code == 204


def test_delete_invalid_job_id_is_silently_noop(client):
    c, server = client
    r = c.delete("/api/jobs/notavalidid")
    assert r.status_code == 204
    assert server.UPLOAD_DIR.exists()


def test_cleanup_orphans_removes_dirs_without_active_job(client):
    c, server = client
    for jid in ["1" * 12, "2" * 12, "3" * 12]:
        (server.UPLOAD_DIR / jid).mkdir(parents=True)
        (server.OUTPUT_DIR / jid).mkdir(parents=True)
    keep = "4" * 12
    server.JOBS[keep] = {"queue": None, "done": False, "result": None, "file": None}
    (server.UPLOAD_DIR / keep).mkdir(parents=True)
    (server.OUTPUT_DIR / keep).mkdir(parents=True)

    server.cleanup_orphans()

    for jid in ["1" * 12, "2" * 12, "3" * 12]:
        assert not (server.UPLOAD_DIR / jid).exists()
        assert not (server.OUTPUT_DIR / jid).exists()
    assert (server.UPLOAD_DIR / keep).exists()
    assert (server.OUTPUT_DIR / keep).exists()


def test_cleanup_orphans_ignores_non_job_id_named_files(client):
    c, server = client
    legacy_file = server.UPLOAD_DIR / "old_user_upload.mp4"
    legacy_file.write_bytes(b"x")
    legacy_dir = server.OUTPUT_DIR / "Comment investir en 2026"
    legacy_dir.mkdir()

    server.cleanup_orphans()

    assert legacy_file.exists()
    assert legacy_dir.exists()


def test_audio_dropped_when_job_ends_transcripts_kept(client):
    _, server = client
    job_id = "b" * 12
    server.JOBS[job_id] = {
        "queue": None, "done": True, "file": None,
        "result": {"files": {"md_raw": "x — Transcript.md", "srt": "x — Transcript.srt"}},
    }
    up = server.UPLOAD_DIR / job_id; up.mkdir()
    (up / "x.m4a").write_bytes(b"a")
    out = server.OUTPUT_DIR / job_id; out.mkdir()
    for name in ("x.wav", "x — Transcript.md", "x — Transcript.srt"):
        (out / name).write_text("t")

    server._drop_job_audio(job_id)

    assert not up.exists()
    assert sorted(p.name for p in out.iterdir()) == ["x — Transcript.md", "x — Transcript.srt"]
