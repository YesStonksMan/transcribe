"""Tests for POST /api/dictate — short push-to-talk clips for Mini-Flow."""
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    for mod in ["server", "transcribe"]:
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import server
    server.JOBS.clear()
    return TestClient(server.app), server


def test_dictate_joins_segments_and_returns_vocabulary(client, monkeypatch):
    c, server = client
    seen = {}

    def fake_transcribe(wav, **options):
        seen.update(options, wav=Path(wav), exists=Path(wav).exists())
        return {"segments": [{"text": " Bonjour"}, {"text": " Marie."}, {"text": "  "}]}

    monkeypatch.setitem(sys.modules, "mlx_whisper", types.SimpleNamespace(transcribe=fake_transcribe))
    monkeypatch.setattr(server._settings_mod, "get_vocabulary", lambda: "Degiro, ETF")

    r = c.post("/api/dictate?lang=fr", content=b"RIFFfakewav")

    assert r.status_code == 200
    assert r.json()["text"] == "Bonjour Marie."
    assert isinstance(r.json()["ms"], int)
    assert seen["exists"] and seen["language"] == "fr"
    assert seen["path_or_hf_repo"] == server.DICTATE_MODEL
    # Word timestamps took 45 % of a dictation (median 532 -> 296 ms, 12-10) and nothing here uses them.
    assert seen["word_timestamps"] is False
    # On a 5 s clip the initial_prompt made Whisper loop ("...", "정말 정말…"):
    # vocabulary goes to the formatter instead of the decoder.
    assert seen.get("initial_prompt") is None
    assert r.json()["vocabulary"] == "Degiro, ETF"
    assert not seen["wav"].exists()  # temp file cleaned up


def test_dictate_rejects_empty_body(client):
    c, _ = client
    assert c.post("/api/dictate", content=b"").status_code == 400


def test_dictate_refuses_while_a_job_runs(client):
    c, server = client
    server.JOBS["x" * 12] = {"done": False}
    assert c.post("/api/dictate", content=b"RIFF").status_code == 409
