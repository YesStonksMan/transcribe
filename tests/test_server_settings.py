"""Endpoint tests for the settings + models + deps surface."""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


class _FakeProc:
    """Drop-in for subprocess.Popen used by brew install."""
    def __init__(self):
        self.stdout = iter(["line 1\n", "line 2\n"])
        self.returncode = 0
    def wait(self):
        return 0


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Fresh server with isolated HF cache + settings.json + dirs."""
    monkeypatch.setenv("HF_HOME", str(tmp_path / ".hf"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    for mod in ["server", "transcribe", "settings"]:
        sys.modules.pop(mod, None)
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import server
    return TestClient(server.app), server


# ---------- GET /api/settings ----------
def test_get_settings_returns_models_deps_default(client):
    c, _ = client
    r = c.get("/api/settings")
    assert r.status_code == 200
    body = r.json()
    assert body["default_model"] == "large-v3"  # fresh home: no mlx-audio venv
    assert len(body["models"]) == 3
    assert [d["id"] for d in body["deps"]] == ["ffmpeg", "yt-dlp", "vad"]
    assert "brew_available" in body
    ids = [m["id"] for m in body["models"]]
    assert ids == ["large-v3-turbo", "large-v3", "qwen3-asr-1.7b"]


# ---------- POST /api/settings/model ----------
def test_post_default_model_persists(client):
    c, _ = client
    r = c.post("/api/settings/model", json={"id": "large-v3-turbo"})
    assert r.status_code == 204
    r2 = c.get("/api/settings")
    assert r2.json()["default_model"] == "large-v3-turbo"


def test_post_default_model_unknown_id_returns_400(client):
    c, _ = client
    r = c.post("/api/settings/model", json={"id": "garbage"})
    assert r.status_code == 400


# ---------- DELETE /api/models/{id} ----------
def test_delete_model_removes_cache_dir(client):
    c, _ = client
    import settings as s
    # Deletes turbo, not large-v3: large-v3 is the current default and the
    # server refuses to delete whatever is selected (409 — covered below).
    snap = s._hf_cache_dir("large-v3-turbo") / "snapshots" / "x"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert s.is_model_installed("large-v3-turbo")

    r = c.delete("/api/models/large-v3-turbo")
    assert r.status_code == 204
    assert not s.is_model_installed("large-v3-turbo")


def test_delete_model_unknown_id_returns_404(client):
    c, _ = client
    r = c.delete("/api/models/garbage")
    assert r.status_code == 404


def test_delete_model_currently_selected_returns_409(client):
    c, _ = client
    import settings as s
    # Pin large-v3 as default so the delete of that model conflicts
    s.set_default_model("large-v3")
    snap = s._hf_cache_dir("large-v3") / "snapshots" / "x"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    r = c.delete("/api/models/large-v3")
    assert r.status_code == 409
    assert s.is_model_installed("large-v3")


# ---------- POST /api/models/{id}/download ----------
def test_post_download_unknown_model_returns_404(client):
    c, _ = client
    r = c.post("/api/models/garbage/download")
    assert r.status_code == 404


def test_post_download_returns_job_id(client, monkeypatch):
    import settings as s
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kw: str(s._hf_cache_dir("large-v3-turbo") / "snapshots" / "fake"),
    )
    c, _ = client
    r = c.post("/api/models/large-v3-turbo/download")
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert len(body["job_id"]) == 12


def test_post_download_refuses_when_disk_full(client, monkeypatch):
    monkeypatch.setattr("settings.disk_free_bytes", lambda _p: 100_000_000)  # 100 MB
    c, _ = client
    r = c.post("/api/models/large-v3/download")
    assert r.status_code == 507


# ---------- POST /api/deps/{name}/install ----------
def test_post_install_dep_unknown_returns_404(client):
    c, _ = client
    r = c.post("/api/deps/garbage/install")
    assert r.status_code == 404


def test_post_install_dep_non_brew_returns_400(client, monkeypatch):
    import settings as s
    from dataclasses import replace
    monkeypatch.setitem(s.DEPS, "ffmpeg", replace(s.DEPS["ffmpeg"], brew_installable=False))
    c, _ = client
    r = c.post("/api/deps/ffmpeg/install")
    assert r.status_code == 400


def test_post_install_dep_brew_unavailable_returns_424(client, monkeypatch):
    monkeypatch.setattr("settings.brew_available", lambda: False)
    c, _ = client
    r = c.post("/api/deps/ffmpeg/install")
    assert r.status_code == 424


def test_post_install_dep_returns_job_id(client, monkeypatch):
    monkeypatch.setattr("settings.brew_available", lambda: True)
    monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: _FakeProc())
    c, _ = client
    r = c.post("/api/deps/ffmpeg/install")
    assert r.status_code == 200
    assert "job_id" in r.json()


# ---------- User-added models ----------

def test_post_model_adds_a_huggingface_repo(client, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "_hf_repo_size", lambda repo: 1_234)
    r = c.post("/api/models", json={"repo": "mlx-community/parakeet-tdt-0.6b-v3"})
    assert r.status_code == 200
    model_id = r.json()["id"]
    listed = {m["id"]: m for m in c.get("/api/settings").json()["models"]}
    assert listed[model_id]["custom"] is True
    assert listed[model_id]["engine"] == "mlx-audio"
    assert listed[model_id]["size_bytes"] == 1_234


def test_post_model_unknown_repo_returns_400(client, monkeypatch):
    c, server = client
    def missing(repo):
        raise LookupError("Repo introuvable sur HuggingFace.")
    monkeypatch.setattr(server, "_hf_repo_size", missing)
    r = c.post("/api/models", json={"repo": "org/does-not-exist"})
    assert r.status_code == 400
    assert "introuvable" in r.json()["detail"]


def test_post_model_malformed_repo_returns_400_without_network(client, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "_hf_repo_size", lambda repo: pytest.fail("no lookup for a bad repo"))
    assert c.post("/api/models", json={"repo": "not a repo"}).status_code == 400


def test_delete_custom_model_also_drops_it_from_the_list(client, monkeypatch):
    c, server = client
    monkeypatch.setattr(server, "_hf_repo_size", lambda repo: 0)
    model_id = c.post("/api/models", json={"repo": "org/model-a"}).json()["id"]
    assert c.delete(f"/api/models/{model_id}").status_code == 204
    ids = [m["id"] for m in c.get("/api/settings").json()["models"]]
    assert model_id not in ids
    assert "large-v3-turbo" in ids   # built-ins only lose their cache, never their entry
