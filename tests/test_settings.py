"""Tests for the Settings module (model + dep catalog, persistence)."""
from pathlib import Path
from unittest.mock import patch

import settings


def test_model_catalog_has_expected_entries():
    ids = [m.id for m in settings.list_models_static()]
    assert ids == ["large-v3-turbo", "large-v3", "qwen3-asr-1.7b"]


def test_model_repo_lookup_returns_huggingface_id():
    assert settings.model_repo("large-v3") == "mlx-community/whisper-large-v3-mlx"
    assert settings.model_repo("large-v3-turbo") == "mlx-community/whisper-large-v3-turbo"


def test_model_repo_unknown_id_raises():
    import pytest
    with pytest.raises(KeyError):
        settings.model_repo("nonexistent")


def test_get_default_model_falls_back_to_qwen3_when_settings_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_settings_file", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(settings, "MLX_AUDIO_PYTHON", tmp_path)  # exists
    assert settings.get_default_model() == settings.DEFAULT_MODEL_FALLBACK == "qwen3-asr-1.7b"


def test_default_model_is_whisper_when_mlx_audio_env_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_settings_file", lambda: tmp_path / "settings.json")
    monkeypatch.setattr(settings, "MLX_AUDIO_PYTHON", tmp_path / "nope")
    assert settings.get_default_model() == "large-v3"


def test_set_default_model_persists_and_get_returns_it(tmp_path, monkeypatch):
    sfile = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "_settings_file", lambda: sfile)
    settings.set_default_model("large-v3-turbo")
    assert sfile.exists()
    raw = sfile.read_text()
    assert "large-v3-turbo" in raw
    assert settings.get_default_model() == "large-v3-turbo"


def test_set_default_model_unknown_id_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_settings_file", lambda: tmp_path / "settings.json")
    import pytest
    with pytest.raises(ValueError):
        settings.set_default_model("garbage-id")


def test_set_default_model_corrupted_file_is_overwritten(tmp_path, monkeypatch):
    sfile = tmp_path / "settings.json"
    sfile.write_text("{not valid json")
    monkeypatch.setattr(settings, "_settings_file", lambda: sfile)
    # Reading falls back to default, writing replaces the bad file
    assert settings.get_default_model() == settings.DEFAULT_MODEL_FALLBACK
    settings.set_default_model("large-v3-turbo")
    assert settings.get_default_model() == "large-v3-turbo"


def test_hf_cache_dir_for_model_uses_huggingface_layout():
    p = settings._hf_cache_dir("large-v3")
    assert p.name == "models--mlx-community--whisper-large-v3-mlx"
    assert p.parent.name == "hub"


def test_is_model_installed_false_when_cache_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_hf_cache_root", lambda: tmp_path / "hub")
    assert settings.is_model_installed("large-v3") is False


def test_is_model_installed_true_when_snapshots_has_content(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_hf_cache_root", lambda: tmp_path / "hub")
    snap = tmp_path / "hub" / "models--mlx-community--whisper-large-v3-mlx" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert settings.is_model_installed("large-v3") is True


def test_uninstall_model_removes_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_hf_cache_root", lambda: tmp_path / "hub")
    snap = tmp_path / "hub" / "models--mlx-community--whisper-large-v3-mlx" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert settings.is_model_installed("large-v3") is True

    settings.uninstall_model("large-v3")
    assert settings.is_model_installed("large-v3") is False


def test_uninstall_model_unknown_id_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_hf_cache_root", lambda: tmp_path / "hub")
    import pytest
    with pytest.raises(KeyError):
        settings.uninstall_model("garbage")


def test_disk_free_bytes_returns_positive_int(tmp_path):
    n = settings.disk_free_bytes(tmp_path)
    assert isinstance(n, int)
    assert n > 0


def test_dep_catalog_has_two_entries():
    ids = [d.id for d in settings.list_deps_static()]
    assert ids == ["ffmpeg", "yt-dlp"]


def test_dep_install_state_missing_when_not_on_path(monkeypatch):
    monkeypatch.setattr(settings.shutil, "which", lambda name: None)
    state = settings.dep_state("ffmpeg")
    assert state.installed is False
    assert state.version is None


def test_dep_install_state_present_with_version(monkeypatch):
    monkeypatch.setattr(settings.shutil, "which", lambda name: "/opt/homebrew/bin/" + name)
    monkeypatch.setattr(settings, "_dep_version_probe",
                        lambda name, path: "ffmpeg version 8.1.1 …" if name == "ffmpeg" else None)
    state = settings.dep_state("ffmpeg")
    assert state.installed is True
    assert "8.1.1" in (state.version or "")


def test_brew_available_true_when_brew_on_path(monkeypatch):
    monkeypatch.setattr(settings.shutil, "which",
                        lambda name: "/opt/homebrew/bin/brew" if name == "brew" else None)
    assert settings.brew_available() is True


def test_brew_available_false_when_missing(monkeypatch):
    monkeypatch.setattr(settings.shutil, "which", lambda name: None)
    assert settings.brew_available() is False



def test_list_models_returns_each_with_install_state(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_hf_cache_root", lambda: tmp_path / "hub")
    monkeypatch.setattr(settings, "_settings_file", lambda: tmp_path / "settings.json")
    # persist a non-fallback default so is_default flags line up with the fixture
    (tmp_path / "settings.json").write_text('{"default_model": "large-v3"}')
    # install large-v3 only
    snap = tmp_path / "hub" / "models--mlx-community--whisper-large-v3-mlx" / "snapshots" / "x"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")

    models = settings.list_models()
    by_id = {m["id"]: m for m in models}

    assert by_id["large-v3"]["installed"] is True
    assert by_id["large-v3-turbo"]["installed"] is False
    assert by_id["large-v3"]["is_default"] is True
    assert by_id["large-v3-turbo"]["is_default"] is False
    assert by_id["large-v3"]["repo"] == "mlx-community/whisper-large-v3-mlx"
    assert by_id["large-v3"]["size_bytes"] == 3_100_000_000


# ---------- custom vocabulary ----------
def test_vocabulary_defaults_to_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_settings_file", lambda: tmp_path / "settings.json")
    assert settings.get_vocabulary() == ""


def test_vocabulary_persists_and_is_trimmed(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_settings_file", lambda: tmp_path / "settings.json")
    settings.set_vocabulary("  Kakashi, ETF, Nasdaq  ")
    assert settings.get_vocabulary() == "Kakashi, ETF, Nasdaq"


def test_vocabulary_over_the_cap_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_settings_file", lambda: tmp_path / "settings.json")
    import pytest
    with pytest.raises(ValueError):
        settings.set_vocabulary("x" * (settings.VOCABULARY_MAX_CHARS + 1))
    # nothing was written
    assert settings.get_vocabulary() == ""


def test_vocabulary_survives_a_model_change(tmp_path, monkeypatch):
    """Both live in the same settings.json — writing one must not drop the other."""
    monkeypatch.setattr(settings, "_settings_file", lambda: tmp_path / "settings.json")
    settings.set_vocabulary("Kakashi")
    settings.set_default_model("large-v3")
    assert settings.get_vocabulary() == "Kakashi"
    assert settings.get_default_model() == "large-v3"


# ---------- Engines + user-added models ----------

def test_engine_is_derived_from_the_repo():
    assert settings.engine_for_repo("mlx-community/whisper-large-v3-mlx") == "mlx-whisper"
    assert settings.engine_for_repo("mlx-community/Qwen3-ASR-1.7B-8bit") == "mlx-audio"
    assert settings.engine_for_repo("mlx-community/parakeet-tdt-0.6b-v3") == "mlx-audio"


def test_qwen3_is_a_builtin_mlx_audio_model():
    spec = settings.get_models()["qwen3-asr-1.7b"]
    assert spec.repo == "mlx-community/Qwen3-ASR-1.7B-8bit"
    assert spec.engine == "mlx-audio"


def test_add_custom_model_persists_and_joins_the_catalog(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "_settings_file", lambda: tmp_path / "settings.json")
    model_id = settings.add_custom_model("mlx-community/whisper-medium-mlx", 1_500_000_000)
    assert model_id == "whisper-medium-mlx"
    spec = settings.get_models()[model_id]
    assert spec.repo == "mlx-community/whisper-medium-mlx"
    assert spec.engine == "mlx-whisper"
    assert spec.custom is True
    listed = {m["id"]: m for m in settings.list_models()}
    assert listed[model_id]["custom"] is True
    assert listed["large-v3"]["custom"] is False
    settings.set_default_model(model_id)
    assert settings.get_default_model() == model_id


def test_add_custom_model_rejects_bad_or_duplicate_repo(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(settings, "_settings_file", lambda: tmp_path / "settings.json")
    for bad in ["", "no-slash", "a/b/c", "https://huggingface.co/x/y", "x/y z"]:
        with pytest.raises(ValueError):
            settings.add_custom_model(bad, 0)
    with pytest.raises(ValueError):
        settings.add_custom_model("mlx-community/whisper-large-v3-mlx", 0)  # builtin
    settings.add_custom_model("org/model-a", 0)
    with pytest.raises(ValueError):
        settings.add_custom_model("org/model-a", 0)


def test_remove_custom_model_drops_it_and_default_falls_back(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(settings, "_settings_file", lambda: tmp_path / "settings.json")
    model_id = settings.add_custom_model("org/model-a", 0)
    settings.set_default_model(model_id)
    settings.remove_custom_model(model_id)
    assert model_id not in settings.get_models()
    assert settings.get_default_model() == settings.DEFAULT_MODEL_FALLBACK
    with pytest.raises(ValueError):
        settings.remove_custom_model("large-v3")   # builtins stay
