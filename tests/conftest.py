"""Add project root to sys.path so tests can import transcribe + server."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.fixture(autouse=True)
def _isolated_settings_file(tmp_path, monkeypatch):
    """Never read the real ~/Library/…/settings.json: user-added models there
    would leak into every catalog assertion. Patched at Path.home, not on the
    settings module: some fixtures re-import settings mid-session."""
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
