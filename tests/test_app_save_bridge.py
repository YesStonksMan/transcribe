"""Regression tests for the native save bridge (app.Api.save_file).

WKWebView's built-in download path (``<a download>`` → WKDownload → save panel)
silently fails in the bundled app, so downloads go through a pywebview JS bridge
instead: the page calls ``window.pywebview.api.save_file(job_id, filename)`` and
Python fetches the bytes from the validated download endpoint, lets the user
pick a destination via a native save panel **defaulting to ~/Downloads**, writes
the bytes there, and reveals the result in Finder. Forcing the default directory
is the crux: NSSavePanel otherwise reuses its last-used folder, which once
buried saved files in an opaque internal directory ("panel appeared, no file").
These tests pin that behaviour without a GUI.
"""
import app


class _FakeResp:
    """Minimal stand-in for the urlopen context manager."""

    def __init__(self, data: bytes):
        self._data = data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self) -> bytes:
        return self._data


class _FakeWindow:
    """Records the save-dialog call and returns a preset destination."""

    def __init__(self, dest):
        self.dest = dest
        self.calls = []

    def create_file_dialog(self, dialog_type, directory="", save_filename="", **kw):
        self.calls.append(
            {"type": dialog_type, "directory": directory, "save_filename": save_filename}
        )
        return self.dest


def _expected_downloads() -> str:
    return str(app.Path.home() / "Downloads")


def test_save_file_opens_panel_in_downloads_and_writes(tmp_path, monkeypatch):
    payload = "Café à 20 ans\n— résumé".encode("utf-8")
    dest = tmp_path / "Café à 20 ans.md"
    monkeypatch.setattr(app.url_request, "urlopen", lambda *a, **k: _FakeResp(payload))
    reveals = []
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: reveals.append(a))

    api = app.Api()
    api._window = _FakeWindow(str(dest))

    res = api.save_file("job123", "Café à 20 ans.md")

    assert res["ok"] is True
    assert res["path"] == str(dest)
    assert dest.read_bytes() == payload
    # Panel was a SAVE dialog, defaulted to ~/Downloads, with the suggested name.
    call = api._window.calls[0]
    assert call["type"] == app.webview.FileDialog.SAVE
    assert call["directory"] == _expected_downloads()
    assert call["save_filename"] == "Café à 20 ans.md"
    # Finder reveal was attempted on the saved file.
    assert reveals and "open" in reveals[0][0] and str(dest) in reveals[0][0]


def test_save_file_returns_cancelled_when_user_dismisses(monkeypatch):
    monkeypatch.setattr(app.url_request, "urlopen", lambda *a, **k: _FakeResp(b"x"))
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: None)

    api = app.Api()
    api._window = _FakeWindow(None)  # user cancelled the panel

    res = api.save_file("job123", "out.md")

    assert res == {"ok": False, "reason": "cancelled"}


def test_save_file_unwraps_sequence_destination(tmp_path, monkeypatch):
    # Some pywebview backends return a tuple/list of paths from the save dialog.
    dest = tmp_path / "out.srt"
    monkeypatch.setattr(app.url_request, "urlopen", lambda *a, **k: _FakeResp(b"sub"))
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: None)

    api = app.Api()
    api._window = _FakeWindow([str(dest)])

    res = api.save_file("job123", "out.srt")

    assert res["ok"] is True
    assert dest.read_bytes() == b"sub"


def test_save_file_reports_fetch_failure(monkeypatch):
    def boom(*a, **k):
        raise app.url_error.URLError("connection refused")

    monkeypatch.setattr(app.url_request, "urlopen", boom)

    api = app.Api()
    api._window = _FakeWindow("/tmp/whatever.md")

    res = api.save_file("job123", "out.md")

    assert res["ok"] is False
    assert res["reason"].startswith("fetch:")
    # The panel was never even opened.
    assert api._window.calls == []


def test_save_file_reveal_failure_is_non_fatal(tmp_path, monkeypatch):
    # Even if `open -R` blows up, the file is saved and we still report ok.
    dest = tmp_path / "out.md"
    monkeypatch.setattr(app.url_request, "urlopen", lambda *a, **k: _FakeResp(b"data"))

    def boom(*a, **k):
        raise OSError("open not found")

    monkeypatch.setattr(app.subprocess, "run", boom)

    api = app.Api()
    api._window = _FakeWindow(str(dest))

    res = api.save_file("job123", "out.md")

    assert res["ok"] is True
    assert dest.read_bytes() == b"data"
