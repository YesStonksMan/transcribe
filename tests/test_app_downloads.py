"""Regression test: the native app must enable WKWebView downloads.

pywebview's ``ALLOW_DOWNLOADS`` setting defaults to ``False``. While it is
``False`` the macOS Cocoa backend (WKWebView) silently *cancels* any navigation
whose response is an attachment / non-renderable MIME type — which is exactly
what every "Télécharger" link is (``application/octet-stream`` +
``Content-Disposition: attachment``). The click does nothing and no file is
saved.

This bug only manifests inside the bundled ``.app`` (WKWebView); it never
appears in a normal dev browser, so it is easy to reintroduce. The native entry
point (``app.py``) must flip the flag on at import time.
"""
import webview


def test_native_app_enables_webview_downloads():
    # Importing the native entry point must enable downloads as a side effect of
    # its module-level configuration (before any window is created).
    import app  # noqa: F401

    assert webview.settings["ALLOW_DOWNLOADS"] is True
