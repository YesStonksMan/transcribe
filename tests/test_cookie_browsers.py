import transcribe as t


def test_instagram_goes_straight_to_cookies():
    assert t._needs_login("https://www.instagram.com/reel/abc/")
    assert t._needs_login("https://instagram.com/p/abc")
    assert not t._needs_login("https://www.youtube.com/watch?v=x")
    assert not t._needs_login("https://notinstagram.com/x")


def test_brave_tried_first_and_safari_skipped_when_unreadable(monkeypatch):
    monkeypatch.setattr(t.os.path, "isdir", lambda p: True)
    monkeypatch.setattr(t, "_safari_cookies_readable", lambda: False)
    browsers = t._browsers_to_try()
    assert browsers[0] == "brave"
    assert "safari" not in browsers
