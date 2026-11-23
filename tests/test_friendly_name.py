"""Tests for _friendly_name_from_meta — ensures generic Instagram titles
('Video by <user>', 'Reel', 'Untitled', …) fall back to a composed unique name."""
from transcribe import _friendly_name_from_meta


def test_real_title_is_used():
    meta = {
        "title": "Rick Astley - Never Gonna Give You Up",
        "uploader": "Rick Astley",
        "id": "dQw4w9WgXcQ",
        "extractor": "youtube",
    }
    assert _friendly_name_from_meta(meta) == "Rick Astley - Never Gonna Give You Up"


def test_instagram_video_by_falls_back_to_composed():
    meta = {
        "title": "Video by some_creator",
        "uploader": "some_creator",
        "id": "DT6DPYwCqMW",
        "extractor": "Instagram",
    }
    result = _friendly_name_from_meta(meta)
    assert "DT6DPYwCqMW" in result, f"Expected ID in result, got: {result}"
    assert result == "Instagram - some_creator - DT6DPYwCqMW"


def test_reel_alone_falls_back():
    meta = {"title": "Reel", "uploader": "u", "id": "ABC123", "extractor": "Instagram"}
    assert "ABC123" in _friendly_name_from_meta(meta)


def test_post_by_falls_back():
    meta = {"title": "Post by someone", "uploader": "someone", "id": "XYZ", "extractor": "Instagram"}
    assert "XYZ" in _friendly_name_from_meta(meta)


def test_untitled_falls_back():
    meta = {"title": "Untitled video", "uploader": "u", "id": "i", "extractor": "x"}
    assert "i" in _friendly_name_from_meta(meta)


def test_empty_title_falls_back():
    meta = {"title": "", "uploader": "user", "id": "vid42", "extractor": "tiktok"}
    assert _friendly_name_from_meta(meta) == "tiktok - user - vid42"


def test_na_title_falls_back():
    meta = {"title": "NA", "uploader": "u", "id": "vid42", "extractor": "tiktok"}
    assert "vid42" in _friendly_name_from_meta(meta)


def test_no_metadata_at_all_fallback():
    meta = {"title": "", "uploader": "", "id": "", "extractor": ""}
    assert _friendly_name_from_meta(meta) == "video"
