from transcribe import build_md_plain, build_md_raw

SEGS = [
    {"start": 0.0, "end": 2.0, "text": " Salut à tous."},
    {"start": 2.3, "end": 4.0, "text": "On parle ETF."},
    {"start": 7.5, "end": 9.0, "text": "Nouveau sujet."},
    {"start": 9.1, "end": 9.5, "text": "   "},
]


def test_raw_has_no_header_and_keeps_timestamps():
    md = build_md_raw(SEGS)
    assert "Transcription brute" not in md
    assert "Langue détectée" not in md
    assert md.startswith("`[00:00:00]` Salut à tous.")
    assert "`[00:00:07]` Nouveau sujet." in md


def test_plain_drops_timestamps_and_breaks_on_pauses():
    md = build_md_plain(SEGS)
    assert md == "Salut à tous. On parle ETF.\n\nNouveau sujet.\n"
