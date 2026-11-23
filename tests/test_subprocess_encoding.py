"""Regression tests: subprocess output must be decoded as UTF-8, not as the
ambient locale encoding.

When the bundled .app is launched from the Dock/Finder, the process inherits a
minimal environment with no LANG/LC_*, so the interpreter's preferred encoding
falls back to ASCII. ``text=True`` without an explicit ``encoding=`` then decodes
yt-dlp / ffprobe output with that ASCII codec — and the moment a video title or
filename contains an accent (byte 0xc3…), it crashes with:

    UnicodeDecodeError: 'ascii' codec can't decode byte 0xc3 ...

yt-dlp and ffprobe always emit UTF-8, so every text-mode subprocess read must
pin ``encoding="utf-8"`` to be correct regardless of how the app was launched.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import transcribe


def test_probe_url_meta_forces_utf8_and_keeps_accents():
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = "Café à 20 ans|creator|id1|tiktok\n"
    with patch.object(transcribe.subprocess, "run", return_value=fake) as run:
        meta = transcribe._probe_url_meta("https://example.com/x")

    assert run.call_args.kwargs.get("encoding") == "utf-8"
    assert meta is not None
    assert meta["title"] == "Café à 20 ans"


def test_run_ytdlp_forces_utf8():
    fake_proc = MagicMock()
    fake_proc.stdout = iter(["title=Café à 20 ans\n", "[download] 100% of 1MiB\n"])
    fake_proc.returncode = 0
    with patch.object(transcribe.subprocess, "Popen", return_value=fake_proc) as popen:
        list(transcribe._run_ytdlp(["yt-dlp", "url"], Path("/tmp")))

    assert popen.call_args.kwargs.get("encoding") == "utf-8"


def test_probe_audio_duration_forces_utf8():
    fake = MagicMock()
    fake.returncode = 0
    fake.stdout = "123.45\n"
    with patch.object(transcribe.subprocess, "run", return_value=fake) as run:
        duration = transcribe._probe_audio_duration(Path("/tmp/clip.wav"))

    assert run.call_args.kwargs.get("encoding") == "utf-8"
    assert duration == 123.45
