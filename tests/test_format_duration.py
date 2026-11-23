"""Tests for _format_duration — the human-readable duration formatter
used in the pre-transcribe ETA status message."""
import pytest

from transcribe import _format_duration


@pytest.mark.parametrize("seconds,expected", [
    (0, "?"),
    (-1, "?"),
    (-100, "?"),
    (1, "1s"),
    (45, "45s"),
    (59, "59s"),
    (60, "1min00"),
    (125, "2min05"),
    (3599, "59min59"),
    (3600, "1h00"),
    (3661, "1h01"),
    (8461.77, "2h21"),       # the 2h21 screen recording that prompted this feature
    (36000, "10h00"),
])
def test_format_duration(seconds, expected):
    assert _format_duration(seconds) == expected
