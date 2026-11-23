"""Splitting of over-long VAD speech regions (_VAD_SPLIT_SRC).

Regression for 12-10-2026: continuous speech (jump-cut Reels, no ≥250 ms pause)
was cut at fixed 18 s offsets — mid-word, and leaving a remainder of any length
(37.9 s → 18 + 18 + 1.9 s). Prompted greedy decodes of such fragments invent
words, so cuts must land on a quiet frame and no piece may be a sliver.
"""
import math
import random

import transcribe

MAX = transcribe.MLX_MAX_CHUNK_S


def _split_fn():
    ns = {}
    exec(transcribe._VAD_SPLIT_SRC, ns)
    return ns["split_region"], ns["SPLIT_FRAME_S"]


def _loud(duration_s, frame_s):
    return [1.0] * round(duration_s / frame_s)


def _check_tiling(pieces, start, end):
    assert pieces[0]["start"] == start and pieces[-1]["end"] == end
    for a, b in zip(pieces, pieces[1:]):
        assert a["end"] == b["start"]
    for p in pieces:
        assert p["end"] - p["start"] <= MAX + 1e-9    # (t + 18) - t is not 18.0 in floats


def test_continuous_38s_region_has_no_short_remainder_and_cuts_on_quiet_frames():
    split_region, frame = _split_fn()
    energy = _loud(37.9, frame)                # the measured case: old split left 1.9 s
    for t in (11.0, 17.0, 25.5):               # 17.0 is quiet too, but outside every search window
        energy[round(t / frame)] = 0.0

    pieces = split_region(0.0, 37.9, MAX, energy)

    _check_tiling(pieces, 0.0, 37.9)
    assert len(pieces) == math.ceil(37.9 / MAX)              # no extra piece
    assert min(p["end"] - p["start"] for p in pieces) >= 3.0
    assert [round(p["end"], 2) for p in pieces[:-1]] == [11.0, 25.5]


def test_just_over_the_limit_splits_in_two_balanced_pieces():
    split_region, frame = _split_fn()
    pieces = split_region(100.0, 118.5, MAX, _loud(18.5, frame))   # old: 18 s + 0.5 s
    _check_tiling(pieces, 100.0, 118.5)
    assert len(pieces) == 2
    assert min(p["end"] - p["start"] for p in pieces) >= 3.0


def test_quiet_frame_past_the_ceiling_is_never_chosen():
    split_region, frame = _split_fn()
    energy = _loud(35.0, frame)
    energy[round(18.4 / frame)] = 0.0          # quietest frame, but the piece would be 18.4 s
    _check_tiling(split_region(0.0, 35.0, MAX, energy), 0.0, 35.0)


def test_cut_on_the_lowest_allowed_frame_does_not_add_a_piece():
    """Rising energy pins every cut to the window's low bound; frame→seconds
    rounding landed it 1e-14 s under, and the rest then needed one piece more."""
    split_region, frame = _split_fn()
    energy = list(range(round(66.26 / frame)))
    pieces = split_region(52.91, 52.91 + 66.26, MAX, energy)
    _check_tiling(pieces, 52.91, 52.91 + 66.26)
    assert len(pieces) == 4


def test_invariants_hold_for_any_length_and_energy():
    split_region, frame = _split_fn()
    rng = random.Random(12102026)
    for _ in range(400):
        start = round(rng.uniform(0, 3600), 3)
        dur = rng.uniform(MAX + 0.01, 8 * MAX)
        energy = [rng.random() for _ in range(int(dur / frame))]
        pieces = split_region(start, start + dur, MAX, energy)
        _check_tiling(pieces, start, start + dur)
        assert len(pieces) == math.ceil(dur / MAX)
        assert min(p["end"] - p["start"] for p in pieces) >= 3.0
