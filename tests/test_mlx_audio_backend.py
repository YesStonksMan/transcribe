"""Tests for the mlx-audio engine (Qwen3-ASR & co., run in an external venv)."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import transcribe

QWEN = "mlx-community/Qwen3-ASR-1.7B-8bit"


class _FakeProc:
    """Stands in for the mlx-audio child process: emits protocol lines."""
    def __init__(self, lines, returncode=0):
        self.stdout = iter(lines)
        self.returncode = returncode
        self.args = None

    def wait(self, timeout=None):
        return self.returncode


def _seg_line(i, segments, language="French"):
    return "__SEG__" + json.dumps({"i": i, "language": language, "segments": segments}) + "\n"


def _popen_capturing(proc, jobs: list):
    """Popen stand-in that records the job JSON (argv[-1]) before the parent's
    temp dir is cleaned up."""
    def fake(argv, **kw):
        jobs.append(json.loads(Path(argv[-1]).read_text()))
        return proc
    return fake


@pytest.fixture
def fake_venv(tmp_path, monkeypatch):
    py = tmp_path / "python"
    py.write_text("")
    monkeypatch.setattr(transcribe, "MLX_AUDIO_PYTHON", py)
    monkeypatch.setattr(transcribe, "_slice_wav", lambda *a: True)
    monkeypatch.setattr(transcribe, "_probe_audio_duration", lambda p: 0.0)
    return py


def test_whisper_repos_stay_on_the_in_process_engine():
    with patch.object(transcribe, "_run_mlx_whisper", return_value=([], "fr", 0.0)) as w, \
         patch.object(transcribe, "_run_mlx_audio") as a:
        transcribe._run_whisper(Path("/tmp/x.wav"), model_repo="mlx-community/whisper-large-v3-mlx")
    w.assert_called_once()
    a.assert_not_called()


def test_non_whisper_repos_route_to_mlx_audio():
    with patch.object(transcribe, "_run_mlx_whisper") as w, \
         patch.object(transcribe, "_run_mlx_audio", return_value=([], "fr", 0.0)) as a:
        transcribe._run_whisper(Path("/tmp/x.wav"), language="fr", model_repo=QWEN, vocabulary="ETF")
    w.assert_not_called()
    assert a.call_args.kwargs["vocabulary"] == "ETF"


def test_missing_venv_raises_a_readable_error(tmp_path, monkeypatch):
    monkeypatch.setattr(transcribe, "MLX_AUDIO_PYTHON", tmp_path / "nope")
    with pytest.raises(RuntimeError, match="mlx-audio"):
        transcribe._run_mlx_audio(Path("/tmp/x.wav"), repo=QWEN)


def test_chunks_are_offset_cleaned_and_reported(fake_venv):
    chunks = [{"start": 0.0, "end": 10.0}, {"start": 0.0, "end": 0.1}, {"start": 30.0, "end": 45.0}]
    proc = _FakeProc([
        "some log noise\n",
        _seg_line(0, [{"start": 0.0, "end": 10.0, "text": " Bonjour Kakashi "}]),
        _seg_line(1, [{"start": 1.0, "end": 15.0, "text": "oui oui oui oui oui"}]),
    ])
    progress, jobs = [], []
    with patch.object(transcribe, "_vad_chunks", return_value=chunks), \
         patch.object(transcribe.subprocess, "Popen", _popen_capturing(proc, jobs)):
        segs, lang, dur = transcribe._run_mlx_audio(
            Path("/tmp/x.wav"), language="fr", repo=QWEN,
            on_chunk=lambda d, t: progress.append((d, t)), vocabulary="Kakashi, ETF\nTrading 212",
        )
    job = jobs[0]

    assert segs == [
        {"start": 0.0, "end": 10.0, "text": "Bonjour Kakashi"},
        {"start": 31.0, "end": 45.0, "text": "oui"},
    ]
    assert (lang, dur) == ("fr", 45.0)
    assert progress == [(0, 2), (1, 2), (2, 2)]          # the 0.1s sliver is dropped
    assert job["repo"] == QWEN
    assert job["language"] == "French"
    assert job["hotwords"] == ["Kakashi", "ETF", "Trading 212"]
    assert len(job["chunks"]) == 2


def test_no_vad_means_one_whole_file_pass_and_language_is_detected(fake_venv):
    proc = _FakeProc([_seg_line(0, [{"start": 0.0, "end": 7.5, "text": "Hello"}], language=["English"])])
    jobs = []
    with patch.object(transcribe, "_vad_chunks", return_value=None), \
         patch.object(transcribe.subprocess, "Popen", _popen_capturing(proc, jobs)):
        segs, lang, dur = transcribe._run_mlx_audio(Path("/tmp/x.wav"), repo=QWEN)
    job = jobs[0]
    assert job["chunks"] == ["/tmp/x.wav"]
    assert job["language"] is None and job["hotwords"] == []
    assert (lang, dur) == ("en", 7.5)


def test_child_failure_surfaces_its_stderr(fake_venv):
    proc = _FakeProc(["Traceback…\n", "ValueError: Model type foo not supported\n"], returncode=1)
    with patch.object(transcribe, "_vad_chunks", return_value=None), \
         patch.object(transcribe.subprocess, "Popen", return_value=proc):
        with pytest.raises(RuntimeError, match="Model type foo not supported"):
            transcribe._run_mlx_audio(Path("/tmp/x.wav"), repo=QWEN)


def test_segments_without_timing_span_their_whole_chunk(fake_venv):
    chunks = [{"start": 20.0, "end": 32.0}]
    proc = _FakeProc([_seg_line(0, [{"start": 0.0, "end": 0.0, "text": "Pas d'horodatage"}])])
    with patch.object(transcribe, "_vad_chunks", return_value=chunks), \
         patch.object(transcribe.subprocess, "Popen", return_value=proc):
        segs, _, dur = transcribe._run_mlx_audio(Path("/tmp/x.wav"), repo=QWEN)
    assert segs == [{"start": 20.0, "end": 32.0, "text": "Pas d'horodatage"}]
    assert dur == 32.0


def test_child_script_reads_dict_segments_and_object_sentences():
    """Qwen3 hands back dict `segments`; Parakeet hands back `sentences` objects."""
    ns = {}
    exec(transcribe._MLX_AUDIO_SEGMENTS_SRC, ns)
    to_segments = ns["to_segments"]

    class Qwen:  text = "a"; segments = [{"start": 0.0, "end": 5.0, "text": "a"}]
    class Sent:  start, end, text = 0.24, 17.36, " b"
    class Parakeet:  text = " b"; sentences = [Sent()]
    class Bare:  text = "c"

    assert to_segments(Qwen()) == [{"start": 0.0, "end": 5.0, "text": "a"}]
    assert to_segments(Parakeet()) == [{"start": 0.24, "end": 17.36, "text": " b"}]
    assert to_segments(Bare()) == [{"start": 0.0, "end": 0.0, "text": "c"}]


# ---------- Qwen3 ForcedAligner (word timing → sentence segments) ----------

def _align_fn():
    ns = {}
    exec(transcribe._MLX_AUDIO_SEGMENTS_SRC, ns)
    return ns["align_segments"]


def test_aligned_words_are_regrouped_into_punctuated_sentences():
    align = _align_fn()
    text = "Moi je joue safe. Livret A, peut-être un appart."
    words = [("Moi", 0.9, 1.1), ("je", 1.1, 1.2), ("joue", 1.2, 1.3), ("safe", 1.3, 1.6),
             ("Livret", 2.0, 2.4), ("A", 2.4, 2.5), ("peutêtre", 2.6, 3.0), ("un", 3.0, 3.1),
             ("appart", 3.1, 3.6)]
    assert align(text, words) == [
        {"start": 0.9, "end": 1.6, "text": "Moi je joue safe."},
        {"start": 2.0, "end": 3.6, "text": "Livret A, peut-être un appart."},
    ]


def test_unpunctuated_text_splits_on_pauses():
    align = _align_fn()
    words = [("au", 0.0, 0.2), ("moins", 0.2, 0.5), ("je", 1.6, 1.7), ("perds", 1.7, 2.0)]
    assert align("au moins je perds", words) == [
        {"start": 0.0, "end": 0.5, "text": "au moins"},
        {"start": 1.6, "end": 2.0, "text": "je perds"},
    ]


def test_alignment_that_does_not_match_the_text_gives_up():
    align = _align_fn()
    assert align("bonjour le monde", [("bonjour", 0.0, 0.5), ("tout", 0.5, 0.8)]) is None
    assert align("bonjour", []) is None


def test_qwen3_asr_repos_get_the_forced_aligner():
    import settings
    assert settings.aligner_for_repo("mlx-community/Qwen3-ASR-1.7B-8bit") == settings.QWEN3_ALIGNER_REPO
    assert settings.aligner_for_repo("mlx-community/parakeet-tdt-0.6b-v3") is None


def test_job_carries_the_aligner_for_qwen3(fake_venv):
    proc, jobs = _FakeProc([]), []
    with patch.object(transcribe, "_vad_chunks", return_value=None), \
         patch.object(transcribe.subprocess, "Popen", _popen_capturing(proc, jobs)):
        transcribe._run_mlx_audio(Path("/tmp/x.wav"), repo=QWEN)
    import settings
    assert jobs[0]["aligner"] == settings.QWEN3_ALIGNER_REPO
