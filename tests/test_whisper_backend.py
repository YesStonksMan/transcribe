"""Tests for the MLX-Whisper backend wrapper in transcribe.py."""
import time
from pathlib import Path
from unittest.mock import patch

import transcribe


def test_run_whisper_returns_segments_language_duration():
    """_run_whisper unpacks mlx_whisper.transcribe's dict into the tuple shape
    that transcribe_stream needs."""
    fake_mlx_result = {
        "text": "Bonjour le monde.",
        "language": "fr",
        "segments": [
            {"start": 0.0, "end": 1.5, "text": "Bonjour"},
            {"start": 1.5, "end": 3.0, "text": "le monde."},
        ],
    }
    with patch("transcribe.mlx_whisper.transcribe", return_value=fake_mlx_result) as m:
        segments, language, duration = transcribe._run_whisper(
            Path("/tmp/fake.wav"), language="fr"
        )

    assert language == "fr"
    assert duration == 3.0
    assert len(segments) == 2
    assert segments[0] == {"start": 0.0, "end": 1.5, "text": "Bonjour"}
    assert segments[1] == {"start": 1.5, "end": 3.0, "text": "le monde."}
    m.assert_called_once()
    call_kwargs = m.call_args.kwargs
    assert call_kwargs["language"] == "fr"
    assert call_kwargs["path_or_hf_repo"] == transcribe.MLX_WHISPER_REPO


def test_run_whisper_handles_empty_segments():
    """Silent/empty audio → empty segments, duration 0.0, no crash."""
    fake_mlx_result = {"text": "", "language": "fr", "segments": []}
    with patch("transcribe.mlx_whisper.transcribe", return_value=fake_mlx_result):
        segments, language, duration = transcribe._run_whisper(
            Path("/tmp/silent.wav"), language=None
        )

    assert segments == []
    assert language == "fr"
    assert duration == 0.0


def test_run_whisper_propagates_language_none():
    """When language=None we let mlx_whisper auto-detect."""
    fake_mlx_result = {"text": "hi", "language": "en", "segments": [
        {"start": 0.0, "end": 0.5, "text": "hi"},
    ]}
    with patch("transcribe.mlx_whisper.transcribe", return_value=fake_mlx_result) as m:
        _, detected, _ = transcribe._run_whisper(Path("/tmp/x.wav"), language=None)

    assert detected == "en"
    assert m.call_args.kwargs["language"] is None


def test_run_whisper_with_heartbeats_yields_pings_while_running():
    """Heartbeats fire while the background thread is still working, and the
    final triple is returned via StopIteration.value (consumed by `yield from`)."""
    sentinel_segments = [{"start": 0.0, "end": 1.0, "text": "hi"}]

    def slow_run_whisper(audio_path, language=None, model_repo=None, on_chunk=None, vocabulary=None):
        time.sleep(0.25)  # long enough for at least 2 heartbeats at interval=0.1
        return (sentinel_segments, "en", 1.0)

    with patch.object(transcribe, "_run_whisper", side_effect=slow_run_whisper):
        gen = transcribe._run_whisper_with_heartbeats(
            Path("/tmp/x.wav"), language=None, interval=0.1
        )
        events = []
        try:
            while True:
                events.append(next(gen))
        except StopIteration as stop:
            final = stop.value

    assert len(events) >= 2, f"expected ≥2 heartbeats, got {len(events)}"
    for ev in events:
        assert ev["type"] == "heartbeat"
        assert ev["stage"] == "transcribe"
        assert ev["elapsed"] >= 0
        assert "écoulé" in ev["msg"]

    assert final == (sentinel_segments, "en", 1.0)


def test_run_whisper_with_heartbeats_propagates_exceptions():
    """If the background thread raises, the exception bubbles to the caller."""
    def boom(audio_path, language=None, model_repo=None, on_chunk=None, vocabulary=None):
        raise RuntimeError("MLX exploded")

    with patch.object(transcribe, "_run_whisper", side_effect=boom):
        gen = transcribe._run_whisper_with_heartbeats(
            Path("/tmp/x.wav"), language=None, interval=0.05
        )
        try:
            for _ in gen:
                pass
        except RuntimeError as e:
            assert str(e) == "MLX exploded"
        else:
            raise AssertionError("expected RuntimeError to bubble up")


def test_run_whisper_uses_provided_repo_over_default():
    fake_mlx_result = {"text": "hi", "language": "en", "segments": [
        {"start": 0.0, "end": 0.5, "text": "hi"},
    ]}
    with patch.object(transcribe, "mlx_whisper") as m:
        m.transcribe.return_value = fake_mlx_result
        transcribe._run_whisper(
            Path("/tmp/x.wav"),
            language=None,
            model_repo="mlx-community/whisper-large-v3-turbo",
        )

    kwargs = m.transcribe.call_args.kwargs
    assert kwargs["path_or_hf_repo"] == "mlx-community/whisper-large-v3-turbo"


def test_run_whisper_default_repo_when_no_override():
    fake_mlx_result = {"text": "", "language": "en", "segments": []}
    with patch.object(transcribe, "mlx_whisper") as m:
        m.transcribe.return_value = fake_mlx_result
        transcribe._run_whisper(Path("/tmp/x.wav"), language=None)

    kwargs = m.transcribe.call_args.kwargs
    assert kwargs["path_or_hf_repo"] == transcribe.MLX_WHISPER_REPO


def test_external_env_strips_python_launcher_vars(monkeypatch):
    """_external_env must remove PYTHONHOME/PYTHONPATH etc. so external Python
    tools (yt-dlp) boot their own interpreter instead of the bundle's."""
    monkeypatch.setenv("PYTHONHOME", "/some/bundle")
    monkeypatch.setenv("PYTHONPATH", "/some/bundle/lib")
    monkeypatch.setenv("PATH", "/opt/homebrew/bin:/usr/bin")
    env = transcribe._external_env()
    assert "PYTHONHOME" not in env
    assert "PYTHONPATH" not in env
    # PATH preserved
    assert env["PATH"] == "/opt/homebrew/bin:/usr/bin"


# ---------- custom vocabulary prompt ----------
def test_vocabulary_prompt_is_none_when_unset():
    assert transcribe._vocabulary_prompt(None) is None
    assert transcribe._vocabulary_prompt("") is None
    assert transcribe._vocabulary_prompt("   ") is None


def test_vocabulary_prompt_wraps_the_terms_in_a_sentence():
    p = transcribe._vocabulary_prompt("Kakashi, ETF", "fr")
    assert "Kakashi, ETF" in p
    # a bare CSV biases worse than something that reads like speech
    assert p.endswith(".") and p != "Kakashi, ETF."


def test_vocabulary_prompt_only_fires_on_french():
    """The carrier sentence is French and the decoder continues whatever it is
    conditioned on, so sending it over non-French audio drags the transcript
    into French. Anything but `fr` — including an unresolved language — must
    yield no prompt at all."""
    assert transcribe._vocabulary_prompt("Kakashi, ETF", "en") is None
    assert transcribe._vocabulary_prompt("Kakashi, ETF", None) is None
    assert transcribe._vocabulary_prompt("Kakashi, ETF", "") is None
    assert transcribe._vocabulary_prompt("Kakashi, ETF", "de") is None
    # case and region suffixes still resolve to French
    assert transcribe._vocabulary_prompt("Kakashi", "FR") is not None
    assert transcribe._vocabulary_prompt("Kakashi", "fr-CA") is not None


def test_vocabulary_reaches_every_chunk(tmp_path):
    """The prompt must be re-sent per chunk: Whisper conditions only its first
    30s window on initial_prompt, and each VAD chunk is its own call."""
    audio = tmp_path / "a.wav"; audio.write_bytes(b"")
    chunks = [{"start": 0.0, "end": 5.0}, {"start": 6.0, "end": 11.0}]
    seen = []

    def fake_transcribe(wav, repo, language, prompt=None):
        seen.append(prompt)
        return [{"start": 0.0, "end": 1.0, "text": "ok"}], "fr"

    with patch.object(transcribe, "_vad_chunks", return_value=chunks), \
         patch.object(transcribe, "_slice_wav", return_value=True), \
         patch.object(transcribe, "_mlx_transcribe_file", side_effect=fake_transcribe):
        transcribe._run_mlx_whisper(audio, language="fr", vocabulary="Kakashi")

    assert len(seen) == 2
    assert all(p and "Kakashi" in p for p in seen)


def test_vocabulary_never_reaches_chunks_on_non_french_audio(tmp_path):
    """The French carrier must not leak onto English audio — that is what made
    an English video transcribe as French (04-08)."""
    audio = tmp_path / "a.wav"; audio.write_bytes(b"")
    chunks = [{"start": 0.0, "end": 5.0}, {"start": 6.0, "end": 11.0}]
    seen = []

    def fake_transcribe(wav, repo, language, prompt=None):
        seen.append(prompt)
        return [{"start": 0.0, "end": 1.0, "text": "ok"}], "en"

    with patch.object(transcribe, "_vad_chunks", return_value=chunks), \
         patch.object(transcribe, "_slice_wav", return_value=True), \
         patch.object(transcribe, "_mlx_transcribe_file", side_effect=fake_transcribe):
        transcribe._run_mlx_whisper(audio, language="en", vocabulary="Kakashi")

    assert len(seen) == 2
    assert all(p is None for p in seen)


def test_language_probe_decides_the_prompt_when_language_is_auto(tmp_path):
    """On auto, the probe result — not the caller — gates the vocabulary."""
    audio = tmp_path / "a.wav"; audio.write_bytes(b"")
    chunks = [{"start": 0.0, "end": 5.0}]
    seen = []

    def fake_transcribe(wav, repo, language, prompt=None):
        seen.append((language, prompt))
        return [{"start": 0.0, "end": 1.0, "text": "ok"}], "en"

    with patch.object(transcribe, "_vad_chunks", return_value=chunks), \
         patch.object(transcribe, "_slice_wav", return_value=True), \
         patch.object(transcribe, "_detect_language_probe", return_value="en"), \
         patch.object(transcribe, "_mlx_transcribe_file", side_effect=fake_transcribe):
        _, detected, _ = transcribe._run_mlx_whisper(
            audio, language=None, vocabulary="Kakashi"
        )

    assert detected == "en"
    assert seen == [("en", None)]   # probe pinned it, and it suppressed the prompt


# ---------- prompt fallback (12-10 regression) ----------
def _fake_decode(prompted_temp, seek=0):
    """mlx_whisper.transcribe stand-in: the prompted decode comes back at
    `prompted_temp`, the unprompted one cleanly at temperature 0."""
    def fake(path, **kw):
        if kw.get("initial_prompt"):
            return {"language": "fr", "segments": [
                {"seek": seek, "start": 0.0, "end": 1.0, "text": "Votre Votre Votre",
                 "temperature": prompted_temp},
            ]}
        return {"language": "fr", "segments": [
            {"seek": 0, "start": 0.0, "end": 1.0, "text": "plus forte", "temperature": 0.0},
        ]}
    return fake


def test_prompted_decode_that_fell_back_is_redone_without_prompt():
    """12-10, real French speech cut mid-word by the 18s VAD split: with the
    vocabulary prompt the greedy decode trips Whisper's own quality gate, the
    temperature fallback then samples "Votre Votre…" / "Sous-titrage Société
    Radio-Canada", while the same chunk decodes cleanly at temperature 0 with
    no prompt. A prompted result that needed fallback is discarded."""
    with patch("transcribe.mlx_whisper.transcribe", side_effect=_fake_decode(1.0)) as m:
        segs, _ = transcribe._mlx_transcribe_file(Path("/tmp/c.wav"), "repo", "fr", "Voici Kakashi.")

    assert [s["text"] for s in segs] == ["plus forte"]
    assert [c.kwargs["initial_prompt"] for c in m.call_args_list] == ["Voici Kakashi.", None]


def test_clean_prompted_decode_is_kept():
    with patch("transcribe.mlx_whisper.transcribe", side_effect=_fake_decode(0.0)) as m:
        segs, _ = transcribe._mlx_transcribe_file(Path("/tmp/c.wav"), "repo", "fr", "Voici Kakashi.")

    assert m.call_count == 1
    assert segs[0]["text"] == "Votre Votre Votre"


def test_fallback_outside_the_prompted_window_is_not_retried():
    """Whisper drops initial_prompt after the first window (seek 0) when
    condition_on_previous_text is False, so a fallback later in a whole-file
    decode is not the prompt's doing — re-decoding would only lose the vocabulary."""
    with patch("transcribe.mlx_whisper.transcribe", side_effect=_fake_decode(1.0, seek=3000)) as m:
        transcribe._mlx_transcribe_file(Path("/tmp/c.wav"), "repo", "fr", "Voici Kakashi.")

    assert m.call_count == 1
