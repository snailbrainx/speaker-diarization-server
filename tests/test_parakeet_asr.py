"""ASR_BACKEND=parakeet: NeMo hypotheses become the same transcription segments as the Whisper path.

No NeMo needed: hypotheses and the model are small fakes with the attributes NeMo's Hypothesis has
(timestamp['word'/'segment'/'timestep'], token_confidence), and VAD / audio decoding are replaced per test.
"""
import sys
import threading
import types

import numpy as np
import pytest

import app.diarization as diarization
from app.diarization import (SpeakerRecognitionEngine, _load_audio_16k, _parakeet_segments, _parakeet_word_confidence,
                             _chunks_at_pauses, _split_by_speaker)

SR = 16000


def _hyp(sentences, conf=True):
    """Fake NeMo hypothesis from [(sentence text, [(word, start, end), ...]), ...]: 10 ms frames, two tokens per word
    (probabilities 0.8 and 1.0) at its first two frames."""
    words, steps, token_conf = [], [], []
    for _, ws in sentences:
        for w, s, e in ws:
            f0, f1 = round(s * 100), round(e * 100)
            words.append({"word": w, "start": s, "end": e, "start_offset": f0, "end_offset": f1})
            steps += [f0, f0 + 1]
            token_conf += [0.8, 1.0]
    return types.SimpleNamespace(
        text=" ".join(t for t, _ in sentences),
        timestamp={"word": words, "timestep": steps,
                   "segment": [{"segment": t, "start": ws[0][1], "end": ws[-1][2]} for t, ws in sentences]},
        token_confidence=token_conf if conf else None)


TWO_SENTENCES = [("Hello there.", [("Hello", 0.1, 0.4), ("there.", 0.5, 0.9)]),
                 ("How are you?", [("How", 2.5, 2.7), ("are", 2.7, 2.8), ("you?", 2.8, 3.1)])]


def test_segments_in_whisper_format_split_at_pauses():
    segs = _parakeet_segments(_hyp(TWO_SENTENCES), offset=10.0)  # 1.6 s pause between the sentences
    assert [s["text"] for s in segs] == ["Hello there.", "How are you?"]
    assert (segs[0]["start"], segs[0]["end"]) == (10.1, 10.9)
    assert (segs[1]["start"], segs[1]["end"]) == pytest.approx((12.5, 13.1))
    w = segs[1]["words"][0]
    assert w["word"] == " How" and w["start"] == pytest.approx(12.5) and w["probability"] == pytest.approx(0.9)
    assert segs[0]["avg_logprob"] == pytest.approx(np.log(0.9))


def test_sentences_without_a_pause_join_up_to_the_length_limit():
    close = [("Hello there.", [("Hello", 0.1, 0.4), ("there.", 0.5, 0.9)]),
             ("How are you?", [("How", 1.5, 1.7), ("are", 1.7, 1.8), ("you?", 1.8, 2.1)]),
             ("Fine.", [("Fine.", 2.5, 31.0)])]  # would make the segment longer than SPLIT_MAX_SEC
    segs = _parakeet_segments(_hyp(close), offset=0.0)
    assert [s["text"] for s in segs] == ["Hello there. How are you?", "Fine."]
    assert len(segs[0]["words"]) == 5 and (segs[0]["start"], segs[0]["end"]) == (0.1, 2.1)


def test_words_kept_when_sentence_split_does_not_line_up():
    h = _hyp(TWO_SENTENCES)
    h.timestamp["segment"][0]["segment"] = "Hello there and more."  # 4 words claimed, 2 present
    segs = _parakeet_segments(h, offset=0.0)
    assert len(segs) == 1 and len(segs[0]["words"]) == 5
    assert segs[0]["text"] == "Hello there. How are you?"


def test_no_confidence_and_empty_hypothesis():
    segs = _parakeet_segments(_hyp(TWO_SENTENCES, conf=False), offset=0.0)
    assert segs[0]["words"][0]["probability"] is None and segs[0]["avg_logprob"] is None
    empty = types.SimpleNamespace(text="", timestamp={"word": [], "segment": [], "timestep": []}, token_confidence=[])
    assert _parakeet_segments(empty, offset=0.0) == []


def test_word_confidence_by_frame_for_glued_tokens():
    # "sense'cause": one word of three tokens, the case NeMo's own word aggregation raises on
    h = types.SimpleNamespace(
        timestamp={"word": [{"word": "sense'cause", "start_offset": 10, "end_offset": 16},
                            {"word": "otherwise", "start_offset": 16, "end_offset": 20}],
                   "timestep": [10, 12, 14, 16, 18]},
        token_confidence=[0.9, 0.6, 0.6, 1.0, 0.5])
    assert _parakeet_word_confidence(h, h.timestamp["word"]) == pytest.approx([0.7, 0.75])
    h.token_confidence = h.token_confidence[:-1]  # tokens and frames do not line up: no confidences
    assert _parakeet_word_confidence(h, h.timestamp["word"]) == [None, None]


def test_short_audio_is_one_chunk_without_vad(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", None)  # importing it would fail
    assert _chunks_at_pauses(np.zeros(5 * SR, dtype=np.float32), max_sec=30) == [(0, 5 * SR)]
    assert _chunks_at_pauses(np.zeros(0, dtype=np.float32), max_sec=30) == []


def _fake_vad(monkeypatch, regions_sec):
    vad = types.ModuleType("faster_whisper.vad")
    vad.VadOptions = lambda **kw: kw
    vad.get_speech_timestamps = lambda audio, opts: [{"start": int(s * SR), "end": int(e * SR)} for s, e in regions_sec]
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", vad)


def test_long_audio_cut_in_pauses_and_fully_covered(monkeypatch):
    _fake_vad(monkeypatch, [(1, 9), (10, 20), (22, 28), (31, 40), (60, 89.9)])
    chunks = _chunks_at_pauses(np.zeros(100 * SR, dtype=np.float32), max_sec=30)
    # cuts in the middle of the pauses (9.5, 21, 29.5, 50), the latest one within 30 s of the chunk start; no pause
    # between 50 and 80 s, so a cut at the limit there
    assert [(s / SR, e / SR) for s, e in chunks] == [(0, 29.5), (29.5, 50), (50, 80), (80, 100)]
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:])) and chunks[-1][1] == 100 * SR


def test_transcribe_parakeet_maps_chunks_back_to_file_time(monkeypatch):
    _fake_vad(monkeypatch, [(2, 10), (40, 48)])
    monkeypatch.setattr(diarization, "_load_audio_16k", lambda path: np.zeros(50 * SR, np.float32))
    monkeypatch.setattr(diarization, "PARAKEET_CHUNK_SEC", 30.0)
    calls = []

    class FakeModel:
        def transcribe(self, arrays, batch_size, timestamps, verbose):
            calls.append([len(a) / SR for a in arrays])
            return [_hyp([("Hi.", [("Hi.", 0.5, 0.8)])]), _hyp([("Bye now.", [("Bye", 0.3, 0.5), ("now.", 0.6, 1.0)])])]

    eng = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    eng._parakeet_model = FakeModel()
    eng._parakeet_lock = threading.Lock()
    eng.clear_gpu_cache_async = lambda *a: None
    segs = eng._transcribe_parakeet("meeting.wav")

    assert calls == [[25, 25]]  # one batch: the file cut in the pause at 25 s
    assert [s["text"] for s in segs] == ["Hi.", "Bye now."]
    assert segs[0]["start"] == pytest.approx(0.5) and segs[1]["words"][1]["end"] == pytest.approx(25 + 1.0)
    # the word-level speaker split works on them unchanged
    turns = [{"start": 0, "end": 25.55, "speaker": "A"}, {"start": 25.55, "end": 50, "speaker": "B"}]
    pieces = _split_by_speaker(segs, turns)
    assert [(p["text"], p["speaker_label"]) for p in pieces] == [("Hi.", "A"), ("Bye", "A"), ("now.", "B")]


def test_load_audio_downmixes_and_resamples(tmp_path):
    sf = pytest.importorskip("soundfile")
    if not hasattr(sf, "write"):
        pytest.skip("soundfile is stubbed")
    t = np.arange(8000 * 2) / 8000
    stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.zeros_like(t)], axis=1).astype(np.float32) * 0.5
    sf.write(tmp_path / "a.wav", stereo, 8000)
    audio = _load_audio_16k(str(tmp_path / "a.wav"))
    assert audio.dtype == np.float32 and audio.ndim == 1 and len(audio) == 2 * SR
    assert 0.2 < np.abs(audio).max() < 0.3  # channel mean of a 0.5 tone and silence


def test_each_whisper_transcription_gets_its_own_batched_pipeline():
    """faster-whisper's batched pipeline carries the last word's end time between batches; one shared by concurrent
    requests aligned one file's words with another's history. The (expensive) model is still shared."""
    fw = pytest.importorskip("faster_whisper")
    if not hasattr(fw, "BatchedInferencePipeline"):
        pytest.skip("faster-whisper is stubbed here")
    eng = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    eng._whisper_model, eng._model_lock = object(), threading.Lock()
    first, second = eng.batched_whisper(), eng.batched_whisper()
    first.last_speech_timestamp = 100.0          # as left by a request still running, or one that raised
    assert second is not first and second.model is first.model is eng._whisper_model
    assert second.last_speech_timestamp == 0.0
