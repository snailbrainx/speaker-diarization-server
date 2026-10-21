"""SNAP_TO_TURNS (segments widened to their speaker's diarization turn) and diarize=False (no diarization)."""
from types import SimpleNamespace

import pytest

import app.diarization as diarization
from app.diarization import SpeakerRecognitionEngine, _one_speaker, _snap_to_turns


def _seg(start, end, label):
    return {"start": start, "end": end, "text": "w", "words": [], "speaker_label": label}


TURNS = [{"start": 0.0, "end": 2.0, "speaker": "A"}, {"start": 1.8, "end": 5.0, "speaker": "B"},
         {"start": 5.0, "end": 9.0, "speaker": "A"}]


def test_widened_to_own_turn_never_into_neighbours():
    segs = _snap_to_turns([_seg(0.5, 1.5, "A"), _seg(2.4, 4.0, "B"), _seg(6.0, 7.0, "A"), _seg(7.5, 8.0, "A")], TURNS)
    # B's turn starts at 1.8 but A's segment before it ends at 1.5; A's last turn ends at 9.0
    assert [(s["start"], s["end"]) for s in segs] == [(0.0, 2.0), (2.0, 5.0), (5.0, 7.5), (7.5, 9.0)]
    assert all(a["end"] <= b["start"] for a, b in zip(segs, segs[1:]))


def test_edges_outside_own_turns_and_overlapping_segments_are_kept():
    segs = [_seg(4.8, 5.4, "B"), _seg(5.2, 6.0, "A"), _seg(9.5, 10.0, "A")]
    out = _snap_to_turns(segs, TURNS)
    # B's segment ends after its turn: end unchanged; A's overlapping start is never moved later (its end still
    # reaches the next segment); a segment past every turn of its speaker stays as it was
    assert [(s["start"], s["end"]) for s in out] == [(1.8, 5.4), (5.2, 9.0), (9.5, 10.0)]
    assert segs[0]["start"] == 4.8  # the input is not modified
    assert _snap_to_turns([_seg(1.0, 2.0, None)], TURNS)[0]["start"] == 1.0


def test_one_speaker_turns_are_the_transcript_segments():
    out = _one_speaker([{"start": 0.0, "end": 1.0}, {"start": 3.0, "end": 4.5}])
    assert out["num_speakers"] == 1 and out["segments"] is out["exclusive_segments"]
    assert [(t["start"], t["end"], t["speaker"]) for t in out["segments"]] == [(0, 1, "SPEAKER_00"), (3, 4.5, "SPEAKER_00")]
    assert _one_speaker([])["num_speakers"] == 0


def _engine(monkeypatch, diarize):
    """The real transcribe_with_diarization over two transcribed words, every model replaced by a fake."""
    words = lambda w, s, e: [{"word": " " + w, "start": s, "end": e}]  # noqa: E731
    eng = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    eng.transcribe = lambda f, audio=None: [{"start": 0.5, "end": 1.0, "text": "hi", "words": words("hi", 0.5, 1.0)},
                                            {"start": 3.0, "end": 4.0, "text": "bye", "words": words("bye", 3.0, 4.0)}]
    eng.diarize = diarize
    eng.clear_gpu_cache_async = lambda *a: None
    eng.embedded = []
    eng._segment_embedding_or_none = lambda f, s: eng.embedded.append((s["start"], s["end"]))
    eng.preload_emotion_audio = lambda f: None
    eng.extract_emotion = lambda *a, **k: None
    eng.labels = []

    def identify(audio_file, segments, diar, known, threshold, db):
        eng.labels.extend(label for _, label, _ in segments)
        return {label: eng._speaker("Sam", 1, True, 0.9) for _, label, _ in segments}

    eng._identify_clusters = identify
    monkeypatch.setattr(diarization, "_in_memory", lambda f: None)
    monkeypatch.setattr(diarization, "get_config", lambda: SimpleNamespace(
        get_settings=lambda: SimpleNamespace(filter_hallucinations=False, emotion_threshold=0.5)))
    return eng


def test_diarize_false_skips_diarization_and_names_the_one_speaker(monkeypatch):
    eng = _engine(monkeypatch, lambda f, audio=None: pytest.fail("diarization ran"))
    out = eng.transcribe_with_diarization("a.wav", cluster_speakers=True, diarize=False)
    assert out["num_speakers"] == 1 and eng.labels == ["SPEAKER_00", "SPEAKER_00"]
    assert [(s["speaker"], s["start"], s["end"]) for s in out["segments"]] == [("Sam", 0.5, 1.0), ("Sam", 3.0, 4.0)]


def test_segments_widened_to_turns_but_voices_embedded_from_their_words(monkeypatch):
    turns = [{"start": 0.0, "end": 2.0, "speaker": "A", "duration": 2.0},
             {"start": 2.5, "end": 5.0, "speaker": "B", "duration": 2.5}]
    eng = _engine(monkeypatch, lambda f, audio=None: {"segments": turns, "exclusive_segments": turns, "num_speakers": 2})
    out = eng.transcribe_with_diarization("a.wav", cluster_speakers=True)
    assert [(s["speaker_label"], s["start"], s["end"]) for s in out["segments"]] == [("A", 0.0, 2.0), ("B", 2.5, 5.0)]
    assert eng.embedded == [(0.5, 1.0), (3.0, 4.0)]
