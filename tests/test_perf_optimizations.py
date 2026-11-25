"""Tests for Phase 1+2A latency & throughput optimizations:
- In-memory 16 kHz ndarray slicing for emotion (no second full-file decode)
- Length-bucketed batched emotion extraction with padding mask
- Periodic GPU cleanup respects active inference slots
- Low-latency live flush override & in-memory audio attachment
- One-speaker (diarize=False) short-reply track inheritance
"""
import threading
import types

import numpy as np
import pytest

from app import diarization, live_tracking as lt, streaming_recorder
from app.diarization import SpeakerRecognitionEngine
from app.streaming_recorder import StreamingRecorder


def test_inference_slot_tracks_active_requests():
    eng = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    eng._active_lock = threading.Lock()
    eng._active_requests = 0
    assert eng._active_requests == 0
    with eng._inference_slot():
        assert eng._active_requests == 1
        with eng._inference_slot():
            assert eng._active_requests == 2
        assert eng._active_requests == 1
    assert eng._active_requests == 0


def test_batched_emotion_handles_length_buckets_and_fallback(monkeypatch):
    eng = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    eng._emotion_model_failed = False
    eng._model_lock = threading.Lock()
    eng.clear_gpu_cache_async = lambda *a: None

    # Stub FunASR wrapper without extract_features -> cleanly falls back to extract_emotion with in-memory ndarray
    seen_preloaded = []

    def fake_extract(audio_file, start, end, extract_embedding=False, preloaded_audio=None):
        seen_preloaded.append((type(preloaded_audio), start, end, extract_embedding))
        return {"emotion_category": "neutral", "emotion_confidence": 0.9}

    eng._emotion_model = types.SimpleNamespace(model=None, kwargs={})
    eng.extract_emotion = fake_extract
    wav = np.zeros(16000 * 5, dtype=np.float32)
    out = eng._extract_emotions_batch(wav, [(0.0, 1.5), (2.0, 4.0)], "x.wav", extract_embedding=True)
    assert len(out) == 2 and all(o["emotion_category"] == "neutral" for o in out)
    assert all(t is np.ndarray for t, _, _, _ in seen_preloaded)


def test_low_latency_min_flush_override_and_in_memory_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming_recorder, "data_path", lambda: str(tmp_path))
    monkeypatch.setattr(streaming_recorder, "_speech_detector", lambda sr, th: (lambda a: float(a.max()) > 0))
    queued = []
    rec = StreamingRecorder(sample_rate=16000, silence_duration=0.25, min_flush_seconds=0.25)
    rec.on_segment_processed = lambda info: queued.append(info)
    rec.start_recording(7)
    # 0.5 s of speech + 0.25 s silence -> flushes immediately (would have waited 2.5 s before)
    rec.process_audio_chunk((16000, np.full(8000, 0.3, np.float32)))
    rec.process_audio_chunk((16000, np.zeros(4000, np.float32)))
    rec.stop_recording()
    rec.cleanup()
    assert len(queued) == 1
    assert "audio_samples" in queued[0] and queued[0]["sample_rate"] == 16000
    assert len(queued[0]["audio_samples"]) == 12000


def test_single_speaker_short_reply_inherits_active_track():
    eng = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    eng._speaker_cache = None
    eng._cache_lock = threading.Lock()
    tracks = lt.SessionTracks()
    v = np.zeros(8, dtype=np.float32)
    v[0] = 1.0
    t = tracks.assign({"SPEAKER_00": (v, 3.0)})["SPEAKER_00"]
    t.speaker_id = 42
    t.own_unknown = False

    eng._all_profiles = lambda db: [{"speaker_id": 42, "speaker_name": "Alice", "profile_type": "general",
                                     "embedding": v}]
    eng._follow_user = lambda *a, **k: None

    # Next chunk: a short "yes" (0.2 s) whose embedding is None
    eng._label_centroids = lambda *a, **k: {"SPEAKER_00": None}
    segments = [({"start": 0.0, "end": 0.2, "text": "yes"}, "SPEAKER_00", None)]
    identities = eng._identify_tracks(None, segments, {"exclusive_segments": []}, tracks, object(), diarize=False)
    who = identities["SPEAKER_00"]
    assert (who["speaker"], who["speaker_id"], who["is_known"]) == ("Alice", 42, True)
    # its voice was never checked: no confidence, and it stays off the track (so correcting it cannot move the track)
    assert who["confidence"] == 0.0
    assert "track" not in who


def test_single_speaker_short_reply_stays_unknown_while_the_track_has_no_speaker():
    eng = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    eng._speaker_cache = None
    eng._cache_lock = threading.Lock()
    tracks = lt.SessionTracks()
    v = np.zeros(8, dtype=np.float32)
    v[0] = 1.0
    tracks.assign({"SPEAKER_00": (v, 3.0)})   # a new voice: its Unknown is only created when the chunk is saved
    eng._all_profiles = lambda db: []
    eng._follow_user = lambda *a, **k: None
    eng._label_centroids = lambda *a, **k: {"SPEAKER_00": None}
    segments = [({"start": 0.0, "end": 0.2, "text": "yes"}, "SPEAKER_00", None)]
    identities = eng._identify_tracks(None, segments, {"exclusive_segments": []}, tracks, object(), diarize=False)
    assert "SPEAKER_00" not in identities   # the caller names it a plain Unknown


def test_in_memory_from_bytes_matches_wav_file(tmp_path):
    import io
    import wave

    pcm = (np.sin(np.linspace(0, 2 * np.pi * 440, 16000)) * 16000).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(pcm.tobytes())
    raw = buf.getvalue()

    if not hasattr(diarization.sf, "read") or not hasattr(diarization.torch, "from_numpy"):
        pytest.skip("real soundfile + torch required")
    res = diarization._in_memory_from_bytes(raw)
    assert res is not None
    assert res["sample_rate"] == 16000
    assert tuple(res["waveform"].shape) == (1, 16000)


def test_process_ephemeral_fast_path_skips_db_embedding_and_emotion(monkeypatch):
    from fastapi.testclient import TestClient
    from app import api
    from app.database import SessionLocal, init_db
    from app.main import app
    from app.models import Conversation, Speaker

    init_db()
    db = SessionLocal()
    before_convs = db.query(Conversation).count()
    before_spks = db.query(Speaker).count()
    db.close()

    called = {}

    class StubEngine:
        def clear_gpu_cache_async(self, reason):
            called["cleanup"] = reason

        def transcribe_with_diarization(
            self,
            audio_file,
            known_speakers=None,
            threshold=0.7,
            db_session=None,
            diarize=True,
            audio=None,
            include_emotion=True,
            identify_speakers=True,
            **kwargs,
        ):
            called.update(
                db_session=db_session,
                diarize=diarize,
                include_emotion=include_emotion,
                identify_speakers=identify_speakers,
            )
            return {
                "num_speakers": 2,
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.2,
                        "text": "hello there",
                        "speaker": "SPEAKER_00",
                        "speaker_label": "SPEAKER_00",
                        "is_known": False,
                        "confidence": 0.0,
                        "words": [{"word": " hello", "start": 0.0, "end": 0.5, "probability": 0.99}],
                        "avg_logprob": -0.05,
                    }
                ],
            }

    app.dependency_overrides[api.get_engine] = lambda: StubEngine()
    try:
        client = TestClient(app)
        resp = client.post(
            "/api/v1/process",
            files={"audio_file": ("clip.wav", b"RIFF0000WAVE", "audio/wav")},
            data={
                "diarize": "true",
                "include_emotion": "false",
                "identify_speakers": "false",
                "save_conversation": "false",
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["id"] == 0
        assert body["status"] == "completed"
        assert body["num_speakers"] == 2
        assert len(body["transcript_segments"]) == 1
        assert body["transcript_segments"][0]["speaker_name"] == "SPEAKER_00"
        assert body["transcript_segments"][0]["emotion_category"] is None
        assert called["db_session"] is None
        assert called["include_emotion"] is False
        assert called["identify_speakers"] is False
    finally:
        app.dependency_overrides.pop(api.get_engine, None)

    db = SessionLocal()
    assert db.query(Conversation).count() == before_convs
    assert db.query(Speaker).count() == before_spks
    db.close()


def test_multimodal_text_emotion_mapping_and_fusion():
    # Verify GoEmotions 28-class aggregation to canonical 7 classes
    id2label = {0: "neutral", 1: "joy", 2: "amusement", 3: "anger", 4: "annoyance", 5: "sadness", 6: "surprise"}
    probs = np.array([0.1, 0.2, 0.2, 0.3, 0.1, 0.05, 0.05], dtype=np.float32)
    mapped = SpeakerRecognitionEngine._map_text_scores_to_canon7(id2label, probs)
    assert abs(sum(mapped.values()) - 1.0) < 1e-6
    assert abs(mapped["happy"] - 0.4) < 1e-6
    assert abs(mapped["angry"] - 0.4) < 1e-6

    # Verify acoustic neutral->happy false positive is corrected by neutral text fusion
    acoustic = {
        "emotion_category": "happy",
        "emotion_confidence": 0.52,
        "probs": {"happy": 0.52, "neutral": 0.44, "sad": 0.02, "angry": 0.02},
    }
    text_probs = {"neutral": 0.90, "happy": 0.04, "sad": 0.02, "angry": 0.01, "fearful": 0.01, "disgusted": 0.01, "surprised": 0.01}
    fused = SpeakerRecognitionEngine._fuse_multimodal_emotion(acoustic, text_probs, weight=0.55)
    assert fused["emotion_category"] == "neutral"
    assert fused["emotion_confidence"] > 0.65
    assert "multimodal" in fused["detector_breakdown"]


def test_stateless_upload_names_unmatched_voices_plainly_and_keeps_their_labels():
    """Nothing is saved, so a voice that matches nobody has no stored Unknown: it is "Unknown" (not a temporary
    Unknown_01 that could read as a stored speaker) with no confidence, and speaker_label tells the voices apart."""
    from fastapi.testclient import TestClient
    from app import api
    from app.main import app

    def seg(start, label, **who):
        return dict({"start": start, "end": start + 1.0, "text": "x", "speaker_label": label, "words": [],
                     "avg_logprob": -0.1}, **who)

    class StubEngine:
        def clear_gpu_cache_async(self, reason):
            pass

        def transcribe_with_diarization(self, *a, **k):
            return {"num_speakers": 3, "segments": [
                seg(0.0, "SPEAKER_00", speaker="Alice", speaker_id=7, is_known=True, confidence=0.81),
                seg(1.0, "SPEAKER_01", speaker="Unknown", is_known=False, confidence=1.0, enroll={"voice": None}),
                seg(2.0, "SPEAKER_02", speaker="Unknown_01", is_known=False, confidence=0.0),
            ]}

    app.dependency_overrides[api.get_engine] = lambda: StubEngine()
    try:
        resp = TestClient(app).post("/api/v1/process", files={"audio_file": ("clip.wav", b"RIFF0000WAVE", "audio/wav")},
                                    data={"save_conversation": "false", "include_emotion": "false"})
    finally:
        app.dependency_overrides.pop(api.get_engine, None)
    assert resp.status_code == 200, resp.text
    got = [(s["speaker_name"], s["speaker_id"], s["confidence"], s["speaker_label"])
           for s in resp.json()["transcript_segments"]]
    assert got == [("Alice", 7, 0.81, "SPEAKER_00"), ("Unknown", None, None, "SPEAKER_01"),
                   ("Unknown", None, None, "SPEAKER_02")]


def test_whisper_reads_in_memory_audio_when_the_upload_has_no_file(monkeypatch):
    """A stateless upload decoded in memory passes only its file name; Whisper then takes the samples."""
    seen = []

    class Whisper:
        def transcribe(self, source, **kwargs):
            seen.append(source)
            return iter(()), types.SimpleNamespace(language="en", language_probability=1.0)

    eng = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    eng._whisper_model = Whisper()
    monkeypatch.setattr(type(eng), "whisper_model", property(lambda self: self._whisper_model))
    eng.clear_gpu_cache_async = lambda *a: None
    monkeypatch.setattr(diarization, "ASR_BACKEND", "whisper")
    monkeypatch.setattr(diarization, "WHISPER_BATCH_SIZE", 0)
    samples = np.zeros(1600, dtype=np.float32)
    audio = {"waveform": [types.SimpleNamespace(numpy=lambda: samples)], "sample_rate": 16000}   # as _in_memory
    eng.transcribe("clip.wav", audio)
    assert isinstance(seen[-1], np.ndarray) and len(seen[-1]) == 1600
    eng.transcribe(__file__, audio)   # a file on disk is still decoded by faster-whisper itself
    assert seen[-1] == __file__


def test_emotion_model_is_put_in_eval_mode_when_loaded(monkeypatch):
    """FunASR only switches to eval mode inside generate(); the direct forward would otherwise keep dropout on."""
    import sys

    class Inner:
        training = True

        def eval(self):
            self.training = False
            return self

    class AutoModel:
        def __init__(self, **kwargs):
            self.model = Inner()

    monkeypatch.setitem(sys.modules, "funasr", types.SimpleNamespace(AutoModel=AutoModel))
    eng = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    eng._emotion_model, eng._emotion_model_failed, eng._model_lock = None, False, threading.Lock()
    assert eng.emotion_model.model.training is False
