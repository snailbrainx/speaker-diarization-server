"""SPEAKER_ASSIGNMENT=cluster: each diarization speaker is identified once, from all of its segments.

Exercises the real SpeakerRecognitionEngine._identify_clusters / _identify_segments with heavy ML modules stubbed at
import time (tests/conftest.py) and a throwaway SQLite database.
"""
import threading

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.diarization import SpeakerRecognitionEngine, _centroid, _split_by_speaker
from app.models import Base, Speaker


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/cluster.db")
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    yield db
    db.close()
    engine.dispose()


@pytest.fixture()
def eng():
    e = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    e._speaker_cache = None
    e._cache_lock = threading.Lock()
    return e


def _vec(*xs):
    v = np.zeros(8, dtype=np.float32)
    v[:len(xs)] = xs
    return v / np.linalg.norm(v)


def _seg(start, end):
    return {"start": start, "end": end, "text": "words", "words": []}


def _enroll(db, name, embedding):
    s = Speaker(name=name)
    s.set_embedding(embedding)
    db.add(s)
    db.flush()
    return s.id


DIAR = {"segments": [], "exclusive_segments": [], "num_speakers": 0}


def test_split_by_speaker_keeps_each_words_diarization_label():
    words = [{"word": " hi", "start": 0.0, "end": 0.4}, {"word": " there", "start": 0.5, "end": 0.9},
             {"word": " yes", "start": 1.2, "end": 1.5}]
    turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}, {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"}]
    pieces = _split_by_speaker([{"start": 0.0, "end": 1.5, "text": "hi there yes", "words": words}], turns)
    assert [(p["text"], p["speaker_label"]) for p in pieces] == [("hi there", "SPEAKER_00"), ("yes", "SPEAKER_01")]


def test_centroid_is_duration_weighted_mean_of_unit_vectors():
    c = _centroid([_vec(1, 0) * 5, _vec(0, 1)], [3.0, 1.0])
    assert np.allclose(c[:2], [0.75, 0.25])


def test_one_new_unknown_per_diarization_speaker(eng, db_session):
    # Noisy per-segment embeddings of two people; per-segment matching would split them.
    segments = [(_seg(0, 2), "SPEAKER_00", _vec(1, 0.3)), (_seg(2, 4), "SPEAKER_01", _vec(0, 1)),
                (_seg(4, 6), "SPEAKER_00", _vec(1, -0.3)), (_seg(6, 7), "SPEAKER_01", _vec(0.2, 1))]
    ids = eng._identify_clusters("a.wav", segments, DIAR, [], 0.30, db_session)
    assert set(ids) == {"SPEAKER_00", "SPEAKER_01"}
    assert ids["SPEAKER_00"]["speaker_id"] != ids["SPEAKER_01"]["speaker_id"]
    assert all(v["speaker"].startswith("Unknown_") and not v["is_known"] for v in ids.values())
    assert db_session.query(Speaker).count() == 2


def test_matches_are_one_to_one_by_total_similarity(eng, db_session):
    alice, bob = _enroll(db_session, "Alice", _vec(1, 0)), _enroll(db_session, "Bob", _vec(0.6, 1))
    # SPEAKER_01 is closer to Alice than to Bob, but SPEAKER_00 is Alice: SPEAKER_01 must get Bob.
    segments = [(_seg(0, 3), "SPEAKER_00", _vec(1, 0.05)), (_seg(3, 6), "SPEAKER_01", _vec(1, 0.4))]
    ids = eng._identify_clusters("a.wav", segments, DIAR, [], 0.30, db_session)
    assert (ids["SPEAKER_00"]["speaker_id"], ids["SPEAKER_01"]["speaker_id"]) == (alice, bob)
    assert ids["SPEAKER_00"]["is_known"] and ids["SPEAKER_01"]["is_known"]


def test_second_speaker_like_a_taken_person_stays_unknown(eng, db_session):
    alice = _enroll(db_session, "Alice", _vec(1, 0))
    segments = [(_seg(0, 3), "SPEAKER_00", _vec(1, 0.05)), (_seg(3, 6), "SPEAKER_01", _vec(1, 0.5))]
    ids = eng._identify_clusters("a.wav", segments, DIAR, [], 0.30, db_session)
    assert ids["SPEAKER_00"]["speaker_id"] == alice
    assert ids["SPEAKER_01"]["speaker"].startswith("Unknown_") and not ids["SPEAKER_01"]["is_known"]


def test_below_threshold_is_unknown_not_a_guess(eng, db_session):
    _enroll(db_session, "Alice", _vec(1, 0))
    ids = eng._identify_clusters("a.wav", [(_seg(0, 3), "SPEAKER_00", _vec(0.25, 1))], DIAR, [], 0.30, db_session)
    assert ids["SPEAKER_00"]["speaker"].startswith("Unknown_")


def test_new_unknown_is_matched_in_the_next_file(eng, db_session):
    first = eng._identify_clusters("a.wav", [(_seg(0, 3), "SPEAKER_00", _vec(1, 0.1))], DIAR, [], 0.30, db_session)
    eng._speaker_cache = None   # next request reloads from the DB
    again = eng._identify_clusters("b.wav", [(_seg(0, 3), "SPEAKER_03", _vec(1, -0.1))], DIAR, [], 0.30, db_session)
    assert again["SPEAKER_03"]["speaker_id"] == first["SPEAKER_00"]["speaker_id"]


def test_without_db_uses_known_speakers_and_temporary_unknown_names(eng):
    known = [(7, "Alice", _vec(1, 0))]
    segments = [(_seg(0, 3), "SPEAKER_00", _vec(1, 0.1)), (_seg(3, 6), "SPEAKER_01", _vec(0, 1))]
    ids = eng._identify_clusters("a.wav", segments, DIAR, known, 0.30, None)
    assert (ids["SPEAKER_00"]["speaker"], ids["SPEAKER_00"]["speaker_id"]) == ("Alice", 7)
    assert (ids["SPEAKER_01"]["speaker"], ids["SPEAKER_01"]["speaker_id"]) == ("Unknown_01", None)


def test_segment_mode_names_unmatched_labels_consistently(eng):
    segments = [(_seg(0, 1), "SPEAKER_00", None), (_seg(1, 2), "SPEAKER_01", None),
                (_seg(2, 3), "SPEAKER_00", None), (_seg(3, 4), None, None)]
    names = [w["speaker"] for w in eng._identify_segments(segments, [], 0.30, None)]
    assert names == ["Unknown_01", "Unknown_02", "Unknown_01", "Unknown_03"]


def test_float_samples_match_reading_the_exported_wav(tmp_path):
    pydub = pytest.importorskip("pydub")
    sf = pytest.importorskip("soundfile")
    from app.diarization import _float_samples
    rng = np.random.default_rng(0)
    stereo = (rng.uniform(-0.9, 0.9, size=(1600, 2)) * 32767).astype(np.int16)
    seg = pydub.AudioSegment(stereo.tobytes(), frame_rate=16000, sample_width=2, channels=2)
    seg.export(tmp_path / "s.wav", format="wav")
    expected = sf.read(tmp_path / "s.wav", dtype="float32")[0].mean(axis=1)
    assert np.allclose(_float_samples(seg), expected, atol=1e-6)


def test_cluster_mode_profile_skips_short_segments_and_weights_by_duration(db_session, monkeypatch):
    from datetime import datetime, timezone
    from app import services
    from app.models import Conversation, ConversationSegment
    monkeypatch.setattr(services, "CLUSTER_SPEAKERS", True)
    speaker_id = _enroll(db_session, "Alice", _vec(0, 1))
    conv = Conversation(title="t", start_time=datetime.now(timezone.utc), status="completed", audio_path=None)
    db_session.add(conv)
    db_session.flush()
    for (start, end), emb in (((0, 0.3), _vec(-1, 0)), ((1, 3), _vec(1, 0.2)), ((4, 10), _vec(1, -0.2))):
        seg = ConversationSegment(conversation_id=conv.id, speaker_id=speaker_id, speaker_name="Alice", text="x",
                                  start_time=datetime.now(timezone.utc), end_time=datetime.now(timezone.utc),
                                  start_offset=start, end_offset=end)
        seg.set_speaker_embedding(emb * 7)
        db_session.add(seg)
    db_session.flush()
    speaker = db_session.get(Speaker, speaker_id)
    assert services.recalculate_speaker_embedding(speaker, db_session, None) == 2   # the 0.3 s segment is left out
    expected = _centroid([_vec(1, 0.2), _vec(1, -0.2)], [2.0, 6.0])
    assert np.allclose(speaker.get_embedding(), expected, atol=1e-5)


def test_saving_a_segment_uses_the_speaker_resolved_for_its_cluster(db_session):
    from datetime import datetime, timezone
    from app.models import Conversation
    from app.services import create_segment_from_result
    unknown_id = _enroll(db_session, "Unknown_123", _vec(1, 0))
    conv = Conversation(title="t", start_time=datetime.now(timezone.utc), status="processing", audio_path=None)
    db_session.add(conv)
    db_session.flush()
    seg = {"start": 0.0, "end": 2.0, "text": "hi", "speaker": "Unknown_123", "speaker_id": unknown_id,
           "is_known": False, "confidence": 1.0, "embedding": _vec(0, 1), "words": []}
    saved = create_segment_from_result(seg, conv.id, conv.start_time, db_session, threshold=0.30)
    assert saved.speaker_id == unknown_id and saved.speaker_name == "Unknown_123"
    assert db_session.query(Speaker).count() == 1   # not auto-enrolled again from the segment's own embedding


def test_live_chunks_never_use_cluster_mode(tmp_path, monkeypatch):
    import asyncio
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from app import streaming_websocket
    from app.models import Conversation
    engine_db = create_engine(f"sqlite:///{tmp_path}/live.db")
    Base.metadata.create_all(engine_db)
    Session = sessionmaker(bind=engine_db)
    db = Session()
    db.add(Conversation(title="live", start_time=datetime.now(timezone.utc), status="recording", processing_token="tok"))
    db.commit()
    conv_id = db.query(Conversation).first().id
    db.close()
    chunk = tmp_path / "chunk.wav"
    chunk.write_bytes(b"RIFF")
    seen = {}

    class FakeEngine:
        def transcribe_with_diarization(self, *_args, **kwargs):
            seen.update(kwargs)
            return {"segments": [], "num_speakers": 0}

        def clear_gpu_cache_async(self, _reason):
            return None

        def clear_speaker_cache(self):
            return None

    class FakeSocket:
        async def send_json(self, _data):
            return None

        async def send_text(self, _data):
            return None

    monkeypatch.setattr(streaming_websocket, "SessionLocal", Session)
    monkeypatch.setattr(streaming_websocket, "get_config",
                        lambda: SimpleNamespace(get_settings=lambda: SimpleNamespace(speaker_threshold=0.30)))
    asyncio.run(streaming_websocket._handle_segment_processed(
        FakeSocket(), conv_id, "tok", {"segment_file": str(chunk), "start_offset": 0.0}, FakeEngine()))
    assert seen.get("cluster_speakers") is False
