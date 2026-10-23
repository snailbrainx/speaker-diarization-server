"""Live speaker tracking (app/live_tracking.py): voices are tracked across a session's chunks and named from all their
speech; a stranger stays one Unknown; a named Unknown takes every segment of the session with it."""
import threading
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import live_tracking as lt
from app.diarization import SpeakerRecognitionEngine
from app.models import AppMetadata, Base, Conversation, ConversationSegment, Speaker


def vec(*xs):
    v = np.zeros(8, dtype=np.float32)
    v[:len(xs)] = xs
    return v / np.linalg.norm(v)


A, B, C = vec(1), vec(0, 1), vec(0, 0, 1)


@pytest.fixture()
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/live.db")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def enroll(db, name, embedding):
    s = Speaker(name=name)
    s.set_embedding(embedding)
    db.add(s)
    db.flush()
    return s.id


def test_chunks_join_the_track_of_their_voice_one_to_one():
    tracks = lt.SessionTracks()
    first = tracks.assign({"S0": (A, 3.0), "S1": (B, 2.0)})
    second = tracks.assign({"S0": (B * 0.9 + A * 0.1, 2.0), "S1": (B, 2.0)})   # two labels like B: only one joins B
    assert first["S0"] is not first["S1"]
    assert second["S0"] is first["S1"] or second["S1"] is first["S1"]
    assert len(tracks.tracks) == 3 and first["S1"].seconds == 4.0


def test_a_track_is_named_only_when_its_voice_is_clear():
    tracks = lt.SessionTracks()
    t = tracks.assign({"S0": (A, 1.0)})["S0"]
    stored = {1: [A], 2: [B]}
    assert tracks.identify(t, stored)[0] == 1                          # one short line, clearly them: named
    assert tracks.identify(t, {1: [A], 2: [A * 0.99 + B * 0.01]}) is None   # two people alike: no guess
    assert tracks.identify(t, {2: [B]}) is None                        # nobody like it: stays Unknown
    assert tracks.identify(t, stored, taken=(1,)) is None              # given to another voice of this chunk
    at = lambda sim: [sim * A + np.sqrt(1 - sim ** 2) * C]             # a stored voice this similar to the track's
    assert tracks.identify(t, {1: at(0.65)})[0] == 1                   # a short line scores ~0.6 against its owner
    assert tracks.identify(t, {1: at(0.58)}) is None


def test_a_second_track_of_one_voice_shares_its_unknown_and_both_follow_its_name():
    tracks = lt.SessionTracks()
    t1 = tracks.assign({"S0": (A, 3.0)})["S0"]
    t1.speaker_id, t1.own_unknown = 101, True                          # the first track's own Unknown
    t2 = lt.Track(id=9, voice_sum=A * 3.0, seconds=3.0, speaker_id=102, own_unknown=True)
    tracks.tracks.append(t2)                                           # the same voice, tracked again
    stored = {101: [A], 102: [A], 7: [B]}                              # its own Unknowns are not "stored before"
    assert tracks.identify(t2, stored) == (101, pytest.approx(1.0))
    tracks.adopt(t2, 101)
    assert tracks.renames == [(102, 101)] and (t2.speaker_id, t2.own_unknown) == (101, False)
    tracks.adopt(t1, 7)                                                # later the Unknown is named
    assert tracks.renames[-1] == (101, 7) and t1.speaker_id == t2.speaker_id == 7


@pytest.fixture()
def eng():
    e = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    e._speaker_cache, e._cache_lock = None, threading.Lock()
    return e


def chunk(eng, db, tracks, voices):
    """One live chunk as the handler runs it: the engine names each voice (no writes), the handler's lease starts
    with a rollback, then saves new Unknowns and voices and gives each segment its track's speaker."""
    from app.streaming_websocket import _save_tracks
    eng._label_centroids = lambda *a, **k: {label: v for label, (v, _) in voices.items()}
    segments = [({"start": 0.0, "end": sec}, label, v) for label, (v, sec) in voices.items()]
    identities = eng._identify_tracks(None, segments, {"exclusive_segments": []}, tracks, db)
    db.rollback()                                                       # claim_segment_persistence
    out = {label: dict(who) for label, who in identities.items()}
    _save_tracks(db, list(out.values()), eng)
    db.commit()
    eng._speaker_cache = None
    return out


def test_live_strangers_get_one_unknown_each_and_known_people_their_name(eng, db):
    known = enroll(db, "Ann", A)
    db.commit()
    tracks = lt.SessionTracks()
    first = chunk(eng, db, tracks, {"S0": (A, 3.0), "S1": (B, 3.0)})
    assert first["S0"]["speaker"] == "Ann" and first["S0"]["speaker_id"] == known
    stranger = first["S1"]["speaker"]
    assert stranger.startswith("Unknown_") and db.query(Speaker).count() == 2   # saved despite the rollback
    again = chunk(eng, db, tracks, {"S0": (B, 2.5)})                   # the stranger again: the same Unknown
    assert again["S0"]["speaker"] == stranger and db.query(Speaker).count() == 2


def test_short_lines_are_named_and_new_voices_are_known_in_the_next_session(eng, db):
    """Someone who speaks in short replies: named from their first line in every session, and a voice nobody named
    is stored at once, so the next session gives it the same Unknown (it was a plain "Unknown" with nothing stored
    before 3 s of speech, and a name needed 2 s: a session of short replies never got either)."""
    ann = enroll(db, "Ann", A)
    db.commit()
    first = chunk(eng, db, lt.SessionTracks(), {"S0": (A, 1.2), "S1": (B, 1.2)})
    assert (first["S0"]["speaker"], first["S0"]["speaker_id"]) == ("Ann", ann)
    stranger = first["S1"]["speaker"]
    assert stranger.startswith("Unknown_") and db.query(Speaker).count() == 2
    later = chunk(eng, db, lt.SessionTracks(), {"S0": (B, 1.1), "S1": (A, 1.0)})   # another session
    assert later["S0"]["speaker"] == stranger and later["S1"]["speaker"] == "Ann"
    assert db.query(Speaker).count() == 2


def test_provisional_segments_take_their_tracks_speaker_later(db):
    from app.streaming_websocket import _relabel_tracks
    conv = Conversation(title="live", start_time=datetime.now(timezone.utc), status="recording")
    db.add(conv)
    db.flush()
    ann = enroll(db, "Ann", A)
    now = datetime.now(timezone.utc)
    seg = ConversationSegment(conversation_id=conv.id, speaker_id=None, speaker_name="Unknown", start_time=now,
                              end_time=now, start_offset=0, end_offset=1)
    db.add(seg)
    db.flush()
    track = lt.Track(id=1, voice_sum=A, seconds=3.0, speaker_id=ann, segment_ids=[seg.id])

    class Eng:
        def clear_speaker_cache(self):
            pass

    updates = _relabel_tracks(db, conv.id, lt.SessionTracks(), [{"track": track}], Eng())
    assert updates == [{"segment_ids": [seg.id], "speaker_id": ann, "speaker_name": "Ann", "is_known": True,
                        "emotions": {}}]   # Ann has no emotion corrections: emotions stay as they were
    db.expire_all()
    assert db.get(ConversationSegment, seg.id).speaker_name == "Ann"


def test_relabel_moves_this_sessions_segments_and_drops_the_empty_unknown(db):
    from app.streaming_websocket import _relabel_tracks
    conv = Conversation(title="live", start_time=datetime.now(timezone.utc), status="recording")
    other = Conversation(title="other", start_time=datetime.now(timezone.utc), status="completed")
    db.add_all([conv, other])
    db.flush()
    ann, unknown = enroll(db, "Ann", A), enroll(db, "Unknown_1", A)
    now = datetime.now(timezone.utc)
    for c in (conv, conv):
        db.add(ConversationSegment(conversation_id=c.id, speaker_id=unknown, speaker_name="Unknown_1",
                                   start_time=now, end_time=now, start_offset=0, end_offset=1))
    db.flush()
    tracks = lt.SessionTracks()
    tracks.renames.append((unknown, ann))

    class Eng:
        cleared = 0

        def clear_speaker_cache(self):
            Eng.cleared += 1

    updates = _relabel_tracks(db, conv.id, tracks, [], Eng())
    assert updates[0]["speaker_name"] == "Ann" and len(updates[0]["segment_ids"]) == 2
    assert {s.speaker_name for s in db.query(ConversationSegment)} == {"Ann"}
    assert db.get(Speaker, unknown) is None and tracks.renames == [] and Eng.cleared == 1


def test_voice_profiles_from_another_model_are_rebuilt_once(db, tmp_path, monkeypatch):
    import soundfile as sf
    from app import services
    wav = tmp_path / "conv.wav"
    sf.write(wav, np.zeros(16000 * 4, dtype=np.float32), 16000)
    conv = Conversation(title="c", start_time=datetime.now(timezone.utc), status="completed", audio_path=str(wav))
    db.add(conv)
    db.flush()
    ann, lost = enroll(db, "Ann", np.ones(512, np.float32)), enroll(db, "Bob", np.ones(512, np.float32))
    now = datetime.now(timezone.utc)
    db.add(ConversationSegment(conversation_id=conv.id, speaker_id=ann, speaker_name="Ann", start_time=now,
                               end_time=now, start_offset=0.5, end_offset=2.5))
    db.flush()

    class Eng:
        embedding_dim = 4

        def _segment_embedding_or_none(self, audio, seg):
            return np.array([1, 0, 0, 0], np.float32)

        def clear_speaker_cache(self):
            pass

    counts = services.migrate_voice_embeddings(db, Eng())          # no record: made by pyannote/embedding
    assert counts == {"segments": 1, "speakers_rebuilt": 1, "speakers_without_audio": ["Bob"]}
    assert db.get(Speaker, ann).get_embedding().shape == (4,)
    assert np.isnan(db.get(Speaker, lost).get_embedding()).all()     # no audio: no voice until named again
    assert db.get(AppMetadata, services.VOICE_MODEL_KEY).value == services.SPEAKER_EMBEDDING_MODEL
    assert services.migrate_voice_embeddings(db, Eng()) is None      # once


def test_emotion_voice_profiles_never_decide_who_someone_is(eng, db):
    from app.models import SpeakerEmotionProfile
    ann, bob = enroll(db, "Ann", A), enroll(db, "Bob", B)
    for emotion in ("happy", "sad", "angry"):   # three corrections: a per-emotion voice sample of Bob's like C
        p = SpeakerEmotionProfile(speaker_id=bob, emotion_category=emotion, sample_count=3, voice_sample_count=3)
        p.set_embedding(np.ones(4, np.float32))
        p.set_voice_embedding(C)
        db.add(p)
    db.commit()
    segments = [({"start": 0.0, "end": 4.0}, "SPEAKER_00", C)]
    ids = eng._identify_clusters("a.wav", segments, {"exclusive_segments": []}, [], 0.60, db)
    assert ids["SPEAKER_00"]["speaker_id"] is None and ids["SPEAKER_00"]["enroll"] is not None   # a new Unknown, not
    # Bob: a voice like C was never Bob's own


def test_a_live_track_follows_the_users_correction(eng, db):
    ann = enroll(db, "Ann", A)
    bob = enroll(db, "Bob", vec(0.9, 0.44))
    db.commit()
    tracks = lt.SessionTracks()
    first = chunk(eng, db, tracks, {"S0": (A, 3.0)})
    track = first["S0"]["track"]
    assert first["S0"]["speaker"] == "Ann"
    conv = Conversation(title="live", start_time=datetime.now(timezone.utc), status="recording")
    db.add(conv)
    db.flush()
    now = datetime.now(timezone.utc)
    seg = ConversationSegment(conversation_id=conv.id, speaker_id=bob, speaker_name="Bob", start_time=now,
                              end_time=now, start_offset=0, end_offset=3)   # the user said this line was Bob
    db.add(seg)
    db.commit()
    track.segment_ids.append(seg.id)
    again = chunk(eng, db, tracks, {"S0": (A, 2.0)})
    assert again["S0"]["speaker"] == "Bob" and again["S0"]["track"] is track


def test_the_fallback_match_never_gives_a_named_persons_name(db):
    from app.diarization import auto_enroll_unknown_speaker
    enroll(db, "Ann", A)
    sid, name = auto_enroll_unknown_speaker(A * 0.6 + B * 0.8, db, threshold=0.30)   # 0.6 like Ann
    assert name.startswith("Unknown_")


def test_stored_voice_changes_drop_every_engines_cache_after_commit(eng, db):
    from app import diarization
    diarization._ENGINES.add(eng)
    eng._speaker_cache = [{"speaker_id": 1}]
    enroll(db, "Ann", A)
    assert eng._speaker_cache is not None      # not before the commit (a reload then would cache the old state)
    db.commit()
    assert eng._speaker_cache is None


def test_track_lookups_read_only_the_sessions_own_segments(db):
    """SQLite gives a deleted or rolled-back segment's id to the next new row, which may belong to another
    conversation: a track's segment ids are only ever looked up in its own session's conversation."""
    from app.streaming_websocket import _relabel_tracks
    live = Conversation(title="live", start_time=datetime.now(timezone.utc), status="recording")
    other = Conversation(title="other", start_time=datetime.now(timezone.utc), status="recording")
    db.add_all([live, other])
    db.flush()
    ann, bob = enroll(db, "Ann", A), enroll(db, "Bob", B)
    now = datetime.now(timezone.utc)
    provisional = ConversationSegment(conversation_id=other.id, speaker_id=None, speaker_name="Unknown",
                                      start_time=now, end_time=now, start_offset=0, end_offset=1)
    bobs = ConversationSegment(conversation_id=other.id, speaker_id=bob, speaker_name="Bob", start_time=now,
                               end_time=now, start_offset=1, end_offset=2)
    db.add_all([provisional, bobs])
    db.flush()

    class Eng:
        def clear_speaker_cache(self):
            pass

    named = lt.Track(id=1, voice_sum=A, seconds=3.0, speaker_id=ann, segment_ids=[provisional.id])
    assert _relabel_tracks(db, live.id, lt.SessionTracks(live.id), [{"track": named}], Eng()) == []
    db.expire_all()
    assert db.get(ConversationSegment, provisional.id).speaker_name == "Unknown"
    unnamed = lt.Track(id=2, voice_sum=A, seconds=3.0, segment_ids=[bobs.id])
    SpeakerRecognitionEngine._follow_user([unnamed], db, {ann: "Ann", bob: "Bob"}, live.id)
    assert unnamed.speaker_id is None                                  # not "corrected" to another session's Bob


def test_a_chunk_that_fails_to_save_leaves_the_tracks_as_they_were(tmp_path, monkeypatch):
    """The tracks change only with a saved chunk: a failed save once left its rolled-back segment ids (and new
    Unknowns and renames) in the session, and a later naming relabelled whatever row SQLite gave those ids next."""
    import asyncio
    from types import SimpleNamespace
    from app import streaming_websocket
    engine_db = create_engine(f"sqlite:///{tmp_path}/live.db")
    Base.metadata.create_all(engine_db)
    Session = sessionmaker(bind=engine_db)
    db = Session()
    conv = Conversation(title="live", start_time=datetime.now(timezone.utc), status="recording",
                        processing_token="tok")
    db.add(conv)
    db.commit()
    conv_id = db.query(Conversation).first().id
    db.close()
    chunk_file = tmp_path / "chunk.wav"
    chunk_file.write_bytes(b"RIFF")
    tracks = lt.SessionTracks(conv_id)

    class FakeEngine:
        save_track_voices = SpeakerRecognitionEngine.save_track_voices

        def transcribe_with_diarization(self, *_args, tracks=None, **_kwargs):
            track = tracks.assign({"S0": (A, 4.0)})["S0"]
            return {"segments": [{"start": 0.0, "end": 4.0, "text": "hello", "words": [], "speaker": "Unknown",
                                  "is_known": False, "confidence": 1.0, "embedding": None, "track": track}],
                    "num_speakers": 1}

        def clear_gpu_cache_async(self, _reason):
            pass

        def clear_speaker_cache(self):
            pass

    class FakeSocket:
        client_state = streaming_websocket.WebSocketState.CONNECTED

        async def send_json(self, _data):
            pass

        async def send_text(self, _data):
            pass

    monkeypatch.setattr(streaming_websocket, "SessionLocal", Session)
    monkeypatch.setattr(streaming_websocket, "get_config",
                        lambda: SimpleNamespace(get_settings=lambda: SimpleNamespace(speaker_threshold=0.30)))

    def run():
        asyncio.run(streaming_websocket._handle_segment_processed(
            FakeSocket(), conv_id, "tok", {"segment_file": str(chunk_file), "start_offset": 0.0}, FakeEngine(),
            tracks=tracks))

    run()                                                               # saved: the track has its Unknown and line
    (track,) = tracks.tracks
    saved_ids, unknown = list(track.segment_ids), track.speaker_id
    assert len(saved_ids) == 1 and unknown is not None

    def fail(*_args, **_kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(streaming_websocket, "_relabel_tracks", fail)
    with pytest.raises(RuntimeError):
        run()                                                           # its segment was flushed, then rolled back
    (track,) = tracks.tracks
    assert (track.segment_ids, track.speaker_id, track.seconds) == (saved_ids, unknown, 4.0)
    db = Session()
    assert db.query(ConversationSegment).count() == 1 and db.query(Speaker).count() == 1
    db.close()
