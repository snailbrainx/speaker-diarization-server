"""Safety tests for existing conversation/speaker operations and audit fixes."""
import asyncio
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (
    Conversation,
    ConversationSegment,
    Speaker,
    SpeakerEmotionProfile,
)


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/conversation.db")

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()
    engine.dispose()


class FakeEngine:
    def clear_speaker_cache(self):
        return None

    def clear_gpu_cache(self):
        return None

    def extract_segment_embedding(self, *_args, **_kwargs):
        raise AssertionError("unsafe chunk extraction was attempted")

    def extract_segment_embeddings_batch(self, *_args, **_kwargs):
        raise AssertionError("unsafe chunk extraction was attempted")

    def extract_emotion(self, *_args, **_kwargs):
        raise AssertionError("unsafe chunk extraction was attempted")


def _conversation_with_segment(db, tmp_path, *, status="recording", speaker=None):
    chunk = tmp_path / "segment.wav"
    chunk.write_bytes(b"RIFF-test")
    conversation = Conversation(
        title="test", start_time=datetime.now(timezone.utc), status=status, audio_path=None
    )
    db.add(conversation)
    db.flush()
    segment = ConversationSegment(
        conversation_id=conversation.id,
        speaker_id=speaker.id if speaker else None,
        speaker_name=speaker.name if speaker else "Unknown_01",
        text="hello",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        start_offset=120.0,
        end_offset=121.0,
        segment_audio_path=str(chunk),
    )
    db.add(segment)
    db.commit()
    return conversation, segment, chunk


def test_active_conversation_delete_is_rejected_and_audio_survives(
    db_session, tmp_path, monkeypatch
):
    from app import conversation_api, services

    conversation, _segment, _chunk = _conversation_with_segment(
        db_session, tmp_path, status="recording"
    )
    monkeypatch.setattr(services, "data_path", lambda: str(tmp_path))
    seg_dir = tmp_path / "stream_segments" / f"conv_{conversation.id}"
    seg_dir.mkdir(parents=True)
    (seg_dir / "seg_0001.wav").write_bytes(b"audio")

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(conversation_api.delete_conversation(conversation.id, db_session))
    assert exc_info.value.status_code == 409
    assert seg_dir.exists()
    assert db_session.query(Conversation).filter_by(id=conversation.id).one()


def test_completed_conversation_delete_removes_segment_directory(
    db_session, tmp_path, monkeypatch
):
    from app import conversation_api, services

    conversation, _segment, _chunk = _conversation_with_segment(
        db_session, tmp_path, status="completed"
    )
    conversation_id = conversation.id
    monkeypatch.setattr(services, "data_path", lambda: str(tmp_path))
    monkeypatch.setattr(conversation_api, "data_path", lambda: str(tmp_path))
    seg_dir = tmp_path / "stream_segments" / f"conv_{conversation_id}"
    seg_dir.mkdir(parents=True)
    (seg_dir / "seg_0001.wav").write_bytes(b"audio")

    result = asyncio.run(conversation_api.delete_conversation(conversation_id, db_session))
    assert "deleted" in result["message"]
    assert not seg_dir.exists()
    assert db_session.query(Conversation).filter_by(id=conversation_id).first() is None


def test_delete_cleans_before_numeric_id_can_be_reused(
    db_session, tmp_path, monkeypatch
):
    from sqlalchemy.exc import IntegrityError
    from sqlalchemy.orm import sessionmaker

    from app import conversation_api

    shared_audio = tmp_path / "recordings" / "conv_reused_full.wav"
    shared_audio.parent.mkdir(parents=True)
    shared_audio.write_bytes(b"old conversation audio")
    conversation = Conversation(
        title="old conversation",
        start_time=datetime.now(timezone.utc),
        status="completed",
        audio_path=str(shared_audio),
    )
    db_session.add(conversation)
    db_session.commit()
    conversation_id = conversation.id

    Session = sessionmaker(bind=db_session.get_bind())
    original_remove = conversation_api.os.remove
    inserted_during_cleanup = []

    def remove_with_id_reuse_probe(path):
        if str(path) == str(shared_audio) and not inserted_during_cleanup:
            probe_db = Session()
            try:
                replacement = Conversation(
                    id=conversation_id,
                    title="replacement conversation",
                    start_time=datetime.now(timezone.utc),
                    status="completed",
                    audio_path=str(shared_audio),
                )
                probe_db.add(replacement)
                try:
                    probe_db.commit()
                except IntegrityError:
                    probe_db.rollback()
                    inserted_during_cleanup.append(False)
                else:
                    inserted_during_cleanup.append(True)
                    shared_audio.write_bytes(b"new conversation audio")
            finally:
                probe_db.close()
        original_remove(path)

    monkeypatch.setattr(conversation_api.os, "remove", remove_with_id_reuse_probe)
    monkeypatch.setattr(conversation_api, "data_path", lambda: str(tmp_path))

    asyncio.run(conversation_api.delete_conversation(conversation_id, db_session))
    assert inserted_during_cleanup == [False]

    replacement = Conversation(
        id=conversation_id,
        title="replacement after cleanup",
        start_time=datetime.now(timezone.utc),
        status="completed",
        audio_path=str(shared_audio),
    )
    db_session.add(replacement)
    db_session.commit()
    shared_audio.write_bytes(b"replacement survives")
    assert shared_audio.exists()
    assert db_session.query(Conversation).filter_by(id=conversation_id).one().id == conversation_id


def test_public_status_mutation_is_forbidden_and_token_still_blocks_delete(
    db_session, tmp_path
):
    from pydantic import ValidationError

    from app import conversation_api
    from app.schemas import ConversationUpdate

    with pytest.raises(ValidationError):
        ConversationUpdate.model_validate({"status": "completed"})

    conversation, _segment, _chunk = _conversation_with_segment(
        db_session, tmp_path, status="completed"
    )
    conversation.processing_token = "authoritative-live-worker"
    db_session.commit()

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(conversation_api.delete_conversation(conversation.id, db_session))
    assert exc_info.value.status_code == 409


def test_reprocess_lease_blocks_concurrent_delete(db_session, tmp_path):
    import threading

    from sqlalchemy.orm import sessionmaker

    from app import conversation_api

    audio_path = tmp_path / "reprocess.wav"
    audio_path.write_bytes(b"fake-audio-for-stub-engine")
    conversation = Conversation(
        title="reprocess lease",
        start_time=datetime.now(timezone.utc),
        status="completed",
        audio_path=str(audio_path),
    )
    db_session.add(conversation)
    db_session.commit()
    conversation_id = conversation.id

    class BlockingEngine:
        def __init__(self):
            self.entered = threading.Event()
            self.release = threading.Event()

        def transcribe_with_diarization(self, *_args, **_kwargs):
            self.entered.set()
            assert self.release.wait(timeout=3)
            return {"segments": [], "num_speakers": 0}

        def clear_gpu_cache(self):
            return None

        def clear_speaker_cache(self):
            return None

    engine = BlockingEngine()
    Session = sessionmaker(bind=db_session.get_bind())

    async def scenario():
        task = asyncio.create_task(conversation_api.reprocess_conversation(
            conversation_id, db_session, engine
        ))
        assert await asyncio.to_thread(engine.entered.wait, 2)

        delete_db = Session()
        try:
            active = delete_db.query(Conversation).filter_by(id=conversation_id).one()
            assert active.status == "processing"
            assert active.processing_token is not None
            with pytest.raises(HTTPException) as exc_info:
                await conversation_api.delete_conversation(conversation_id, delete_db)
            assert exc_info.value.status_code == 409
        finally:
            delete_db.close()

        engine.release.set()
        return await task

    result = asyncio.run(scenario())
    assert result["segments"] == 0
    db_session.expire_all()
    stored = db_session.query(Conversation).filter_by(id=conversation_id).one()
    assert stored.status == "completed"
    assert stored.processing_token is None


def test_reprocess_flushes_new_segments_before_orphan_cleanup(db_session, tmp_path):
    from app import conversation_api

    # Match app.database.SessionLocal; the production defect is hidden by
    # SQLAlchemy's default autoflush=True.
    db_session.autoflush = False
    audio_path = tmp_path / "reprocess-unknown.wav"
    audio_path.write_bytes(b"fake-audio-for-stub-engine")
    unknown = Speaker(name="Unknown_existing")
    unknown.set_embedding(np.ones(4, dtype=np.float32))
    db_session.add(unknown)
    db_session.flush()
    conversation = Conversation(
        title="reprocess unknown",
        start_time=datetime.now(timezone.utc),
        status="completed",
        audio_path=str(audio_path),
    )
    db_session.add(conversation)
    db_session.flush()
    db_session.add(ConversationSegment(
        conversation_id=conversation.id,
        speaker_id=unknown.id,
        speaker_name=unknown.name,
        text="old",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        start_offset=0.0,
        end_offset=1.0,
    ))
    db_session.commit()

    class ExistingUnknownEngine:
        def transcribe_with_diarization(self, *_args, **_kwargs):
            return {
                "segments": [{
                    "speaker": unknown.name,
                    "is_known": True,
                    "confidence": 0.9,
                    "text": "new",
                    "start": 0.0,
                    "end": 1.0,
                }],
                "num_speakers": 1,
            }

        def clear_gpu_cache(self):
            return None

    result = asyncio.run(conversation_api.reprocess_conversation(
        conversation.id, db_session, ExistingUnknownEngine()
    ))
    assert result["segments"] == 1
    db_session.expire_all()
    surviving = db_session.query(Speaker).filter_by(name="Unknown_existing").one()
    replacement = db_session.query(ConversationSegment).filter_by(
        conversation_id=conversation.id
    ).one()
    assert replacement.speaker_id == surviving.id
    assert replacement.speaker_name == surviving.name


def test_recalculate_emotions_decodes_conversation_once(db_session, tmp_path):
    from app import conversation_api

    audio_path = tmp_path / "conversation.wav"
    audio_path.write_bytes(b"fake-audio-for-stub-engine")
    speaker = Speaker(name="DecodeOnce")
    speaker.set_embedding(np.ones(4, dtype=np.float32))
    db_session.add(speaker)
    db_session.flush()
    conversation = Conversation(
        title="decode once",
        start_time=datetime.now(timezone.utc),
        status="completed",
        audio_path=str(audio_path),
    )
    db_session.add(conversation)
    db_session.flush()
    for index in range(3):
        db_session.add(ConversationSegment(
            conversation_id=conversation.id,
            speaker_id=speaker.id,
            speaker_name=speaker.name,
            text=f"segment {index}",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            start_offset=float(index),
            end_offset=float(index + 1),
        ))
    db_session.commit()
    conversation_id = conversation.id

    marker = object()

    class CountingEngine:
        preload_calls = 0
        extract_calls = 0

        def preload_emotion_audio(self, path):
            assert path == str(audio_path)
            self.preload_calls += 1
            return marker

        def extract_emotion(
            self, _path, _start, _end, *, extract_embedding, preloaded_audio
        ):
            assert extract_embedding is True
            assert preloaded_audio is marker
            self.extract_calls += 1
            return {"emotion_category": "happy", "emotion_confidence": 0.8}

        def clear_gpu_cache(self):
            return None

    engine = CountingEngine()
    result = asyncio.run(conversation_api.recalculate_emotions(
        conversation_id, db_session, engine
    ))
    assert result["updated"] == 3
    assert engine.preload_calls == 1
    assert engine.extract_calls == 3
    db_session.expire_all()
    stored = db_session.query(Conversation).filter_by(id=conversation_id).one()
    assert stored.status == "completed"
    assert stored.processing_token is None


def test_emotion_profile_rebuild_preloads_once_per_audio_path(db_session, tmp_path):
    from app.services import recalculate_emotion_profile

    audio_path = tmp_path / "profile-source.wav"
    audio_path.write_bytes(b"fake-audio-for-stub-engine")
    speaker = Speaker(name="ProfileDecodeOnce")
    speaker.set_embedding(np.ones(4, dtype=np.float32))
    db_session.add(speaker)
    db_session.flush()
    conversation = Conversation(
        title="profile decode once",
        start_time=datetime.now(timezone.utc),
        status="completed",
        audio_path=str(audio_path),
    )
    db_session.add(conversation)
    db_session.flush()
    for index in range(2):
        db_session.add(ConversationSegment(
            conversation_id=conversation.id,
            speaker_id=speaker.id,
            speaker_name=speaker.name,
            text=f"corrected {index}",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            start_offset=float(index),
            end_offset=float(index + 1),
            emotion_category="happy",
            emotion_corrected=True,
            emotion_misidentified=False,
        ))
    db_session.commit()
    speaker_id = speaker.id

    marker = object()

    class CountingEngine:
        preload_calls = 0
        extract_calls = 0

        def clear_speaker_cache(self):
            return None

        def preload_emotion_audio(self, path):
            assert path == str(audio_path)
            self.preload_calls += 1
            return marker

        def extract_emotion(self, _path, _start, _end, **kwargs):
            assert kwargs["preloaded_audio"] is marker
            self.extract_calls += 1
            return {"embedding": np.ones(4, dtype=np.float32)}

    engine = CountingEngine()
    assert recalculate_emotion_profile(
        speaker_id, "happy", db_session, engine
    ) == "created"
    assert engine.preload_calls == 1
    assert engine.extract_calls == 2


def test_identify_retroactive_emotion_preloads_once_per_audio_path(
    db_session, tmp_path, monkeypatch
):
    from app import conversation_api
    from app.schemas import IdentifySpeakerRequest

    audio_path = tmp_path / "identify-source.wav"
    audio_path.write_bytes(b"fake-audio-for-stub-engine")
    speaker = Speaker(name="Known")
    speaker.set_embedding(np.ones(4, dtype=np.float32))
    db_session.add(speaker)
    db_session.flush()
    profile = SpeakerEmotionProfile(
        emotion_category="happy",
        sample_count=1,
    )
    profile.set_embedding(np.ones(4, dtype=np.float32))
    speaker.emotion_profiles.append(profile)
    conversation = Conversation(
        title="identify preload",
        start_time=datetime.now(timezone.utc),
        status="completed",
        audio_path=str(audio_path),
    )
    db_session.add(conversation)
    db_session.flush()
    segments = []
    for index in range(2):
        segment = ConversationSegment(
            conversation_id=conversation.id,
            speaker_name="Unknown_01",
            text=f"segment {index}",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            start_offset=float(index),
            end_offset=float(index + 1),
            emotion_category="happy",
            emotion_corrected=False,
        )
        db_session.add(segment)
        segments.append(segment)
    db_session.commit()

    marker = object()

    class CountingEngine:
        preload_calls = 0
        extract_calls = 0

        def preload_emotion_audio(self, path):
            assert path == str(audio_path)
            self.preload_calls += 1
            return marker

        def extract_emotion(
            self, path, _start, _end, *, extract_embedding, preloaded_audio
        ):
            assert path == str(audio_path)
            assert extract_embedding is True
            assert preloaded_audio is marker
            self.extract_calls += 1
            return {"embedding": np.ones(4, dtype=np.float32)}

        def match_emotion_to_profile(self, *_args, **_kwargs):
            return None

        def clear_gpu_cache(self):
            return None

    monkeypatch.setattr(
        conversation_api, "recalculate_speaker_embedding", lambda *_args: 0
    )
    monkeypatch.setattr(
        conversation_api, "cleanup_orphaned_unknowns", lambda *_args, **_kwargs: []
    )
    engine = CountingEngine()
    request = IdentifySpeakerRequest(speaker_id=speaker.id, enroll=False)
    result = asyncio.run(conversation_api.identify_speaker_in_segment(
        conversation.id, segments[0].id, request, db_session, engine
    ))
    assert result["segments_updated"] == 2
    assert engine.preload_calls == 1
    assert engine.extract_calls == 2


def test_identify_only_works_with_chunk_audio_without_extraction(db_session, tmp_path):
    from app import conversation_api
    from app.schemas import IdentifySpeakerRequest

    speaker = Speaker(name="Bob")
    speaker.set_embedding(np.ones(4, dtype=np.float32))
    db_session.add(speaker)
    db_session.commit()
    conversation, segment, _chunk = _conversation_with_segment(
        db_session, tmp_path, status="recording"
    )

    request = IdentifySpeakerRequest(speaker_id=speaker.id, enroll=False)
    asyncio.run(conversation_api.identify_speaker_in_segment(
        conversation.id, segment.id, request, db_session, FakeEngine()
    ))
    db_session.expire_all()
    updated = db_session.query(ConversationSegment).filter_by(id=segment.id).one()
    assert updated.speaker_id == speaker.id
    assert updated.speaker_name == "Bob"


def _segment_for(db, conversation, *, speaker=None, name=None, offset=0.0):
    segment = ConversationSegment(
        conversation_id=conversation.id,
        speaker_id=speaker.id if speaker else None,
        speaker_name=speaker.name if speaker else name,
        text="hello",
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc),
        start_offset=offset,
        end_offset=offset + 1.0,
    )
    db.add(segment)
    db.flush()
    return segment


def test_identify_auto_enrolled_unknown_propagates_across_conversations(db_session, tmp_path):
    """Auto-enrolled Unknown_{timestamp} speakers own one globally unique
    Speaker row reused across recordings; identifying them must follow that
    row into every conversation, not just the current one."""
    from app import conversation_api
    from app.schemas import IdentifySpeakerRequest

    auto = Speaker(name="Unknown_1723180000000000")
    auto.set_embedding(np.ones(4, dtype=np.float32))
    bob = Speaker(name="Bob")
    bob.set_embedding(np.ones(4, dtype=np.float32))
    db_session.add_all([auto, bob])
    db_session.commit()

    conversation_a, target, _chunk = _conversation_with_segment(
        db_session, tmp_path, status="completed", speaker=auto
    )
    sibling_a = _segment_for(db_session, conversation_a, speaker=auto, offset=10.0)
    conversation_b = Conversation(
        title="other recording", start_time=datetime.now(timezone.utc), status="completed"
    )
    db_session.add(conversation_b)
    db_session.flush()
    sibling_b = _segment_for(db_session, conversation_b, speaker=auto, offset=0.0)
    db_session.commit()

    request = IdentifySpeakerRequest(speaker_id=bob.id, enroll=False)
    asyncio.run(conversation_api.identify_speaker_in_segment(
        conversation_a.id, target.id, request, db_session, FakeEngine()
    ))
    db_session.expire_all()
    for seg_id in (target.id, sibling_a.id, sibling_b.id):
        seg = db_session.query(ConversationSegment).filter_by(id=seg_id).one()
        assert seg.speaker_id == bob.id, f"segment {seg_id} was not propagated"
        assert seg.speaker_name == "Bob"


def test_identify_embeddingless_unknown_stays_conversation_scoped(db_session, tmp_path):
    """Unknown_XX labels restart at 01 in every recording, so identifying one
    must never relabel another conversation's unrelated Unknown_01."""
    from app import conversation_api
    from app.schemas import IdentifySpeakerRequest

    bob = Speaker(name="Bob")
    bob.set_embedding(np.ones(4, dtype=np.float32))
    db_session.add(bob)
    db_session.commit()

    conversation_a, target, _chunk = _conversation_with_segment(
        db_session, tmp_path, status="completed"
    )
    sibling_a = _segment_for(db_session, conversation_a, name="Unknown_01", offset=10.0)
    conversation_b = Conversation(
        title="unrelated recording", start_time=datetime.now(timezone.utc), status="completed"
    )
    db_session.add(conversation_b)
    db_session.flush()
    unrelated_b = _segment_for(db_session, conversation_b, name="Unknown_01", offset=0.0)
    db_session.commit()

    request = IdentifySpeakerRequest(speaker_id=bob.id, enroll=False)
    asyncio.run(conversation_api.identify_speaker_in_segment(
        conversation_a.id, target.id, request, db_session, FakeEngine()
    ))
    db_session.expire_all()
    assert db_session.query(ConversationSegment).filter_by(id=target.id).one().speaker_name == "Bob"
    assert db_session.query(ConversationSegment).filter_by(id=sibling_a.id).one().speaker_name == "Bob"
    untouched = db_session.query(ConversationSegment).filter_by(id=unrelated_b.id).one()
    assert untouched.speaker_name == "Unknown_01"
    assert untouched.speaker_id is None


def test_enrollment_refuses_chunk_audio_with_conversation_offsets(db_session, tmp_path):
    from app import conversation_api
    from app.schemas import IdentifySpeakerRequest

    conversation, segment, _chunk = _conversation_with_segment(
        db_session, tmp_path, status="recording"
    )
    request = IdentifySpeakerRequest(speaker_name="New", enroll=True)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(conversation_api.identify_speaker_in_segment(
            conversation.id, segment.id, request, db_session, FakeEngine()
        ))
    assert exc_info.value.status_code == 409


def test_recalculation_skips_unsafe_chunk_fallback(db_session, tmp_path):
    from app.services import recalculate_emotion_profile, recalculate_speaker_embedding

    speaker = Speaker(name="Alice")
    speaker.set_embedding(np.ones(4, dtype=np.float32))
    db_session.add(speaker)
    db_session.commit()
    _conversation, segment, _chunk = _conversation_with_segment(
        db_session, tmp_path, status="recording", speaker=speaker
    )
    segment.emotion_corrected = True
    segment.emotion_category = "happy"
    db_session.commit()

    engine = FakeEngine()
    assert recalculate_speaker_embedding(speaker, db_session, engine) == 0
    assert recalculate_emotion_profile(speaker.id, "happy", db_session, engine) is None


def test_deleted_speaker_keeps_non_relabelable_tombstone(db_session, tmp_path, monkeypatch):
    from app import api

    speaker = Speaker(name="Alice")
    speaker.set_embedding(np.ones(4, dtype=np.float32))
    db_session.add(speaker)
    db_session.commit()
    _conversation, segment, _chunk = _conversation_with_segment(
        db_session, tmp_path, status="completed", speaker=speaker
    )
    monkeypatch.setattr(api, "get_engine", lambda: FakeEngine())

    asyncio.run(api.delete_speaker(speaker.id, db_session))
    db_session.expire_all()
    updated = db_session.query(ConversationSegment).filter_by(id=segment.id).one()
    assert updated.speaker_id is None
    assert updated.speaker_name == "Deleted_Alice"
    assert not updated.speaker_name.startswith("Unknown_")
