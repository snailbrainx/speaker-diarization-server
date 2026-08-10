"""Safety tests for existing conversation/speaker operations and audit fixes."""
import asyncio
from datetime import datetime, timezone

import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Conversation, ConversationSegment, Speaker


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
    monkeypatch.setattr(services, "data_path", lambda: str(tmp_path))
    seg_dir = tmp_path / "stream_segments" / f"conv_{conversation.id}"
    seg_dir.mkdir(parents=True)
    (seg_dir / "seg_0001.wav").write_bytes(b"audio")

    result = asyncio.run(conversation_api.delete_conversation(conversation.id, db_session))
    assert "deleted" in result["message"]
    assert not seg_dir.exists()
    assert db_session.query(Conversation).filter_by(id=conversation.id).first() is None


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
