"""Live word times are stored like file ones, from the conversation start."""
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models import Conversation

@pytest.fixture()
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    yield db
    db.close()
    engine.dispose()


def test_live_word_times_are_stored_from_the_conversation_start(db_session):
    from app.services import create_segment_from_result
    conversation = Conversation(title="live", start_time=datetime.now(timezone.utc), status="recording")
    db_session.add(conversation)
    db_session.flush()
    words = [{"word": " hello", "start": 0.5, "end": 0.9, "probability": 0.9},
             {"word": " there", "start": 1.0, "end": None, "probability": 0.8}]
    seg = {"start": 0.5, "end": 1.4, "text": "hello there", "speaker": "Unknown", "words": words}
    saved = create_segment_from_result(seg, conversation.id, conversation.start_time, db_session, threshold=0.3,
                                       start_offset_base=120.0)
    assert (saved.start_offset, saved.end_offset) == (120.5, 121.4)
    assert [(w["start"], w["end"]) for w in saved.words] == [(120.5, 120.9), (121.0, None)]
    assert words[0]["start"] == 0.5   # the chunk's own result is left as it was
