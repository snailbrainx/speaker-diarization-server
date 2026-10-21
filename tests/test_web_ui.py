"""A conversation's recording can be fetched (and seeked) whole, and live word times are stored like file
ones, from the conversation start."""
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models import Conversation

@pytest.fixture()
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    yield db
    db.close()
    engine.dispose()


def test_a_conversation_recording_is_served_whole_and_by_range(db_session, tmp_path):
    from app import conversation_api
    wav = tmp_path / "talk.wav"
    wav.write_bytes(b"RIFF" + bytes(range(256)) * 40)
    conversation = Conversation(title="t", start_time=datetime.now(timezone.utc), status="completed", audio_path=str(wav))
    unrecorded = Conversation(title="u", start_time=datetime.now(timezone.utc), status="recording", audio_path=None)
    db_session.add_all([conversation, unrecorded])
    db_session.commit()
    app = FastAPI()
    app.include_router(conversation_api.router)
    app.dependency_overrides[get_db] = lambda: db_session
    client = TestClient(app)

    whole = client.get(f"/conversations/{conversation.id}/audio")
    assert whole.status_code == 200 and whole.content == wav.read_bytes()
    assert whole.headers["content-type"] in ("audio/wav", "audio/x-wav")
    part = client.get(f"/conversations/{conversation.id}/audio", headers={"range": "bytes=100-199"})
    assert part.status_code == 206 and part.content == wav.read_bytes()[100:200]
    assert client.get(f"/conversations/{unrecorded.id}/audio").status_code == 404
    assert client.get("/conversations/999/audio").status_code == 404



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
