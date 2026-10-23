"""The web interface (app/static) and the API it needs: the page is served at /ui/ and browsers opening / are sent
there, every module it imports exists, a conversation's recording can be fetched (and seeked) whole, and live word
times are stored like file ones, from the conversation start."""
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.models import Conversation

STATIC = Path(__file__).resolve().parents[1] / "app" / "static"


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    yield db
    db.close()
    engine.dispose()


def test_the_web_interface_is_served_and_browsers_are_sent_to_it():
    from app.main import app
    client = TestClient(app)   # no lifespan: no models are loaded
    page = client.get("/ui/")
    assert page.status_code == 200 and "js/main.js" in page.text
    assert client.get("/ui/js/main.js").headers["content-type"].startswith(("text/javascript", "application/javascript"))
    browser = client.get("/", headers={"accept": "text/html,application/xhtml+xml"}, follow_redirects=False)
    assert browser.status_code == 307 and browser.headers["location"] == "/ui/"
    assert client.get("/").json()["ui"] == "/ui/"   # API clients and the health check still get JSON


def test_every_module_the_page_loads_exists():
    html = (STATIC / "index.html").read_text()
    wanted = [STATIC / m for m in re.findall(r'(?:src|href)="([^":#]+)"', html)]
    for js in STATIC.rglob("*.js"):
        text = js.read_text()
        wanted += [js.parent / m for m in re.findall(r'from "(\.[^"]+)"', text)]
        wanted += [js.parent / m for m in re.findall(r'new URL\("(\.[^"]+)", import\.meta\.url\)', text)]
    missing = sorted({str(p.resolve().relative_to(STATIC)) for p in wanted if not p.resolve().is_file()})
    assert len(wanted) > 20 and missing == []


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
