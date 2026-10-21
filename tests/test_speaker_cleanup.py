"""Cleaning up Unknown voices: speakers come with their seconds of speech and when they were last heard, an Unknown
with the closest person and the closest other Unknown, a speaker's segments to listen to, and a bulk delete of the
stray Unknowns of a few words that keeps their segments."""
import asyncio
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import api
from app.models import Base, Conversation, ConversationSegment, Speaker


def vec(*xs):
    v = np.zeros(8, dtype=np.float32)
    v[:len(xs)] = xs
    return v / np.linalg.norm(v)


@pytest.fixture()
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/speakers.db")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def add_speaker(db, name, voice):
    s = Speaker(name=name)
    s.set_embedding(voice)
    db.add(s)
    db.flush()
    return s


def add_lines(db, conv, speaker, *spans):
    t0 = conv.start_time
    for start, end in spans:
        db.add(ConversationSegment(conversation_id=conv.id, speaker_id=speaker.id, speaker_name=speaker.name,
                                   text=f"{speaker.name} {start}", start_offset=start, end_offset=end,
                                   start_time=t0 + timedelta(seconds=start), end_time=t0 + timedelta(seconds=end)))
    db.flush()


@pytest.fixture()
def people(db):
    conv = Conversation(title="chat", start_time=datetime(2026, 10, 21, 12, tzinfo=timezone.utc), status="completed")
    db.add(conv)
    db.flush()
    ann = add_speaker(db, "Ann", vec(1))
    ann_again = add_speaker(db, "Unknown_100", vec(1, 0.3))      # Ann heard once more before anyone named her
    stray = add_speaker(db, "Unknown_200", vec(0, 1))            # a few words of someone else
    add_lines(db, conv, ann, (0, 4), (10, 12))
    add_lines(db, conv, ann_again, (20, 30))
    add_lines(db, conv, stray, (40, 41.5))
    db.commit()
    return conv, ann, ann_again, stray


def test_speakers_come_with_their_speech_and_when_they_were_last_heard(db, people):
    conv, ann, _, stray = people
    rows = {s.name: s for s in asyncio.run(api.list_speakers(db))}
    assert rows["Ann"].segment_count == 2 and rows["Ann"].speech_seconds == pytest.approx(6.0)
    assert rows["Unknown_200"].speech_seconds == pytest.approx(1.5)
    assert rows["Ann"].last_heard.replace(tzinfo=timezone.utc) == conv.start_time.replace(tzinfo=timezone.utc) + timedelta(seconds=12)


def test_an_unknown_comes_with_the_person_and_the_unknown_it_sounds_like(db, people):
    _, ann, ann_again, stray = people
    matches = {m.speaker_name: m for m in asyncio.run(api.unknown_voice_matches(db))}
    assert set(matches) == {"Unknown_100", "Unknown_200"}            # named people are not listed
    close = matches["Unknown_100"]
    assert close.person.speaker_id == ann.id and close.person.similarity == pytest.approx(float(vec(1) @ vec(1, 0.3)), abs=1e-3)
    assert close.unknown.speaker_id == stray.id
    assert matches["Unknown_200"].person.similarity < 0.1


def test_a_speakers_segments_newest_or_longest_first(db, people):
    conv, ann, _, _ = people
    recent = asyncio.run(api.list_speaker_segments(ann.id, db=db))
    assert [s.start_offset for s in recent] == [10, 0] and recent[0].conversation_title == "chat"
    longest = asyncio.run(api.list_speaker_segments(ann.id, limit=1, order="longest", db=db))
    assert [(s.start_offset, s.end_offset) for s in longest] == [(0, 4)]
    with pytest.raises(HTTPException) as e:
        asyncio.run(api.list_speaker_segments(999, db=db))
    assert e.value.status_code == 404


def test_deleting_stray_unknowns_keeps_longer_ones_and_every_segment(db, people, monkeypatch):
    _, ann, ann_again, stray = people

    class Engine:
        def clear_speaker_cache(self):
            pass

    monkeypatch.setattr(api, "get_engine", lambda: Engine())
    r = asyncio.run(api.delete_all_unknown_speakers(max_seconds=5, db=db))
    assert r["deleted_count"] == 1
    assert {s.name for s in db.query(Speaker)} == {"Ann", "Unknown_100"}
    assert db.query(ConversationSegment).count() == 4                 # the stray's line stays, without a speaker
    assert db.query(ConversationSegment).filter_by(speaker_id=None).one().text == "Unknown_200 40"
