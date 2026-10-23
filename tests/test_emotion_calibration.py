"""Personal emotion calibration (app/emotion_calibration.py): a speaker's emotion corrections shift how the emotion
model's scores are read for them, so a few lines corrected to neutral stop their ordinary voice being called happy
while their clearly angry or surprised lines stay what they are."""
import asyncio
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import emotion_calibration as ec
from app.diarization import SpeakerRecognitionEngine
from app.models import Base, Conversation, ConversationSegment, Speaker

L = ec.LABELS


def logits(**scores):
    z = np.zeros(len(L))
    for name, v in scores.items():
        z[L.index(name)] = v
    return z


def test_no_corrections_means_the_models_own_answer():
    assert ec.fit([], []) is None
    assert ec.fit([logits(happy=3)], ["not-an-emotion"]) is None


def test_corrections_move_borderline_lines_and_not_clear_ones():
    plain = [logits(happy=3.0, neutral=2.0), logits(happy=2.8, neutral=2.1), logits(happy=3.2, neutral=1.9)]
    cal = ec.fit(plain, ["neutral"] * 3)
    assert cal.corrections == 3
    assert ec.decide(logits(happy=3.1, neutral=2.0), cal)["emotion"] == "neutral"   # another ordinary line
    assert ec.decide(logits(angry=8.0, neutral=1.0), cal)["emotion"] == "angry"     # clearly angry
    assert ec.decide(logits(happy=9.0, neutral=1.0), cal)["emotion"] == "happy"     # clearly happy
    out = ec.decide(logits(happy=3.1, neutral=2.0), cal)
    assert out["breakdown"]["emotion2vec"]["emotion"] == "happy"
    assert out["breakdown"]["personal_calibration"]["corrections"] == 3


def test_the_pull_to_zero_keeps_a_single_correction_small():
    one = ec.fit([logits(happy=6.0, neutral=1.0)], ["neutral"])
    assert np.abs(one.offsets).max() < 3.0
    assert ec.decide(logits(happy=6.0, neutral=1.0), one)["emotion"] == "happy"   # far from the line it corrected


# ---- through the engine and the database: what the app fits on ----

@pytest.fixture()
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/emo.db")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture()
def eng():
    """An engine whose emotion head is W = identity on the first nine of 16 dimensions: a line's 'embedding' carries
    its logits there, and a large shared part elsewhere (the same person's voice, whatever they feel)."""
    e = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    W = np.zeros((len(L), 16))
    W[:, :len(L)] = np.eye(len(L))
    e._emotion_head_cache = (W, np.zeros(len(L)))
    return e


def embedding(**scores):
    v = np.zeros(16, dtype=np.float32)
    v[:len(L)] = logits(**scores)
    v[len(L):] = 20.0   # the speaker's own voice, the same in every line
    return v


def add_line(db, conv, speaker, emb, emotion, corrected=False, **flags):
    seg = ConversationSegment(conversation_id=conv.id, speaker_id=speaker.id, speaker_name=speaker.name, text="x",
                              start_time=conv.start_time, end_time=conv.start_time, start_offset=0, end_offset=1,
                              emotion_category=emotion, emotion_corrected=corrected, **flags)
    seg.set_emotion_embedding(emb)
    db.add(seg)
    db.flush()
    return seg


@pytest.fixture()
def ann(db):
    conv = Conversation(title="c", start_time=datetime.now(timezone.utc), status="completed")
    ann = Speaker(name="Ann")
    ann.set_embedding(np.ones(4, dtype=np.float32))
    db.add_all([conv, ann])
    db.flush()
    for z in ((3.0, 2.0), (2.9, 2.0), (3.2, 2.1)):   # ordinary lines the model called happy, corrected to neutral
        add_line(db, conv, ann, embedding(happy=z[0], neutral=z[1]), "neutral", corrected=True)
    db.commit()
    return conv, ann


def test_three_neutral_corrections_do_not_make_everything_neutral(eng, db, ann):
    """The previous personal profiles compared embeddings: one person's lines are alike whatever they feel, so after
    three corrections to neutral nearly every line matched the neutral profile (RAVDESS: 92.5% -> 18% right)."""
    _, speaker = ann
    cal = eng.speaker_emotion_calibration(speaker.id, db)
    assert cal.corrections == 3
    angry = embedding(angry=6.0, neutral=1.0)
    assert np.dot(angry, embedding(happy=3.0, neutral=2.0)) / (np.linalg.norm(angry) * np.linalg.norm(embedding(happy=3.0, neutral=2.0))) > 0.9
    assert eng.calibrated_emotion(angry, cal)["emotion"] == "angry"
    assert eng.calibrated_emotion(embedding(surprised=5.0, happy=1.0), cal)["emotion"] == "surprised"
    assert eng.calibrated_emotion(embedding(happy=3.1, neutral=2.0), cal)["emotion"] == "neutral"


def test_corrections_left_out_of_learning_are_not_fitted(eng, db, ann):
    conv, speaker = ann
    add_line(db, conv, speaker, embedding(happy=3.0), "angry", corrected=True, emotion_misidentified=True)
    add_line(db, conv, speaker, embedding(happy=3.0), "sad", corrected=True, is_misidentified=True)
    add_line(db, conv, speaker, embedding(happy=3.0), "happy")   # not corrected
    db.commit()
    assert eng.speaker_emotion_calibration(speaker.id, db).corrections == 3
    assert eng.speaker_emotion_calibration(None, db) is None


def test_forgetting_a_speakers_emotions_stops_learning_from_their_corrections(eng, db, ann):
    from app import conversation_api
    _, speaker = ann
    asyncio.run(conversation_api.reset_speaker_emotion_profiles(speaker.id, None, db))
    assert eng.speaker_emotion_calibration(speaker.id, db) is None
    assert db.query(ConversationSegment).filter_by(emotion_category="neutral").count() == 3   # labels stay


def test_live_lines_given_to_a_speaker_later_are_read_with_their_corrections(eng, db, ann):
    """A live line that only gets its speaker when the session's track is named was first read without them."""
    from app.streaming_websocket import _reread_emotions
    conv, speaker = ann
    late = add_line(db, conv, speaker, embedding(happy=3.1, neutral=2.0), "happy")
    angry = add_line(db, conv, speaker, embedding(angry=6.0, neutral=1.0), "angry")
    reread = _reread_emotions(db, eng, [late.id, angry.id], speaker.id)
    assert {i: e["emotion_category"] for i, e in reread.items()} == {late.id: "neutral", angry.id: "angry"}
    assert late.detector_breakdown and angry.emotion_category == "angry"
