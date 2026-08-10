"""Regression tests for backup/restore atomicity (OPUS-001/KIMI-001/GLM-002)
and profile-overwrite protection (OPUS-004/QWEN-017).

All failures reproduced against pinned SHA 700976f; all pass after the fix.
"""
import json
import os

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Speaker


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")

    # Mirror app.database production pragmas so FK cascades behave identically.
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()

    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    yield db
    db.close()
    engine.dispose()


@pytest.fixture()
def backup_dir(tmp_path, monkeypatch):
    from app import backup_api
    d = tmp_path / "backups"
    d.mkdir()
    monkeypatch.setattr(backup_api, "_BACKUPS_DIR", str(d))
    return d


def _make_speaker(db, name="Alice"):
    import numpy as np
    s = Speaker(name=name)
    s.set_embedding(np.ones(512, dtype=np.float32))
    db.add(s)
    db.commit()
    return s


class TestRestoreAtomicity:
    def test_failed_restore_preserves_existing_speakers(self, db_session, backup_dir, tmp_path, monkeypatch):
        """Malformed profile must NOT wipe the existing database (was P0)."""
        from app import backup_api

        _make_speaker(db_session, "Original")

        # Profile whose rebuild raises mid-way (duplicate names)
        bad = {
            "name": "BadProfile",
            "speakers": [
                {"id": 1, "name": "Duplicate", "embedding": [0.1] * 4},
                {"id": 2, "name": "Duplicate", "embedding": [0.2] * 4},
            ],
            "segments": [],
        }
        path = backup_dir / "profile_bad.json"
        path.write_text(json.dumps(bad))
        monkeypatch.chdir(tmp_path)

        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            import asyncio
            asyncio.run(backup_api.restore_from_file(filename="profile_bad.json", db=db_session))

        survivors = db_session.query(Speaker).all()
        names = [s.name for s in survivors]
        assert "Original" in names, (
            f"failed restore destroyed existing speakers; remaining={names}"
        )

    @staticmethod
    async def _restore(backup_api, filename, db):
        return await backup_api.restore_from_file(filename=filename, db=db)

    def test_malformed_json_restore_preserves_data(self, db_session, backup_dir, tmp_path, monkeypatch):
        """A syntactically broken JSON file must fail before any DB change."""
        from app import backup_api
        _make_speaker(db_session, "Keeper")

        path = backup_dir / "profile_broken.json"
        path.write_text("{not valid json")
        monkeypatch.chdir(tmp_path)

        import asyncio

        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            asyncio.run(backup_api.restore_from_file(filename="profile_broken.json", db=db_session))

        names = [s.name for s in db_session.query(Speaker).all()]
        assert "Keeper" in names

    def test_duplicate_names_rejected_before_wipe(self, db_session, backup_dir, tmp_path, monkeypatch):
        from fastapi import HTTPException

        from app import backup_api
        _make_speaker(db_session, "Keeper")

        bad = {
            "name": "Dupes",
            "speakers": [
                {"id": 1, "name": "Same"},
                {"id": 2, "name": "Same"},
            ],
        }
        (backup_dir / "profile_dupes.json").write_text(json.dumps(bad))
        monkeypatch.chdir(tmp_path)

        import asyncio
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(backup_api.restore_from_file(filename="profile_dupes.json", db=db_session))
        assert exc_info.value.status_code == 400

        names = [s.name for s in db_session.query(Speaker).all()]
        assert names == ["Keeper"]


class TestCheckpointRestoreRoundTrip:
    def test_checkpoint_restore_preserves_emotion_profiles(self, db_session, backup_dir, monkeypatch):
        """SOL-003: checkpoints used to omit emotion profiles, so restoring one
        silently deleted all learned emotion state. Round-trip must be lossless."""
        import numpy as np

        from app import backup_api
        from app.models import SpeakerEmotionProfile

        speaker = _make_speaker(db_session, "Trained")
        prof = SpeakerEmotionProfile(
            speaker_id=speaker.id,
            emotion_category="happy",
            sample_count=4,
        )
        prof.set_embedding(np.full(1024, 0.25, dtype=np.float32))
        db_session.add(prof)
        db_session.commit()

        monkeypatch.chdir(backup_dir.parent)
        import asyncio
        asyncio.run(backup_api.create_checkpoint(profile_name="rt", db=db_session))
        ckpt_file = backup_dir / next(
            f for f in os.listdir(backup_dir) if f.startswith("checkpoint_rt_")
        )
        assert ckpt_file.exists()

        # Checkpoint must actually contain the emotion profile now.
        payload = json.loads(ckpt_file.read_text())
        assert payload["speakers"][0].get("emotion_profiles"), (
            "checkpoint must serialize emotion profiles or restore destroys them"
        )

        asyncio.run(backup_api.restore_from_file(filename=ckpt_file.name, db=db_session))
        db_session.expire_all()
        profiles = db_session.query(SpeakerEmotionProfile).all()
        assert len(profiles) == 1, f"emotion profiles lost after checkpoint restore: {len(profiles)}"
        assert profiles[0].emotion_category == "happy"
        assert profiles[0].sample_count == 4


class TestProfileOverwriteProtection:
    def test_create_refuses_existing_name(self, db_session, backup_dir, monkeypatch):
        import asyncio

        from fastapi import HTTPException

        from app import backup_api
        from app.backup_api import CreateProfileRequest

        monkeypatch.chdir(backup_dir.parent)
        req = CreateProfileRequest(name="mine", description="first")
        asyncio.run(backup_api.create_profile(request=req, db=db_session))

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(backup_api.create_profile(request=req, db=db_session))
        assert exc_info.value.status_code == 409

    def test_duplicate_refuses_existing_name(self, db_session, backup_dir, monkeypatch):
        import asyncio

        from fastapi import HTTPException

        from app import backup_api
        from app.backup_api import CreateProfileRequest

        monkeypatch.chdir(backup_dir.parent)
        req = CreateProfileRequest(name="copy", description="x")
        asyncio.run(backup_api.duplicate_profile(request=req, db=db_session))
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(backup_api.duplicate_profile(request=req, db=db_session))
        assert exc_info.value.status_code == 409

    def test_patch_update_still_overwrites(self, db_session, backup_dir, monkeypatch):
        import asyncio

        from app import backup_api
        from app.backup_api import CreateProfileRequest, UpdateProfileRequest

        monkeypatch.chdir(backup_dir.parent)
        asyncio.run(backup_api.create_profile(
            request=CreateProfileRequest(name="upd", description="v1"), db=db_session))
        # PATCH is the sanctioned overwrite path — must succeed.
        result = asyncio.run(backup_api.update_profile(
            profile_name="upd", request=UpdateProfileRequest(description="v2"), db=db_session))
        assert result["message"]

    def test_atomic_no_clobber_has_exactly_one_concurrent_winner(self, backup_dir):
        """The publish primitive itself must close the exists/replace race."""
        import threading

        from fastapi import HTTPException

        from app import backup_api

        path = str(backup_dir / "profile_race.json")
        barrier = threading.Barrier(2)
        successes = []
        errors = []

        def writer(value):
            barrier.wait()
            try:
                backup_api._dump_json(path, {"winner": value}, allow_overwrite=False)
                successes.append(value)
            except HTTPException as exc:
                errors.append(exc.status_code)

        threads = [threading.Thread(target=writer, args=(value,)) for value in (1, 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(successes) == 1
        assert errors == [409]
        assert json.loads((backup_dir / "profile_race.json").read_text())["winner"] in {1, 2}


class TestRestoreValidationAndMapping:
    def test_invalid_settings_rejected_before_database_mutation(
        self, db_session, backup_dir, tmp_path, monkeypatch
    ):
        import asyncio

        from fastapi import HTTPException

        from app import backup_api

        _make_speaker(db_session, "Keeper")
        payload = {
            "name": "InvalidSettings",
            "settings": {"speaker_threshold": 5.0},
            "speakers": [{"id": 7, "name": "Replacement"}],
            "segments": [],
        }
        (backup_dir / "profile_invalid_settings.json").write_text(json.dumps(payload))
        monkeypatch.chdir(tmp_path)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(backup_api.restore_from_file(
                filename="profile_invalid_settings.json", db=db_session
            ))
        assert exc_info.value.status_code == 400
        db_session.expire_all()
        assert [speaker.name for speaker in db_session.query(Speaker).all()] == ["Keeper"]

    def test_settings_io_failure_reports_partial_success_truthfully(
        self, db_session, backup_dir, tmp_path, monkeypatch
    ):
        import asyncio

        from app import backup_api
        from app.config import VoiceSettings

        class FailingConfig:
            def get_settings(self):
                return VoiceSettings()

            def update_settings(self, _updates):
                raise OSError("read-only filesystem")

            def reload_settings(self):
                raise AssertionError("reload must not run after failed save")

        _make_speaker(db_session, "Old")
        payload = {
            "name": "GoodDBBadSettingsIO",
            "settings": {"speaker_threshold": 0.44},
            "speakers": [{"id": 7, "name": "Restored", "embedding": [1.0, 0.0]}],
            "segments": [],
        }
        (backup_dir / "profile_settings_io.json").write_text(json.dumps(payload))
        monkeypatch.setattr(backup_api, "get_config", lambda: FailingConfig())
        monkeypatch.chdir(tmp_path)

        result = asyncio.run(backup_api.restore_from_file(
            filename="profile_settings_io.json", db=db_session
        ))
        assert result["settings_restored"] is False
        assert "Database restored" in result["settings_warning"]
        db_session.expire_all()
        assert [speaker.name for speaker in db_session.query(Speaker).all()] == ["Restored"]

    def test_cross_database_id_collision_remaps_by_backup_name(
        self, db_session, backup_dir, tmp_path, monkeypatch
    ):
        import asyncio
        from datetime import datetime, timezone

        from app import backup_api
        from app.models import Conversation, ConversationSegment

        _make_speaker(db_session, "Alice")
        bob = _make_speaker(db_session, "Bob")
        conversation = Conversation(
            title="mapping", start_time=datetime.now(timezone.utc), status="completed"
        )
        db_session.add(conversation)
        db_session.flush()
        segment = ConversationSegment(
            conversation_id=conversation.id,
            speaker_id=bob.id,
            speaker_name="Bob",
            text="hello",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            start_offset=0.0,
            end_offset=1.0,
        )
        db_session.add(segment)
        db_session.commit()

        payload = {
            "name": "ForeignIds",
            "speakers": [
                {"id": 7, "name": "Alice", "embedding": [1.0, 0.0]},
                {"id": 8, "name": "Bob", "embedding": [0.0, 1.0]},
            ],
            "segments": [{
                "id": segment.id,
                "conversation_id": conversation.id,
                # Numeric 2 collides with target Bob, but the backup itself
                # says this segment belongs to Alice.
                "speaker_id": 2,
                "speaker_name": "Alice",
            }],
        }
        (backup_dir / "profile_foreign_ids.json").write_text(json.dumps(payload))
        monkeypatch.chdir(tmp_path)

        result = asyncio.run(backup_api.restore_from_file(
            filename="profile_foreign_ids.json", db=db_session
        ))
        db_session.expire_all()
        restored_segment = db_session.query(ConversationSegment).filter_by(id=segment.id).one()
        restored_speaker = db_session.query(Speaker).filter_by(id=restored_segment.speaker_id).one()
        assert restored_speaker.name == "Alice"
        assert restored_segment.speaker_name == "Alice"
        assert result["segments_remapped_by_name"] == 1
