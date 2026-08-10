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

    def test_failure_after_destructive_sql_rolls_back_original_state(
        self, db_session, backup_dir, tmp_path, monkeypatch
    ):
        """A rebuild failure after DELETE has begun must restore old rows."""
        import asyncio

        from fastapi import HTTPException

        from app import backup_api

        _make_speaker(db_session, "Original")
        payload = {
            "name": "FailsMidRebuild",
            "speakers": [{
                "id": 7,
                "name": "Replacement",
                "embedding": [1.0, 0.0],
                # Missing emotion_category raises after the speaker wipe and
                # replacement INSERT have already executed.
                "emotion_profiles": [{"embedding": [0.2, 0.3]}],
            }],
            "segments": [],
        }
        (backup_dir / "profile_mid_rebuild.json").write_text(json.dumps(payload))
        monkeypatch.chdir(tmp_path)

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(backup_api.restore_from_file(
                filename="profile_mid_rebuild.json", db=db_session
            ))
        assert exc_info.value.status_code == 500
        db_session.expire_all()
        assert [speaker.name for speaker in db_session.query(Speaker).all()] == ["Original"]


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

    def test_concurrent_checkpoint_requests_publish_distinct_files(
        self, backup_dir, monkeypatch
    ):
        """Same-profile requests in one clock tick must never overwrite."""
        import asyncio
        import threading
        from concurrent.futures import ThreadPoolExecutor
        from datetime import datetime as RealDateTime
        from datetime import timezone

        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app import backup_api
        from app.database import Base

        engine = create_engine(
            f"sqlite:///{backup_dir.parent / 'checkpoint-race.db'}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)

        class FrozenDateTime:
            @classmethod
            def now(cls, tz=None):
                return RealDateTime(
                    2026, 8, 10, 12, 34, 56, 123456, tzinfo=tz or timezone.utc
                )

        monkeypatch.setattr(backup_api, "datetime", FrozenDateTime)
        barrier = threading.Barrier(2)

        def create_one():
            db = Session()
            try:
                barrier.wait()
                return asyncio.run(backup_api.create_checkpoint("race", db))
            finally:
                db.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(create_one) for _ in range(2)]
            results = [future.result() for future in futures]
        engine.dispose()

        filenames = {result["filename"] for result in results}
        assert len(filenames) == 2
        assert all((backup_dir / filename).exists() for filename in filenames)


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

    def test_foreign_database_id_collision_cannot_touch_local_segment_state(
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
            "database_namespace": "00000000-0000-0000-0000-foreign000000",
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
        # The foreign payload says Alice, but the target row's own local name
        # remains Bob and is safely reconnected to restored Bob by name.
        assert restored_speaker.name == "Bob"
        assert restored_segment.speaker_name == "Bob"
        assert result["segment_namespace_match"] is False
        assert result["segments_skipped_namespace"] == 1
        assert result["segments_remapped_from_local_names"] == 1
        assert result["segments_remapped_by_name"] == 1

    def test_same_database_snapshot_can_replay_segment_state(
        self, db_session, backup_dir, tmp_path, monkeypatch
    ):
        import asyncio
        from datetime import datetime, timezone

        from app import backup_api
        from app.models import Conversation, ConversationSegment

        alice = _make_speaker(db_session, "Alice")
        bob = _make_speaker(db_session, "Bob")
        conversation = Conversation(
            title="same namespace", start_time=datetime.now(timezone.utc), status="completed"
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
        namespace = backup_api._get_database_namespace(db_session)

        payload = {
            "name": "LocalSnapshot",
            "database_namespace": namespace,
            "speakers": [
                {"id": alice.id, "name": "Alice", "embedding": [1.0, 0.0]},
                {"id": bob.id, "name": "Bob", "embedding": [0.0, 1.0]},
            ],
            "segments": [{
                "id": segment.id,
                "conversation_id": conversation.id,
                "snapshot_uuid": segment.snapshot_uuid,
                "conversation_snapshot_uuid": conversation.snapshot_uuid,
                "speaker_id": alice.id,
                "speaker_name": "Alice",
                "is_misidentified": True,
            }],
        }
        (backup_dir / "profile_local_snapshot.json").write_text(json.dumps(payload))
        monkeypatch.chdir(tmp_path)

        result = asyncio.run(backup_api.restore_from_file(
            filename="profile_local_snapshot.json", db=db_session
        ))
        db_session.expire_all()
        restored = db_session.query(ConversationSegment).filter_by(id=segment.id).one()
        assert restored.speaker_name == "Alice"
        assert restored.is_misidentified is True
        assert result["segment_namespace_match"] is True
        assert result["segments_skipped_namespace"] == 0

    def test_same_database_stale_snapshot_ids_cannot_touch_reused_rows(
        self, db_session, backup_dir, tmp_path, monkeypatch
    ):
        import asyncio
        from datetime import datetime, timezone

        from app import backup_api
        from app.models import Conversation, ConversationSegment, Speaker

        alice = _make_speaker(db_session, "Alice")
        bob = _make_speaker(db_session, "Bob")
        old_conversation = Conversation(
            title="old", start_time=datetime.now(timezone.utc), status="completed"
        )
        db_session.add(old_conversation)
        db_session.flush()
        old_segment = ConversationSegment(
            conversation_id=old_conversation.id,
            speaker_id=alice.id,
            speaker_name="Alice",
            text="old segment",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            start_offset=0.0,
            end_offset=1.0,
            is_misidentified=True,
        )
        db_session.add(old_segment)
        db_session.commit()

        namespace = backup_api._get_database_namespace(db_session)
        old_conversation_id = old_conversation.id
        old_segment_id = old_segment.id
        payload = {
            "name": "StaleLocalSnapshot",
            "database_namespace": namespace,
            "speakers": [
                {"id": alice.id, "name": "Alice", "embedding": [1.0, 0.0]},
                {"id": bob.id, "name": "Bob", "embedding": [0.0, 1.0]},
            ],
            "segments": [{
                "id": old_segment_id,
                "conversation_id": old_conversation_id,
                "snapshot_uuid": old_segment.snapshot_uuid,
                "conversation_snapshot_uuid": old_conversation.snapshot_uuid,
                "speaker_id": alice.id,
                "speaker_name": "Alice",
                "is_misidentified": True,
            }],
        }

        db_session.delete(old_segment)
        db_session.delete(old_conversation)
        db_session.commit()

        replacement_conversation = Conversation(
            id=old_conversation_id,
            title="replacement",
            start_time=datetime.now(timezone.utc),
            status="completed",
        )
        db_session.add(replacement_conversation)
        db_session.flush()
        replacement_segment = ConversationSegment(
            id=old_segment_id,
            conversation_id=replacement_conversation.id,
            speaker_id=bob.id,
            speaker_name="Bob",
            text="replacement segment",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            start_offset=0.0,
            end_offset=1.0,
            is_misidentified=False,
        )
        db_session.add(replacement_segment)
        db_session.commit()
        assert replacement_segment.snapshot_uuid != payload["segments"][0]["snapshot_uuid"]
        assert replacement_conversation.snapshot_uuid != payload["segments"][0]["conversation_snapshot_uuid"]

        (backup_dir / "profile_stale_local_snapshot.json").write_text(json.dumps(payload))
        monkeypatch.chdir(tmp_path)
        result = asyncio.run(backup_api.restore_from_file(
            filename="profile_stale_local_snapshot.json", db=db_session
        ))

        db_session.expire_all()
        restored_segment = db_session.query(ConversationSegment).filter_by(
            id=old_segment_id
        ).one()
        restored_speaker = db_session.query(Speaker).filter_by(
            id=restored_segment.speaker_id
        ).one()
        assert restored_speaker.name == "Bob"
        assert restored_segment.speaker_name == "Bob"
        assert restored_segment.is_misidentified is False
        assert result["segments_not_found"] == 1
        assert result["segments_remapped_from_local_names"] == 1

    def test_same_database_legacy_integer_ids_are_not_replayed(
        self, db_session, backup_dir, tmp_path, monkeypatch
    ):
        import asyncio
        from datetime import datetime, timezone

        from app import backup_api
        from app.models import Conversation, ConversationSegment, Speaker

        alice = _make_speaker(db_session, "Alice")
        bob = _make_speaker(db_session, "Bob")
        conversation = Conversation(
            title="legacy", start_time=datetime.now(timezone.utc), status="completed"
        )
        db_session.add(conversation)
        db_session.flush()
        segment = ConversationSegment(
            conversation_id=conversation.id,
            speaker_id=bob.id,
            speaker_name="Bob",
            text="current local row",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            start_offset=0.0,
            end_offset=1.0,
        )
        db_session.add(segment)
        db_session.commit()

        payload = {
            "name": "LegacyLocalSnapshot",
            "database_namespace": backup_api._get_database_namespace(db_session),
            "speakers": [
                {"id": alice.id, "name": "Alice", "embedding": [1.0, 0.0]},
                {"id": bob.id, "name": "Bob", "embedding": [0.0, 1.0]},
            ],
            # Legacy snapshots have only reusable SQLite integer IDs.
            "segments": [{
                "id": segment.id,
                "conversation_id": conversation.id,
                "speaker_id": alice.id,
                "speaker_name": "Alice",
                "is_misidentified": True,
            }],
        }
        (backup_dir / "profile_legacy_local_snapshot.json").write_text(json.dumps(payload))
        monkeypatch.chdir(tmp_path)
        result = asyncio.run(backup_api.restore_from_file(
            filename="profile_legacy_local_snapshot.json", db=db_session
        ))

        db_session.expire_all()
        restored_segment = db_session.query(ConversationSegment).filter_by(id=segment.id).one()
        restored_speaker = db_session.query(Speaker).filter_by(
            id=restored_segment.speaker_id
        ).one()
        assert restored_speaker.name == "Bob"
        assert restored_segment.speaker_name == "Bob"
        assert restored_segment.is_misidentified is False
        assert result["segments_skipped_identity"] == 1

    def test_pre_namespace_legacy_file_replays_verified_segment_state(
        self, db_session, backup_dir, tmp_path, monkeypatch
    ):
        """Files written before database namespaces existed must still be able
        to revert segment assignments — via id verified against conversation
        and offsets — instead of silently reducing restore to a no-op."""
        import asyncio
        from datetime import datetime, timezone

        from app import backup_api
        from app.models import Conversation, ConversationSegment

        alice = _make_speaker(db_session, "Alice")
        bob = _make_speaker(db_session, "Bob")
        conversation = Conversation(
            title="legacy", start_time=datetime.now(timezone.utc), status="completed"
        )
        db_session.add(conversation)
        db_session.flush()
        segment = ConversationSegment(
            conversation_id=conversation.id,
            speaker_id=bob.id,
            speaker_name="Bob",
            text="reassigned after the backup was taken",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            start_offset=3.25,
            end_offset=7.5,
        )
        db_session.add(segment)
        db_session.commit()

        # Exactly what pre-namespace backup_api serialized: no
        # database_namespace, no snapshot UUIDs, only integer identities.
        payload = {
            "name": "PreNamespaceCheckpoint",
            "speakers": [
                {"id": alice.id, "name": "Alice", "embedding": [1.0, 0.0]},
                {"id": bob.id, "name": "Bob", "embedding": [0.0, 1.0]},
            ],
            "segments": [{
                "id": segment.id,
                "conversation_id": conversation.id,
                "speaker_id": alice.id,
                "speaker_name": "Alice",
                "is_misidentified": True,
                "start_offset": 3.25,
                "end_offset": 7.5,
            }],
        }
        (backup_dir / "checkpoint_pre_namespace.json").write_text(json.dumps(payload))
        monkeypatch.chdir(tmp_path)
        result = asyncio.run(backup_api.restore_from_file(
            filename="checkpoint_pre_namespace.json", db=db_session
        ))

        db_session.expire_all()
        restored_segment = db_session.query(ConversationSegment).filter_by(id=segment.id).one()
        restored_speaker = db_session.query(Speaker).filter_by(
            id=restored_segment.speaker_id
        ).one()
        assert restored_speaker.name == "Alice"
        assert restored_segment.speaker_name == "Alice"
        assert restored_segment.is_misidentified is True
        assert result["segments_replayed_by_legacy_identity"] == 1
        assert result["legacy_restore_warning"] is not None

    def test_legacy_record_failing_offset_verification_is_not_replayed(
        self, db_session, backup_dir, tmp_path, monkeypatch
    ):
        """A legacy integer id whose row no longer matches the record's
        conversation/offsets (id reuse) must not be trusted; the live row is
        reconnected from its own current name instead."""
        import asyncio
        from datetime import datetime, timezone

        from app import backup_api
        from app.models import Conversation, ConversationSegment

        alice = _make_speaker(db_session, "Alice")
        bob = _make_speaker(db_session, "Bob")
        conversation = Conversation(
            title="legacy-reused", start_time=datetime.now(timezone.utc), status="completed"
        )
        db_session.add(conversation)
        db_session.flush()
        segment = ConversationSegment(
            conversation_id=conversation.id,
            speaker_id=bob.id,
            speaker_name="Bob",
            text="different row now occupying that id",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            start_offset=100.0,
            end_offset=104.0,
        )
        db_session.add(segment)
        db_session.commit()

        payload = {
            "name": "PreNamespaceStale",
            "speakers": [
                {"id": alice.id, "name": "Alice", "embedding": [1.0, 0.0]},
                {"id": bob.id, "name": "Bob", "embedding": [0.0, 1.0]},
            ],
            "segments": [{
                "id": segment.id,
                "conversation_id": conversation.id,
                "speaker_id": alice.id,
                "speaker_name": "Alice",
                "is_misidentified": True,
                # Offsets from the (deleted) row that originally held this id.
                "start_offset": 3.25,
                "end_offset": 7.5,
            }],
        }
        (backup_dir / "checkpoint_pre_namespace_stale.json").write_text(json.dumps(payload))
        monkeypatch.chdir(tmp_path)
        result = asyncio.run(backup_api.restore_from_file(
            filename="checkpoint_pre_namespace_stale.json", db=db_session
        ))

        db_session.expire_all()
        restored_segment = db_session.query(ConversationSegment).filter_by(id=segment.id).one()
        restored_speaker = db_session.query(Speaker).filter_by(
            id=restored_segment.speaker_id
        ).one()
        assert restored_speaker.name == "Bob"
        assert restored_segment.speaker_name == "Bob"
        assert restored_segment.is_misidentified is False
        assert result["segments_replayed_by_legacy_identity"] == 0
        assert result["segments_not_found"] == 1

    def test_mixed_era_same_namespace_records_without_uuids_still_replay(
        self, db_session, backup_dir, tmp_path, monkeypatch
    ):
        """A file that names THIS database but whose records predate snapshot
        UUIDs (mid-upgrade builds) is strictly safer than a pure-legacy file;
        it must replay through the same verified integer identity instead of
        silently skipping every record."""
        import asyncio
        from datetime import datetime, timezone

        from app import backup_api
        from app.models import Conversation, ConversationSegment

        alice = _make_speaker(db_session, "Alice")
        bob = _make_speaker(db_session, "Bob")
        conversation = Conversation(
            title="mixed-era", start_time=datetime.now(timezone.utc), status="completed"
        )
        db_session.add(conversation)
        db_session.flush()
        segment = ConversationSegment(
            conversation_id=conversation.id,
            speaker_id=bob.id,
            speaker_name="Bob",
            text="reassigned after the checkpoint",
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            start_offset=12.0,
            end_offset=15.5,
        )
        db_session.add(segment)
        db_session.commit()

        payload = {
            "name": "MixedEraCheckpoint",
            "database_namespace": backup_api._get_database_namespace(db_session),
            "speakers": [
                {"id": alice.id, "name": "Alice", "embedding": [1.0, 0.0]},
                {"id": bob.id, "name": "Bob", "embedding": [0.0, 1.0]},
            ],
            "segments": [{
                "id": segment.id,
                "conversation_id": conversation.id,
                "speaker_id": alice.id,
                "speaker_name": "Alice",
                "is_misidentified": True,
                "start_offset": 12.0,
                "end_offset": 15.5,
            }],
        }
        (backup_dir / "checkpoint_mixed_era.json").write_text(json.dumps(payload))
        monkeypatch.chdir(tmp_path)
        result = asyncio.run(backup_api.restore_from_file(
            filename="checkpoint_mixed_era.json", db=db_session
        ))

        db_session.expire_all()
        restored_segment = db_session.query(ConversationSegment).filter_by(id=segment.id).one()
        assert restored_segment.speaker_name == "Alice"
        assert restored_segment.is_misidentified is True
        assert result["segments_replayed_by_legacy_identity"] == 1
        assert result["segment_namespace_match"] is True
        assert result["legacy_restore_warning"] is not None
