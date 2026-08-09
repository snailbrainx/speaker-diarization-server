"""Regression tests for backup/restore atomicity (OPUS-001/KIMI-001/GLM-002)
and profile-overwrite protection (OPUS-004/QWEN-017).

All failures reproduced against pinned SHA 700976f; all pass after the fix.
"""
import json

import pytest
from app.database import Base
from app.models import Speaker
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
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
        from app import backup_api
        from fastapi import HTTPException
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


class TestProfileOverwriteProtection:
    def test_create_refuses_existing_name(self, db_session, backup_dir, monkeypatch):
        import asyncio

        from app import backup_api
        from app.backup_api import CreateProfileRequest
        from fastapi import HTTPException

        monkeypatch.chdir(backup_dir.parent)
        req = CreateProfileRequest(name="mine", description="first")
        asyncio.run(backup_api.create_profile(request=req, db=db_session))

        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(backup_api.create_profile(request=req, db=db_session))
        assert exc_info.value.status_code == 409

    def test_duplicate_refuses_existing_name(self, db_session, backup_dir, monkeypatch):
        import asyncio

        from app import backup_api
        from app.backup_api import CreateProfileRequest
        from fastapi import HTTPException

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
