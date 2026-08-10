"""
API endpoints for Voice Profiles and Checkpoints.

Profiles and checkpoints are JSON snapshots of the speaker + segment state stored
under the `backups/` directory. All disk I/O, JSON serialization, and bulk SQL
reads are offloaded to worker threads so the event loop stays responsive when a
profile contains thousands of segments.
"""
import asyncio
import io
import json
import logging
import math
import os
import re
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .database import get_db
from .models import (
    AppMetadata,
    Conversation,
    ConversationSegment,
    Speaker,
    SpeakerEmotionProfile,
)
from .config import VoiceSettings, get_config

router = APIRouter(prefix="/profiles", tags=["Voice Profiles"])

logger = logging.getLogger(__name__)

_BACKUPS_DIR = "backups"
_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}(?:_\d{6}_[0-9a-f]{8})?$")
_DATABASE_NAMESPACE_KEY = "database_namespace_uuid"


def _get_database_namespace(db: Session) -> str:
    """Return this database's persistent UUID without committing caller state."""
    bind = db.get_bind()
    candidate = str(uuid.uuid4())
    try:
        # Use an independent transaction: snapshot creation must not commit
        # unrelated request-session changes, while the namespace itself must
        # survive a later restore rollback. A unique PK arbitrates concurrent
        # first-use requests.
        with bind.begin() as conn:
            existing = conn.execute(
                select(AppMetadata.value).where(
                    AppMetadata.key == _DATABASE_NAMESPACE_KEY
                )
            ).scalar_one_or_none()
            if existing is not None:
                return existing
            conn.execute(insert(AppMetadata).values(
                key=_DATABASE_NAMESPACE_KEY,
                value=candidate,
            ))
            return candidate
    except IntegrityError:
        # Another concurrent request committed the singleton first.
        with bind.connect() as conn:
            existing = conn.execute(
                select(AppMetadata.value).where(
                    AppMetadata.key == _DATABASE_NAMESPACE_KEY
                )
            ).scalar_one()
        return existing


class CreateProfileRequest(BaseModel):
    name: str
    description: Optional[str] = None


class UpdateProfileRequest(BaseModel):
    description: Optional[str] = None


def sanitize_filename(name: str) -> str:
    """Sanitize a user-supplied profile name for safe use as a filename stem.

    Keeps alphanumerics plus ` -_.`, collapses `..` so a user can't traverse
    out of the backups dir, and falls back to `unnamed` on empty input.
    """
    cleaned = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_', '.'))
    cleaned = cleaned.strip().replace(' ', '_')
    cleaned = re.sub(r'\.{2,}', '.', cleaned).lstrip('.')
    return cleaned or "unnamed"


def _safe_backup_path(filename: str) -> str:
    """Resolve `filename` inside the backups directory, rejecting traversal."""
    backups_abs = os.path.realpath(_BACKUPS_DIR)
    candidate = os.path.realpath(os.path.join(_BACKUPS_DIR, filename))
    if candidate != backups_abs and not candidate.startswith(backups_abs + os.sep):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return candidate


def _profile_path(safe_name: str) -> str:
    return _safe_backup_path(f"profile_{safe_name}.json")


def _checkpoint_path(safe_name: str, timestamp: str) -> str:
    return _safe_backup_path(f"checkpoint_{safe_name}_{timestamp}.json")


def _tunable_settings(source) -> dict[str, Any]:
    """Extract every VoiceSettings field from `source` as a dict.

    Driving the schema off `VoiceSettings.model_fields` keeps save / create /
    checkpoint / restore in sync automatically — a new tunable doesn't need to
    be added in three places.
    """
    return {field: getattr(source, field) for field in VoiceSettings.model_fields}


def _serialize_speakers(db: Session, include_emotion_profiles: bool) -> list:
    speakers_out = []
    for speaker in db.query(Speaker).all():
        embedding = speaker.get_embedding()
        entry = {
            "id": speaker.id,
            "name": speaker.name,
            "embedding": embedding.tolist() if embedding is not None else None,
        }
        if include_emotion_profiles:
            entry["emotion_threshold"] = speaker.emotion_threshold
            emotion_profiles = []
            for prof in speaker.emotion_profiles:
                voice_emb = prof.get_voice_embedding()
                emotion_profiles.append({
                    "emotion_category": prof.emotion_category,
                    "embedding": prof.get_embedding().tolist(),
                    "sample_count": prof.sample_count,
                    "confidence_threshold": prof.confidence_threshold,
                    "voice_embedding": voice_emb.tolist() if voice_emb is not None else None,
                    "voice_sample_count": prof.voice_sample_count,
                    "voice_threshold": prof.voice_threshold,
                })
            entry["emotion_profiles"] = emotion_profiles
        speakers_out.append(entry)
    return speakers_out


def _serialize_segments(db: Session) -> list:
    return [
        {
            "id": seg.id,
            "conversation_id": seg.conversation_id,
            "snapshot_uuid": seg.snapshot_uuid,
            "conversation_snapshot_uuid": conversation_snapshot_uuid,
            "speaker_id": seg.speaker_id,
            "speaker_name": seg.speaker_name,
            "is_misidentified": seg.is_misidentified,
            "start_offset": seg.start_offset,
            "end_offset": seg.end_offset,
        }
        for seg, conversation_snapshot_uuid in db.query(
            ConversationSegment, Conversation.snapshot_uuid
        ).join(
            Conversation,
            Conversation.id == ConversationSegment.conversation_id,
        ).all()
    ]


def _dump_json(path: str, payload: dict, *, allow_overwrite: bool = True) -> None:
    """Publish JSON atomically, optionally refusing an existing destination.

    For create/duplicate operations ``allow_overwrite=False`` hard-links a
    fully written sibling tempfile into place. ``os.link`` is atomic and fails
    if the destination already exists, closing the check-then-replace race that
    let two concurrent creates both report success.
    """
    import tempfile as _tempfile
    target_dir = os.path.dirname(path) or "."
    fd, tmp = _tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=target_dir
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        if allow_overwrite:
            os.replace(tmp, path)
        else:
            try:
                os.link(tmp, path)
            except FileExistsError:
                raise HTTPException(
                    status_code=409,
                    detail="Profile already exists; use the update endpoint or choose a different name",
                )
            os.unlink(tmp)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _read_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


# OPUS-014: listing profiles/checkpoints used to fully parse EVERY backup file
# on each request just to count speakers/segments. Cache the parsed summary per
# (path, mtime_ns, size) — backups are immutable once written, so the stat
# tuple is a sound cache key.
_SUMMARY_CACHE: dict[str, tuple[tuple, dict]] = {}
_SUMMARY_CACHE_MAX = 512


def _backup_summary(filepath: str) -> dict | None:
    """Parsed summary of a backup file, cached by stat signature."""
    try:
        st = os.stat(filepath)
    except OSError:
        return None
    key = (filepath, st.st_mtime_ns, st.st_size)
    cached = _SUMMARY_CACHE.get(filepath)
    if cached and cached[0] == key:
        return cached[1]
    try:
        data = _read_json(filepath)
    except (json.JSONDecodeError, OSError, KeyError):
        return None
    summary = {
        "name": data.get("name", os.path.basename(filepath)),
        "description": data.get("description", ""),
        "filename": os.path.basename(filepath),
        "timestamp": data.get("timestamp", ""),
        "speakers_count": len(data.get("speakers", [])),
        "segments_count": len(data.get("segments", [])),
        "created_at": datetime.fromtimestamp(st.st_ctime, tz=timezone.utc).isoformat(),
    }
    if len(_SUMMARY_CACHE) >= _SUMMARY_CACHE_MAX:
        _SUMMARY_CACHE.clear()
    _SUMMARY_CACHE[filepath] = (key, summary)
    return summary


def save_current_state(profile_name: str, description: str, db: Session, allow_overwrite: bool = False) -> dict:
    """Save current speaker/segment state to profile file. Blocking — call via to_thread.

    allow_overwrite=False refuses to clobber an existing snapshot of the same
    name (409). Silent overwrites are how operators lose their only backup:
    `POST /profiles` or `/duplicate` with a reused name used to replace the
    stored speakers+segments with whatever the DB holds at that moment —
    including an EMPTY state right after a failed restore.
    """
    safe_name = sanitize_filename(profile_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(_BACKUPS_DIR, exist_ok=True)
    profile_file = _profile_path(safe_name)
    if not allow_overwrite and os.path.exists(profile_file):
        raise HTTPException(
            status_code=409,
            detail=f"Profile '{profile_name}' already exists; use the update endpoint or choose a different name",
        )

    settings = get_config().get_settings()
    database_namespace = _get_database_namespace(db)
    speakers = _serialize_speakers(db, include_emotion_profiles=True)
    segments = _serialize_segments(db)

    profile_data = {
        "timestamp": timestamp,
        "database_namespace": database_namespace,
        "name": profile_name,
        "description": description,
        "type": "profile",
        "settings": _tunable_settings(settings),
        "speakers": speakers,
        "segments": segments,
    }
    _dump_json(profile_file, profile_data, allow_overwrite=allow_overwrite)

    return {
        "filename": os.path.basename(profile_file),
        "speakers_count": len(speakers),
        "segments_count": len(segments),
        "timestamp": timestamp,
    }


@router.post("")
async def create_profile(request: CreateProfileRequest, db: Session = Depends(get_db)):
    """Create a new EMPTY voice profile with default settings."""
    def _work() -> dict:
        safe_name = sanitize_filename(request.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(_BACKUPS_DIR, exist_ok=True)
        profile_file = _profile_path(safe_name)
        if os.path.exists(profile_file):
            raise HTTPException(
                status_code=409,
                detail=f"Profile '{request.name}' already exists; choose a different name",
            )
        defaults = VoiceSettings()
        profile_data = {
            "timestamp": timestamp,
            "database_namespace": _get_database_namespace(db),
            "name": request.name,
            "description": request.description or "",
            "type": "profile",
            "settings": _tunable_settings(defaults),
            "speakers": [],
            "segments": [],
        }
        _dump_json(profile_file, profile_data, allow_overwrite=False)
        return {
            "message": f"Empty profile '{request.name}' created successfully",
            "name": request.name,
            "description": request.description or "",
            "filename": os.path.basename(profile_file),
            "speakers_count": 0,
            "segments_count": 0,
            "timestamp": timestamp,
        }

    try:
        return await asyncio.to_thread(_work)
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to create profile")


@router.post("/duplicate")
async def duplicate_profile(request: CreateProfileRequest, db: Session = Depends(get_db)):
    """Duplicate current state into a new profile (speakers + segments + settings)."""
    try:
        result = await asyncio.to_thread(save_current_state, request.name, request.description or "", db, False)
        return {
            "message": f"Profile '{request.name}' duplicated successfully",
            "name": request.name,
            "description": request.description or "",
            **result,
        }
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to duplicate profile")


@router.patch("/{profile_name}")
async def update_profile(
    profile_name: str,
    request: UpdateProfileRequest,
    db: Session = Depends(get_db),
):
    """Update existing profile with current state."""
    def _load_description() -> str:
        safe_name = sanitize_filename(profile_name)
        profile_file = _profile_path(safe_name)
        if request.description is not None:
            return request.description
        if os.path.exists(profile_file):
            try:
                return _read_json(profile_file).get("description", "")
            except Exception:
                return ""
        return ""

    try:
        description = await asyncio.to_thread(_load_description)
        # PATCH is the explicit update path: overwrite is intended here.
        result = await asyncio.to_thread(save_current_state, profile_name, description, db, True)
        return {
            "message": f"Profile '{profile_name}' updated successfully",
            "name": profile_name,
            "description": description,
            **result,
        }
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to update profile")


def _scan_profiles() -> list:
    if not os.path.exists(_BACKUPS_DIR):
        return []
    profiles = []
    for filename in os.listdir(_BACKUPS_DIR):
        if not filename.startswith("profile_") or not filename.endswith(".json"):
            continue
        summary = _backup_summary(os.path.join(_BACKUPS_DIR, filename))
        if summary is None:
            continue
        profiles.append(summary)
    profiles.sort(key=lambda x: x["name"])
    return profiles


@router.get("")
async def list_profiles():
    """List all voice profiles."""
    return {"profiles": await asyncio.to_thread(_scan_profiles)}


@router.delete("/{profile_name}")
async def delete_profile(profile_name: str):
    """Delete a profile and all its checkpoints."""
    def _work() -> dict:
        safe_name = sanitize_filename(profile_name)
        profile_file = _profile_path(safe_name)
        if not os.path.exists(profile_file):
            raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found")
        os.remove(profile_file)

        deleted_checkpoints = 0
        if os.path.exists(_BACKUPS_DIR):
            prefix = f"checkpoint_{safe_name}_"
            for filename in os.listdir(_BACKUPS_DIR):
                if filename.startswith(prefix) and filename.endswith(".json"):
                    os.remove(os.path.join(_BACKUPS_DIR, filename))
                    deleted_checkpoints += 1
        return {
            "message": f"Profile '{profile_name}' and {deleted_checkpoints} checkpoints deleted",
            "deleted_checkpoints": deleted_checkpoints,
        }

    return await asyncio.to_thread(_work)


@router.post("/{profile_name}/checkpoints")
async def create_checkpoint(profile_name: str, db: Session = Depends(get_db)):
    """Create a checkpoint (snapshot) of current profile state."""
    def _work() -> dict:
        safe_name = sanitize_filename(profile_name)
        timestamp = (
            f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        os.makedirs(_BACKUPS_DIR, exist_ok=True)
        checkpoint_file = _checkpoint_path(safe_name, timestamp)

        description = ""
        profile_file = _profile_path(safe_name)
        if os.path.exists(profile_file):
            try:
                description = _read_json(profile_file).get("description", "")
            except Exception:
                pass

        settings = get_config().get_settings()
        # Checkpoints include emotion profiles: restore is a destructive
        # wipe-and-rebuild, so anything the checkpoint omits is permanently
        # deleted by a successful restore (SOL-003). Keeping them makes the
        # checkpoint→restore round-trip lossless.
        speakers = _serialize_speakers(db, include_emotion_profiles=True)
        segments = _serialize_segments(db)

        checkpoint_data = {
            "timestamp": timestamp,
            "database_namespace": _get_database_namespace(db),
            "profile_name": profile_name,
            "description": description,
            "type": "checkpoint",
            "settings": _tunable_settings(settings),
            "speakers": speakers,
            "segments": segments,
        }
        _dump_json(checkpoint_file, checkpoint_data, allow_overwrite=False)
        return {
            "message": f"Checkpoint created for profile '{profile_name}'",
            "filename": os.path.basename(checkpoint_file),
            "timestamp": timestamp,
            "speakers_count": len(speakers),
            "segments_count": len(segments),
        }

    try:
        return await asyncio.to_thread(_work)
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to create checkpoint")


def _scan_checkpoints(profile_name: str) -> list:
    if not os.path.exists(_BACKUPS_DIR):
        return []
    safe_name = sanitize_filename(profile_name)
    prefix = f"checkpoint_{safe_name}_"
    checkpoints = []
    for filename in os.listdir(_BACKUPS_DIR):
        if not filename.startswith(prefix) or not filename.endswith(".json"):
            continue
        summary = _backup_summary(os.path.join(_BACKUPS_DIR, filename))
        if summary is None:
            continue
        checkpoints.append({
            "filename": filename,
            "timestamp": summary["timestamp"],
            "profile_name": profile_name,
            "speakers_count": summary["speakers_count"],
            "segments_count": summary["segments_count"],
            "created_at": summary["created_at"],
        })
    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    return checkpoints


@router.get("/{profile_name}/checkpoints")
async def list_checkpoints(profile_name: str):
    """List all checkpoints for a specific profile."""
    return {"checkpoints": await asyncio.to_thread(_scan_checkpoints, profile_name)}


@router.delete("/{profile_name}/checkpoints/{timestamp}")
async def delete_checkpoint(profile_name: str, timestamp: str):
    """Delete a specific checkpoint."""
    if not _TIMESTAMP_RE.match(timestamp):
        raise HTTPException(status_code=400, detail="Invalid timestamp format")

    def _work() -> dict:
        safe_name = sanitize_filename(profile_name)
        checkpoint_file = _checkpoint_path(safe_name, timestamp)
        if not os.path.exists(checkpoint_file):
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        os.remove(checkpoint_file)
        return {"message": "Checkpoint deleted"}

    return await asyncio.to_thread(_work)


@router.post("/restore")
async def restore_from_file(filename: str, db: Session = Depends(get_db)):
    """Restore speakers/segments from a profile or checkpoint file."""
    base = os.path.basename(filename)
    if not (base.startswith("profile_") or base.startswith("checkpoint_")) or not base.endswith(".json"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    filepath = _safe_backup_path(base)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    def _work() -> dict:
        # Parse and validate BEFORE touching the database: a malformed file
        # must never leave the DB wiped (previous code committed the wipe
        # first and rolled back nothing on failure).
        data = _read_json(filepath)

        speakers_in = data.get("speakers", [])
        names_in = [s.get("name") for s in speakers_in]
        if len(names_in) != len(set(names_in)):
            raise HTTPException(
                status_code=400,
                detail="Profile contains duplicate speaker names; restore aborted before any data was modified",
            )

        local_namespace = _get_database_namespace(db)
        source_namespace = data.get("database_namespace")
        segment_namespace_match = (
            isinstance(source_namespace, str)
            and source_namespace == local_namespace
        )
        segments_in = data.get("segments", [])

        # Validate the settings block before touching the database. Invalid
        # settings used to raise only after the DB commit, returning a
        # misleading 500 even though the destructive restore had succeeded.
        validated_settings = None
        if "settings" in data:
            if not isinstance(data["settings"], dict):
                raise HTTPException(status_code=400, detail="Profile settings must be an object")
            config = get_config()
            candidate = _tunable_settings(config.get_settings())
            candidate.update({
                key: value
                for key, value in data["settings"].items()
                if key in VoiceSettings.model_fields
            })
            try:
                validated_settings = VoiceSettings(**candidate)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=f"Invalid profile settings: {exc}")

        # Everything below is ONE transaction: wipe, rebuild speakers and
        # emotion profiles, and segment remap commit together. Any failure
        # rolls the database back to its pre-restore state.
        # Explicit deletes — never rely on DB-level CASCADE, which only works
        # when PRAGMA foreign_keys is ON for this exact connection.
        db.query(SpeakerEmotionProfile).delete()
        db.query(ConversationSegment).update({"speaker_id": None})
        db.query(Speaker).delete()
        # Bulk deletes bypass ORM identity bookkeeping. Clear stale objects
        # before SQLite reuses low integer IDs for restored speakers.
        db.expunge_all()

        speaker_id_map: dict[int, int] = {}
        for speaker_data in speakers_in:
            old_id = speaker_data["id"]
            speaker = Speaker(name=speaker_data["name"])
            if speaker_data.get("embedding"):
                speaker.set_embedding(np.array(speaker_data["embedding"], dtype=np.float32))
            speaker.emotion_threshold = speaker_data.get("emotion_threshold")
            db.add(speaker)
            db.flush()
            speaker_id_map[old_id] = speaker.id

            for prof_data in speaker_data.get("emotion_profiles", []):
                profile = SpeakerEmotionProfile(
                    speaker_id=speaker.id,
                    emotion_category=prof_data["emotion_category"],
                    sample_count=prof_data.get("sample_count", 1),
                    confidence_threshold=prof_data.get("confidence_threshold"),
                    voice_sample_count=prof_data.get("voice_sample_count", 0),
                    voice_threshold=prof_data.get("voice_threshold"),
                )
                if prof_data.get("embedding"):
                    profile.set_embedding(np.array(prof_data["embedding"], dtype=np.float32))
                if prof_data.get("voice_embedding"):
                    profile.set_voice_embedding(np.array(prof_data["voice_embedding"], dtype=np.float32))
                db.add(profile)

        db.flush()
        new_id_by_name = {s.name: s.id for s in db.query(Speaker).all()}
        new_name_by_id = {speaker_id: name for name, speaker_id in new_id_by_name.items()}

        # Legacy files predate database namespaces entirely; only a file that
        # explicitly names a DIFFERENT database is known-foreign and must never
        # have its row identities trusted at all.
        is_foreign_snapshot = (
            source_namespace is not None and not segment_namespace_match
        )
        is_legacy_snapshot = source_namespace is None

        segments_updated = 0
        segments_unmapped = 0
        segments_remapped_by_name = 0
        segments_remapped_from_local_names = 0
        segments_not_found = 0
        segments_skipped_namespace = len(segments_in) if is_foreign_snapshot else 0
        segments_skipped_identity = 0
        segments_replayed_by_legacy_identity = 0
        legacy_records_seen = 0
        directly_restored_segment_ids: set[int] = set()

        def _apply_backup_segment_state(segment, seg_data) -> None:
            """Replay one backup record onto a positively identified live row."""
            nonlocal segments_updated, segments_unmapped, segments_remapped_by_name
            old_speaker_id = seg_data.get("speaker_id")
            new_speaker_id = None
            if old_speaker_id and old_speaker_id in speaker_id_map:
                new_speaker_id = speaker_id_map[old_speaker_id]
            elif old_speaker_id and seg_data.get("speaker_name"):
                # Never interpret a backup ID in the target database's
                # unrelated ID namespace. The backup's denormalised name is
                # the only stable fallback available.
                new_speaker_id = new_id_by_name.get(seg_data["speaker_name"])
                if new_speaker_id is not None:
                    segments_remapped_by_name += 1

            if new_speaker_id is not None:
                segment.speaker_id = new_speaker_id
                # Keep denormalised name consistent with the restored FK,
                # even if a malformed backup segment carries a stale name.
                segment.speaker_name = new_name_by_id[new_speaker_id]
            else:
                segment.speaker_id = None
                segment.speaker_name = seg_data.get("speaker_name")
                if old_speaker_id:
                    segments_unmapped += 1
            segment.is_misidentified = seg_data.get("is_misidentified", False)
            segments_updated += 1

        def _int_field(value) -> Optional[int]:
            # bool is an int subclass; a JSON `true` must not pass as an id.
            return value if isinstance(value, int) and not isinstance(value, bool) else None

        def _offset_field(value) -> Optional[float]:
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
                return float(value)
            return None

        def _verified_legacy_replay(seg_data) -> None:
            """Replay a record that has no snapshot UUIDs — acceptable only
            under verification: the live row found by integer id must also
            match the record's conversation FK and exact offsets, which a
            reused SQLite id practically never satisfies. NaN/inf offsets are
            rejected up front (every NaN comparison is False, which would
            otherwise wave the record through the epsilon guards)."""
            nonlocal legacy_records_seen, segments_skipped_identity, \
                segments_not_found, segments_replayed_by_legacy_identity
            legacy_records_seen += 1
            seg_id = _int_field(seg_data.get("id"))
            conv_id = _int_field(seg_data.get("conversation_id"))
            start = _offset_field(seg_data.get("start_offset"))
            end = _offset_field(seg_data.get("end_offset"))
            if seg_id is None or conv_id is None or start is None or end is None:
                segments_skipped_identity += 1
                return
            segment = db.get(ConversationSegment, seg_id)
            if segment is not None and segment.id in directly_restored_segment_ids:
                # Duplicate record for a row already replayed; first one wins.
                segments_skipped_identity += 1
                return
            if (
                segment is None
                or segment.conversation_id != conv_id
                or segment.start_offset is None
                or segment.end_offset is None
                or abs(segment.start_offset - start) > 1e-3
                or abs(segment.end_offset - end) > 1e-3
            ):
                segments_not_found += 1
                return
            directly_restored_segment_ids.add(segment.id)
            segments_replayed_by_legacy_identity += 1
            _apply_backup_segment_state(segment, seg_data)

        # Even in the same database, SQLite integer IDs may be reused after a
        # row is deleted. Replay historical state only through persistent row
        # UUIDs. Foreign snapshots remain portable for speakers and settings,
        # but their integer IDs are never used to identify live rows.
        for seg_data in segments_in if segment_namespace_match else []:
            segment_snapshot_uuid = seg_data.get("snapshot_uuid")
            conversation_snapshot_uuid = seg_data.get("conversation_snapshot_uuid")
            if not (
                isinstance(segment_snapshot_uuid, str)
                and segment_snapshot_uuid
                and isinstance(conversation_snapshot_uuid, str)
                and conversation_snapshot_uuid
            ):
                # Mixed-era record: the file names THIS database but the row
                # was serialized before snapshot UUIDs existed. Same-DB is
                # verified, so verified integer identity is safe here too.
                _verified_legacy_replay(seg_data)
                continue
            segment = (
                db.query(ConversationSegment)
                .join(
                    Conversation,
                    Conversation.id == ConversationSegment.conversation_id,
                )
                .filter(
                    ConversationSegment.snapshot_uuid == segment_snapshot_uuid,
                    Conversation.snapshot_uuid == conversation_snapshot_uuid,
                )
                .first()
            )
            if not segment:
                segments_not_found += 1
                continue
            directly_restored_segment_ids.add(segment.id)
            _apply_backup_segment_state(segment, seg_data)

        # Legacy files (written before database namespaces existed) would
        # otherwise lose ALL segment replay — silently reducing a checkpoint
        # restore to a no-op for exactly the assignments the operator wants
        # reverted. A truly foreign legacy file fails per-row verification and
        # degrades to the name-based reconnect below.
        for seg_data in segments_in if is_legacy_snapshot else []:
            _verified_legacy_replay(seg_data)

        # The target rows' own denormalised names are local evidence. Reconnect
        # every row not authoritatively replayed above; this safely covers
        # foreign, legacy and stale same-database snapshots without consulting
        # reusable integer IDs.
        for segment in db.query(ConversationSegment).all():
            if segment.id in directly_restored_segment_ids:
                continue
            new_speaker_id = new_id_by_name.get(segment.speaker_name)
            if new_speaker_id is None:
                continue
            segment.speaker_id = new_speaker_id
            segments_remapped_by_name += 1
            segments_remapped_from_local_names += 1
            segments_updated += 1

        db.commit()

        # Active streams keep an in-memory speaker cache keyed by the OLD
        # speaker IDs; after a restore those IDs are gone. Drop the cache so
        # the next match reloads from the restored database (OPUS-015/QWEN-006).
        try:
            from .api import get_engine
            get_engine().clear_speaker_cache()
        except Exception:  # noqa: BLE001 - model/cache availability must not undo a committed restore
            logger.info("Speaker cache invalidation skipped after restore")

        # A JSON settings file and SQLite cannot share one transaction. The DB
        # restore is already committed, so a later settings I/O failure must be
        # reported as explicit partial success rather than a false 500.
        settings_restored = True
        settings_warning = None
        if validated_settings is not None:
            try:
                get_config().update_settings(validated_settings.model_dump())
                get_config().reload_settings()
            except Exception as exc:
                settings_restored = False
                settings_warning = f"Database restored, but settings could not be persisted: {exc}"
                logger.exception("Profile database restored but settings persistence failed")

        legacy_warning = None
        if legacy_records_seen:
            legacy_warning = (
                f"{legacy_records_seen} segment record(s) predate snapshot "
                "UUIDs; replay used verified integer identity (id + "
                f"conversation + offsets). {segments_replayed_by_legacy_identity} "
                f"of {legacy_records_seen} passed verification."
            )

        profile_name = data.get("name") or data.get("profile_name", "Unknown")
        return {
            "message": f"Restored profile '{profile_name}'",
            "speakers_restored": len(speaker_id_map),
            "segments_updated": segments_updated,
            "segments_unmapped": segments_unmapped,
            "segments_remapped_by_name": segments_remapped_by_name,
            "segments_remapped_from_local_names": segments_remapped_from_local_names,
            "segments_not_found": segments_not_found,
            "segments_skipped_namespace": segments_skipped_namespace,
            "segments_skipped_identity": segments_skipped_identity,
            "segments_replayed_by_identity": len(directly_restored_segment_ids),
            "segments_replayed_by_legacy_identity": segments_replayed_by_legacy_identity,
            "segment_namespace_match": segment_namespace_match,
            "legacy_restore_warning": legacy_warning,
            "settings_restored": settings_restored,
            "settings_warning": settings_warning,
        }

    try:
        return await asyncio.to_thread(_work)
    except HTTPException:
        raise
    except Exception:
        db.rollback()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Restore failed")


@router.get("/download/{profile_name}")
async def download_profile(profile_name: str):
    """Download a single profile."""
    safe_name = sanitize_filename(profile_name)
    profile_file = _profile_path(safe_name)
    if not os.path.exists(profile_file):
        raise HTTPException(status_code=404, detail="Profile not found")
    return FileResponse(
        path=profile_file,
        media_type="application/json",
        filename=f"{profile_name}.json",
    )


def _zip_all_profiles() -> io.BytesIO:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename in os.listdir(_BACKUPS_DIR):
            if filename.startswith("profile_") and filename.endswith(".json"):
                zip_file.write(os.path.join(_BACKUPS_DIR, filename), filename)
    buffer.seek(0)
    return buffer


@router.get("/download-all")
async def download_all_profiles():
    """Download all profiles as a ZIP file."""
    if not os.path.exists(_BACKUPS_DIR):
        raise HTTPException(status_code=404, detail="No profiles found")
    zip_buffer = await asyncio.to_thread(_zip_all_profiles)
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=voice_profiles.zip"},
    )


@router.post("/import")
async def import_profile(file: UploadFile = File(...)):
    """Import a profile from uploaded JSON file."""
    if not file.filename or not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    try:
        contents = await file.read()
        data = json.loads(contents)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")

    if "speakers" not in data or "name" not in data:
        raise HTTPException(status_code=400, detail="Invalid profile file format")

    def _work() -> dict:
        profile_name = data.get("name", "Imported")
        safe_name = sanitize_filename(profile_name)
        profile_file = _profile_path(safe_name)
        os.makedirs(_BACKUPS_DIR, exist_ok=True)
        _dump_json(profile_file, data, allow_overwrite=False)
        return {
            "message": f"Profile '{profile_name}' imported successfully",
            "name": profile_name,
            "filename": os.path.basename(profile_file),
            "speakers_count": len(data.get("speakers", [])),
            "segments_count": len(data.get("segments", [])),
        }

    try:
        return await asyncio.to_thread(_work)
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Import failed")
