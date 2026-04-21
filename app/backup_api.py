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
import os
import re
import traceback
import zipfile
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import get_db
from .models import Speaker, ConversationSegment, SpeakerEmotionProfile
from .config import VoiceSettings, get_config

router = APIRouter(prefix="/profiles", tags=["Voice Profiles"])

_BACKUPS_DIR = "backups"
_TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


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


def _tunable_settings(source) -> Dict[str, Any]:
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
            "speaker_id": seg.speaker_id,
            "speaker_name": seg.speaker_name,
            "is_misidentified": seg.is_misidentified,
            "start_offset": seg.start_offset,
            "end_offset": seg.end_offset,
        }
        for seg in db.query(ConversationSegment).all()
    ]


def _dump_json(path: str, payload: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def _read_json(path: str) -> dict:
    with open(path, 'r') as f:
        return json.load(f)


def save_current_state(profile_name: str, description: str, db: Session) -> dict:
    """Save current speaker/segment state to profile file. Blocking — call via to_thread."""
    safe_name = sanitize_filename(profile_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(_BACKUPS_DIR, exist_ok=True)
    profile_file = _profile_path(safe_name)

    settings = get_config().get_settings()
    speakers = _serialize_speakers(db, include_emotion_profiles=True)
    segments = _serialize_segments(db)

    profile_data = {
        "timestamp": timestamp,
        "name": profile_name,
        "description": description,
        "type": "profile",
        "settings": _tunable_settings(settings),
        "speakers": speakers,
        "segments": segments,
    }
    _dump_json(profile_file, profile_data)

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
        defaults = VoiceSettings()
        profile_data = {
            "timestamp": timestamp,
            "name": request.name,
            "description": request.description or "",
            "type": "profile",
            "settings": _tunable_settings(defaults),
            "speakers": [],
            "segments": [],
        }
        _dump_json(profile_file, profile_data)
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
        result = await asyncio.to_thread(save_current_state, request.name, request.description or "", db)
        return {
            "message": f"Profile '{request.name}' duplicated successfully",
            "name": request.name,
            "description": request.description or "",
            **result,
        }
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
        result = await asyncio.to_thread(save_current_state, profile_name, description, db)
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
        filepath = os.path.join(_BACKUPS_DIR, filename)
        try:
            data = _read_json(filepath)
        except (json.JSONDecodeError, OSError, KeyError):
            continue
        profiles.append({
            "name": data.get("name", filename),
            "description": data.get("description", ""),
            "filename": filename,
            "timestamp": data.get("timestamp", ""),
            "speakers_count": len(data.get("speakers", [])),
            "segments_count": len(data.get("segments", [])),
            "created_at": datetime.fromtimestamp(os.stat(filepath).st_ctime).isoformat(),
        })
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
        # Checkpoints don't save emotion profiles (lightweight by design) but do
        # store every tunable setting so they can be cleanly reverted.
        speakers = _serialize_speakers(db, include_emotion_profiles=False)
        segments = _serialize_segments(db)

        checkpoint_data = {
            "timestamp": timestamp,
            "profile_name": profile_name,
            "description": description,
            "type": "checkpoint",
            "settings": _tunable_settings(settings),
            "speakers": speakers,
            "segments": segments,
        }
        _dump_json(checkpoint_file, checkpoint_data)
        return {
            "message": f"Checkpoint created for profile '{profile_name}'",
            "filename": os.path.basename(checkpoint_file),
            "timestamp": timestamp,
            "speakers_count": len(speakers),
            "segments_count": len(segments),
        }

    try:
        return await asyncio.to_thread(_work)
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
        filepath = os.path.join(_BACKUPS_DIR, filename)
        try:
            data = _read_json(filepath)
        except (json.JSONDecodeError, OSError, KeyError):
            continue
        checkpoints.append({
            "filename": filename,
            "timestamp": data.get("timestamp", ""),
            "profile_name": profile_name,
            "speakers_count": len(data.get("speakers", [])),
            "segments_count": len(data.get("segments", [])),
            "created_at": datetime.fromtimestamp(os.stat(filepath).st_ctime).isoformat(),
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
        data = _read_json(filepath)

        # Restore any settings fields the file provides, leaving others at their
        # current values. This is safer than back-filling every missing setting
        # with a hard-coded default — checkpoints that predate a new setting
        # won't clobber the user's current value.
        if "settings" in data:
            config = get_config()
            current = _tunable_settings(config.get_settings())
            current.update({
                k: v for k, v in data["settings"].items()
                if k in VoiceSettings.model_fields
            })
            config.update_settings(current)
            config.reload_settings()

        db.query(ConversationSegment).update({"speaker_id": None})
        db.query(Speaker).delete()
        db.commit()

        speaker_id_map: Dict[int, int] = {}
        for speaker_data in data.get("speakers", []):
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

        db.commit()

        segments_updated = 0
        for seg_data in data.get("segments", []):
            segment = db.query(ConversationSegment).filter(
                ConversationSegment.id == seg_data["id"]
            ).first()
            if segment:
                old_speaker_id = seg_data.get("speaker_id")
                if old_speaker_id and old_speaker_id in speaker_id_map:
                    segment.speaker_id = speaker_id_map[old_speaker_id]
                segment.speaker_name = seg_data.get("speaker_name")
                segment.is_misidentified = seg_data.get("is_misidentified", False)
                segments_updated += 1
        db.commit()

        profile_name = data.get("name") or data.get("profile_name", "Unknown")
        return {
            "message": f"Restored profile '{profile_name}'",
            "speakers_restored": len(speaker_id_map),
            "segments_updated": segments_updated,
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
        _dump_json(profile_file, data)
        return {
            "message": f"Profile '{profile_name}' imported successfully",
            "name": profile_name,
            "filename": os.path.basename(profile_file),
            "speakers_count": len(data.get("speakers", [])),
            "segments_count": len(data.get("segments", [])),
        }

    try:
        return await asyncio.to_thread(_work)
    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Import failed")
