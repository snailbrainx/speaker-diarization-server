"""
API endpoints for Voice Profiles and Checkpoints
"""
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import json
import os
import numpy as np
import zipfile
import io
from datetime import datetime

from .database import get_db
from .models import Speaker, ConversationSegment, SpeakerEmotionProfile

router = APIRouter(prefix="/profiles", tags=["Voice Profiles"])


class CreateProfileRequest(BaseModel):
    name: str
    description: Optional[str] = None


class UpdateProfileRequest(BaseModel):
    description: Optional[str] = None


def sanitize_filename(name: str) -> str:
    """Sanitize profile name for use as filename"""
    return "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).strip().replace(' ', '_')


def save_current_state(profile_name: str, description: str, db: Session):
    """Save current speaker/segment state to profile file"""
    from .config import get_config

    safe_name = sanitize_filename(profile_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs("backups", exist_ok=True)
    profile_file = f"backups/profile_{safe_name}.json"

    # Get current settings
    config = get_config()  # Use global singleton
    settings = config.get_settings()

    # Collect all data
    profile_data = {
        "timestamp": timestamp,
        "name": profile_name,
        "description": description,
        "type": "profile",
        "settings": {
            "speaker_threshold": settings.speaker_threshold,
            "context_padding": settings.context_padding,
            "silence_duration": settings.silence_duration,
            "filter_hallucinations": settings.filter_hallucinations,
            "emotion_threshold": settings.emotion_threshold,
        },
        "speakers": [],
        "segments": []
    }

    # Export all speakers with embeddings and emotion profiles
    speakers = db.query(Speaker).all()
    for speaker in speakers:
        embedding = speaker.get_embedding()

        # Export emotion profiles for this speaker
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
                "voice_threshold": prof.voice_threshold
            })

        profile_data["speakers"].append({
            "id": speaker.id,
            "name": speaker.name,
            "embedding": embedding.tolist() if embedding is not None else None,
            "emotion_threshold": speaker.emotion_threshold,
            "emotion_profiles": emotion_profiles
        })

    # Export all segment assignments
    segments = db.query(ConversationSegment).all()
    for seg in segments:
        profile_data["segments"].append({
            "id": seg.id,
            "conversation_id": seg.conversation_id,
            "speaker_id": seg.speaker_id,
            "speaker_name": seg.speaker_name,
            "is_misidentified": seg.is_misidentified,
            "start_offset": seg.start_offset,
            "end_offset": seg.end_offset
        })

    # Write to file
    with open(profile_file, 'w') as f:
        json.dump(profile_data, f, indent=2)

    return {
        "filename": os.path.basename(profile_file),
        "speakers_count": len(speakers),
        "segments_count": len(segments),
        "timestamp": timestamp
    }


@router.post("")
@router.post("/")
async def create_profile(
    request: CreateProfileRequest,
    db: Session = Depends(get_db)
):
    """Create a new EMPTY voice profile with default settings"""
    from .config import get_config, VoiceSettings

    try:
        safe_name = sanitize_filename(request.name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs("backups", exist_ok=True)
        profile_file = f"backups/profile_{safe_name}.json"

        # Get default settings (not current settings!)
        default_settings = VoiceSettings()

        # Create empty profile with defaults
        profile_data = {
            "timestamp": timestamp,
            "name": request.name,
            "description": request.description or "",
            "type": "profile",
            "settings": {
                "speaker_threshold": default_settings.speaker_threshold,
                "context_padding": default_settings.context_padding,
                "silence_duration": default_settings.silence_duration,
                "filter_hallucinations": default_settings.filter_hallucinations,
            },
            "speakers": [],  # Empty!
            "segments": []   # Empty!
        }

        # Write to file
        with open(profile_file, 'w') as f:
            json.dump(profile_data, f, indent=2)

        return {
            "message": f"Empty profile '{request.name}' created successfully",
            "name": request.name,
            "description": request.description or "",
            "filename": os.path.basename(profile_file),
            "speakers_count": 0,
            "segments_count": 0,
            "timestamp": timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create profile: {str(e)}")


@router.post("/duplicate")
async def duplicate_profile(
    request: CreateProfileRequest,
    db: Session = Depends(get_db)
):
    """Duplicate current state into a new profile (speakers + segments + settings)"""
    try:
        result = save_current_state(request.name, request.description or "", db)
        return {
            "message": f"Profile '{request.name}' duplicated successfully",
            "name": request.name,
            "description": request.description or "",
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to duplicate profile: {str(e)}")


@router.patch("/{profile_name}")
async def update_profile(
    profile_name: str,
    request: UpdateProfileRequest,
    db: Session = Depends(get_db)
):
    """Update existing profile with current state"""
    try:
        # Read existing profile to get description if not provided
        safe_name = sanitize_filename(profile_name)
        profile_file = f"backups/profile_{safe_name}.json"

        description = request.description
        if description is None and os.path.exists(profile_file):
            with open(profile_file, 'r') as f:
                data = json.load(f)
                description = data.get("description", "")

        result = save_current_state(profile_name, description or "", db)
        return {
            "message": f"Profile '{profile_name}' updated successfully",
            "name": profile_name,
            "description": description,
            **result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update profile: {str(e)}")


@router.get("")
@router.get("/")
async def list_profiles():
    """List all voice profiles"""
    if not os.path.exists("backups"):
        return {"profiles": []}

    profiles = []
    for filename in os.listdir("backups"):
        if not filename.startswith("profile_") or not filename.endswith(".json"):
            continue

        filepath = os.path.join("backups", filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                profiles.append({
                    "name": data.get("name", filename),
                    "description": data.get("description", ""),
                    "filename": filename,
                    "timestamp": data.get("timestamp", ""),
                    "speakers_count": len(data.get("speakers", [])),
                    "segments_count": len(data.get("segments", [])),
                    "created_at": datetime.fromtimestamp(os.stat(filepath).st_ctime).isoformat()
                })
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    profiles.sort(key=lambda x: x["name"])
    return {"profiles": profiles}


@router.delete("/{profile_name}")
async def delete_profile(profile_name: str):
    """Delete a profile and all its checkpoints"""
    safe_name = sanitize_filename(profile_name)
    profile_file = f"backups/profile_{safe_name}.json"

    if not os.path.exists(profile_file):
        raise HTTPException(status_code=404, detail=f"Profile '{profile_name}' not found")

    # Delete profile
    os.remove(profile_file)

    # Delete all checkpoints for this profile
    deleted_checkpoints = 0
    if os.path.exists("backups"):
        for filename in os.listdir("backups"):
            if filename.startswith(f"checkpoint_{safe_name}_") and filename.endswith(".json"):
                os.remove(os.path.join("backups", filename))
                deleted_checkpoints += 1

    return {
        "message": f"Profile '{profile_name}' and {deleted_checkpoints} checkpoints deleted",
        "deleted_checkpoints": deleted_checkpoints
    }


@router.post("/{profile_name}/checkpoints")
async def create_checkpoint(profile_name: str, db: Session = Depends(get_db)):
    """Create a checkpoint (snapshot) of current profile state"""
    from .config import get_config

    try:
        safe_name = sanitize_filename(profile_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        os.makedirs("backups", exist_ok=True)
        checkpoint_file = f"backups/checkpoint_{safe_name}_{timestamp}.json"

        # Read profile description if exists
        profile_file = f"backups/profile_{safe_name}.json"
        description = ""
        if os.path.exists(profile_file):
            with open(profile_file, 'r') as f:
                data = json.load(f)
                description = data.get("description", "")

        # Get current settings
        config = get_config()  # Use global singleton
        settings = config.get_settings()

        # Collect all data
        checkpoint_data = {
            "timestamp": timestamp,
            "profile_name": profile_name,
            "description": description,
            "type": "checkpoint",
            "settings": {
                "speaker_threshold": settings.speaker_threshold,
                "context_padding": settings.context_padding,
                "silence_duration": settings.silence_duration,
                "filter_hallucinations": settings.filter_hallucinations,
            },
            "speakers": [],
            "segments": []
        }

        # Export all speakers with embeddings
        speakers = db.query(Speaker).all()
        for speaker in speakers:
            embedding = speaker.get_embedding()
            checkpoint_data["speakers"].append({
                "id": speaker.id,
                "name": speaker.name,
                "embedding": embedding.tolist() if embedding is not None else None
            })

        # Export all segment assignments
        segments = db.query(ConversationSegment).all()
        for seg in segments:
            checkpoint_data["segments"].append({
                "id": seg.id,
                "conversation_id": seg.conversation_id,
                "speaker_id": seg.speaker_id,
                "speaker_name": seg.speaker_name,
                "is_misidentified": seg.is_misidentified,
                "start_offset": seg.start_offset,
                "end_offset": seg.end_offset
            })

        # Write to file
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        return {
            "message": f"Checkpoint created for profile '{profile_name}'",
            "filename": os.path.basename(checkpoint_file),
            "timestamp": timestamp,
            "speakers_count": len(speakers),
            "segments_count": len(segments)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create checkpoint: {str(e)}")


@router.get("/{profile_name}/checkpoints")
async def list_checkpoints(profile_name: str):
    """List all checkpoints for a specific profile"""
    if not os.path.exists("backups"):
        return {"checkpoints": []}

    safe_name = sanitize_filename(profile_name)
    checkpoints = []

    for filename in os.listdir("backups"):
        if not filename.startswith(f"checkpoint_{safe_name}_") or not filename.endswith(".json"):
            continue

        filepath = os.path.join("backups", filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                checkpoints.append({
                    "filename": filename,
                    "timestamp": data.get("timestamp", ""),
                    "profile_name": profile_name,
                    "speakers_count": len(data.get("speakers", [])),
                    "segments_count": len(data.get("segments", [])),
                    "created_at": datetime.fromtimestamp(os.stat(filepath).st_ctime).isoformat()
                })
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"checkpoints": checkpoints}


@router.delete("/{profile_name}/checkpoints/{timestamp}")
async def delete_checkpoint(profile_name: str, timestamp: str):
    """Delete a specific checkpoint"""
    safe_name = sanitize_filename(profile_name)
    checkpoint_file = f"backups/checkpoint_{safe_name}_{timestamp}.json"

    if not os.path.exists(checkpoint_file):
        raise HTTPException(status_code=404, detail="Checkpoint not found")

    os.remove(checkpoint_file)
    return {"message": f"Checkpoint deleted"}


@router.post("/restore")
async def restore_from_file(filename: str, db: Session = Depends(get_db)):
    """Restore speakers/segments from a profile or checkpoint file"""
    from .config import get_config

    filepath = os.path.join("backups", filename)

    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Restore settings if present (with defaults for missing fields)
        if "settings" in data:
            config = get_config()  # Use global singleton

            # Ensure all settings have defaults for backwards compatibility
            settings_to_restore = {
                "speaker_threshold": 0.30,
                "context_padding": 0.15,
                "silence_duration": 0.5,
                "filter_hallucinations": True,
                "emotion_threshold": 0.6,
            }
            # Override with values from profile
            settings_to_restore.update(data["settings"])

            config.update_settings(settings_to_restore)
            config.reload_settings()  # Force reload from disk

        # Delete all existing speakers first (for clean slate)
        # IMPORTANT: With foreign keys enabled, we must NULL out segment references first
        from .models import ConversationSegment
        db.query(ConversationSegment).update({"speaker_id": None})
        db.query(Speaker).delete()
        db.commit()

        # Restore speakers and embeddings from profile
        speaker_id_map = {}
        speakers_restored = 0

        for speaker_data in data["speakers"]:
            old_id = speaker_data["id"]

            # Create speaker (all were deleted above)
            speaker = Speaker(name=speaker_data["name"])
            if speaker_data.get("embedding"):
                embedding = np.array(speaker_data["embedding"], dtype=np.float32)
                speaker.set_embedding(embedding)

            # Restore emotion threshold if present
            speaker.emotion_threshold = speaker_data.get("emotion_threshold")

            db.add(speaker)
            db.flush()
            speaker_id_map[old_id] = speaker.id

            # Restore emotion profiles for this speaker
            for prof_data in speaker_data.get("emotion_profiles", []):
                profile = SpeakerEmotionProfile(
                    speaker_id=speaker.id,
                    emotion_category=prof_data["emotion_category"],
                    sample_count=prof_data.get("sample_count", 1),
                    confidence_threshold=prof_data.get("confidence_threshold"),
                    voice_sample_count=prof_data.get("voice_sample_count", 0),
                    voice_threshold=prof_data.get("voice_threshold")
                )
                if prof_data.get("embedding"):
                    emb = np.array(prof_data["embedding"], dtype=np.float32)
                    profile.set_embedding(emb)
                if prof_data.get("voice_embedding"):
                    voice_emb = np.array(prof_data["voice_embedding"], dtype=np.float32)
                    profile.set_voice_embedding(voice_emb)
                db.add(profile)

            speakers_restored += 1

        db.commit()

        # Restore segment assignments
        segments_updated = 0
        for seg_data in data["segments"]:
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
            "speakers_restored": speakers_restored,
            "segments_updated": segments_updated
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Restore failed: {str(e)}")


@router.get("/download/{profile_name}")
async def download_profile(profile_name: str):
    """Download a single profile"""
    safe_name = sanitize_filename(profile_name)
    profile_file = f"backups/profile_{safe_name}.json"

    if not os.path.exists(profile_file):
        raise HTTPException(status_code=404, detail="Profile not found")

    return FileResponse(
        path=profile_file,
        media_type="application/json",
        filename=f"{profile_name}.json"
    )


@router.get("/download-all")
async def download_all_profiles():
    """Download all profiles as a ZIP file"""
    if not os.path.exists("backups"):
        raise HTTPException(status_code=404, detail="No profiles found")

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename in os.listdir("backups"):
            if filename.startswith("profile_") and filename.endswith(".json"):
                filepath = os.path.join("backups", filename)
                zip_file.write(filepath, filename)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=voice_profiles.zip"}
    )


@router.post("/import")
async def import_profile(file: UploadFile = File(...)):
    """Import a profile from uploaded JSON file"""
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="Only JSON files are supported")

    try:
        contents = await file.read()
        data = json.loads(contents)

        # Validate it's a profile file
        if "speakers" not in data or "name" not in data:
            raise HTTPException(status_code=400, detail="Invalid profile file format")

        # Save to backups directory
        profile_name = data.get("name", "Imported")
        safe_name = sanitize_filename(profile_name)
        profile_file = f"backups/profile_{safe_name}.json"

        os.makedirs("backups", exist_ok=True)
        with open(profile_file, 'w') as f:
            json.dump(data, f, indent=2)

        return {
            "message": f"Profile '{profile_name}' imported successfully",
            "name": profile_name,
            "filename": os.path.basename(profile_file),
            "speakers_count": len(data.get("speakers", [])),
            "segments_count": len(data.get("segments", []))
        }

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {str(e)}")
