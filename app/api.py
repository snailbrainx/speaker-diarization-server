from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
import asyncio
from datetime import datetime, timedelta
import json
import torch
from pydub import AudioSegment

from .database import get_db
from .models import Speaker, Conversation, ConversationSegment
from .schemas import (
    SpeakerCreate, SpeakerResponse, SpeakerRename,
    StatusResponse, ConversationResponse
)
from .diarization import SpeakerRecognitionEngine
from .config import get_config

router = APIRouter()

# Initialize speaker recognition engine (singleton)
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = SpeakerRecognitionEngine()
    return engine


@router.get("/status", response_model=StatusResponse)
async def get_status():
    """Get system status"""
    return StatusResponse(
        status="online",
        message="Speaker diarization service is running",
        gpu_available=torch.cuda.is_available(),
        device=str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    )


@router.get("/speakers", response_model=List[SpeakerResponse])
async def list_speakers(db: Session = Depends(get_db)):
    """List all enrolled speakers with segment counts"""
    from sqlalchemy import func

    # Query speakers with segment counts
    speakers_with_counts = db.query(
        Speaker,
        func.count(ConversationSegment.id).label('segment_count')
    ).outerjoin(
        ConversationSegment, Speaker.id == ConversationSegment.speaker_id
    ).group_by(Speaker.id).all()

    # Convert to response format
    result = []
    for speaker, segment_count in speakers_with_counts:
        speaker_dict = {
            "id": speaker.id,
            "name": speaker.name,
            "created_at": speaker.created_at,
            "updated_at": speaker.updated_at,
            "segment_count": segment_count
        }
        result.append(SpeakerResponse(**speaker_dict))

    return result


@router.post("/speakers/enroll", response_model=SpeakerResponse)
async def enroll_speaker(
    name: str = Form(...),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Enroll a new speaker with audio sample

    Args:
        name: Speaker name
        audio_file: Audio file containing speaker's voice (10-30 seconds recommended)
    """
    # Check if speaker already exists
    existing = db.query(Speaker).filter(Speaker.name == name).first()
    if existing:
        raise HTTPException(status_code=400, detail=f"Speaker '{name}' already exists")

    # Save audio file temporarily
    data_path = os.getenv("DATA_PATH", "/app/data")
    os.makedirs(f"{data_path}/temp", exist_ok=True)
    temp_path = f"{data_path}/temp/{audio_file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        # Convert MP3 to WAV if needed (speaker enrollment may receive MP3 uploads)
        wav_temp_path = None
        if temp_path.lower().endswith('.mp3'):
            try:
                print(f"Converting uploaded MP3 to WAV for speaker enrollment...")
                audio = AudioSegment.from_file(temp_path)
                wav_temp_path = temp_path.rsplit('.', 1)[0] + '_enrolled.wav'
                audio.export(wav_temp_path, format='wav')
                # Use WAV for embedding extraction
                extraction_path = wav_temp_path
                print(f"Conversion successful: {wav_temp_path}")
            except Exception as e:
                print(f"Warning: Failed to convert MP3 to WAV: {e}")
                extraction_path = temp_path  # Fall back to MP3
        else:
            extraction_path = temp_path

        # Extract embedding (run in thread to avoid blocking event loop)
        embedding = await asyncio.to_thread(engine.extract_embedding, extraction_path)

        # Create speaker in database
        speaker = Speaker(name=name)
        speaker.set_embedding(embedding)

        db.add(speaker)
        db.commit()
        db.refresh(speaker)

        # Clear GPU cache after embedding extraction
        engine.clear_gpu_cache()

        return speaker

    finally:
        # Clean up temp files
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if wav_temp_path and os.path.exists(wav_temp_path):
            os.remove(wav_temp_path)


@router.patch("/speakers/{speaker_id}/rename", response_model=SpeakerResponse)
async def rename_speaker(
    speaker_id: int,
    rename_data: SpeakerRename,
    db: Session = Depends(get_db)
):
    """
    Rename a speaker (useful for AI agents to label unknown speakers)

    Args:
        speaker_id: Speaker ID
        rename_data: New name for the speaker
    """
    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    # Check if new name already exists
    existing = db.query(Speaker).filter(Speaker.name == rename_data.new_name).first()
    if existing and existing.id != speaker_id:
        raise HTTPException(
            status_code=400,
            detail=f"Speaker '{rename_data.new_name}' already exists"
        )

    old_name = speaker.name
    speaker.name = rename_data.new_name
    speaker.updated_at = datetime.utcnow()

    # UPDATE ALL PAST CONVERSATION SEGMENTS
    from .models import ConversationSegment
    updated_segments = db.query(ConversationSegment).filter(
        ConversationSegment.speaker_id == speaker_id
    ).update({"speaker_name": rename_data.new_name})

    db.commit()
    db.refresh(speaker)

    print(f"✓ Renamed speaker '{old_name}' → '{rename_data.new_name}' (updated {updated_segments} past segments)")

    return speaker


@router.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: int, db: Session = Depends(get_db)):
    """Delete a speaker"""
    from .models import SpeakerEmotionProfile, ConversationSegment

    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    # 1. Set speaker_id to NULL in segments (SQLite FK constraint is NO ACTION, not SET NULL)
    db.query(ConversationSegment).filter(
        ConversationSegment.speaker_id == speaker_id
    ).update({"speaker_id": None}, synchronize_session=False)

    # 2. Delete emotion profiles
    db.query(SpeakerEmotionProfile).filter(
        SpeakerEmotionProfile.speaker_id == speaker_id
    ).delete(synchronize_session=False)

    # 3. Delete speaker
    db.delete(speaker)
    db.commit()

    return {"message": f"Speaker '{speaker.name}' deleted successfully"}


@router.delete("/speakers/unknown/all")
async def delete_all_unknown_speakers(db: Session = Depends(get_db)):
    """Delete all speakers with names starting with 'Unknown_'"""
    from .services import delete_unknown_speakers

    deleted_count, _ = delete_unknown_speakers(db)
    db.commit()

    return {
        "message": f"Deleted {deleted_count} unknown speakers",
        "deleted_count": deleted_count
    }


@router.post("/process", response_model=ConversationResponse)
async def process_audio(
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Process uploaded audio file with speaker diarization and recognition.
    Creates a new Conversation with segments.
    """
    # Save audio file
    data_path = os.getenv("DATA_PATH", "/app/data")
    os.makedirs(f"{data_path}/recordings", exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    # Save uploaded file with timestamp
    base_filename = audio_file.filename or "upload"
    temp_filename = f"uploaded_{timestamp}_{base_filename}"
    temp_path = f"{data_path}/recordings/{temp_filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    # Convert to WAV for reliable processing
    if not temp_path.endswith('.wav'):
        try:
            audio = AudioSegment.from_file(temp_path)
            wav_filename = temp_filename.rsplit('.', 1)[0] + '.wav'
            file_path = f"{data_path}/recordings/{wav_filename}"
            audio.export(file_path, format='wav')
            # Remove original after conversion
            os.remove(temp_path)
        except Exception as e:
            # If conversion fails, use original file
            print(f"Warning: Failed to convert to WAV: {e}")
            file_path = temp_path
    else:
        file_path = temp_path

    # Create conversation entry
    start_time = datetime.utcnow()
    conversation = Conversation(
        title=f"Uploaded: {audio_file.filename}",
        audio_path=file_path,
        start_time=start_time,
        status="processing"
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    try:
        # Get known speakers
        speakers = db.query(Speaker).all()
        known_speakers = [(s.id, s.name, s.get_embedding()) for s in speakers]

        # Get threshold from config
        config = get_config()
        settings = config.get_settings()
        threshold = settings.speaker_threshold

        # Process audio with transcription (run in thread to avoid blocking event loop)
        result = await asyncio.to_thread(
            engine.transcribe_with_diarization,
            file_path,
            known_speakers,
            threshold=threshold,
            db_session=db
        )

        # Create conversation segments
        from .services import create_segment_from_result
        for seg in result["segments"]:
            create_segment_from_result(
                seg=seg,
                conversation_id=conversation.id,
                conv_start=start_time,
                db=db,
                threshold=threshold,
            )

        # Update conversation metadata
        conversation.status = "completed"
        conversation.num_segments = len(result["segments"])
        conversation.num_speakers = result["num_speakers"]
        if result["segments"]:
            conversation.duration = max(s["end"] for s in result["segments"])
            conversation.end_time = start_time + timedelta(seconds=conversation.duration)

        db.commit()
        db.refresh(conversation)

        # Clear GPU cache after processing
        engine.clear_gpu_cache()

        # Return conversation
        return conversation

    except Exception as e:
        conversation.status = "failed"
        db.commit()
        raise HTTPException(status_code=500, detail=str(e))


