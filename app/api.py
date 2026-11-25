from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy import func
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
import tempfile
import asyncio
import logging
from datetime import timedelta
import numpy as np
import torch
from pydub import AudioSegment

logger = logging.getLogger(__name__)

from .database import get_db, utc_now
from .models import Speaker, SpeakerEmotionProfile, Conversation, ConversationSegment
from .schemas import (
    SpeakerResponse, SpeakerRename, SpeakerSegment, UnknownVoiceMatches, VoiceMatch,
    StatusResponse, ConversationResponse, ConversationSegmentResponse,
)
from .diarization import SpeakerRecognitionEngine, _in_memory_from_bytes
from .config import get_config
from .conversation_lifecycle import (
    fail_processing_lease,
    finish_processing_lease,
    new_processing_token,
)
from .services import (
    create_segment_from_result,
    data_path,
    delete_unknown_speakers,
    load_known_speakers,
)

router = APIRouter()


def _maybe_clear_gpu_cache(engine, reason: str) -> None:
    """Prefer threshold-gated async GPU cleanup so uploads don't trigger blocking CUDA syncs."""
    clear_async = getattr(engine, "clear_gpu_cache_async", None)
    if clear_async is not None:
        clear_async(reason)
    else:
        engine.clear_gpu_cache()


def _convert_audio(src: str, dst: str) -> None:
    """Blocking pydub conversion — always called via asyncio.to_thread."""
    AudioSegment.from_file(src).export(dst, format="wav")


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
    """List all enrolled speakers with segment counts, seconds of speech and when they were last heard"""
    speakers_with_counts = db.query(
        Speaker,
        func.count(ConversationSegment.id).label('segment_count'),
        func.coalesce(func.sum(ConversationSegment.end_offset - ConversationSegment.start_offset), 0.0),
        func.max(ConversationSegment.end_time),
    ).outerjoin(
        ConversationSegment, Speaker.id == ConversationSegment.speaker_id
    ).group_by(Speaker.id).all()

    # Convert to response format
    result = []
    for speaker, segment_count, speech_seconds, last_heard in speakers_with_counts:
        speaker_dict = {
            "id": speaker.id,
            "name": speaker.name,
            "created_at": speaker.created_at,
            "updated_at": speaker.updated_at,
            "segment_count": segment_count,
            "speech_seconds": float(speech_seconds or 0.0),
            "last_heard": last_heard,
        }
        result.append(SpeakerResponse(**speaker_dict))

    return result


@router.get("/speakers/unknown/matches", response_model=List[UnknownVoiceMatches])
async def unknown_voice_matches(db: Session = Depends(get_db)):
    """Every Unknown voice with the closest named speaker and the closest other Unknown (cosine similarity of the
    stored voices): an Unknown that is close to someone is probably them, and can be merged by identifying one of its
    segments as that person."""
    voices = []
    for speaker in db.query(Speaker).all():
        v = speaker.get_embedding() if speaker.embedding is not None else None
        if v is None or not len(v) or np.isnan(v).any() or not np.linalg.norm(v):
            continue
        voices.append((speaker, v / np.linalg.norm(v)))
    # voices of another model (before the startup migration) cannot be compared: keep the current model's size
    dims = [len(v) for _, v in voices]
    voices = [(s, v) for s, v in voices if len(v) == max(set(dims), key=dims.count)]
    if not voices:
        return []
    matrix = np.stack([v for _, v in voices])
    result = []
    for i, (speaker, v) in enumerate(voices):
        if not speaker.name.startswith("Unknown_"):
            continue
        sims = matrix @ v
        best = {}
        for j in np.argsort(-sims):
            other = voices[j][0]
            if j == i:
                continue
            kind = "unknown" if other.name.startswith("Unknown_") else "person"
            if kind not in best:
                best[kind] = VoiceMatch(speaker_id=other.id, speaker_name=other.name, similarity=round(float(sims[j]), 4))
            if len(best) == 2:
                break
        result.append(UnknownVoiceMatches(speaker_id=speaker.id, speaker_name=speaker.name, **best))
    return result


@router.get("/speakers/{speaker_id}/segments", response_model=List[SpeakerSegment])
async def list_speaker_segments(
    speaker_id: int,
    limit: int = 20,
    order: str = "recent",
    db: Session = Depends(get_db),
):
    """A speaker's segments, newest first (order=recent) or longest first (order=longest): to hear who a voice is,
    or to identify one of its segments (which moves an Unknown's segments with it)."""
    if not db.query(Speaker.id).filter(Speaker.id == speaker_id).first():
        raise HTTPException(status_code=404, detail="Speaker not found")
    sort = ((ConversationSegment.end_offset - ConversationSegment.start_offset).desc() if order == "longest"
            else ConversationSegment.end_time.desc())
    rows = db.query(ConversationSegment, Conversation.title).join(
        Conversation, Conversation.id == ConversationSegment.conversation_id
    ).filter(ConversationSegment.speaker_id == speaker_id).order_by(sort).limit(max(1, min(limit, 500))).all()
    return [SpeakerSegment(id=s.id, conversation_id=s.conversation_id, conversation_title=title, text=s.text,
                           start_offset=s.start_offset, end_offset=s.end_offset, start_time=s.start_time)
            for s, title in rows]


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

    # Save audio file temporarily (sanitize filename to prevent path traversal)
    temp_dir = os.path.join(data_path(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    safe_filename = os.path.basename(audio_file.filename or "upload")
    # Unique temp name: concurrent enrolls with the same client filename used
    # to write to the same path and delete each other's file mid-stream.
    import uuid as _uuid
    temp_path = os.path.join(temp_dir, f"{_uuid.uuid4().hex[:12]}_{safe_filename}")

    def _stream_upload():
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
    await asyncio.to_thread(_stream_upload)

    try:
        # Convert MP3 to WAV if needed (speaker enrollment may receive MP3 uploads).
        # pydub.from_file shells out to ffmpeg and is blocking — run off the event loop.
        wav_temp_path = None
        if temp_path.lower().endswith('.mp3'):
            wav_temp_path = temp_path.rsplit('.', 1)[0] + '_enrolled.wav'
            try:
                await asyncio.to_thread(_convert_audio, temp_path, wav_temp_path)
                extraction_path = wav_temp_path
            except Exception as e:
                logger.warning(f"Failed to convert MP3 to WAV: {e}")
                wav_temp_path = None
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

        # Make active streams aware of the new speaker immediately; their
        # in-memory cache was loaded before this enrollment (OPUS-015).
        try:
            engine.add_speaker_to_cache(speaker.id, speaker.name, embedding)
        except Exception:  # noqa: BLE001 - cache failures must fall back to a full reload
            engine.clear_speaker_cache()

        # Clear GPU cache asynchronously if above threshold
        _maybe_clear_gpu_cache(engine, "enroll")

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
    Rename a speaker (useful for MCP clients to label unknown speakers)

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
    speaker.updated_at = utc_now()

    updated_segments = db.query(ConversationSegment).filter(
        ConversationSegment.speaker_id == speaker_id
    ).update({"speaker_name": rename_data.new_name})

    db.commit()
    db.refresh(speaker)

    # A live WS stream caches profiles by name; drop the cache so matches use the new name.
    get_engine().clear_speaker_cache()

    logger.info(f"✓ Renamed speaker '{old_name}' → '{rename_data.new_name}' (updated {updated_segments} past segments)")

    return speaker


@router.delete("/speakers/{speaker_id}")
async def delete_speaker(speaker_id: int, db: Session = Depends(get_db)):
    """Delete a speaker"""
    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    # 1. Set speaker_id to NULL in segments (SQLite FK constraint is NO ACTION, not SET NULL).
    # Preserve a non-active tombstone instead of erasing attribution history;
    # it deliberately does not start with Unknown_ so retroactive relabelling
    # cannot match it.
    deleted_name = f"Deleted_{speaker.name}"
    db.query(ConversationSegment).filter(
        ConversationSegment.speaker_id == speaker_id
    ).update({"speaker_id": None, "speaker_name": deleted_name}, synchronize_session=False)

    # 2. Delete emotion profiles
    db.query(SpeakerEmotionProfile).filter(
        SpeakerEmotionProfile.speaker_id == speaker_id
    ).delete(synchronize_session=False)

    # Capture name before delete — post-commit attribute access can
    # trigger an expired-instance reload that raises.
    name = speaker.name

    db.delete(speaker)
    db.commit()

    get_engine().clear_speaker_cache()

    return {"message": f"Speaker '{name}' deleted successfully"}


@router.delete("/speakers/unknown/all")
async def delete_all_unknown_speakers(max_seconds: Optional[float] = None, db: Session = Depends(get_db)):
    """Delete all speakers with names starting with 'Unknown_' (with max_seconds: only those with at most that many
    seconds of speech). Their segments stay, without a speaker."""
    deleted_count, _ = delete_unknown_speakers(db, engine=get_engine(), max_seconds=max_seconds)
    db.commit()

    return {
        "message": f"Deleted {deleted_count} unknown speakers",
        "deleted_count": deleted_count
    }


@router.post("/process", response_model=ConversationResponse)
async def process_audio(
    audio_file: UploadFile = File(...),
    diarize: bool = Form(True),
    include_emotion: bool = Form(True),
    identify_speakers: bool = Form(True),
    save_conversation: bool = Form(True),
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Process uploaded audio file with speaker diarization and recognition.
    Creates a new Conversation with segments unless save_conversation=false (stateless/ephemeral fast path:
    decodes WAV/FLAC/OGG directly in memory, skips SQLite writes and never enrolls Unknown speakers).
    diarize=false skips speaker diarization when one person is talking; identify_speakers=false skips voice
    embedding extraction and returns raw diarization speaker labels; include_emotion=false skips emotion inference.
    """
    base_filename = os.path.basename(audio_file.filename or "upload")
    start_time = utc_now()
    threshold = get_config().get_settings().speaker_threshold

    if not save_conversation:
        raw_bytes = await asyncio.to_thread(audio_file.file.read)
        in_mem_audio = await asyncio.to_thread(_in_memory_from_bytes, raw_bytes)
        tmp_dir_ctx = None
        file_path = base_filename
        try:
            if in_mem_audio is None:
                tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="sds-ephemeral-")
                raw_path = os.path.join(tmp_dir_ctx.name, base_filename)
                with open(raw_path, "wb") as buffer:
                    buffer.write(raw_bytes)
                if not raw_path.lower().endswith(".wav"):
                    wav_path = raw_path.rsplit(".", 1)[0] + ".wav"
                    try:
                        await asyncio.to_thread(_convert_audio, raw_path, wav_path)
                        file_path = wav_path
                    except Exception as e:
                        logger.warning(f"Failed to convert ephemeral upload to WAV: {e}")
                        file_path = raw_path
                else:
                    file_path = raw_path

            result = await asyncio.to_thread(
                engine.transcribe_with_diarization,
                file_path,
                None,
                threshold=threshold,
                db_session=db if identify_speakers else None,
                diarize=diarize,
                audio=in_mem_audio,
                include_emotion=include_emotion,
                identify_speakers=identify_speakers,
            )
        except Exception:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail="Audio processing failed")
        finally:
            if tmp_dir_ctx is not None:
                tmp_dir_ctx.cleanup()

        _maybe_clear_gpu_cache(engine, "upload_ephemeral_complete")

        segs_out = []
        for idx, seg in enumerate(result["segments"], start=1):
            segs_out.append(
                ConversationSegmentResponse(
                    id=idx,
                    conversation_id=0,
                    speaker_id=seg.get("speaker_id"),
                    speaker_name=seg.get("speaker") or seg.get("speaker_label"),
                    text=seg.get("text"),
                    start_time=start_time + timedelta(seconds=seg["start"]),
                    end_time=start_time + timedelta(seconds=seg["end"]),
                    start_offset=seg["start"],
                    end_offset=seg["end"],
                    confidence=seg.get("confidence"),
                    emotion_category=seg.get("emotion_category"),
                    emotion_confidence=seg.get("emotion_confidence"),
                    detector_breakdown=seg.get("detector_breakdown"),
                    words=seg.get("words"),
                    avg_logprob=seg.get("avg_logprob"),
                    is_overlap=bool(seg.get("overlap", False)),
                )
            )
        duration = max((s["end"] for s in result["segments"]), default=0.0)
        return ConversationResponse(
            id=0,
            title=f"Uploaded: {audio_file.filename}",
            start_time=start_time,
            end_time=start_time + timedelta(seconds=duration) if result["segments"] else start_time,
            duration=duration if result["segments"] else None,
            status="completed",
            audio_format="wav",
            num_segments=len(segs_out),
            num_speakers=result["num_speakers"],
            transcript_segments=segs_out,
        )

    # Save audio file
    recordings_dir = os.path.join(data_path(), "recordings")
    os.makedirs(recordings_dir, exist_ok=True)
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    # uuid suffix avoids collisions between uploads with the same name in the
    # same second (they used to overwrite each other mid-stream).
    import uuid as _uuid
    temp_filename = f"uploaded_{timestamp}_{_uuid.uuid4().hex[:8]}_{base_filename}"
    temp_path = os.path.join(recordings_dir, temp_filename)

    def _stream_upload():
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
    await asyncio.to_thread(_stream_upload)

    # Convert to WAV for reliable processing (blocking ffmpeg shell-out)
    if not temp_path.lower().endswith('.wav'):
        wav_filename = temp_filename.rsplit('.', 1)[0] + '.wav'
        file_path = os.path.join(recordings_dir, wav_filename)
        try:
            await asyncio.to_thread(_convert_audio, temp_path, file_path)
            os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to convert to WAV: {e}")
            file_path = temp_path
    else:
        file_path = temp_path

    # Create conversation entry
    processing_token = new_processing_token()
    conversation = Conversation(
        title=f"Uploaded: {audio_file.filename}",
        audio_path=file_path,
        start_time=start_time,
        status="processing",
        processing_token=processing_token,
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    try:
        # Process audio with transcription (run in thread to avoid blocking event loop).
        # db_session gives the engine access to its cached profiles without an extra full-table scan.
        result = await asyncio.to_thread(
            engine.transcribe_with_diarization,
            file_path,
            None,
            threshold=threshold,
            db_session=db,
            diarize=diarize,
            include_emotion=include_emotion,
            identify_speakers=identify_speakers,
        )

        for seg in result["segments"]:
            create_segment_from_result(
                seg=seg,
                conversation_id=conversation.id,
                conv_start=start_time,
                db=db,
                threshold=threshold,
            )

        # Update conversation metadata
        conversation.num_segments = len(result["segments"])
        conversation.num_speakers = result["num_speakers"]
        if result["segments"]:
            conversation.duration = max(s["end"] for s in result["segments"])
            conversation.end_time = start_time + timedelta(seconds=conversation.duration)

        if not finish_processing_lease(
            db,
            conversation.id,
            processing_token,
            status="completed",
        ):
            raise RuntimeError("Upload processing lease lost")
        db.refresh(conversation)

        # Queue async GPU cleanup only if VRAM exceeds threshold (avoids per-request CUDA sync stalls)
        _maybe_clear_gpu_cache(engine, "upload_complete")

        # Return conversation
        return conversation

    except Exception:
        import traceback
        traceback.print_exc()
        # Roll back half-built segments and release only our own operation.
        fail_processing_lease(db, conversation.id, processing_token)
        engine.clear_speaker_cache()  # reload from what is committed
        raise HTTPException(status_code=500, detail="Audio processing failed")


