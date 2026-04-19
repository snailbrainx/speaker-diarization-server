"""
API endpoints for conversation management
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
import asyncio
import json
import os

from .database import get_db, utc_now
from .models import Conversation, ConversationSegment, Speaker, SpeakerEmotionProfile
from .schemas import (
    ConversationResponse,
    ConversationsListResponse,
    ConversationUpdate,
    IdentifySpeakerRequest,
    ToggleMisidentifiedRequest,
)
from .diarization import SpeakerRecognitionEngine
from .api import get_engine
from .config import get_config
from .services import (
    cleanup_orphaned_unknowns,
    create_segment_from_result,
    data_path,
    load_known_speakers,
    recalculate_emotion_profile,
    recalculate_speaker_embedding,
    resolve_audio_path,
)
import logging
import numpy as np

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/conversations", tags=["Conversations"])


@router.get("", response_model=ConversationsListResponse)
async def list_conversations(
    skip: int = 0,
    limit: int = 100,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all conversations with pagination and filtering.
    Returns lightweight summaries without segments for better performance.
    """
    query = db.query(Conversation).order_by(Conversation.start_time.desc())

    if status:
        query = query.filter(Conversation.status == status)

    # Get total count
    total = query.count()

    # Get paginated results (no segments loaded)
    conversations = query.offset(skip).limit(limit).all()

    return ConversationsListResponse(
        conversations=conversations,
        total=total,
        skip=skip,
        limit=limit
    )


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Get conversation details with all segments"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return conversation


@router.patch("/{conversation_id}", response_model=ConversationResponse)
async def update_conversation(
    conversation_id: int,
    update_data: ConversationUpdate,
    db: Session = Depends(get_db)
):
    """Update conversation metadata"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if update_data.title is not None:
        conversation.title = update_data.title
    if update_data.status is not None:
        conversation.status = update_data.status

    db.commit()
    db.refresh(conversation)
    return conversation


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: int, db: Session = Depends(get_db)):
    """Delete conversation and associated audio"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Delete audio file
    if conversation.audio_path and os.path.exists(conversation.audio_path):
        os.remove(conversation.audio_path)

    db.delete(conversation)
    db.commit()

    return {"message": f"Conversation {conversation_id} deleted"}


@router.post("/{conversation_id}/reprocess")
async def reprocess_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """Re-process conversation with current speaker profiles (works with MP3)"""
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not conversation.audio_path or not os.path.exists(conversation.audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    known_speakers = load_known_speakers(db)

    # Get threshold from config (consistent with all other endpoints)
    config = get_config()
    settings = config.get_settings()
    threshold = settings.speaker_threshold
    result = await asyncio.to_thread(
        engine.transcribe_with_diarization,
        conversation.audio_path,
        known_speakers,
        threshold=threshold,
        db_session=db  # Enable personalized emotion matching
    )

    # Delete old segments (synchronize_session=False avoids FK constraint issues)
    db.query(ConversationSegment).filter(
        ConversationSegment.conversation_id == conversation_id
    ).delete(synchronize_session=False)

    # Create new segments
    conv_start = conversation.start_time

    for seg in result["segments"]:
        create_segment_from_result(
            seg, conversation_id, conv_start, db, threshold
        )

    # Update conversation stats
    conversation.status = "completed"
    conversation.num_segments = len(result["segments"])
    conversation.num_speakers = result["num_speakers"]

    db.commit()

    # Clear GPU cache after reprocessing
    engine.clear_gpu_cache()

    return {"message": "Conversation reprocessed", "segments": len(result["segments"])}


@router.post("/{conversation_id}/recalculate-emotions")
async def recalculate_emotions(
    conversation_id: int,
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Recalculate emotions for all segments using current emotion profiles
    WITHOUT re-running diarization or transcription (preserves manual work)
    """
    conversation = db.query(Conversation).filter(
        Conversation.id == conversation_id
    ).first()

    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not conversation.audio_path or not os.path.exists(conversation.audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Get all segments for this conversation
    segments = db.query(ConversationSegment).filter(
        ConversationSegment.conversation_id == conversation_id
    ).all()

    # Preload speakers referenced by these segments (avoids a per-segment query)
    speaker_ids = {s.speaker_id for s in segments if s.speaker_id}
    speakers_by_id = (
        {sp.id: sp for sp in db.query(Speaker).filter(Speaker.id.in_(speaker_ids)).all()}
        if speaker_ids else {}
    )

    updated_count = 0
    skipped_count = 0
    audio_file = conversation.audio_path

    for segment in segments:
        # Skip if no speaker or manually corrected (respect user corrections)
        if not segment.speaker_id or segment.emotion_corrected:
            skipped_count += 1
            continue

        try:
            # Re-extract emotion with personalized matching (off the event loop)
            emotion_data = await asyncio.to_thread(
                engine.extract_emotion,
                audio_file,
                segment.start_offset,
                segment.end_offset,
                extract_embedding=True,
            )

            if not emotion_data:
                skipped_count += 1
                continue

            speaker = speakers_by_id.get(segment.speaker_id)

            if speaker and speaker.emotion_profiles:
                # Use dual-detector matching if profiles exist
                voice_emb = segment.get_speaker_embedding()
                emotion_emb = emotion_data.get('embedding')

                if voice_emb is not None and emotion_emb is not None:
                
                    global_threshold = get_config().get_settings().emotion_threshold

                    dual_result = engine.match_emotion_dual_detector(
                        emotion_embedding=emotion_emb,
                        voice_embedding=voice_emb,
                        speaker_emotion_profiles=speaker.emotion_profiles,
                        global_threshold=global_threshold,
                        speaker_threshold=speaker.emotion_threshold,
                        generic_emotion=emotion_data['emotion_category'],
                        generic_confidence=emotion_data['emotion_confidence']
                    )

                    # Update segment with final decision
                    final = dual_result['final_decision']
                    segment.emotion_category = final['emotion']
                    segment.emotion_confidence = final['confidence']
                    segment.detector_breakdown = json.dumps(dual_result)  # Store breakdown for clients
                    updated_count += 1
                else:
                    # Fall back to generic emotion2vec result
                    segment.emotion_category = emotion_data['emotion_category']
                    segment.emotion_confidence = emotion_data['emotion_confidence']
                    segment.detector_breakdown = None  # No dual-detector used
                    updated_count += 1
            else:
                # No profiles, use generic emotion2vec result
                segment.emotion_category = emotion_data['emotion_category']
                segment.emotion_confidence = emotion_data['emotion_confidence']
                segment.detector_breakdown = None  # No dual-detector used
                updated_count += 1

        except Exception as e:
            logger.warning(f"Failed to recalculate emotion for segment {segment.id}: {e}")
            skipped_count += 1

    db.commit()

    # Clear GPU cache
    engine.clear_gpu_cache()

    return {
        "message": "Emotions recalculated",
        "updated": updated_count,
        "skipped": skipped_count,
        "total": len(segments)
    }


@router.post("/{conversation_id}/segments/{segment_id}/identify")
async def identify_speaker_in_segment(
    conversation_id: int,
    segment_id: int,
    request: IdentifySpeakerRequest,
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Identify speaker in segment and optionally enroll them

    Args:
        request: Request body with speaker_id, speaker_name, and enroll flag
    """
    speaker_id = request.speaker_id
    speaker_name = request.speaker_name
    enroll = request.enroll
    segment = db.query(ConversationSegment).filter(
        ConversationSegment.id == segment_id,
        ConversationSegment.conversation_id == conversation_id
    ).first()

    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    conversation = segment.conversation

    # Determine which audio file to use for embedding extraction
    # CRITICAL: Database offsets (start_offset/end_offset) are ALWAYS conversation-relative!
    # They represent seconds from the conversation start, NOT from individual segment files.
    # Therefore, we MUST use the full conversation audio file where these offsets are valid.
    start_time = segment.start_offset
    end_time = segment.end_offset

    audio_file = resolve_audio_path(conversation, segment)
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found (neither conversation audio nor segment audio exists)")
    if audio_file == segment.segment_audio_path:
        logger.info(f"⚠️ WARNING: Using segment audio with conversation-relative offsets - may extract wrong audio!")

    # Store the old speaker name and ID for propagation and embedding recalculation
    old_speaker_name = segment.speaker_name
    old_speaker_id = segment.speaker_id

    # Extract embedding FIRST if enrolling (needed for new speakers, off the event loop)
    embedding = None
    if enroll:
        try:
            embedding = await asyncio.to_thread(
                engine.extract_segment_embedding,
                audio_file,
                start_time,
                end_time,
            )
        except Exception:
            raise HTTPException(
                status_code=500,
                detail="Failed to extract speaker embedding"
            )

    # Get or create speaker
    speaker = None
    merge_msg = ""

    if speaker_id:
        # Existing speaker by ID - load from DB
        speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
        if not speaker:
            raise HTTPException(status_code=404, detail="Speaker not found")
    elif speaker_name:
        # Try to find existing speaker by name
        speaker = db.query(Speaker).filter(Speaker.name == speaker_name).first()

    # At this point, speaker is either:
    # - Existing speaker (found by ID or name)
    # - None (need to create new)

    if speaker:
        # Existing speaker - we'll recalculate embedding after updating segments
        merge_msg = ""
    else:
        # New speaker - must have name and embedding
        if not speaker_name:
            raise HTTPException(status_code=400, detail="speaker_name required for new speaker")

        if not enroll or embedding is None:
            raise HTTPException(status_code=400, detail="Must enroll new speaker (enroll=True)")

        # Create new speaker with embedding
        speaker = Speaker(name=speaker_name)
        speaker.set_embedding(embedding)
        db.add(speaker)
        db.flush()  # Get ID without committing
        merge_msg = " (initial enrollment)"

    # Update THIS segment
    segment.speaker_id = speaker.id
    segment.speaker_name = speaker.name
    segment.confidence = 1.0  # Manually identified

    # UPDATE ALL OTHER SEGMENTS with the same old speaker name (retroactive identification!)
    # SAFETY: Only do retroactive updates for Unknown speakers!
    # If old speaker is already identified (Tommy, Diamond, etc.), only update THIS segment
    updated_count = 0
    if old_speaker_name and old_speaker_name != speaker.name and old_speaker_name.startswith("Unknown_"):
        updated_count = db.query(ConversationSegment).filter(
            ConversationSegment.speaker_name == old_speaker_name,
            ConversationSegment.id != segment_id  # Don't update the one we just did
        ).update({
            "speaker_id": speaker.id,
            "speaker_name": speaker.name
        })

    # CRITICAL: Flush segment updates so emotion recalculation queries see the new speaker_id
    db.flush()

    # Everything below touches GPU (emotion extraction) and runs O(speaker_segments)
    # SQL. Wrap it in a single worker hop so the event loop is only blocked by the
    # initial embedding extraction above. The handler awaits here, so no other
    # coroutine is racing the `db` Session.
    def _retroactive_updates() -> tuple[str, int]:
        merge_suffix = ""
        emb_count = recalculate_speaker_embedding(speaker, db, engine)
        if emb_count:
            logger.info(f"✓ Recalculated embedding for '{speaker.name}' (added segment {segment_id}, now {emb_count} total segments)")
            merge_suffix = f" (recalculated from {emb_count} non-misidentified segments)"

        # Recalculate OLD speaker's embedding to exclude this segment, unless they'll be deleted anyway
        if (old_speaker_id and old_speaker_id != speaker.id
                and not (old_speaker_name and old_speaker_name.startswith("Unknown_"))):
            old_speaker = db.query(Speaker).filter(Speaker.id == old_speaker_id).first()
            if old_speaker:
                old_emb_count = recalculate_speaker_embedding(old_speaker, db, engine)
                if old_emb_count:
                    logger.info(f"✓ Recalculated embedding for '{old_speaker.name}' (removed segment {segment_id})")
                else:
                    logger.info(f"⚠️ No valid segments remaining for '{old_speaker.name}' after removing segment {segment_id}")

        if segment.emotion_corrected and not segment.emotion_misidentified and segment.emotion_category:
            emotion_category = segment.emotion_category
            logger.info(f"🎭 Recalculating emotion profiles for '{emotion_category}' (segment moved from {old_speaker_name} to {speaker.name})")

            new_result = recalculate_emotion_profile(speaker.id, emotion_category, db, engine)
            if new_result:
                logger.info(f"  ✓ {new_result.capitalize()} '{speaker.name}' emotion profile '{emotion_category}' (segment {segment_id})")

            if (old_speaker_id and old_speaker_id != speaker.id
                    and not (old_speaker_name and old_speaker_name.startswith("Unknown_"))):
                old_result = recalculate_emotion_profile(old_speaker_id, emotion_category, db, engine)
                if old_result:
                    logger.info(f"  ✓ {old_result.capitalize()} old speaker emotion profile '{emotion_category}' (removed segment {segment_id})")

        db.flush()

        logger.info(f"🔍 Starting cleanup check for orphaned Unknown speakers...")
        deleted_unknowns = cleanup_orphaned_unknowns(db, engine=engine)
        for name in deleted_unknowns:
            logger.info(f"🗑️ Auto-deleted orphaned speaker: {name}")

        if deleted_unknowns:
            if len(deleted_unknowns) == 1:
                merge_suffix += f" (auto-deleted orphaned {deleted_unknowns[0]})"
            else:
                merge_suffix += f" (auto-deleted {len(deleted_unknowns)} orphaned Unknown speakers)"

        db.commit()
        db.refresh(segment)

        # Re-detect emotions using personalized profiles (Unknown→Known transition)
        emotions_updated = 0
        if speaker.emotion_profiles:
            profiles = [
                (prof.emotion_category, prof.get_embedding(), prof.confidence_threshold)
                for prof in speaker.emotion_profiles
            ]
            global_threshold = get_config().get_settings().emotion_threshold
            identified_segments = db.query(ConversationSegment).filter(
                ConversationSegment.speaker_id == speaker.id,
                ConversationSegment.conversation_id == conversation_id,
            ).all()

            for seg in identified_segments:
                if not seg.emotion_category or seg.emotion_corrected:
                    continue
                original_emotion = seg.emotion_category
                emotion_embedding = seg.get_emotion_embedding()
                if emotion_embedding is None or np.isnan(emotion_embedding).any():
                    seg_audio = resolve_audio_path(seg.conversation, seg)
                    if seg_audio:
                        try:
                            emotion_data = engine.extract_emotion(
                                seg_audio, seg.start_offset, seg.end_offset, extract_embedding=True
                            )
                            if emotion_data and 'embedding' in emotion_data:
                                emotion_embedding = emotion_data.get('embedding')
                        except Exception as e:
                            logger.info(f"  ⚠️ Could not extract emotion for segment {seg.id}: {e}")
                            continue

                if emotion_embedding is not None and not np.isnan(emotion_embedding).any():
                    match = engine.match_emotion_to_profile(
                        emotion_embedding, profiles, global_threshold,
                        speaker_threshold=speaker.emotion_threshold,
                    )
                    if match:
                        matched_emotion, confidence = match
                        if matched_emotion != original_emotion:
                            logger.info(f"  ✓ Segment {seg.id}: {original_emotion} → {matched_emotion} ({confidence:.2%} personalized match)")
                            seg.emotion_category = matched_emotion
                            seg.emotion_confidence = confidence
                            emotions_updated += 1

            if emotions_updated > 0:
                logger.info(f"✅ Updated {emotions_updated} emotion(s) using personalized profiles")
        else:
            logger.info(f"  ℹ️ No emotion profiles found for {speaker.name} - keeping generic detections")

        db.commit()
        db.refresh(segment)
        engine.clear_gpu_cache()
        return merge_suffix, emotions_updated

    merge_suffix, _ = await asyncio.to_thread(_retroactive_updates)
    merge_msg += merge_suffix

    return {
        "message": f"Speaker identified as {speaker.name}{merge_msg}. Updated {updated_count + 1} segment(s) total.",
        "speaker_id": speaker.id,
        "enrolled": enroll,
        "segments_updated": updated_count + 1
    }


@router.patch("/{conversation_id}/segments/{segment_id}/misidentified")
async def toggle_segment_misidentified(
    conversation_id: int,
    segment_id: int,
    request: ToggleMisidentifiedRequest,
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Toggle misidentification status for a segment and recalculate speaker embedding

    When a segment is marked as misidentified, it's excluded from the speaker's
    embedding calculation, improving recognition accuracy.
    """
    segment = db.query(ConversationSegment).filter(
        ConversationSegment.id == segment_id,
        ConversationSegment.conversation_id == conversation_id
    ).first()

    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    segment.is_misidentified = request.is_misidentified

    # Flush to ensure the change is visible to subsequent queries
    db.flush()

    # If segment has a speaker, recalculate their embedding
    if segment.speaker_id:
        speaker = db.query(Speaker).filter(Speaker.id == segment.speaker_id).first()

        if speaker:
            emb_count = recalculate_speaker_embedding(speaker, db, engine)
            if emb_count:
                logger.info(f"✓ Recalculated embedding for '{speaker.name}' from {emb_count} non-misidentified segments")
            else:
                logger.info(f"⚠️ No valid segments remaining for '{speaker.name}' after marking segment {segment_id} as misidentified")

    db.commit()
    db.refresh(segment)

    # Clear GPU cache after embedding extractions
    engine.clear_gpu_cache()

    status_text = "marked as misidentified" if request.is_misidentified else "unmarked as misidentified"
    return {
        "message": f"Segment {segment_id} {status_text}",
        "is_misidentified": segment.is_misidentified,
        "embedding_recalculated": segment.speaker_id is not None
    }


@router.patch("/{conversation_id}/segments/{segment_id}/emotion-misidentified")
async def toggle_emotion_misidentified(
    conversation_id: int,
    segment_id: int,
    request: ToggleMisidentifiedRequest,
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Toggle emotion misidentification status for a segment and recalculate emotion profile

    When a segment's emotion correction is marked as misidentified, it's excluded from the
    speaker's emotion profile calculation, allowing you to fix mistakes in emotion learning.
    """
    segment = db.query(ConversationSegment).filter(
        ConversationSegment.id == segment_id,
        ConversationSegment.conversation_id == conversation_id
    ).first()

    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Only process if segment has an emotion correction
    if not segment.emotion_corrected:
        raise HTTPException(
            status_code=400,
            detail="Segment has no emotion correction to mark as misidentified"
        )

    segment.emotion_misidentified = request.is_misidentified

    # Flush so subsequent same-session queries see the new value
    db.flush()

    # If segment has a speaker and emotion, recalculate emotion profile
    if segment.speaker_id and segment.emotion_category:
        speaker = db.query(Speaker).filter(Speaker.id == segment.speaker_id).first()

        if speaker:
            emotion_category = segment.emotion_category
            result = recalculate_emotion_profile(speaker.id, emotion_category, db, engine)
            if result == "updated":
                logger.info(f"✓ Recalculated emotion profile '{emotion_category}' for '{speaker.name}'")
            elif result == "created":
                logger.info(f"✓ Created emotion profile '{emotion_category}' for '{speaker.name}'")
            elif result == "deleted":
                logger.info(f"⚠️ Deleted emotion profile '{emotion_category}' for '{speaker.name}' - no valid corrections remaining")

    db.commit()
    db.refresh(segment)

    # Clear GPU cache after embedding extractions
    engine.clear_gpu_cache()

    status_text = "marked as misidentified" if request.is_misidentified else "unmarked as misidentified"
    return {
        "message": f"Emotion correction for segment {segment_id} {status_text}",
        "emotion_misidentified": segment.emotion_misidentified,
        "emotion_profile_recalculated": segment.speaker_id is not None and segment.emotion_category is not None
    }


@router.get("/segments/{segment_id}/audio")
async def get_segment_audio(
    segment_id: int,
    db: Session = Depends(get_db)
):
    """
    Extract and serve audio for a specific conversation segment.

    Uses ffmpeg to extract the segment's time range from the full conversation audio.
    Returns WAV audio file.
    """
    logger.info(f"🎵 Audio request for segment {segment_id}")

    segment = db.query(ConversationSegment).filter(ConversationSegment.id == segment_id).first()
    if not segment:
        logger.info(f"❌ Segment {segment_id} not found in database")
        raise HTTPException(status_code=404, detail="Segment not found")

    conversation = segment.conversation

    # Determine source audio file and check if we need extraction
    # CRITICAL: Streaming segment files (seg_XXXX.wav) contain the RAW VAD-triggered audio chunk.
    # After diarization, ONE segment file may contain MULTIPLE speaker segments.
    # We MUST extract the specific time range, not serve the whole file!

    # First check: Can we use full conversation audio? (Best option)
    use_conversation_audio = conversation.audio_path and os.path.exists(conversation.audio_path)

    # Second check: Use segment file if conversation audio doesn't exist yet (during streaming)
    use_segment_audio = segment.segment_audio_path and os.path.exists(segment.segment_audio_path)

    if not use_conversation_audio and not use_segment_audio:
        logger.info(f"❌ No audio file found for segment {segment_id}")
        logger.info(f"  segment_audio_path: {segment.segment_audio_path}")
        logger.info(f"  conversation.audio_path: {conversation.audio_path}")
        raise HTTPException(status_code=404, detail="Audio file not found")

    # Prefer full conversation audio (offsets are conversation-relative)
    if use_conversation_audio:
        source_audio = conversation.audio_path
        start_time = segment.start_offset
        end_time = segment.end_offset
        logger.info(f"  Using conversation audio: {source_audio}")
        logger.info(f"  Offsets: {start_time:.2f}s - {end_time:.2f}s (conversation-relative)")
    else:
        # Fallback: Use segment file with file-relative offsets
        # Need to calculate the segment's position within its segment file
        source_audio = segment.segment_audio_path
        # TODO: Calculate file-relative offsets from segment file metadata
        # For now, serve entire segment file (may contain extra audio)
        logger.info(f"  ⚠️ Using segment audio (may contain multiple segments): {source_audio}")
        start_time = 0  # Start of segment file
        # Get duration from file
        from pydub import AudioSegment as AS
        audio = AS.from_file(source_audio)
        end_time = len(audio) / 1000.0  # Convert ms to seconds
        logger.info(f"  Serving entire segment file: 0s - {end_time:.2f}s")

    # Create temporary directory for extracted segments
    temp_dir = os.path.join(data_path(), "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"segment_{segment_id}_{int(datetime.now().timestamp())}.wav")

    try:
        # Use ffmpeg to extract the specific time range with small padding at end
        duration = end_time - start_time
        duration_with_padding = duration + 0.25  # Add 250ms to avoid cutting off last word
        logger.info(f"  Extracting {duration_with_padding:.2f}s from offset {start_time:.2f}s")
        logger.info(f"  Output: {temp_path}")

        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y",
            "-ss", str(start_time),
            "-t", str(duration_with_padding),
            "-i", source_audio,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            temp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_bytes = await proc.communicate()
        if proc.returncode != 0:
            logger.error(f"FFmpeg error:{stderr_bytes.decode(errors='replace')}")
            raise HTTPException(status_code=500, detail="Audio extraction failed")

        if not os.path.exists(temp_path):
            logger.info(f"❌ Extraction failed - temp file not created")
            raise HTTPException(status_code=500, detail="Audio extraction failed")

        file_size = os.path.getsize(temp_path)
        logger.info(f"✅ Extracted successfully ({file_size} bytes)")

        # Return the extracted audio file with cache control headers
        from starlette.background import BackgroundTask

        # Clean up temp file after sending
        def cleanup():
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    logger.info(f"🗑️  Cleaned up {temp_path}")
            except Exception as e:
                logger.info(f"Failed to cleanup temp file {temp_path}: {e}")

        return FileResponse(
            path=temp_path,
            media_type="audio/wav",
            filename=f"segment_{segment_id}.wav",
            background=BackgroundTask(cleanup),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.info(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Audio extraction failed")


# ============================================================================
# EMOTION ENDPOINTS (Personalized Emotion Detection)
# ============================================================================

@router.post("/{conversation_id}/segments/{segment_id}/correct-emotion")
async def correct_emotion_in_segment(
    conversation_id: int,
    segment_id: int,
    corrected_emotion: str = Query(..., description="Correct emotion category"),
    learn: bool = Query(True, description="Learn from this correction"),
    db: Session = Depends(get_db),
    engine: SpeakerRecognitionEngine = Depends(get_engine)
):
    """
    Correct emotion in a segment and optionally learn from the correction.

    This enables personalized emotion detection by building speaker-specific emotion profiles.

    Args:
        corrected_emotion: The correct emotion category (angry, happy, sad, neutral, fearful, surprised, disgusted, other)
        learn: If True, extract embedding and update speaker's emotion profile (default: True)

    Returns:
        Success message with details about learning
    """
    # Validate segment exists
    segment = db.query(ConversationSegment).filter(
        ConversationSegment.id == segment_id,
        ConversationSegment.conversation_id == conversation_id
    ).first()

    if not segment:
        raise HTTPException(status_code=404, detail="Segment not found")

    # Must have a known speaker to create emotion profile
    if not segment.speaker_id:
        raise HTTPException(
            status_code=400,
            detail="Cannot create emotion profile for unknown speaker. Identify speaker first."
        )

    old_emotion = segment.emotion_category
    old_emotion_corrected = segment.emotion_corrected
    conversation = segment.conversation

    # Get audio file for embedding extraction
    audio_file = resolve_audio_path(conversation, segment)
    if not audio_file:
        raise HTTPException(status_code=404, detail="Audio file not found for this segment")

    # Extract emotion embedding if learning
    emotion_embedding = None
    if learn:
        # Try stored embedding first (FAST - no audio extraction needed!)
        emotion_embedding = segment.get_emotion_embedding()

        if emotion_embedding is None or np.isnan(emotion_embedding).any():
            # Extract from audio if not cached (SLOW - fallback only, off the event loop)
            try:
                logger.info(f"  ℹ️ Extracting emotion embedding from audio for segment {segment_id} (not cached)")
                emotion_data = await asyncio.to_thread(
                    engine.extract_emotion,
                    audio_file,
                    segment.start_offset,
                    segment.end_offset,
                    True,
                )

                if emotion_data:
                    emotion_embedding = emotion_data.get('embedding')

                if emotion_embedding is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to extract emotion embedding for learning"
                    )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail="Failed to extract emotion embedding"
                )
        else:
            logger.info(f"  ✓ Using cached emotion embedding for segment {segment_id}")

    # Update segment FIRST so recalculation of OLD profile correctly excludes this segment
    segment.emotion_category = corrected_emotion
    segment.emotion_confidence = 1.0  # Manual correction = 100% confidence
    segment.emotion_corrected = True
    segment.emotion_corrected_at = utc_now()
    db.flush()

    # CRITICAL: If changing from one emotion to another, recalculate OLD emotion profile
    # to exclude this segment (like speaker identification does)
    # Do this whenever old_emotion exists, regardless of old_emotion_corrected status,
    # because reprocessing with personalized matching can set emotions without corrected=True
    if learn and old_emotion and old_emotion != corrected_emotion:
        old_result = await asyncio.to_thread(
            recalculate_emotion_profile, segment.speaker_id, old_emotion, db, engine
        )
        if old_result == "updated":
            logger.info(f"✓ Recalculated '{old_emotion}' profile (removed segment {segment_id})")
        elif old_result == "deleted":
            logger.info(f"⚠️ Deleted emotion profile '{old_emotion}' - no valid corrections remaining after removing segment {segment_id}")

    # Learn from correction if requested
    merge_msg = ""
    sample_count = 0
    voice_samples = 0
    if learn and emotion_embedding is not None:
        # Get or create emotion profile
        profile = db.query(SpeakerEmotionProfile).filter(
            SpeakerEmotionProfile.speaker_id == segment.speaker_id,
            SpeakerEmotionProfile.emotion_category == corrected_emotion
        ).first()

        if profile:
            # MERGE EMOTION embeddings (weighted average)
            existing_emb = profile.get_embedding()

            # Weighted average: existing embedding has more weight based on sample count
            weight = profile.sample_count / (profile.sample_count + 1)
            merged_emb = (existing_emb * weight) + (emotion_embedding * (1 - weight))

            profile.set_embedding(merged_emb)
            profile.sample_count += 1
            profile.updated_at = utc_now()

            sample_count = profile.sample_count
            logger.info(f"✓ Merged segment {segment_id} into '{corrected_emotion}' profile (now {sample_count} emotion samples)")
        else:
            # Create new profile
            profile = SpeakerEmotionProfile(
                speaker_id=segment.speaker_id,
                emotion_category=corrected_emotion,
                sample_count=1,
                voice_sample_count=0
            )
            profile.set_embedding(emotion_embedding)
            db.add(profile)

            sample_count = 1
            logger.info(f"✓ Created new '{corrected_emotion}' profile with segment {segment_id}")
        
        # NEW: Also merge VOICE embedding for this emotion (Detector 2 data)
        voice_emb = segment.get_speaker_embedding()
        if voice_emb is not None and not np.isnan(voice_emb).any():
            existing_voice_emb = profile.get_voice_embedding()

            if existing_voice_emb is not None and not np.isnan(existing_voice_emb).any():
                # Merge with existing voice profile for this emotion
                voice_weight = profile.voice_sample_count / (profile.voice_sample_count + 1)
                merged_voice = (existing_voice_emb * voice_weight) + (voice_emb * (1 - voice_weight))
                profile.set_voice_embedding(merged_voice)
                profile.voice_sample_count += 1
                logger.info(f"  → Also merged voice embedding (now {profile.voice_sample_count} voice samples)")
            else:
                # First voice sample for this emotion
                profile.set_voice_embedding(voice_emb)
                profile.voice_sample_count = 1
                logger.info(f"  → Added first voice sample for '{corrected_emotion}' profile")

            voice_samples = profile.voice_sample_count
            
            # Also update generic speaker profile (keeps it current)
            speaker = db.query(Speaker).filter(Speaker.id == segment.speaker_id).first()
            if speaker:
                existing_speaker_emb = speaker.get_embedding()
                # Get all non-misidentified segments for this speaker
                all_segments = db.query(ConversationSegment).filter(
                    ConversationSegment.speaker_id == speaker.id,
                    ConversationSegment.is_misidentified == False
                ).count()
                
                if all_segments > 0:
                    speaker_weight = (all_segments - 1) / all_segments
                    merged_speaker = (existing_speaker_emb * speaker_weight) + (voice_emb * (1 - speaker_weight))
                    speaker.set_embedding(merged_speaker)
        
        merge_msg = f" (emotion: {sample_count} samples, voice: {voice_samples} samples)"

    db.commit()
    db.refresh(segment)

    # Clear GPU cache
    engine.clear_gpu_cache()

    speaker = db.query(Speaker).filter(Speaker.id == segment.speaker_id).first()

    # Determine if this was a correction or confirmation
    is_confirmation = old_emotion == corrected_emotion
    action_msg = "confirmed" if is_confirmation else f"corrected from '{old_emotion}' to '{corrected_emotion}'"
    
    return {
        "message": f"Emotion {action_msg}{merge_msg}",
        "old_emotion": old_emotion,
        "new_emotion": corrected_emotion,
        "learned": learn,
        "sample_count": sample_count,
        "speaker_name": speaker.name if speaker else None
    }


@router.delete("/speakers/{speaker_id}/emotion-profiles")
async def reset_speaker_emotion_profiles(
    speaker_id: int,
    emotion_category: Optional[str] = Query(None, description="Specific emotion to reset (or all if not specified)"),
    db: Session = Depends(get_db)
):
    """
    Reset emotion profiles for a speaker.

    Args:
        emotion_category: If specified, only reset this emotion. If None, reset all emotions.

    Returns:
        Number of profiles deleted
    """
    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    query = db.query(SpeakerEmotionProfile).filter(
        SpeakerEmotionProfile.speaker_id == speaker_id
    )

    if emotion_category:
        query = query.filter(SpeakerEmotionProfile.emotion_category == emotion_category)
        deleted = query.delete()
        db.commit()
        return {
            "message": f"Reset emotion profile '{emotion_category}' for speaker '{speaker.name}'",
            "speaker_name": speaker.name,
            "emotion_category": emotion_category,
            "deleted": deleted
        }
    else:
        deleted = query.delete()
        db.commit()
        return {
            "message": f"Reset all emotion profiles for speaker '{speaker.name}'",
            "speaker_name": speaker.name,
            "deleted": deleted
        }


@router.get("/speakers/{speaker_id}/emotion-threshold")
async def get_speaker_emotion_threshold(
    speaker_id: int,
    db: Session = Depends(get_db)
):
    """Get speaker's custom emotion threshold (or global default)"""
    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")


    global_threshold = get_config().get_settings().emotion_threshold

    return {
        "speaker_id": speaker_id,
        "speaker_name": speaker.name,
        "custom_threshold": speaker.emotion_threshold,
        "effective_threshold": speaker.emotion_threshold or global_threshold,
        "using_global": speaker.emotion_threshold is None
    }


@router.patch("/speakers/{speaker_id}/emotion-threshold")
async def set_speaker_emotion_threshold(
    speaker_id: int,
    threshold: Optional[float] = Query(None, ge=0.3, le=1.0, description="Custom threshold (0.3-1.0) or null for global"),
    db: Session = Depends(get_db)
):
    """
    Set speaker's custom emotion threshold.

    Args:
        threshold: Custom threshold (0.3-1.0) or None to use global default
                  Higher = stricter matching (1.0 = perfect match required)

    Returns:
        Updated threshold settings
    """
    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    speaker.emotion_threshold = threshold
    db.commit()


    global_threshold = get_config().get_settings().emotion_threshold

    return {
        "message": f"Updated emotion threshold for '{speaker.name}'",
        "speaker_name": speaker.name,
        "custom_threshold": threshold,
        "effective_threshold": threshold or global_threshold,
        "using_global": threshold is None
    }


@router.get("/speakers/{speaker_id}/emotion-profiles")
async def get_speaker_emotion_profiles(
    speaker_id: int,
    db: Session = Depends(get_db)
):
    """Get all emotion profiles for a speaker"""
    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")

    profiles = db.query(SpeakerEmotionProfile).filter(
        SpeakerEmotionProfile.speaker_id == speaker_id
    ).all()

    return {
        "speaker_id": speaker_id,
        "speaker_name": speaker.name,
        "emotion_threshold": speaker.emotion_threshold,
        "profiles": [
            {
                "emotion_category": prof.emotion_category,
                "sample_count": prof.sample_count,
                "voice_sample_count": prof.voice_sample_count,
                "confidence_threshold": prof.confidence_threshold,
                "voice_threshold": prof.voice_threshold,
                "created_at": prof.created_at,
                "updated_at": prof.updated_at
            }
            for prof in profiles
        ]
    }




def _get_speaker_emotion_profile(speaker_id: int, emotion_category: str, db: Session) -> tuple:
    """Shared lookup for the two threshold endpoints."""
    speaker = db.query(Speaker).filter(Speaker.id == speaker_id).first()
    if not speaker:
        raise HTTPException(status_code=404, detail="Speaker not found")
    profile = db.query(SpeakerEmotionProfile).filter(
        SpeakerEmotionProfile.speaker_id == speaker_id,
        SpeakerEmotionProfile.emotion_category == emotion_category,
    ).first()
    if not profile:
        raise HTTPException(
            status_code=404,
            detail=f"Emotion profile '{emotion_category}' not found for speaker '{speaker.name}'. Create it by correcting an emotion first.",
        )
    return speaker, profile


@router.patch("/speakers/{speaker_id}/emotion-profiles/{emotion_category}/threshold")
async def set_emotion_profile_threshold(
    speaker_id: int,
    emotion_category: str,
    threshold: Optional[float] = Query(
        None, ge=0.3, le=1.0,
        description="Emotion-match threshold (0.3-1.0) or null to fall back to speaker/global"
    ),
    db: Session = Depends(get_db),
):
    """Set the per-emotion confidence threshold applied to emotion2vec matches."""
    speaker, profile = _get_speaker_emotion_profile(speaker_id, emotion_category, db)
    profile.confidence_threshold = threshold
    db.commit()
    return {
        "message": f"Updated {emotion_category} emotion threshold for '{speaker.name}'",
        "speaker_name": speaker.name,
        "emotion_category": emotion_category,
        "threshold": threshold,
    }


@router.patch("/speakers/{speaker_id}/emotion-profiles/{emotion_category}/voice-threshold")
async def set_emotion_profile_voice_threshold(
    speaker_id: int,
    emotion_category: str,
    threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0,
        description="Voice-profile match threshold (0.0-1.0) or null to fall back to speaker/global"
    ),
    db: Session = Depends(get_db),
):
    """Set the per-emotion voice-profile threshold (Detector 2, 512-D pyannote embeddings)."""
    speaker, profile = _get_speaker_emotion_profile(speaker_id, emotion_category, db)
    profile.voice_threshold = threshold
    db.commit()
    return {
        "message": f"Updated {emotion_category} voice threshold for '{speaker.name}'",
        "speaker_name": speaker.name,
        "emotion_category": emotion_category,
        "voice_threshold": threshold,
    }
