"""
Shared service functions for segment creation, speaker management, and embedding operations.
Extracted from api.py, conversation_api.py, and streaming_websocket.py to eliminate duplication.
"""
import os
import json
import numpy as np
from datetime import timedelta
from typing import Optional, Tuple, List
from sqlalchemy.orm import Session

from .models import Speaker, ConversationSegment, SpeakerEmotionProfile


def resolve_audio_path(conversation, segment=None) -> Optional[str]:
    """
    Resolve the best audio file path for a segment.
    Prefers full conversation audio (where offsets are valid),
    falls back to segment audio file.

    Returns:
        Path string or None if no audio available
    """
    if conversation.audio_path and os.path.exists(conversation.audio_path):
        return conversation.audio_path
    if segment and segment.segment_audio_path and os.path.exists(segment.segment_audio_path):
        return segment.segment_audio_path
    return None


def create_segment_from_result(
    seg: dict,
    conversation_id: int,
    conv_start,
    db: Session,
    threshold: float,
    segment_audio_path: str = None,
    start_offset_base: float = 0.0,
    engine=None,
) -> ConversationSegment:
    """
    Create a ConversationSegment from a diarization result dict.

    Handles speaker identification, unknown auto-enrollment,
    word serialization, and embedding caching.

    Args:
        seg: Dict from transcribe_with_diarization result
        conversation_id: Parent conversation ID
        conv_start: Conversation start datetime
        db: Database session
        threshold: Speaker similarity threshold
        segment_audio_path: Optional path to segment WAV file (for streaming)
        start_offset_base: Offset to add to segment times (for streaming)
        engine: SpeakerRecognitionEngine (needed for cache updates during streaming)

    Returns:
        ConversationSegment (added to session but not committed)
    """
    from .diarization import auto_enroll_unknown_speaker

    speaker_id = None
    speaker_name = seg["speaker"]
    confidence = seg.get("confidence", 0.0)

    if seg.get("is_known"):
        speaker = db.query(Speaker).filter(Speaker.name == speaker_name).first()
        if speaker:
            speaker_id = speaker.id
    else:
        embedding = seg.get("embedding")
        if embedding is not None and speaker_name and speaker_name.startswith("Unknown_"):
            speaker_id, speaker_name = auto_enroll_unknown_speaker(
                embedding, db, threshold=threshold
            )
            # Add to speaker cache if engine provided (streaming mode)
            if speaker_id and engine and hasattr(engine, 'add_speaker_to_cache'):
                engine.add_speaker_to_cache(
                    speaker_id=speaker_id,
                    speaker_name=speaker_name,
                    embedding=embedding,
                    profile_type='general'
                )
            confidence = 1.0 if speaker_id else confidence

    words_json = json.dumps(seg["words"]) if seg.get("words") else None

    seg_start = start_offset_base + seg["start"]
    seg_end = start_offset_base + seg["end"]

    segment = ConversationSegment(
        conversation_id=conversation_id,
        speaker_id=speaker_id,
        speaker_name=speaker_name,
        text=seg.get("text", ""),
        start_time=conv_start + timedelta(seconds=seg_start),
        end_time=conv_start + timedelta(seconds=seg_end),
        start_offset=seg_start,
        end_offset=seg_end,
        confidence=confidence,
        emotion_category=seg.get("emotion_category"),
        emotion_confidence=seg.get("emotion_confidence"),
        detector_breakdown=json.dumps(seg["detector_breakdown"]) if seg.get("detector_breakdown") else None,
        segment_audio_path=segment_audio_path,
        words_data=words_json,
        avg_logprob=seg.get("avg_logprob")
    )

    if seg.get("embedding") is not None:
        segment.set_speaker_embedding(seg["embedding"])
    if seg.get("emotion_embedding") is not None:
        segment.set_emotion_embedding(seg["emotion_embedding"])

    db.add(segment)
    return segment


def recalculate_speaker_embedding(
    speaker: Speaker,
    db: Session,
    engine,
) -> int:
    """
    Recalculate a speaker's embedding from all their non-misidentified segments.
    Uses cached embeddings where available, falls back to audio extraction.

    Returns:
        Number of embeddings used, or 0 if no valid segments found
    """
    segments = db.query(ConversationSegment).filter(
        ConversationSegment.speaker_id == speaker.id,
        ConversationSegment.is_misidentified == False
    ).all()

    if not segments:
        return 0

    embeddings = []
    batch_segments = []

    for seg in segments:
        stored = seg.get_speaker_embedding()
        if stored is not None and not np.isnan(stored).any():
            embeddings.append(stored)
        else:
            audio_path = resolve_audio_path(seg.conversation, seg)
            if audio_path:
                batch_segments.append({
                    'audio_file': audio_path,
                    'start_time': seg.start_offset,
                    'end_time': seg.end_offset
                })

    if batch_segments:
        extracted = engine.extract_segment_embeddings_batch(batch_segments)
        embeddings.extend([e for e in extracted if e is not None and not np.isnan(e).any()])

    if not embeddings:
        return 0

    speaker.set_embedding(np.mean(embeddings, axis=0))
    return len(embeddings)


def recalculate_emotion_profile(
    speaker_id: int,
    emotion_category: str,
    db: Session,
    engine,
) -> Optional[str]:
    """
    Recalculate a speaker's emotion profile from all corrected, non-misidentified segments.

    Returns:
        "updated", "created", "deleted", or None if nothing changed
    """
    segments = db.query(ConversationSegment).filter(
        ConversationSegment.speaker_id == speaker_id,
        ConversationSegment.emotion_corrected == True,
        ConversationSegment.emotion_misidentified == False,
        ConversationSegment.emotion_category == emotion_category
    ).all()

    emotion_embeddings = []
    voice_embeddings = []

    for seg in segments:
        # Emotion embedding
        stored_emb = seg.get_emotion_embedding()
        if stored_emb is not None and not np.isnan(stored_emb).any():
            emotion_embeddings.append(stored_emb)
        else:
            audio_path = resolve_audio_path(seg.conversation, seg)
            if audio_path:
                try:
                    data = engine.extract_emotion(audio_path, seg.start_offset, seg.end_offset, extract_embedding=True)
                    if data and 'embedding' in data and not np.isnan(data['embedding']).any():
                        emotion_embeddings.append(data['embedding'])
                except Exception as e:
                    print(f"Warning: Could not extract emotion embedding for segment {seg.id}: {e}")

        # Voice embedding
        voice_emb = seg.get_speaker_embedding()
        if voice_emb is not None and not np.isnan(voice_emb).any():
            voice_embeddings.append(voice_emb)

    profile = db.query(SpeakerEmotionProfile).filter(
        SpeakerEmotionProfile.speaker_id == speaker_id,
        SpeakerEmotionProfile.emotion_category == emotion_category
    ).first()

    if emotion_embeddings:
        avg_emb = np.mean(emotion_embeddings, axis=0)
        avg_voice = np.mean(voice_embeddings, axis=0) if voice_embeddings else None

        if profile:
            profile.set_embedding(avg_emb)
            profile.sample_count = len(emotion_embeddings)
            if avg_voice is not None:
                profile.set_voice_embedding(avg_voice)
                profile.voice_sample_count = len(voice_embeddings)
            else:
                profile.set_voice_embedding(None)
                profile.voice_sample_count = 0
            return "updated"
        else:
            profile = SpeakerEmotionProfile(
                speaker_id=speaker_id,
                emotion_category=emotion_category,
                sample_count=len(emotion_embeddings),
                voice_sample_count=len(voice_embeddings) if voice_embeddings else 0
            )
            profile.set_embedding(avg_emb)
            if avg_voice is not None:
                profile.set_voice_embedding(avg_voice)
            db.add(profile)
            return "created"
    elif profile:
        db.delete(profile)
        return "deleted"

    return None


def delete_unknown_speakers(db: Session) -> Tuple[int, List[str]]:
    """
    Delete all speakers with names starting with 'Unknown_'.
    Handles FK cleanup (nullify segments, delete emotion profiles).

    Returns:
        Tuple of (deleted_count, list of deleted names)
    """
    unknowns = db.query(Speaker).filter(Speaker.name.like("Unknown_%")).all()
    if not unknowns:
        return 0, []

    ids = [s.id for s in unknowns]
    names = [s.name for s in unknowns]

    db.query(ConversationSegment).filter(
        ConversationSegment.speaker_id.in_(ids)
    ).update({"speaker_id": None}, synchronize_session=False)

    db.query(SpeakerEmotionProfile).filter(
        SpeakerEmotionProfile.speaker_id.in_(ids)
    ).delete(synchronize_session=False)

    for speaker in unknowns:
        db.delete(speaker)

    return len(unknowns), names


def cleanup_orphaned_unknowns(db: Session) -> List[str]:
    """
    Delete Unknown_* speakers that have zero segments assigned.

    Returns:
        List of deleted speaker names
    """
    unknowns = db.query(Speaker).filter(Speaker.name.like("Unknown_%")).all()
    deleted = []

    for speaker in unknowns:
        count = db.query(ConversationSegment).filter(
            ConversationSegment.speaker_id == speaker.id
        ).count()
        if count == 0:
            db.delete(speaker)
            deleted.append(speaker.name)

    return deleted
