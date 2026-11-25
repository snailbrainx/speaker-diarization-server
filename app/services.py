"""
Shared service functions for segment creation, speaker management, and embedding operations.
Extracted from api.py, conversation_api.py, and streaming_websocket.py to eliminate duplication.
"""
import os
import json
import logging
import numpy as np
from datetime import timedelta
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import exists, func
from sqlalchemy.orm import Session, joinedload

from .models import AppMetadata, Speaker, ConversationSegment, SpeakerEmotionProfile
from .diarization import (CENTROID_MAX_WEIGHT_SEC, CLUSTER_SPEAKERS, MIN_SEGMENT_DURATION_SEC,
                          SPEAKER_EMBEDDING_MODEL, _centroid, _create_unknown_speaker, _in_memory,
                          auto_enroll_unknown_speaker)

logger = logging.getLogger(__name__)


def data_path() -> str:
    """Single source of truth for the data directory (audio, temp, recordings)."""
    return os.getenv("DATA_PATH", "./data")


def load_known_speakers(db: Session) -> List[Tuple[int, str, Any]]:
    """Return the (id, name, embedding) tuple list consumed by the engine."""
    speakers = db.query(Speaker).all()
    return [(s.id, s.name, s.get_embedding()) for s in speakers]


def resolve_audio_path(conversation, segment=None) -> str | None:
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


def uses_segment_audio_fallback(conversation, segment, audio_path: str | None) -> bool:
    """Return whether ``audio_path`` is the per-chunk streaming WAV.

    Segment offsets are conversation-relative. They are safe only with the full
    conversation recording; using them against a VAD chunk silently extracts
    the wrong tail of that file.
    """
    if not audio_path or not segment or not segment.segment_audio_path:
        return False
    return os.path.realpath(audio_path) == os.path.realpath(segment.segment_audio_path)


def create_segment_from_result(
    seg: dict,
    conversation_id: int,
    conv_start,
    db: Session,
    threshold: float,
    segment_audio_path: str = None,
    start_offset_base: float = 0.0,
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

    Returns:
        ConversationSegment (added to session but not committed)
    """
    speaker_id = None
    speaker_name = seg["speaker"]
    confidence = seg.get("confidence", 0.0)

    if seg.get("speaker_id") is not None:
        # Identified once for the whole diarization speaker (SPEAKER_ASSIGNMENT=cluster) or by its live track
        speaker_id = seg["speaker_id"]
    elif seg.get("enroll") is not None:
        # SPEAKER_ASSIGNMENT=cluster: a diarization speaker nobody matched becomes one new Unknown with its voice,
        # created when its first segment is saved; its other segments share the same "enroll" dict and speaker.
        enroll = seg["enroll"]
        if "speaker" not in enroll:
            enroll["speaker"] = _create_unknown_speaker(enroll["voice"], db)
        speaker_id, speaker_name = enroll["speaker"]
        confidence = 1.0
    elif seg.get("is_known"):
        speaker = db.query(Speaker).filter(Speaker.name == speaker_name).first()
        if speaker:
            speaker_id = speaker.id
    else:
        embedding = seg.get("embedding")
        if embedding is not None and speaker_name and speaker_name.startswith("Unknown_"):
            # (not added to the engine's shared cache: every engine reloads it once this is committed)
            speaker_id, speaker_name = auto_enroll_unknown_speaker(
                embedding, db, threshold=threshold
            )
            confidence = 1.0 if speaker_id else confidence

    words = seg.get("words")
    if words and start_offset_base:
        # A live chunk's word times count from the chunk; stored like the segment's, from the conversation start
        words = [dict(w, **{k: w[k] + start_offset_base for k in ("start", "end") if w.get(k) is not None})
                 for w in words]
    words_json = json.dumps(words) if words else None

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
        avg_logprob=seg.get("avg_logprob"),
        is_overlap=bool(seg.get("overlap")),
    )

    if seg.get("embedding") is not None:
        segment.set_speaker_embedding(seg["embedding"])
    if seg.get("emotion_embedding") is not None:
        segment.set_emotion_embedding(seg["emotion_embedding"])

    db.add(segment)
    return segment


def _invalidate_speaker_cache(engine) -> None:
    """Tell a streaming engine to forget its speaker-profile cache after a mutation.

    Safe to call with any engine-like object or None; just a noop if the engine
    doesn't expose clear_speaker_cache (e.g. in batch mode).
    """
    if engine is None:
        return
    clear = getattr(engine, "clear_speaker_cache", None)
    if clear is not None:
        clear()


def recalculate_speaker_embedding(
    speaker: Speaker,
    db: Session,
    engine,
) -> int:
    """
    Recalculate a speaker's embedding from all their non-misidentified segments.
    Uses cached embeddings where available, falls back to audio extraction.
    Segments recovered from overlapping speech are left out: their audio holds two voices.

    Returns:
        Number of embeddings used, or 0 if no valid segments found
    """
    _invalidate_speaker_cache(engine)
    # joinedload prevents an N+1 on seg.conversation when embeddings are missing.
    segments = (
        db.query(ConversationSegment)
        .options(joinedload(ConversationSegment.conversation))
        .filter(
            ConversationSegment.speaker_id == speaker.id,
            ConversationSegment.is_misidentified == False,
            ConversationSegment.is_overlap == False,   # two voices at once: never anyone's voice sample
        )
        .all()
    )

    if not segments:
        return 0

    if CLUSTER_SPEAKERS:
        # Build the profile the way cluster mode matches speakers: segments too short to embed reliably are left out
        # (the pipeline never embeds them; re-extracted they are noise), the rest weighted by duration.
        segments = [s for s in segments if s.end_offset - s.start_offset >= MIN_SEGMENT_DURATION_SEC]
        if not segments:
            return 0

    embeddings = []
    weights = []
    batch_segments = []

    for seg in segments:
        stored = seg.get_speaker_embedding()
        if stored is not None and not np.isnan(stored).any():
            embeddings.append(stored)
            weights.append(min(seg.end_offset - seg.start_offset, CENTROID_MAX_WEIGHT_SEC))
        else:
            audio_path = resolve_audio_path(seg.conversation, seg)
            if audio_path and not uses_segment_audio_fallback(seg.conversation, seg, audio_path):
                batch_segments.append({
                    'audio_file': audio_path,
                    'start_time': seg.start_offset,
                    'end_time': seg.end_offset
                })
            elif audio_path:
                logger.warning(
                    "Skipping speaker embedding re-extraction for segment %s: "
                    "only chunk audio is available for conversation-relative offsets",
                    seg.id,
                )

    if batch_segments:
        extracted = engine.extract_segment_embeddings_batch(batch_segments)
        for e, seg in zip(extracted, batch_segments):
            if e is not None and not np.isnan(e).any():
                embeddings.append(e)
                weights.append(min(seg['end_time'] - seg['start_time'], CENTROID_MAX_WEIGHT_SEC))

    if not embeddings:
        return 0

    speaker.set_embedding(_centroid(embeddings, weights) if CLUSTER_SPEAKERS else np.mean(embeddings, axis=0))
    return len(embeddings)


def emotion_correction_counts(db: Session, speaker_id: int) -> Dict[str, int]:
    """How many of a speaker's lines were corrected to each emotion and are learned from (the same rows the
    personal emotion calibration is fitted on)."""
    rows = (
        db.query(ConversationSegment.emotion_category, func.count(ConversationSegment.id))
        .filter(
            ConversationSegment.speaker_id == speaker_id,
            ConversationSegment.emotion_corrected == True,   # noqa: E712
            ConversationSegment.emotion_misidentified == False,   # noqa: E712
            ConversationSegment.is_misidentified == False,   # noqa: E712
            ConversationSegment.emotion_category.isnot(None),
        )
        .group_by(ConversationSegment.emotion_category)
        .all()
    )
    return {emotion: count for emotion, count in rows}


def delete_unknown_speakers(db: Session, engine=None, max_seconds: Optional[float] = None) -> Tuple[int, List[str]]:
    """
    Delete all speakers with names starting with 'Unknown_' (with max_seconds: only those whose segments last at most
    that long in all, the stray voices of a few words).
    Handles FK cleanup (nullify segments, delete emotion profiles).

    Returns:
        Tuple of (deleted_count, list of deleted names)
    """
    _invalidate_speaker_cache(engine)
    query = db.query(Speaker).filter(Speaker.name.like("Unknown_%"))
    if max_seconds is not None:
        speech = (db.query(func.coalesce(func.sum(ConversationSegment.end_offset - ConversationSegment.start_offset), 0.0))
                  .filter(ConversationSegment.speaker_id == Speaker.id).correlate(Speaker).scalar_subquery())
        query = query.filter(speech <= max_seconds)
    unknowns = query.all()
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


VOICE_MODEL_KEY = "speaker_embedding_model"
LEGACY_VOICE_MODEL = "pyannote/embedding"


def migrate_voice_embeddings(db: Session, engine, commit: bool = True) -> Optional[Dict[str, Any]]:
    """Stored voices made by another voice-embedding model (SPEAKER_EMBEDDING_MODEL changed, or a database from before
    the setting, made with pyannote/embedding) cannot be compared with new ones. Once: every segment of a speaker is
    embedded again from its recording, then each speaker's voice, and the voice part of each emotion profile, is
    rebuilt from those segments (emotion2vec embeddings do not depend on the voice model and are kept). A speaker
    whose recordings are gone keeps their name and segments but has no voice (NaN, skipped by matching) until they
    are identified again; their names are logged. Returns counts, or None when the stored voices are current.
    commit=False leaves the commit to the caller (a restore converts the restored voices in its own transaction)."""
    row = db.get(AppMetadata, VOICE_MODEL_KEY)
    has_voices = (db.query(Speaker.id).first() is not None
                  or db.query(ConversationSegment.id).filter(ConversationSegment.speaker_embedding.isnot(None)).first())
    made_by = row.value if row else (LEGACY_VOICE_MODEL if has_voices else SPEAKER_EMBEDDING_MODEL)
    if made_by == SPEAKER_EMBEDDING_MODEL:
        if row is None:
            db.add(AppMetadata(key=VOICE_MODEL_KEY, value=SPEAKER_EMBEDDING_MODEL))
            if commit:
                db.commit()
        return None
    logger.info(f"Voice profiles were made by {made_by}; rebuilding them with {SPEAKER_EMBEDDING_MODEL}...")
    db.query(ConversationSegment).update({ConversationSegment.speaker_embedding: None}, synchronize_session=False)
    counts: Dict[str, Any] = {"segments": 0, "speakers_rebuilt": 0, "speakers_without_audio": []}
    by_conversation: Dict[int, List[ConversationSegment]] = {}
    for seg in db.query(ConversationSegment).filter(ConversationSegment.speaker_id.isnot(None),
                                                    ConversationSegment.is_overlap == False).all():  # noqa: E712
        by_conversation.setdefault(seg.conversation_id, []).append(seg)
    for segments in by_conversation.values():
        path = resolve_audio_path(segments[0].conversation)
        if not path or uses_segment_audio_fallback(segments[0].conversation, segments[0], path):
            continue
        try:
            audio = _in_memory(path)
        except Exception as e:  # noqa: BLE001 - one unreadable recording must not stop the others (or startup)
            logger.warning(f"Voice rebuild: cannot read {path} ({e}); its segments are skipped")
            continue
        duration = audio["waveform"].shape[-1] / audio["sample_rate"]
        for seg in segments:
            if seg.end_offset > duration:   # past the end of the recording: no audio of its own
                continue
            emb = engine._segment_embedding_or_none(audio, {"start": seg.start_offset, "end": seg.end_offset})
            seg.set_speaker_embedding(emb)
            counts["segments"] += emb is not None
    db.flush()
    for speaker in db.query(Speaker).all():
        if recalculate_speaker_embedding(speaker, db, engine):
            counts["speakers_rebuilt"] += 1
        else:
            speaker.set_embedding(np.full(engine.embedding_dim, np.nan, dtype=np.float32))
            counts["speakers_without_audio"].append(speaker.name)
    if row is None:
        db.add(AppMetadata(key=VOICE_MODEL_KEY, value=SPEAKER_EMBEDDING_MODEL))
    else:
        row.value = SPEAKER_EMBEDDING_MODEL
    if commit:
        db.commit()
    else:
        db.flush()
    _invalidate_speaker_cache(engine)
    if counts["speakers_without_audio"]:
        logger.warning(f"Voice rebuild: no recordings left for {', '.join(counts['speakers_without_audio'])}: "
                       "identify each of them once in a new recording to recognise them again")
    logger.info(f"Voice profiles rebuilt: {counts['speakers_rebuilt']} speakers from {counts['segments']} segments")
    return counts


def cleanup_orphaned_unknowns(db: Session, engine=None) -> List[str]:
    """
    Delete Unknown_* speakers that have zero segments assigned.

    Returns:
        List of deleted speaker names
    """
    _invalidate_speaker_cache(engine)
    has_segments = exists().where(ConversationSegment.speaker_id == Speaker.id)
    orphans = db.query(Speaker).filter(
        Speaker.name.like("Unknown_%"),
        ~has_segments,
    ).all()

    deleted = [s.name for s in orphans]
    for speaker in orphans:
        db.delete(speaker)
    return deleted
