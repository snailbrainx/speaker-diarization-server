"""
WebSocket endpoint for real-time audio streaming and transcription.
Integrates with StreamingRecorder for live speaker diarization.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from starlette.websockets import WebSocketState
from sqlalchemy.orm import Session
import numpy as np
import json
import asyncio
import logging
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from typing import Optional

from .database import SessionLocal, get_db, utc_now
from .diarization import LIVE_SPEAKER_TRACKING
from .live_tracking import SessionTracks
from .models import Conversation, ConversationSegment, Speaker
from .streaming_recorder import StreamingRecorder
from .config import get_config
from .conversation_lifecycle import (
    begin_finalization,
    claim_segment_persistence,
    fail_processing_lease,
    new_processing_token,
)
from .services import create_segment_from_result, load_known_speakers
import os

logger = logging.getLogger(__name__)


def _timeout_from_env(name: str, default: float, minimum: float) -> float:
    """Parse a timeout env var; a malformed or non-positive value must degrade
    to the default rather than crash startup or disable streaming waits."""
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %s", name, raw, default)
        return default
    if not math.isfinite(value) or value < minimum:
        logger.warning("%s must be finite and >= %s; using %s", name, minimum, default)
        return default
    return value


SEGMENT_HANDLER_TIMEOUT_SECONDS = _timeout_from_env("SEGMENT_HANDLER_TIMEOUT_SECONDS", 600.0, 1.0)
WS_SEND_TIMEOUT_SECONDS = _timeout_from_env("WS_SEND_TIMEOUT_SECONDS", 10.0, 0.1)
_FINALIZE_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="stream-finalize")


def convert_numpy_to_native(obj):
    """Recursively convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_to_native(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# WebSocket streams use 48 kHz float32 PCM by default — same rate as the browser's
# MediaRecorder default. The start message may pick another: {"sample_rate": 16000,
# "format": "int16"} is a sixth of the bandwidth. The models resample internally.
WS_SAMPLE_RATE = 48000
WS_FORMATS = {"float32": np.float32, "int16": np.int16}
# Noise floor below which a frame is treated as silence. Tuned against typical
# near-field microphones on a quiet channel; adjust in StreamingRecorder config
# if you need different behaviour per-deployment.
WS_SILENCE_THRESHOLD = 0.005

router = APIRouter(prefix="/streaming", tags=["Streaming"])


def get_engine():
    """Get shared speaker recognition engine (preloaded at startup)"""
    from .api import get_engine as get_api_engine
    return get_api_engine()


async def send_message(websocket: WebSocket, message_type: str, data: dict) -> bool:
    """Send one JSON message and report whether it reached the socket."""
    try:
        if websocket.client_state != WebSocketState.CONNECTED:
            logger.info(f"⚪ WebSocket not connected, skipping {message_type} message")
            return False
        message = {
            "type": message_type,
            "data": data,
            "timestamp": utc_now().isoformat(),
        }
        logger.info(
            "🔌 Sending WebSocket message: type=%s, data_keys=%s",
            message_type,
            list(data.keys()) if isinstance(data, dict) else "not-dict",
        )
        await websocket.send_json(message)
        logger.info(f"✅ Successfully sent {message_type} message")
        return True
    except WebSocketDisconnect:
        logger.info(
            f"⚪ Client disconnected, skipping {message_type} message "
            "(expected during shutdown)"
        )
        return False
    except Exception as exc:  # noqa: BLE001 - socket implementations vary
        logger.error(f"Error sending {message_type} message: {exc}")
        return False


async def _close_websocket_bounded(websocket: WebSocket) -> None:
    try:
        if websocket.client_state == WebSocketState.CONNECTED:
            await asyncio.wait_for(
                websocket.close(code=1011),
                timeout=WS_SEND_TIMEOUT_SECONDS,
            )
    except (asyncio.TimeoutError, RuntimeError, WebSocketDisconnect):
        logger.info("WebSocket close did not complete cleanly")


async def _send_message_bounded(
    websocket: WebSocket,
    message_type: str,
    data: dict,
) -> bool:
    """Deliver with bounded backpressure; close a socket that stops reading."""
    try:
        sent = await asyncio.wait_for(
            send_message(websocket, message_type, data),
            timeout=WS_SEND_TIMEOUT_SECONDS,
        )
        if not sent:
            await _close_websocket_bounded(websocket)
        return sent
    except asyncio.TimeoutError:
        logger.warning("Timed out sending websocket %s message", message_type)
        await _close_websocket_bounded(websocket)
        return False


def _make_segment_callback(
    websocket: WebSocket,
    conversation_id: int,
    processing_token: str,
    loop,
    diarize: bool = True,
    tracks: Optional[SessionTracks] = None,
):
    """Bridge the recorder worker to the async handler with a hard deadline."""
    def segment_callback(seg_info):
        coroutine = _handle_segment_processed(
            websocket,
            conversation_id,
            processing_token,
            seg_info,
            get_engine(),
            diarize=diarize,
            tracks=tracks,
        )
        try:
            future = asyncio.run_coroutine_threadsafe(coroutine, loop)
        except RuntimeError:
            # Scheduling failed before ownership transferred to the event loop.
            coroutine.close()
            logger.info("Segment %s dropped: event loop closed", seg_info.get("id"))
            raise RuntimeError("event loop closed before segment handler could be scheduled")

        try:
            future.result(timeout=SEGMENT_HANDLER_TIMEOUT_SECONDS)
        except FutureTimeoutError as exc:
            # A running asyncio.to_thread task cannot be safely killed. Leave
            # the handler future alive so it owns its DB session to completion,
            # but release recorder finalisation and record the timeout.
            future.add_done_callback(
                lambda done: done.exception() if not done.cancelled() else None
            )
            raise TimeoutError(
                f"Segment {seg_info.get('id')} exceeded "
                f"{SEGMENT_HANDLER_TIMEOUT_SECONDS}s processing deadline"
            ) from exc
        except Exception:
            # Let StreamingRecorder count the failed segment and mark the
            # conversation completed_with_errors during finalisation.
            raise

    return segment_callback


async def _stop_and_concatenate(recorder: StreamingRecorder):
    """Drain a recorder without consuming the loop's default executor."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_FINALIZE_EXECUTOR, recorder.stop_recording)
    stats = recorder.get_stats()
    audio_path = await loop.run_in_executor(
        _FINALIZE_EXECUTOR, recorder.concatenate_segments
    )
    return stats, audio_path


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time audio streaming.

    Protocol:
    - Client → Server: Binary audio chunks (ArrayBuffer)
    - Server → Client: JSON messages (status, segment, error)
    """
    await websocket.accept()
    conversation_id: Optional[int] = None
    conversation: Optional[Conversation] = None
    recorder: Optional[StreamingRecorder] = None
    processing_token: Optional[str] = None

    try:
        # Wait for initial "start" message
        init_message = await websocket.receive_json()

        if init_message.get("type") != "start":
            await _send_message_bounded(
                websocket, "error", {"message": "Expected 'start' message"}
            )
            await _close_websocket_bounded(websocket)
            return
        sample_rate = init_message.get("sample_rate") or WS_SAMPLE_RATE
        sample_dtype = WS_FORMATS.get(init_message.get("format") or "float32")
        if sample_dtype is None or not isinstance(sample_rate, int) or not 8000 <= sample_rate <= 48000:
            await _send_message_bounded(websocket, "error", {
                "message": "Unsupported audio: format must be float32 or int16, sample_rate 8000-48000"})
            await _close_websocket_bounded(websocket)
            return
        sample_size = np.dtype(sample_dtype).itemsize

        # Create conversation
        processing_token = new_processing_token()
        conversation = Conversation(
            title=f"Live Recording {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            start_time=utc_now(),
            status="recording",
            processing_token=processing_token,
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        conversation_id = conversation.id
        assert conversation_id is not None

        # Initialize recorder
        config = get_config()
        settings = config.get_settings()

        recorder = StreamingRecorder(
            sample_rate=sample_rate,
            silence_threshold=WS_SILENCE_THRESHOLD,
            silence_duration=settings.silence_duration,
        )

        # Get event loop for scheduling async tasks from background threads
        loop = asyncio.get_running_loop()

        # Block until persistence completes, but through a bounded bridge so a
        # stalled client or engine cannot hang finalisation forever.
        # "diarize": false in the start message: one person talking, no speaker diarization per chunk
        # LIVE_SPEAKER_TRACKING: the session's speakers are tracked across chunks and named from all their speech
        recorder.on_segment_processed = _make_segment_callback(
            websocket, conversation_id, processing_token, loop, diarize=bool(init_message.get("diarize", True)),
            tracks=SessionTracks(conversation_id) if LIVE_SPEAKER_TRACKING else None,
        )

        recorder.start_recording(conversation_id)

        # Load speaker cache for fast matching (avoids DB queries per segment)
        engine = get_engine()
        cache_size = engine.load_speaker_cache(db)
        logger.info(f"🚀 Speaker cache loaded: {cache_size} profiles ready for streaming")

        # Send confirmation
        started_sent = await _send_message_bounded(websocket, "started", {
            "conversation_id": conversation_id,
            "sample_rate": sample_rate,
            "message": "Recording started"
        })
        if not started_sent:
            await _finalize_recording(
                conversation_id, processing_token, recorder, conversation, db, None
            )
            return

        # Main loop: receive and process audio chunks
        while True:
            try:
                # Receive message
                message = await websocket.receive()

                if "bytes" in message:
                    # Binary audio chunk (PCM in the start message's format)
                    audio_bytes = message["bytes"]
                    logger.info(f"📦 Received audio chunk: {len(audio_bytes)} bytes")

                    # Clients occasionally send truncated frames — skip instead of crashing.
                    if len(audio_bytes) % sample_size != 0:
                        logger.info(f"⚠️ Dropping misaligned audio chunk ({len(audio_bytes)} bytes)")
                        continue

                    audio_array = np.frombuffer(audio_bytes, dtype=sample_dtype)
                    logger.info(f"🔊 Converted to audio array: {len(audio_array)} samples")

                    # Process chunk (StreamingRecorder expects tuple of (sample_rate, audio_data))
                    result = recorder.process_audio_chunk((sample_rate, audio_array))
                    logger.info(f"📊 VAD: {result['speech_detected']}, Level: {result['audio_level']:.3f}")

                    # Send status update
                    status_sent = await _send_message_bounded(websocket, "status", {
                        "vad_active": result["speech_detected"],
                        "audio_level": float(result["audio_level"]),
                        "stats": {
                            "buffer_size": result.get("buffer_size", 0),
                            "segments_processed": result.get("segments_processed", 0),
                            "total_audio_seconds": float(result.get("cumulative_offset", 0.0)),
                        }
                    })
                    if not status_sent:
                        break

                elif "text" in message:
                    # JSON message (e.g., stop command)
                    data = json.loads(message["text"])

                    if data.get("type") == "stop":
                        # Stop recording
                        break

            except WebSocketDisconnect:
                logger.info(f"WebSocket disconnected for conversation {conversation_id}")
                break
            except Exception as e:
                logger.error(f"Error processingaudio chunk: {e}")
                import traceback
                traceback.print_exc()
                await _send_message_bounded(
                    websocket, "error", {"message": str(e)}
                )
                break  # Exit loop on error

        # Cleanup: stop recording and finalize
        await _finalize_recording(
            conversation_id,
            processing_token,
            recorder,
            conversation,
            db,
            websocket,
        )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected during initialization")
        if conversation_id and processing_token and recorder and conversation:
            await _finalize_recording(
                conversation_id, processing_token, recorder, conversation, db, None
            )
    except Exception as e:
        logger.info(f"WebSocket error: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await _send_message_bounded(websocket, "error", {"message": str(e)})
        if conversation_id and processing_token and recorder and conversation:
            await _finalize_recording(
                conversation_id, processing_token, recorder, conversation, db, None
            )
    finally:
        # Close WebSocket if still open (ignore if already closed)
        await _close_websocket_bounded(websocket)


async def _handle_segment_processed(
    websocket: WebSocket,
    conversation_id: int,
    processing_token: str,
    segment_info: dict,
    engine,
    diarize: bool = True,
    tracks: Optional[SessionTracks] = None,
):
    """
    Callback when StreamingRecorder finishes processing a segment.
    Runs diarization + transcription (off the event loop), saves to DB, sends to client.
    With tracks (LIVE_SPEAKER_TRACKING), a track that has just been named has its earlier segments relabelled and
    the client is told with a "speaker_update" message.

    Owns its own DB session: the request-scoped session held by the WebSocket
    handler is not safe to share with this thread-scheduled coroutine, and the
    blocking GPU call below runs in a worker thread that also needs to touch the
    session for speaker/emotion profile lookups.

    The session's tracks change only with a saved chunk: if anything fails before the commit they go back to how
    they were (the chunk's segment ids, new Unknowns and renames were rolled back, and SQLite can reuse those ids).
    """
    db = SessionLocal()
    tracks_before = tracks.snapshot() if tracks is not None else None
    committed = False
    try:
        # Get conversation
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id,
            Conversation.processing_token == processing_token,
            Conversation.status == "recording",
        ).first()

        if not conversation:
            return

        # Get segment file path
        segment_file = segment_info["segment_file"]
        start_offset = segment_info["start_offset"]

        if not os.path.exists(segment_file):
            logger.info(f"Segment file not found: {segment_file}")
            return

        known_speakers = load_known_speakers(db)

        # Get threshold from config
        config = get_config()
        settings = config.get_settings()
        threshold = settings.speaker_threshold

        # Process with diarization + transcription — heavy GPU work, run off the event loop
        result = await asyncio.to_thread(
            engine.transcribe_with_diarization,
            segment_file,
            known_speakers,
            threshold=threshold,
            db_session=db,
            # A live chunk holds a few seconds of speech: never cluster mode; the session's tracks (if on) name
            # each voice from all its speech so far, else every segment is matched on its own.
            cluster_speakers=False,
            diarize=diarize,
            tracks=tracks,
        )

        # Timeout only stops the recorder waiting; inference can still finish
        # later. Acquire a conditional write lease before creating any rows.
        # This succeeds before finalisation (which then waits for our commit),
        # or fails after finalisation has revoked the generation token.
        if not claim_segment_persistence(db, conversation_id, processing_token):
            logger.warning(
                "Discarding late segment %s after conversation %s finalized",
                segment_info.get("id"),
                conversation_id,
            )
            return

        # Save segments to database
        conv_start = conversation.start_time
        segments_data = []
        _save_tracks(db, result["segments"], engine)

        logger.info(f"📝 Processing {len(result['segments'])} segment(s) from transcription")

        for seg in result["segments"]:
            segment = create_segment_from_result(
                seg=seg,
                conversation_id=conversation_id,
                conv_start=conv_start,
                db=db,
                threshold=threshold,
                segment_audio_path=segment_file,
                start_offset_base=start_offset,
            )
            db.flush()
            if seg.get("track") is not None:
                seg["track"].segment_ids.append(segment.id)

            # Build response data from the created segment object
            emotion_conf = segment.emotion_confidence
            detector_breakdown = seg.get("detector_breakdown")

            segments_data.append({
                "segment_id": segment.id,
                "speaker_name": segment.speaker_name,
                "text": segment.text,
                "start_offset": float(segment.start_offset),
                "end_offset": float(segment.end_offset),
                "confidence": float(segment.confidence) if segment.confidence is not None else 0.0,
                "emotion_category": segment.emotion_category,
                "emotion_confidence": float(emotion_conf) if emotion_conf is not None else None,
                "detector_breakdown": convert_numpy_to_native(detector_breakdown) if detector_breakdown else None,
                "is_known": seg.get("is_known", False),
                "words": segment.words or [],
                "avg_logprob": segment.avg_logprob
            })

        # after this chunk's segments exist, so cleaning up orphaned Unknowns cannot take a new track's Unknown
        speaker_updates = _relabel_tracks(db, conversation_id, tracks, result["segments"], engine)

        # Update conversation stats with an atomic SQL increment. A
        # read-modify-write on the ORM object across independent sessions
        # loses concurrent increments (two segment handlers → stored value 1).
        db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).update(
            {"num_segments": Conversation.num_segments + len(result["segments"])},
            synchronize_session=False,
        )

        db.commit()
        committed = True

        # Send segments to client with bounded backpressure. Persistence is
        # already committed; a stuck browser must not pin recorder shutdown.
        logger.info(f"📤 Sending {len(segments_data)} segment(s) to client")
        for update in speaker_updates:
            await _send_message_bounded(websocket, "speaker_update", update)
        for seg_data in segments_data:
            logger.info(f"   → Segment: {seg_data['speaker_name']}: {seg_data['text'][:50]}...")
            await _send_message_bounded(websocket, "segment", seg_data)

        # Queue async GPU cleanup (non-blocking)
        engine.clear_gpu_cache_async("segment_complete")

    except Exception as e:
        logger.error(f"Error processingsegment: {e}")
        import traceback
        traceback.print_exc()
        db.rollback()
        if tracks is not None and not committed:
            tracks.restore(tracks_before)
        engine.clear_speaker_cache()  # reload from what is committed
        await _send_message_bounded(websocket, "error", {"message": "Segment processing error"})
        raise
    finally:
        db.close()


def _save_tracks(db: Session, segments: list, engine) -> None:
    """Live tracks of this chunk: new voices get their Unknown and own Unknowns their grown voice (saved here, inside
    the persistence lease, never during processing), and every segment takes its track's speaker."""
    tracks = list({id(s["track"]): s["track"] for s in segments if s.get("track") is not None}.values())
    if not tracks:
        return
    names = dict(db.query(Speaker.id, Speaker.name).filter(Speaker.id.in_([t.speaker_id for t in tracks])).all())
    for track in tracks:
        if track.speaker_id is not None and track.speaker_id not in names:   # deleted while this chunk ran
            track.speaker_id, track.own_unknown = None, False
    engine.save_track_voices(tracks, db)
    db.flush()
    names = dict(db.query(Speaker.id, Speaker.name).filter(Speaker.id.in_([t.speaker_id for t in tracks])).all())
    for seg in segments:
        track = seg.get("track")
        if track is None:
            continue
        if track.speaker_id in names:
            seg["speaker_id"], seg["speaker"] = track.speaker_id, names[track.speaker_id]
            seg["is_known"] = not seg["speaker"].startswith("Unknown_")
        else:   # provisional: a plain Unknown for now
            seg.pop("speaker_id", None)
            seg["speaker"], seg["is_known"] = "Unknown", False


def _relabel_tracks(db: Session, conversation_id: int, tracks: Optional[SessionTracks], segments: list,
                    engine) -> list:
    """Live tracks of this chunk that now have a speaker take their earlier plain-"Unknown" (provisional) segments;
    Unknowns of this session that have just become a named or matched speaker (tracks.renames) take this session's
    segments that carry them, and the Unknown, left without segments, is deleted. Returns one "speaker_update"
    message per change: {segment_ids, speaker_id, speaker_name, is_known}."""
    updates = []
    for track in {id(s["track"]): s["track"] for s in segments if s.get("track") is not None}.values():
        if track.speaker_id is None or not track.segment_ids:
            continue
        rows = db.query(ConversationSegment).filter(ConversationSegment.conversation_id == conversation_id,
                                                    ConversationSegment.id.in_(track.segment_ids),
                                                    ConversationSegment.speaker_id.is_(None),
                                                    ConversationSegment.speaker_name == "Unknown")
        ids = [r.id for r in rows]
        if ids:
            speaker = db.query(Speaker).filter(Speaker.id == track.speaker_id).first()
            rows.update({"speaker_id": speaker.id, "speaker_name": speaker.name}, synchronize_session=False)
            updates.append({"segment_ids": ids, "speaker_id": speaker.id, "speaker_name": speaker.name,
                            "is_known": not speaker.name.startswith("Unknown_")})
    while tracks is not None and tracks.renames:
        old, new = tracks.renames.pop(0)
        speaker = db.query(Speaker).filter(Speaker.id == new).first()
        if speaker is None:
            continue
        rows = db.query(ConversationSegment).filter(ConversationSegment.conversation_id == conversation_id,
                                                    ConversationSegment.speaker_id == old)
        ids = [r.id for r in rows]
        rows.update({"speaker_id": speaker.id, "speaker_name": speaker.name}, synchronize_session=False)
        db.flush()
        if not db.query(ConversationSegment.id).filter(ConversationSegment.speaker_id == old).first():
            db.query(Speaker).filter(Speaker.id == old, Speaker.name.like("Unknown_%")).delete(synchronize_session=False)
        if ids:
            updates.append({"segment_ids": ids, "speaker_id": speaker.id, "speaker_name": speaker.name,
                            "is_known": not speaker.name.startswith("Unknown_")})
        logger.info(f"🔁 Live session: {len(ids)} earlier segment(s) relabelled to {speaker.name}")
    if updates:
        db.flush()
        engine.clear_speaker_cache()
    return updates


async def _finalize_recording(
    conversation_id: int,
    processing_token: str,
    recorder: StreamingRecorder,
    conversation: Conversation,
    db: Session,
    websocket: Optional[WebSocket]
):
    """Drain workers and publish final state under the streaming lease."""
    try:
        logger.info(f"Finalizing recording for conversation {conversation_id}")

        # Use a dedicated pool: holding a default-executor slot here while the
        # segment coroutine itself awaits asyncio.to_thread creates a classic
        # pool-saturation deadlock under concurrent stops.
        recorder_stats, concatenation = await _stop_and_concatenate(recorder)

        # Conditional UPDATE both revokes late writers and obtains SQLite's
        # write lock.  A handler that acquired the lock first commits before
        # this point; one that finishes inference later sees a revoked token and
        # discards its result. Keep this transaction open through final status.
        if not begin_finalization(db, conversation_id, processing_token):
            logger.info(
                "Conversation %s was already finalized or lost its processing lease",
                conversation_id,
            )
            return

        db.expire_all()
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()
        if conversation is None:
            db.rollback()
            return

        concatenation_failed = concatenation.status in {"partial", "failed"}
        if concatenation.status in {"success", "partial"}:
            if concatenation.path and os.path.exists(concatenation.path):
                # Keep WAV (avoids pyannote's MP3 boundary inaccuracies).
                conversation.audio_path = concatenation.path
                conversation.audio_format = "wav"
            else:
                concatenation_failed = True

        segment_errors = recorder_stats.get("segments_failed", 0)
        processing_errors = segment_errors + int(concatenation_failed)
        conversation.status = (
            "completed_with_errors" if processing_errors else "completed"
        )
        conversation.end_time = utc_now()

        last_segment = db.query(ConversationSegment).filter(
            ConversationSegment.conversation_id == conversation_id
        ).order_by(ConversationSegment.end_offset.desc()).first()
        if last_segment:
            conversation.duration = last_segment.end_offset

        conversation.num_speakers = db.query(
            ConversationSegment.speaker_name
        ).filter(
            ConversationSegment.conversation_id == conversation_id
        ).distinct().count()

        # Commits token revocation and all final metadata atomically.
        db.commit()

        if concatenation.status == "partial":
            completion_message = (
                "Recording completed, but full audio contains only readable segments"
            )
        elif concatenation_failed:
            completion_message = "Recording completed, but full audio publication failed"
        elif segment_errors:
            completion_message = "Recording completed with segment-processing errors"
        elif concatenation.status == "empty":
            completion_message = "Recording completed with no speech segments"
        else:
            completion_message = "Recording completed and saved"

        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            await _send_message_bounded(websocket, "completed", {
                "conversation_id": conversation_id,
                "status": conversation.status,
                "num_segments": conversation.num_segments,
                "num_speakers": conversation.num_speakers,
                "processing_errors": processing_errors,
                "concatenation_status": concatenation.status,
                "duration": conversation.duration,
                "message": completion_message,
            })

        logger.info(
            "Recording finalized: %s segments, %s speakers, %s errors",
            conversation.num_segments,
            conversation.num_speakers,
            processing_errors,
        )

        try:
            get_engine().clear_gpu_cache()
        except Exception as exc:  # noqa: BLE001 - cleanup cannot un-finalize data
            logger.warning("Post-finalization GPU cleanup failed: %s", exc)

    except Exception as e:
        logger.error(f"Error finalizing recording: {e}")
        import traceback
        traceback.print_exc()
        fail_processing_lease(db, conversation_id, processing_token)

        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            await _send_message_bounded(
                websocket, "error", {"message": "Finalization error"}
            )
