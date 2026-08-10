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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime
from typing import Optional

from .database import SessionLocal, get_db, utc_now
from .models import Conversation, ConversationSegment
from .streaming_recorder import StreamingRecorder
from .config import get_config
from .services import create_segment_from_result, load_known_speakers
import os

logger = logging.getLogger(__name__)

SEGMENT_HANDLER_TIMEOUT_SECONDS = float(os.getenv("SEGMENT_HANDLER_TIMEOUT_SECONDS") or "600")
WS_SEND_TIMEOUT_SECONDS = float(os.getenv("WS_SEND_TIMEOUT_SECONDS") or "10")
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

# WebSocket streams use 48 kHz float32 PCM — same rate as the browser's
# MediaRecorder default. The StreamingRecorder resamples internally.
WS_SAMPLE_RATE = 48000
# Noise floor below which a frame is treated as silence. Tuned against typical
# near-field microphones on a quiet channel; adjust in StreamingRecorder config
# if you need different behaviour per-deployment.
WS_SILENCE_THRESHOLD = 0.005

router = APIRouter(prefix="/streaming", tags=["Streaming"])


def get_engine():
    """Get shared speaker recognition engine (preloaded at startup)"""
    from .api import get_engine as get_api_engine
    return get_api_engine()


async def send_message(websocket: WebSocket, message_type: str, data: dict):
    """Send JSON message to WebSocket client"""
    try:
        # Check if WebSocket is still connected
        if websocket.client_state == WebSocketState.CONNECTED:
            message = {
                "type": message_type,
                "data": data,
                "timestamp": utc_now().isoformat()
            }
            logger.info(f"🔌 Sending WebSocket message: type={message_type}, data_keys={list(data.keys()) if isinstance(data, dict) else 'not-dict'}")
            await websocket.send_json(message)
            logger.info(f"✅ Successfully sent {message_type} message")
        else:
            logger.info(f"⚪ WebSocket not connected, skipping {message_type} message")
    except WebSocketDisconnect:
        # Expected during stop/cleanup - client disconnected before we could send
        logger.info(f"⚪ Client disconnected, skipping {message_type} message (expected during shutdown)")
    except Exception as e:
        # Unexpected errors
        logger.error(f"ERROR sending{message_type} message: {e}")
        import traceback
        traceback.print_exc()


async def _send_message_bounded(websocket: WebSocket, message_type: str, data: dict):
    """Best-effort websocket delivery that cannot stall persistence/finalise."""
    try:
        await asyncio.wait_for(
            send_message(websocket, message_type, data),
            timeout=WS_SEND_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.warning("Timed out sending websocket %s message", message_type)


def _make_segment_callback(websocket: WebSocket, conversation_id: int, loop):
    """Bridge the recorder worker to the async handler with a hard deadline."""
    def segment_callback(seg_info):
        coroutine = _handle_segment_processed(
            websocket, conversation_id, seg_info, get_engine()
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
    recorder: Optional[StreamingRecorder] = None

    try:
        # Wait for initial "start" message
        init_message = await websocket.receive_json()

        if init_message.get("type") != "start":
            await send_message(websocket, "error", {"message": "Expected 'start' message"})
            await websocket.close()
            return

        # Create conversation
        conversation = Conversation(
            title=f"Live Recording {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            start_time=utc_now(),
            status="recording"
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
            sample_rate=WS_SAMPLE_RATE,
            silence_threshold=WS_SILENCE_THRESHOLD,
            silence_duration=settings.silence_duration,
        )

        # Get event loop for scheduling async tasks from background threads
        loop = asyncio.get_running_loop()

        # Block until persistence completes, but through a bounded bridge so a
        # stalled client or engine cannot hang finalisation forever.
        recorder.on_segment_processed = _make_segment_callback(
            websocket, conversation_id, loop
        )

        recorder.start_recording(conversation_id)

        # Load speaker cache for fast matching (avoids DB queries per segment)
        engine = get_engine()
        cache_size = engine.load_speaker_cache(db)
        logger.info(f"🚀 Speaker cache loaded: {cache_size} profiles ready for streaming")

        # Send confirmation
        await send_message(websocket, "started", {
            "conversation_id": conversation_id,
            "sample_rate": WS_SAMPLE_RATE,
            "message": "Recording started"
        })

        # Main loop: receive and process audio chunks
        while True:
            try:
                # Receive message
                message = await websocket.receive()

                if "bytes" in message:
                    # Binary audio chunk (float32 PCM, 4 bytes/sample)
                    audio_bytes = message["bytes"]
                    logger.info(f"📦 Received audio chunk: {len(audio_bytes)} bytes")

                    # Clients occasionally send truncated frames — skip instead of crashing.
                    if len(audio_bytes) % 4 != 0:
                        logger.info(f"⚠️ Dropping misaligned audio chunk ({len(audio_bytes)} bytes)")
                        continue

                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    logger.info(f"🔊 Converted to audio array: {len(audio_array)} samples")

                    # Process chunk (StreamingRecorder expects tuple of (sample_rate, audio_data))
                    result = recorder.process_audio_chunk((WS_SAMPLE_RATE, audio_array))
                    logger.info(f"📊 VAD: {result['speech_detected']}, Level: {result['audio_level']:.3f}")

                    # Send status update
                    await send_message(websocket, "status", {
                        "vad_active": result["speech_detected"],
                        "audio_level": float(result["audio_level"]),
                        "stats": {
                            "buffer_size": result.get("buffer_size", 0),
                            "segments_processed": result.get("segments_processed", 0),
                            "total_audio_seconds": float(result.get("cumulative_offset", 0.0)),
                        }
                    })

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
                try:
                    await send_message(websocket, "error", {"message": str(e)})
                except Exception:
                    pass  # Connection already closed
                break  # Exit loop on error

        # Cleanup: stop recording and finalize
        await _finalize_recording(conversation_id, recorder, conversation, db, websocket)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected during initialization")
        if conversation_id and recorder:
            await _finalize_recording(conversation_id, recorder, conversation, db, None)
    except Exception as e:
        logger.info(f"WebSocket error: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await send_message(websocket, "error", {"message": str(e)})
    finally:
        # Close WebSocket if still open (ignore if already closed)
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close()
        except RuntimeError:
            # WebSocket already closed, ignore
            pass


async def _handle_segment_processed(
    websocket: WebSocket,
    conversation_id: int,
    segment_info: dict,
    engine
):
    """
    Callback when StreamingRecorder finishes processing a segment.
    Runs diarization + transcription (off the event loop), saves to DB, sends to client.

    Owns its own DB session: the request-scoped session held by the WebSocket
    handler is not safe to share with this thread-scheduled coroutine, and the
    blocking GPU call below runs in a worker thread that also needs to touch the
    session for speaker/emotion profile lookups.
    """
    db = SessionLocal()
    try:
        # Get conversation
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_id
        ).first()

        if not conversation:
            return

        # Get segment file path
        segment_file = segment_info["segment_file"]
        start_offset = segment_info["start_offset"]
        end_offset = segment_info["end_offset"]

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
        )

        # Save segments to database
        conv_start = conversation.start_time
        segments_data = []

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
                engine=engine,
            )
            db.flush()

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
                "words": seg.get("words", []),
                "avg_logprob": segment.avg_logprob
            })

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

        # Send segments to client with bounded backpressure. Persistence is
        # already committed; a stuck browser must not pin recorder shutdown.
        logger.info(f"📤 Sending {len(segments_data)} segment(s) to client")
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
        await _send_message_bounded(websocket, "error", {"message": "Segment processing error"})
        raise
    finally:
        db.close()


async def _finalize_recording(
    conversation_id: int,
    recorder: StreamingRecorder,
    conversation: Conversation,
    db: Session,
    websocket: Optional[WebSocket]
):
    """
    Finalize recording: stop recorder, concatenate segments, convert to MP3.
    """
    try:
        logger.info(f"Finalizing recording for conversation {conversation_id}")

        # Use a dedicated pool: holding a default-executor slot here while the
        # segment coroutine itself awaits asyncio.to_thread creates a classic
        # pool-saturation deadlock under concurrent stops.
        recorder_stats, full_audio_path = await _stop_and_concatenate(recorder)

        # Refresh BEFORE mutating: the per-segment callback owns its own Session
        # and commits num_segments from a worker thread, so this handler's cached
        # instance is stale. Refresh() overwrites every attribute, so any dirty
        # change made before this line would be silently discarded.
        db.refresh(conversation)

        if full_audio_path and os.path.exists(full_audio_path):
            # Keep WAV file (no MP3 conversion - WAV avoids pyannote 24ms boundary bug)
            conversation.audio_path = full_audio_path
            conversation.audio_format = "wav"

        processing_errors = recorder_stats.get("segments_failed", 0)
        conversation.status = "completed_with_errors" if processing_errors else "completed"
        conversation.end_time = utc_now()

        last_segment = db.query(ConversationSegment).filter(
            ConversationSegment.conversation_id == conversation_id
        ).order_by(ConversationSegment.end_offset.desc()).first()
        if last_segment:
            conversation.duration = last_segment.end_offset

        speaker_count = db.query(ConversationSegment.speaker_name).filter(
            ConversationSegment.conversation_id == conversation_id
        ).distinct().count()
        conversation.num_speakers = speaker_count

        db.commit()

        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            await _send_message_bounded(websocket, "completed", {
                "conversation_id": conversation_id,
                "status": conversation.status,
                "num_segments": conversation.num_segments,
                "num_speakers": conversation.num_speakers,
                "processing_errors": processing_errors,
                "duration": conversation.duration,
                "message": (
                    "Recording completed with segment-processing errors"
                    if processing_errors
                    else "Recording completed and saved"
                ),
            })

        logger.info(
            "Recording finalized: %s segments, %s speakers, %s errors",
            conversation.num_segments,
            conversation.num_speakers,
            processing_errors,
        )

        engine = get_engine()
        engine.clear_gpu_cache()

    except Exception as e:
        logger.info(f"Error finalizing recording: {e}")
        import traceback
        traceback.print_exc()
        conversation.status = "failed"
        db.commit()

        if websocket and websocket.client_state == WebSocketState.CONNECTED:
            await _send_message_bounded(websocket, "error", {"message": "Finalization error"})
