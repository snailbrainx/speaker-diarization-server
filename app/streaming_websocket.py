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
from datetime import datetime, timedelta
from typing import Dict, Optional

from .database import SessionLocal, get_db
from .models import Conversation, ConversationSegment, Speaker
from .streaming_recorder import StreamingRecorder
from .config import get_config
from .services import create_segment_from_result
import os


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

router = APIRouter(prefix="/streaming", tags=["Streaming"])

# Active WebSocket connections (conversation_id -> WebSocket)
active_connections: Dict[int, WebSocket] = {}

# Active recorders (conversation_id -> StreamingRecorder)
active_recorders: Dict[int, StreamingRecorder] = {}


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
                "timestamp": datetime.utcnow().isoformat()
            }
            print(f"🔌 Sending WebSocket message: type={message_type}, data_keys={list(data.keys()) if isinstance(data, dict) else 'not-dict'}")
            await websocket.send_json(message)
            print(f"✅ Successfully sent {message_type} message")
        else:
            print(f"⚪ WebSocket not connected, skipping {message_type} message")
    except WebSocketDisconnect:
        # Expected during stop/cleanup - client disconnected before we could send
        print(f"⚪ Client disconnected, skipping {message_type} message (expected during shutdown)")
    except Exception as e:
        # Unexpected errors
        print(f"❌ ERROR sending {message_type} message: {e}")
        import traceback
        traceback.print_exc()


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
            start_time=datetime.utcnow(),
            status="recording"
        )
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

        conversation_id = conversation.id
        active_connections[conversation_id] = websocket

        # Initialize recorder
        config = get_config()
        settings = config.get_settings()

        recorder = StreamingRecorder(
            sample_rate=48000,
            silence_threshold=0.005,
            silence_duration=settings.silence_duration,
            max_workers=2
        )

        # Get event loop for scheduling async tasks from background threads
        loop = asyncio.get_running_loop()

        # Set callback after initialization
        # Use asyncio.run_coroutine_threadsafe to schedule from background thread.
        # The coroutine creates its own DB session — the request-scoped `db` is not
        # thread-safe and may conflict with the main handler's usage.
        def segment_callback(seg_info):
            asyncio.run_coroutine_threadsafe(
                _handle_segment_processed(websocket, conversation_id, seg_info, get_engine()),
                loop
            )

        recorder.on_segment_processed = segment_callback

        recorder.start_recording(conversation_id)
        active_recorders[conversation_id] = recorder

        # Load speaker cache for fast matching (avoids DB queries per segment)
        engine = get_engine()
        cache_size = engine.load_speaker_cache(db)
        print(f"🚀 Speaker cache loaded: {cache_size} profiles ready for streaming")

        # Send confirmation
        await send_message(websocket, "started", {
            "conversation_id": conversation_id,
            "sample_rate": 48000,
            "message": "Recording started"
        })

        # Main loop: receive and process audio chunks
        while True:
            try:
                # Receive message
                message = await websocket.receive()

                if "bytes" in message:
                    # Binary audio chunk
                    audio_bytes = message["bytes"]
                    print(f"📦 Received audio chunk: {len(audio_bytes)} bytes")

                    # Convert bytes to numpy array (assuming float32)
                    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    print(f"🔊 Converted to audio array: {len(audio_array)} samples")

                    # Process chunk (StreamingRecorder expects tuple of (sample_rate, audio_data))
                    result = recorder.process_audio_chunk((48000, audio_array))
                    print(f"📊 VAD: {result['speech_detected']}, Level: {result['audio_level']:.3f}")

                    # Send status update
                    await send_message(websocket, "status", {
                        "vad_active": result["speech_detected"],
                        "audio_level": float(result["audio_level"]),
                        "stats": {
                            "buffer_size": result.get("buffer_size", 0),
                            "segments_processed": result.get("segments_processed", 0),
                            "total_audio_seconds": result.get("segments_processed", 0) * 2.0  # Estimate
                        }
                    })

                elif "text" in message:
                    # JSON message (e.g., stop command)
                    data = json.loads(message["text"])

                    if data.get("type") == "stop":
                        # Stop recording
                        break

            except WebSocketDisconnect:
                print(f"WebSocket disconnected for conversation {conversation_id}")
                break
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
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
        print(f"WebSocket disconnected during initialization")
        if conversation_id and recorder:
            await _finalize_recording(conversation_id, recorder, conversation, db, None)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await send_message(websocket, "error", {"message": str(e)})
    finally:
        # Cleanup
        if conversation_id:
            active_connections.pop(conversation_id, None)
            active_recorders.pop(conversation_id, None)

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
            print(f"Segment file not found: {segment_file}")
            return

        # Get known speakers
        speakers = db.query(Speaker).all()
        known_speakers = [(s.id, s.name, s.get_embedding()) for s in speakers]

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

        print(f"📝 Processing {len(result['segments'])} segment(s) from transcription")

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

        # Update conversation stats (increment rather than re-count)
        conversation.num_segments = (conversation.num_segments or 0) + len(result["segments"])

        db.commit()

        # Send segments to client
        print(f"📤 Sending {len(segments_data)} segment(s) to client")
        for seg_data in segments_data:
            print(f"   → Segment: {seg_data['speaker_name']}: {seg_data['text'][:50]}...")
            await send_message(websocket, "segment", seg_data)

        # Queue async GPU cleanup (non-blocking)
        engine.clear_gpu_cache_async("segment_complete")

    except Exception as e:
        print(f"Error processing segment: {e}")
        await send_message(websocket, "error", {"message": "Segment processing error"})
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
        print(f"Finalizing recording for conversation {conversation_id}")

        # Stop recorder and wait for queue to finish
        recorder.stop_recording()

        # Concatenate segments
        full_audio_path = recorder.concatenate_segments()

        if full_audio_path and os.path.exists(full_audio_path):
            # Keep WAV file (no MP3 conversion - WAV avoids pyannote 24ms boundary bug)
            conversation.audio_path = full_audio_path
            conversation.audio_format = "wav"

        # Update conversation status
        conversation.status = "completed"
        conversation.end_time = datetime.utcnow()

        # Calculate duration
        if conversation.num_segments and conversation.num_segments > 0:
            last_segment = db.query(ConversationSegment).filter(
                ConversationSegment.conversation_id == conversation_id
            ).order_by(ConversationSegment.end_offset.desc()).first()

            if last_segment:
                conversation.duration = last_segment.end_offset

        # Count speakers
        speaker_count = db.query(ConversationSegment.speaker_name).filter(
            ConversationSegment.conversation_id == conversation_id
        ).distinct().count()
        conversation.num_speakers = speaker_count

        db.commit()

        # Send completion message
        if websocket and websocket.client_state.CONNECTED:
            await send_message(websocket, "completed", {
                "conversation_id": conversation_id,
                "num_segments": conversation.num_segments,
                "num_speakers": conversation.num_speakers,
                "duration": conversation.duration,
                "message": "Recording completed and saved"
            })

        print(f"Recording finalized: {conversation.num_segments} segments, {conversation.num_speakers} speakers")

        # Force GPU cleanup after recording stops (free up VRAM)
        engine = get_engine()
        print(f"🧹 Forcing GPU cleanup after recording stop...")
        engine.clear_gpu_cache()  # Blocking cleanup to free VRAM immediately
        print(f"✅ GPU cleanup complete")

    except Exception as e:
        print(f"Error finalizing recording: {e}")
        conversation.status = "failed"
        db.commit()

        if websocket and websocket.client_state.CONNECTED:
            await send_message(websocket, "error", {"message": f"Finalization error: {str(e)}"})
