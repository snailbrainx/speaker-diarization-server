"""
Streaming audio recorder with queue-based background processing.
"""
import logging
import numpy as np
import os
import queue
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor, wait as futures_wait
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .services import data_path

logger = logging.getLogger(__name__)


class StreamingRecorder:
    """Handles continuous audio streaming with background processing."""

    MIN_SEGMENT_SECONDS = 0.5
    LONG_SEGMENT_WARN_SECONDS = 60.0

    def __init__(
        self,
        sample_rate: int = 48000,
        silence_threshold: float = 0.005,
        silence_duration: Optional[float] = None,
        max_workers: int = 1,
    ):
        """
        Args:
            sample_rate: audio sample rate (Hz)
            silence_threshold: RMS energy threshold for VAD
            silence_duration: seconds of silence before flushing a segment
                (falls back to config.silence_duration if None)
            max_workers: background processing threads. Kept at 1 by default
                because the pyannote+Whisper engine holds the GIL through its
                CUDA calls, so extra workers just queue without parallelism.
        """
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold

        from .config import get_config
        settings = get_config().get_settings()
        self.silence_duration = silence_duration if silence_duration is not None else settings.silence_duration

        # State
        self.is_recording = False
        self.conversation_id: Optional[int] = None

        # Audio buffering
        self.current_buffer: List[np.ndarray] = []
        self.last_speech_time = time.time()
        self.chunk_count = 0
        self.speech_detected = False

        self.cumulative_offset = 0.0  # seconds of audio flushed so far

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_futures: List = []
        # Guards current_buffer, total_segments, segments_queued, segment_paths,
        # cumulative_offset, and last_speech_time against concurrent flushes
        # (process_audio_chunk from the WS thread vs. stop_recording's tail flush).
        self._lock = threading.Lock()

        # Callbacks
        self.on_segment_processed: Optional[Callable] = None
        self.on_audio_level: Optional[Callable] = None

        # Stats
        self.total_segments = 0
        self.segments_queued = 0
        self.segments_processed = 0

        # Paths of flushed segments (for later concatenation)
        self.segment_paths: List[str] = []

    def _segments_dir(self) -> str:
        return os.path.join(data_path(), "stream_segments", f"conv_{self.conversation_id}")

    def start_recording(self, conversation_id: int):
        """Start recording for a conversation."""
        # Pick up any runtime change to silence_duration
        from .config import get_config
        self.silence_duration = get_config().get_settings().silence_duration

        self.is_recording = True
        self.conversation_id = conversation_id
        self.current_buffer = []
        self.last_speech_time = time.time()
        self.chunk_count = 0
        self.total_segments = 0
        self.segments_processed = 0
        self.segments_queued = 0
        self.segment_paths = []
        self.cumulative_offset = 0.0
        self.processing_futures = []

        os.makedirs(self._segments_dir(), exist_ok=True)
        logger.info(f"🎤 Streaming recorder started for conversation {conversation_id} (silence: {self.silence_duration}s)")

    def stop_recording(self):
        """Stop recording and wait for all queued segments to finish processing.

        Safe to call from a worker thread (via asyncio.to_thread).
        """
        logger.info("⏹️ Stopping streaming recorder...")

        # Flush any remaining buffer under the lock (races with process_audio_chunk)
        with self._lock:
            if self.current_buffer:
                self._flush_locked()

        self.is_recording = False

        # Wait on the actual futures — the counter approach deadlocked if a
        # worker raised before incrementing, and .exception() surfaces errors.
        futures = list(self.processing_futures)
        logger.info(f"⏳ Waiting for {len(futures)} segment(s) to finish processing...")
        done, _ = futures_wait(futures)
        for fut in done:
            err = fut.exception()
            if err:
                logger.info(f"⚠️ Segment worker raised: {err!r}")

        logger.info("✅ Streaming recorder stopped")

    def process_audio_chunk(self, audio_chunk: tuple) -> Dict:
        """Process an incoming audio chunk.

        Args:
            audio_chunk: (sample_rate, audio_data) tuple.
        """
        if not self.is_recording or audio_chunk is None:
            return {
                "status": "not_recording",
                "audio_level": 0.0,
                "segments_queued": 0,
                "segments_processed": 0,
            }

        _, audio_data = audio_chunk

        # Mono-down if needed
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32) / 32768.0

        energy = float(np.sqrt(np.mean(audio_data ** 2)))

        if self.on_audio_level:
            self.on_audio_level(energy)

        with self._lock:
            self.current_buffer.append(audio_data)
            self.chunk_count += 1

            if energy > self.silence_threshold:
                self.last_speech_time = time.time()
                self.speech_detected = True
            else:
                self.speech_detected = False
                silence_elapsed = time.time() - self.last_speech_time
                if silence_elapsed >= self.silence_duration and len(self.current_buffer) > 10:
                    self._flush_locked()

            buffer_size = len(self.current_buffer)
            cumulative_offset = self.cumulative_offset

        return {
            "status": "recording",
            "audio_level": energy,
            "speech_detected": self.speech_detected,
            "segments_queued": self.segments_queued,
            "segments_processed": self.segments_processed,
            "buffer_size": buffer_size,
            "cumulative_offset": cumulative_offset,
        }

    def _flush_locked(self):
        """Flush current_buffer to a segment file. Caller must hold self._lock."""
        if not self.current_buffer:
            return

        segment_audio = np.concatenate(self.current_buffer)
        # Always clear the buffer and reset the silence clock — even if we
        # decide not to flush, so pure-silence clients don't spin.
        self.current_buffer = []
        self.last_speech_time = time.time()

        duration = len(segment_audio) / self.sample_rate
        if duration < self.MIN_SEGMENT_SECONDS:
            logger.info(f"⏭️ Skipping segment (too short: {duration:.2f}s)")
            return
        if duration > self.LONG_SEGMENT_WARN_SECONDS:
            logger.info(f"⚠️ Long segment detected: {duration:.1f}s - processing may take longer")

        avg_energy = float(np.sqrt(np.mean(segment_audio ** 2)))
        if avg_energy < self.silence_threshold * 2:
            logger.info(f"⏭️ Skipping segment (mostly silence, energy: {avg_energy:.4f})")
            return

        # Normalize
        max_val = float(np.abs(segment_audio).max())
        if max_val > 0:
            segment_audio = segment_audio * (0.9 / max_val)

        segment_id = self.total_segments + 1
        segment_path = os.path.join(self._segments_dir(), f"seg_{segment_id:04d}.wav")

        with wave.open(segment_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes((segment_audio * 32767).astype(np.int16).tobytes())

        start_offset = self.cumulative_offset
        end_offset = self.cumulative_offset + duration
        self.cumulative_offset = end_offset

        segment_info = {
            "id": segment_id,
            "path": segment_path,
            "segment_file": segment_path,
            "conversation_id": self.conversation_id,
            "duration": duration,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "timestamp": datetime.now(),
        }

        self.segment_paths.append(segment_path)
        self.total_segments += 1
        self.segments_queued += 1

        future = self.executor.submit(self._process_segment_worker, segment_info)
        self.processing_futures.append(future)

        logger.info(f"📦 Queued segment {segment_id} ({duration:.1f}s, offset {start_offset:.1f}-{end_offset:.1f}s)")

    def _process_segment_worker(self, segment_info: Dict):
        """Background worker for a single segment. Increments segments_processed
        in finally so stop_recording can no longer deadlock on an exception."""
        try:
            if self.on_segment_processed:
                self.on_segment_processed(segment_info)
        except Exception as e:
            logger.error(f"Error processingsegment {segment_info['id']}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self._lock:
                self.segments_processed += 1
                done = self.segments_processed
                total = self.segments_queued
            logger.info(f"✅ Processed segment {segment_info['id']} ({done}/{total})")

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "is_recording": self.is_recording,
                "total_segments": self.total_segments,
                "segments_queued": self.segments_queued,
                "segments_processed": self.segments_processed,
                "buffer_chunks": len(self.current_buffer),
            }

    def concatenate_segments(self) -> Optional[str]:
        """Stream all segment WAV files into a single conversation WAV.

        Streams frame-by-frame instead of loading the whole recording into
        memory — a 2-hour mono 48 kHz session is ~700 MB otherwise.
        """
        if not self.segment_paths or self.conversation_id is None:
            return None

        try:
            logger.info(f"🔗 Concatenating {len(self.segment_paths)} segments...")

            recordings_dir = os.path.join(data_path(), "recordings")
            os.makedirs(recordings_dir, exist_ok=True)
            output_path = os.path.join(recordings_dir, f"conv_{self.conversation_id}_full.wav")

            out_wf = None
            total_frames = 0
            try:
                for seg_path in self.segment_paths:
                    if not os.path.exists(seg_path):
                        logger.info(f"⚠️ Segment not found: {seg_path}")
                        continue

                    with wave.open(seg_path, "rb") as in_wf:
                        if out_wf is None:
                            out_wf = wave.open(output_path, "wb")
                            out_wf.setnchannels(in_wf.getnchannels())
                            out_wf.setsampwidth(in_wf.getsampwidth())
                            out_wf.setframerate(in_wf.getframerate())
                            sample_rate = in_wf.getframerate()

                        frames = in_wf.readframes(in_wf.getnframes())
                        out_wf.writeframes(frames)
                        total_frames += in_wf.getnframes()
            finally:
                if out_wf is not None:
                    out_wf.close()

            if total_frames == 0:
                logger.info("⚠️ No valid segments to concatenate")
                # Remove empty file if one was created
                if os.path.exists(output_path) and os.path.getsize(output_path) == 0:
                    os.remove(output_path)
                return None

            duration = total_frames / sample_rate
            logger.info(f"✅ Concatenated conversation saved: {output_path} ({duration:.1f}s)")
            return output_path

        except Exception as e:
            logger.info(f"❌ Error concatenating segments: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup(self):
        """Release the thread pool."""
        self.executor.shutdown(wait=True)
