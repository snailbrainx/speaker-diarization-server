"""Regression tests for the streaming finalisation race (OPUS-002/GLM-001/QWEN-005).

At pinned SHA 700976f, stop_recording() returned while async segment handlers
were still scheduled-but-not-finished; num_segments increments were lost under
concurrency. After the fix: stop_recording blocks until handlers complete, and
num_segments increments are atomic SQL updates.
"""
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pytest

from app.streaming_recorder import StreamingRecorder


@pytest.fixture(autouse=True)
def _stub_services(tmp_path, monkeypatch):
    """Stable temp data dir for the recorder (never the real ./data)."""
    from app import services
    monkeypatch.setattr(services, "data_path", lambda: str(tmp_path))
    yield


class TestStopRecordingWaitsForAsyncHandlers:
    def _drive_one_segment(self, recorder):
        loud = np.full(16000, 0.5, dtype=np.float32)   # 1 s speech
        silence = np.zeros(3200, dtype=np.float32)      # 0.2 s silence
        recorder.process_audio_chunk((16000, loud))
        recorder.process_audio_chunk((16000, silence))

    def test_stop_blocks_until_async_handler_finished(self, monkeypatch):
        """Exercise the production callback bridge, not a hand-copied sketch."""
        state = {"finished": 0}

        async def scenario():
            from app import streaming_websocket

            loop = asyncio.get_running_loop()
            recorder = StreamingRecorder(max_workers=1, sample_rate=16000)
            recorder.silence_duration = 0.05

            async def handler(_websocket, _conversation_id, _seg, _engine):
                await asyncio.sleep(0.3)  # stand-in for GPU transcription
                state["finished"] += 1

            monkeypatch.setattr(streaming_websocket, "_handle_segment_processed", handler)
            monkeypatch.setattr(streaming_websocket, "get_engine", lambda: object())
            recorder.on_segment_processed = streaming_websocket._make_segment_callback(
                object(), 4242, loop
            )
            recorder.start_recording(4242)
            await asyncio.to_thread(self._drive_one_segment, recorder)
            await asyncio.sleep(0.05)
            await asyncio.to_thread(recorder.stop_recording)
            recorder.cleanup()
            return recorder

        recorder = asyncio.run(scenario())
        assert state["finished"] == recorder.segments_queued == 1, (
            f"stop_recording returned with {state['finished']}/{recorder.segments_queued} "
            "async handlers finished — finalisation would use incomplete data"
        )

    def test_handler_timeout_releases_stop_and_counts_failure(self, monkeypatch):
        async def scenario():
            from app import streaming_websocket

            loop = asyncio.get_running_loop()
            recorder = StreamingRecorder(max_workers=1, sample_rate=16000)
            recorder.silence_duration = 0.05

            async def stalled_handler(*_args):
                await asyncio.sleep(60)

            monkeypatch.setattr(streaming_websocket, "_handle_segment_processed", stalled_handler)
            monkeypatch.setattr(streaming_websocket, "get_engine", lambda: object())
            monkeypatch.setattr(streaming_websocket, "SEGMENT_HANDLER_TIMEOUT_SECONDS", 0.05)
            recorder.on_segment_processed = streaming_websocket._make_segment_callback(
                object(), 4242, loop
            )
            recorder.start_recording(4242)
            await asyncio.to_thread(self._drive_one_segment, recorder)
            await asyncio.wait_for(asyncio.to_thread(recorder.stop_recording), timeout=1.0)
            stats = recorder.get_stats()
            recorder.cleanup()
            return stats

        stats = asyncio.run(scenario())
        assert stats["segments_processed"] == 1
        assert stats["segments_failed"] == 1

    def test_dedicated_finalize_pool_breaks_default_pool_cycle(self):
        """Two stops may each need inner default-executor work without deadlock."""
        async def scenario():
            from app.streaming_websocket import _stop_and_concatenate

            loop = asyncio.get_running_loop()
            loop.set_default_executor(ThreadPoolExecutor(max_workers=2))

            class NestedRecorder:
                def stop_recording(self):
                    inner = asyncio.run_coroutine_threadsafe(
                        asyncio.to_thread(lambda: None), loop
                    )
                    inner.result(timeout=1.0)

                def get_stats(self):
                    return {"segments_failed": 0}

                def concatenate_segments(self):
                    return None

            await asyncio.wait_for(
                asyncio.gather(
                    _stop_and_concatenate(NestedRecorder()),
                    _stop_and_concatenate(NestedRecorder()),
                ),
                timeout=2.0,
            )

        asyncio.run(scenario())

    def test_num_segments_atomic_increment(self, tmp_path):
        """Concurrent segment handlers must not lose increments (was RMW race)."""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app.database import Base
        from app.models import Conversation

        engine = create_engine(f"sqlite:///{tmp_path}/conv.db")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)

        db0 = Session()
        conv = Conversation(title="t", start_time=datetime.now(), status="recording")  # noqa: DTZ005 (schema uses naive UTC)
        db0.add(conv)
        db0.commit()
        conv_id = conv.id
        db0.close()

        def add_segments(n):
            db = Session()
            for _ in range(n):
                db.query(Conversation).filter(Conversation.id == conv_id).update(
                    {"num_segments": Conversation.num_segments + 1},
                    synchronize_session=False,
                )
                db.commit()
            db.close()

        threads = [threading.Thread(target=add_segments, args=(5,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        db = Session()
        final = db.query(Conversation).filter(Conversation.id == conv_id).first().num_segments
        db.close()
        engine.dispose()
        assert final == 20, f"lost increments: stored {final}, expected 20"


class TestRecorderHardening:
    def test_nan_chunk_dropped_not_buffered(self):
        recorder = StreamingRecorder(max_workers=1, sample_rate=16000)
        recorder.start_recording(1)
        nan_chunk = np.full(1600, np.nan, dtype=np.float32)
        result = recorder.process_audio_chunk((16000, nan_chunk))
        assert result["status"] == "recording"
        assert result["buffer_size"] == 0, "non-finite audio must not enter the buffer"
        recorder.cleanup()

    def test_buffer_cap_force_flushes(self, tmp_path):
        recorder = StreamingRecorder(max_workers=1, sample_rate=16000)
        recorder.MAX_BUFFER_SECONDS = 0.5
        recorder.silence_duration = 999  # never flush on silence
        recorder.start_recording(2)
        loud = np.full(16000, 0.5, dtype=np.float32)  # 1 s each
        recorder.process_audio_chunk((16000, loud))
        recorder.process_audio_chunk((16000, loud))
        # Second 1 s chunk pushed buffer past the 0.5 s cap -> forced flush
        assert recorder.segments_queued >= 1, "buffer cap must force a flush"
        stats = recorder.get_stats()
        assert stats["buffered_samples"] == sum(len(chunk) for chunk in recorder.current_buffer)
        assert stats["buffered_samples"] == 0
        recorder.cleanup()
