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
    from app import services, streaming_recorder
    monkeypatch.setattr(services, "data_path", lambda: str(tmp_path))
    monkeypatch.setattr(streaming_recorder, "data_path", lambda: str(tmp_path))
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

            async def handler(
                _websocket, _conversation_id, _processing_token, _seg, _engine
            ):
                await asyncio.sleep(0.3)  # stand-in for GPU transcription
                state["finished"] += 1

            monkeypatch.setattr(streaming_websocket, "_handle_segment_processed", handler)
            monkeypatch.setattr(streaming_websocket, "get_engine", lambda: object())
            recorder.on_segment_processed = streaming_websocket._make_segment_callback(
                object(), 4242, "test-token", loop
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
                object(), 4242, "test-token", loop
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

    def test_timed_out_handler_finishing_late_cannot_write_after_finalization(
        self, tmp_path, monkeypatch
    ):
        """Exercise recorder callback → real handler → real finalizer lease gate."""
        from types import SimpleNamespace

        from sqlalchemy import create_engine, event
        from sqlalchemy.orm import sessionmaker

        from app import streaming_websocket
        from app.database import Base
        from app.models import Conversation, ConversationSegment

        engine_db = create_engine(
            f"sqlite:///{tmp_path}/late-writer.db",
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(engine_db, "connect")
        def _pragmas(dbapi_conn, _record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        Base.metadata.create_all(engine_db)
        Session = sessionmaker(bind=engine_db)
        db = Session()
        token = "stream-generation-token"
        conversation = Conversation(
            title="late writer",
            start_time=datetime.now(),  # noqa: DTZ005 - schema stores naive UTC
            status="recording",
            processing_token=token,
        )
        db.add(conversation)
        db.commit()
        conversation_id = conversation.id

        class BlockingEngine:
            def __init__(self):
                self.entered = threading.Event()
                self.release = threading.Event()
                self.finished = threading.Event()

            def transcribe_with_diarization(self, *_args, **_kwargs):
                self.entered.set()
                assert self.release.wait(timeout=3)
                self.finished.set()
                return {
                    "segments": [{
                        "start": 0.0,
                        "end": 1.0,
                        "text": "must be discarded",
                        "speaker": "Unknown_01",
                        "is_known": False,
                    }],
                    "num_speakers": 1,
                }

            def clear_gpu_cache(self):
                return None

            def clear_gpu_cache_async(self, _reason):
                return None

        fake_engine = BlockingEngine()
        monkeypatch.setattr(streaming_websocket, "SessionLocal", Session)
        monkeypatch.setattr(streaming_websocket, "get_engine", lambda: fake_engine)
        monkeypatch.setattr(
            streaming_websocket,
            "get_config",
            lambda: SimpleNamespace(
                get_settings=lambda: SimpleNamespace(speaker_threshold=0.35)
            ),
        )
        monkeypatch.setattr(
            streaming_websocket, "SEGMENT_HANDLER_TIMEOUT_SECONDS", 0.05
        )

        async def scenario():
            loop = asyncio.get_running_loop()
            recorder = StreamingRecorder(max_workers=1, sample_rate=16000)
            recorder.silence_duration = 999
            recorder.on_segment_processed = streaming_websocket._make_segment_callback(
                object(), conversation_id, token, loop
            )
            recorder.start_recording(conversation_id)
            loud = np.full(16000, 0.5, dtype=np.float32)
            recorder.process_audio_chunk((16000, loud))

            await streaming_websocket._finalize_recording(
                conversation_id,
                token,
                recorder,
                conversation,
                db,
                None,
            )
            assert fake_engine.entered.is_set()
            fake_engine.release.set()
            assert await asyncio.to_thread(fake_engine.finished.wait, 2)
            await asyncio.sleep(0.1)
            recorder.cleanup()

        asyncio.run(scenario())
        db.expire_all()
        stored = db.query(Conversation).filter_by(id=conversation_id).one()
        assert stored.status == "completed_with_errors"
        assert stored.processing_token is None
        assert stored.num_segments == 0
        assert db.query(ConversationSegment).filter_by(
            conversation_id=conversation_id
        ).count() == 0
        db.close()
        engine_db.dispose()

    def test_concatenation_failure_is_not_reported_as_success(
        self, tmp_path, monkeypatch
    ):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker

        from app import streaming_websocket
        from app.database import Base
        from app.models import Conversation
        from app.streaming_recorder import ConcatenationResult

        engine_db = create_engine(f"sqlite:///{tmp_path}/concat-failure.db")
        Base.metadata.create_all(engine_db)
        Session = sessionmaker(bind=engine_db)
        db = Session()
        token = "concat-token"
        conversation = Conversation(
            title="concat failure",
            start_time=datetime.now(),  # noqa: DTZ005
            status="recording",
            processing_token=token,
        )
        db.add(conversation)
        db.commit()
        conversation_id = conversation.id

        class FailedRecorder:
            def stop_recording(self):
                return None

            def get_stats(self):
                return {"segments_failed": 0}

            def concatenate_segments(self):
                return ConcatenationResult(status="failed", error="disk full")

        class Socket:
            client_state = streaming_websocket.WebSocketState.CONNECTED

            def __init__(self):
                self.messages = []

            async def send_json(self, message):
                self.messages.append(message)

        class Engine:
            def clear_gpu_cache(self):
                return None

        socket = Socket()
        monkeypatch.setattr(streaming_websocket, "get_engine", lambda: Engine())
        asyncio.run(streaming_websocket._finalize_recording(
            conversation_id,
            token,
            FailedRecorder(),
            conversation,
            db,
            socket,
        ))
        db.expire_all()
        stored = db.query(Conversation).filter_by(id=conversation_id).one()
        assert stored.status == "completed_with_errors"
        assert stored.audio_path is None
        assert stored.processing_token is None
        completed = [m for m in socket.messages if m["type"] == "completed"]
        assert completed[0]["data"]["concatenation_status"] == "failed"
        assert "failed" in completed[0]["data"]["message"]
        db.close()
        engine_db.dispose()

    def test_bounded_send_closes_non_reading_socket(self, monkeypatch):
        from app import streaming_websocket

        class StuckSocket:
            client_state = streaming_websocket.WebSocketState.CONNECTED

            def __init__(self):
                self.closed = False

            async def send_json(self, _message):
                await asyncio.Event().wait()

            async def close(self, code):
                assert code == 1011
                self.closed = True

        socket = StuckSocket()
        monkeypatch.setattr(streaming_websocket, "WS_SEND_TIMEOUT_SECONDS", 0.01)
        sent = asyncio.run(streaming_websocket._send_message_bounded(
            socket, "status", {"ok": True}
        ))
        assert sent is False
        assert socket.closed is True


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

    def test_configured_buffer_cap_reaches_runtime(self, monkeypatch):
        monkeypatch.setenv("MAX_STREAM_BUFFER_SECONDS", "7.5")
        recorder = StreamingRecorder(max_workers=1, sample_rate=16000)
        assert recorder.MAX_BUFFER_SECONDS == 7.5
        recorder.cleanup()

    def test_empty_and_large_finite_frames_emit_finite_json_levels(self):
        recorder = StreamingRecorder(max_workers=1, sample_rate=16000)
        recorder.start_recording(9)
        empty = recorder.process_audio_chunk(
            (16000, np.array([], dtype=np.float32))
        )
        huge = recorder.process_audio_chunk(
            (16000, np.array([np.finfo(np.float32).max], dtype=np.float32))
        )
        assert empty["audio_level"] == 0.0
        assert np.isfinite(huge["audio_level"])
        recorder.cleanup()
