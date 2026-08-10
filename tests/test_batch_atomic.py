"""Atomic batch-output publication tests."""

import pytest


def test_atomic_write_replaces_only_after_complete_success(tmp_path):
    from batch_process import _atomic_write

    target = tmp_path / "result.txt"
    target.write_text("old")

    def fail_after_partial(stream):
        stream.write("partial")
        raise RuntimeError("injected failure")

    with pytest.raises(RuntimeError, match="injected"):
        _atomic_write(str(target), fail_after_partial, encoding="utf-8")

    assert target.read_text() == "old"
    assert list(tmp_path.glob("result.txt.*.tmp")) == []


def test_atomic_write_publishes_complete_content(tmp_path):
    from batch_process import _atomic_write

    target = tmp_path / "result.txt"
    _atomic_write(str(target), lambda stream: stream.write("complete"), encoding="utf-8")
    assert target.read_text() == "complete"


def test_real_worker_resumes_after_crash_between_transcript_and_json(
    tmp_path, monkeypatch
):
    """Fault-inject the real worker where JSON (the resume marker) publishes."""
    import json
    import logging
    import queue
    import sys
    from pathlib import Path

    from pydub import AudioSegment

    import batch_process
    from app import diarization

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source = input_dir / "sample.mp3"
    source.write_bytes(b"stub input")

    class FakeAudio:
        def __len__(self):
            return 1_000

        def export(self, path, format):
            assert format == "wav"
            Path(path).write_bytes(b"stub wav")

    class FakeEngine:
        calls = 0
        diarization_pipeline = object()
        embedding_model = object()
        whisper_model = object()
        emotion_model = object()

        def transcribe_with_diarization(self, *_args, **_kwargs):
            type(self).calls += 1
            return {
                "num_speakers": 1,
                "segments": [{
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "Speaker",
                    "text": "hello",
                    "confidence": 0.9,
                }],
            }

    monkeypatch.setattr(diarization, "SpeakerRecognitionEngine", FakeEngine)
    monkeypatch.setattr(
        AudioSegment,
        "from_file",
        staticmethod(lambda _path: FakeAudio()),
    )

    original_atomic_write = batch_process._atomic_write

    def fail_json_publish(path, writer, *, encoding=None):
        if str(path).endswith(".json"):
            raise RuntimeError("injected crash before completion marker")
        return original_atomic_write(path, writer, encoding=encoding)

    def run_worker():
        file_queue = queue.Queue()
        result_queue = queue.Queue()
        file_queue.put(str(source))
        file_queue.put(None)
        stdout, stderr = sys.stdout, sys.stderr
        try:
            batch_process.worker_main(
                0,
                file_queue,
                result_queue,
                [],
                str(input_dir),
                str(output_dir),
                0.35,
                str(tmp_path),
            )
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            logging.disable(logging.NOTSET)

    monkeypatch.setattr(batch_process, "_atomic_write", fail_json_publish)
    run_worker()
    txt_path = output_dir / "results" / "sample.txt"
    json_path = output_dir / "results" / "sample.json"
    assert txt_path.exists()
    assert not json_path.exists()

    # The missing JSON marker must cause the production worker to regenerate
    # and publish both outputs on the next run rather than skip the input.
    monkeypatch.setattr(batch_process, "_atomic_write", original_atomic_write)
    run_worker()
    assert FakeEngine.calls == 2
    assert "hello" in txt_path.read_text()
    assert json.loads(json_path.read_text())["segments"][0]["text"] == "hello"
