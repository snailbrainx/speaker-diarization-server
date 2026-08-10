"""Runtime configuration and unchanged-path tests.

These cover startup/default behaviour, dynamic Settings API reads, emotion-model
absence, and deployment-document consistency without loading any ML model.
"""
import threading
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_empty_context_padding_env_does_not_crash_engine(monkeypatch):
    from app import diarization

    monkeypatch.setenv("CONTEXT_PADDING", "")
    fake_config = SimpleNamespace(
        get_settings=lambda: SimpleNamespace(context_padding=0.15)
    )
    monkeypatch.setattr(diarization, "get_config", lambda: fake_config)
    monkeypatch.setattr(diarization.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(diarization.torch, "device", lambda name: name, raising=False)
    monkeypatch.setattr(diarization.threading.Thread, "start", lambda _thread: None)

    engine = diarization.SpeakerRecognitionEngine()
    assert engine.context_padding == 0.15


def test_embedding_uses_latest_runtime_context_padding(monkeypatch):
    from app import diarization

    state = {"padding": 0.25}
    fake_config = SimpleNamespace(
        get_settings=lambda: SimpleNamespace(context_padding=state["padding"])
    )
    monkeypatch.setattr(diarization, "get_config", lambda: fake_config)
    monkeypatch.setattr(
        diarization.sf,
        "info",
        lambda _path: SimpleNamespace(duration=10.0),
    )
    monkeypatch.setattr(diarization.torch, "no_grad", lambda: nullcontext())

    captured = []

    class FakeEmbeddingModel:
        def crop(self, _audio_file, segment):
            start = getattr(segment, "start", segment[0])
            end = getattr(segment, "end", segment[1])
            captured.append((start, end))
            return np.ones(4, dtype=np.float32)

    engine = diarization.SpeakerRecognitionEngine.__new__(
        diarization.SpeakerRecognitionEngine
    )
    engine._embedding_model = FakeEmbeddingModel()
    engine._model_lock = threading.Lock()
    engine.context_padding = 0.0

    engine.extract_segment_embedding("unused.wav", 1.0, 2.0)
    assert captured[-1] == (0.75, 2.25)

    state["padding"] = 0.40
    engine.extract_segment_embedding("unused.wav", 1.0, 2.0)
    assert captured[-1] == (0.60, 2.40)
    assert engine.context_padding == 0.40


def test_emotion_preload_skips_decode_when_model_unavailable(monkeypatch):
    from app import diarization

    engine = diarization.SpeakerRecognitionEngine.__new__(
        diarization.SpeakerRecognitionEngine
    )
    engine._emotion_model = None
    engine._emotion_model_failed = True
    engine._model_lock = threading.Lock()

    def forbidden_decode(_path):
        raise AssertionError("audio decode must not run without an emotion model")

    monkeypatch.setattr(diarization.AudioSegment, "from_file", forbidden_decode)
    assert engine._preload_emotion_audio("unused.wav") is None


def test_gpu_environment_documentation_matches_compose_variables():
    root = Path(__file__).resolve().parents[1]
    compose = (root / "docker-compose.yml").read_text()
    example = (root / ".env.example").read_text()

    assert "${GPU_COUNT:-all}" in compose
    assert "GPU_COUNT=1" in example
    assert "NVIDIA_VISIBLE_DEVICES=GPU-" in example
    assert "GPU_DEVICE_ID=" not in example
    for variable in (
        "MAX_STREAM_BUFFER_SECONDS",
        "SEGMENT_HANDLER_TIMEOUT_SECONDS",
        "WS_SEND_TIMEOUT_SECONDS",
    ):
        assert f"{variable}=${{{variable}:-" in compose
        assert f"{variable}=" in example
