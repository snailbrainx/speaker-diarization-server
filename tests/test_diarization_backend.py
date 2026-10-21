"""DIARIZATION_BACKEND=speakrs: speakrs' turns become the same diarization result as pyannote's."""
import threading
from types import SimpleNamespace

import numpy as np

import app.diarization as diarization
from app.diarization import SpeakerRecognitionEngine


def test_speakrs_turns_in_the_pyannote_result_format(monkeypatch):
    seen = []
    turn = lambda s, e, spk: SimpleNamespace(start=s, end=e, speaker=spk)  # noqa: E731

    class FakeSpeakrs:
        def diarize_samples(self, samples):
            seen.append(samples)
            return SimpleNamespace(segments=[turn(0.0, 2.0, "SPEAKER_00"), turn(1.5, 3.0, "SPEAKER_01")],
                                   exclusive_segments=[turn(0.0, 1.5, "SPEAKER_00"), turn(1.5, 3.0, "SPEAKER_01")])

    monkeypatch.setattr(diarization, "DIARIZATION_BACKEND", "speakrs")
    eng = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    eng._diarization_pipeline, eng._speakrs_lock = FakeSpeakrs(), threading.Lock()
    eng.clear_gpu_cache_async = lambda *a: None
    audio = {"waveform": np.ones((1, 48000), dtype=np.float32), "sample_rate": 16000}
    out = eng.diarize("a.wav", audio)
    assert seen[0].shape == (48000,)  # the decoded samples, not the file
    assert out["num_speakers"] == 2
    assert [(t["start"], t["end"], t["speaker"], t["duration"]) for t in out["segments"]] == \
        [(0.0, 2.0, "SPEAKER_00", 2.0), (1.5, 3.0, "SPEAKER_01", 1.5)]
    assert [(t["start"], t["end"]) for t in out["exclusive_segments"]] == [(0.0, 1.5), (1.5, 3.0)]
