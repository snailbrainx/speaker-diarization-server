"""Regression tests for the dual-detector emotion combiner
(OPUS-003/KIMI-003) — an absent voice detector must NOT be treated as a
"neutral" prediction that cancels a confident emotion2vec label.

Exercises the real SpeakerRecognitionEngine.match_emotion_dual_detector with
heavy ML modules stubbed at import time (tests/conftest.py).
"""
import numpy as np
import pytest


class FakeProfile:
    """Minimal stand-in for SpeakerEmotionProfile rows."""

    def __init__(self, emotion_category, embedding=None, voice_embedding=None,
                 voice_sample_count=0, confidence_threshold=None, voice_threshold=None):
        self.emotion_category = emotion_category
        self._embedding = embedding
        self._voice_embedding = voice_embedding
        self.voice_sample_count = voice_sample_count
        self.confidence_threshold = confidence_threshold
        self.voice_threshold = voice_threshold

    def get_embedding(self):
        return self._embedding

    def get_voice_embedding(self):
        return self._voice_embedding


@pytest.fixture()
def engine(monkeypatch):
    from app.diarization import SpeakerRecognitionEngine
    # Bypass __init__ (it touches config/model locks); we only need the
    # combiner method, which is stateless.
    eng = SpeakerRecognitionEngine.__new__(SpeakerRecognitionEngine)
    return eng


def _unit_vec(seed_value, dim):
    v = np.full(dim, seed_value, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_missing_voice_detector_keeps_confident_emotion(engine):
    """Detector-1 says 'happy' at 0.9; detector-2 has no usable voice profile.
    Old code synthesised d2='neutral' and collapsed the result to neutral."""
    emotion_emb = _unit_vec(1.0, 1024)
    voice_emb = _unit_vec(0.5, 512)
    profiles = [
        FakeProfile("happy", embedding=emotion_emb, confidence_threshold=0.6,
                    voice_sample_count=1),  # below MIN_VOICE_SAMPLES
    ]
    result = engine.match_emotion_dual_detector(
        emotion_emb, voice_emb, profiles,
        global_threshold=0.6, speaker_threshold=0.3,
    )
    final = result["final_decision"]
    assert final["emotion"] == "happy", (
        f"absent voice detector must not neutralise a confident label: {final}"
    )
    assert final["voice_profile_available"] is False
    assert result["voice_profile_detector"] is None


def test_voice_detector_below_min_samples_is_not_a_second_opinion(engine):
    emotion_emb = _unit_vec(1.0, 1024)
    voice_emb = _unit_vec(0.5, 512)
    profiles = [
        FakeProfile("happy", embedding=emotion_emb, confidence_threshold=0.6,
                    voice_embedding=voice_emb, voice_sample_count=2),
    ]
    result = engine.match_emotion_dual_detector(
        emotion_emb, voice_emb, profiles, global_threshold=0.6, speaker_threshold=0.3)
    assert result["final_decision"]["emotion"] == "happy"


def test_below_threshold_voice_match_remains_visible_for_diagnostics(engine):
    emotion_emb = _unit_vec(1.0, 1024)
    voice_emb = np.zeros(512, dtype=np.float32)
    voice_emb[0] = 1.0
    profile_voice_emb = np.zeros(512, dtype=np.float32)
    profile_voice_emb[1] = 1.0
    profiles = [FakeProfile(
        "happy",
        embedding=emotion_emb,
        confidence_threshold=0.6,
        voice_embedding=profile_voice_emb,
        voice_sample_count=5,
        voice_threshold=0.3,
    )]

    result = engine.match_emotion_dual_detector(
        emotion_emb,
        voice_emb,
        profiles,
        global_threshold=0.6,
        speaker_threshold=0.3,
    )
    assert result["final_decision"]["emotion"] == "happy"
    assert result["final_decision"]["voice_profile_available"] is False
    diagnostics = result["voice_profile_detector"]
    assert diagnostics is not None
    assert diagnostics["emotion"] is None
    assert diagnostics["matches"][0]["emotion"] == "happy"
    assert diagnostics["matches"][0]["similarity"] == pytest.approx(0.0)


def test_both_detectors_agree_high_confidence(engine):
    emotion_emb = _unit_vec(1.0, 1024)
    voice_emb = _unit_vec(0.5, 512)
    profiles = [
        FakeProfile("happy", embedding=emotion_emb, confidence_threshold=0.6,
                    voice_embedding=voice_emb, voice_sample_count=5,
                    voice_threshold=0.3),
    ]
    result = engine.match_emotion_dual_detector(
        emotion_emb, voice_emb, profiles, global_threshold=0.6, speaker_threshold=0.3)
    final = result["final_decision"]
    assert final["emotion"] == "happy"
    assert final["voice_profile_available"] is True


def test_strong_voice_override_still_works(engine):
    """A strong, well-sampled voice match may still override detector-1."""
    emotion_emb = _unit_vec(1.0, 1024)          # d1 will match 'happy'
    voice_emb = _unit_vec(0.5, 512)
    profiles = [
        FakeProfile("happy", embedding=emotion_emb, confidence_threshold=0.6,
                    voice_sample_count=1),
        FakeProfile("angry", embedding=np.full(1024, 0.0, dtype=np.float32),
                    voice_embedding=voice_emb, voice_sample_count=5,
                    voice_threshold=0.3),
    ]
    result = engine.match_emotion_dual_detector(
        emotion_emb, voice_emb, profiles, global_threshold=0.6, speaker_threshold=0.3)
    final = result["final_decision"]
    assert final["emotion"] == "angry", f"strong voice match must override: {final}"
    assert final["reason"].startswith("Voice strong")


def test_neutral_detector1_stays_neutral(engine):
    emotion_emb = _unit_vec(1.0, 1024)
    voice_emb = _unit_vec(0.5, 512)
    result = engine.match_emotion_dual_detector(
        emotion_emb, voice_emb, [],
        generic_emotion="neutral", generic_confidence=0.2)
    assert result["final_decision"]["emotion"] == "neutral"
