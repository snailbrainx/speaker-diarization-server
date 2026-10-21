"""The live recorder's clock is the audio itself and speech is found by a VAD (Silero), not by loudness."""
import numpy as np
import pytest

from app import streaming_recorder
from app.streaming_recorder import StreamingRecorder


@pytest.fixture(autouse=True)
def _tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming_recorder, "data_path", lambda: str(tmp_path))


def _recorder(monkeypatch, speech):
    """A recorder whose speech detector says `speech(packet)`."""
    monkeypatch.setattr(streaming_recorder, "_speech_detector", lambda sr, threshold: speech)
    rec = StreamingRecorder(max_workers=1, sample_rate=16000)
    rec.silence_duration = 0.5
    rec.start_recording(1)
    return rec


def test_a_pause_cuts_by_the_audio_received_not_the_wall_clock(monkeypatch):
    rec = _recorder(monkeypatch, lambda a: a.max() > 0)
    talk, quiet = np.full(4000, 0.3, np.float32), np.zeros(4000, np.float32)   # 0.25 s packets, sent at once
    for _ in range(12):                      # 3 s of speech
        rec.process_audio_chunk((16000, talk))
    rec.process_audio_chunk((16000, quiet))
    assert rec.segments_queued == 0           # 0.25 s of pause: not yet
    rec.process_audio_chunk((16000, quiet))
    assert rec.segments_queued == 1           # 0.5 s of pause, though no time has passed
    rec.cleanup()


def test_a_pause_does_not_cut_a_short_buffer(monkeypatch):
    rec = _recorder(monkeypatch, lambda a: a.max() > 0)
    for _ in range(4):                       # 1 s of speech, then a long pause
        rec.process_audio_chunk((16000, np.full(4000, 0.3, np.float32)))
    for _ in range(4):
        rec.process_audio_chunk((16000, np.zeros(4000, np.float32)))
    assert rec.segments_queued == 0           # under MIN_FLUSH_SECONDS: keeps buffering
    rec.cleanup()


def test_quiet_speech_is_kept_and_loud_noise_without_speech_is_not(monkeypatch):
    voiced = {"on": True}
    rec = _recorder(monkeypatch, lambda a: voiced["on"])
    for _ in range(12):                      # quiet but voiced (a loudness threshold dropped this)
        rec.process_audio_chunk((16000, np.full(4000, 0.002, np.float32)))
    voiced["on"] = False
    for _ in range(3):
        rec.process_audio_chunk((16000, np.full(4000, 0.002, np.float32)))
    assert rec.segments_queued == 1
    rec.LONG_TURN_SECONDS = 2.0
    for _ in range(12):                      # loud, but no speech in it: cut, and not processed
        rec.process_audio_chunk((16000, np.full(4000, 0.4, np.float32)))
    assert rec.segments_queued == 1
    rec.cleanup()


def test_silero_frames_carry_their_context_across_packets(monkeypatch):
    calls = []

    class FakeModel:
        def __call__(self, audio):
            calls.append(len(audio))
            frames = audio.reshape(-1, 512)
            return (frames.max(axis=1) > 0).astype(np.float32)[:, None]

    monkeypatch.setattr("faster_whisper.vad.get_vad_model", lambda: FakeModel())
    vad = streaming_recorder._SileroSpeech(48000)
    assert vad(np.zeros(4800, np.float32)) is False          # 0.1 s at 48 kHz = 1600 samples at 16 kHz: 3 frames
    assert calls == [512 + 3 * 512] and len(vad.pending) == 64
    assert vad(np.full(4800, 0.1, np.float32)) is True
    assert calls[1] == 512 + 3 * 512                           # the previous frame first, then 3 new ones


def test_skipped_silence_keeps_real_times_and_the_recording_lines_up(monkeypatch):
    import wave
    rec = _recorder(monkeypatch, lambda a: a.max() > 0)
    talk, quiet = np.full(4000, 0.3, np.float32), np.zeros(4000, np.float32)
    for packets in ([talk] * 12 + [quiet] * 2, [quiet] * 12, [talk] * 12 + [quiet] * 2):
        for p in packets:                    # 3.5 s (speech), 3 s (silence: skipped), 3.5 s (speech)
            rec.process_audio_chunk((16000, p))
    rec.stop_recording()
    assert rec.segments_queued == 2
    # the second chunk starts at 6.0 s (with the last 0.5 s of the silence), not 3.5 s: the skipped silence counts
    assert sorted(rec.segment_offsets.values()) == [0.0, pytest.approx(6.0)]
    out = rec.concatenate_segments()
    assert out.status == "success"
    with wave.open(out.path) as wf:
        assert wf.getnframes() == int(10.0 * 16000)                           # silence in the skipped gap
    rec.cleanup()
