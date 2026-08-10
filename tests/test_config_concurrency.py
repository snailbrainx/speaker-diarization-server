"""Regression tests for config concurrency + persistence (OPUS-005/KIMI-005/GLM-006).

Before the fix: 7/8 concurrent update threads raised FileNotFoundError on the
fixed "<file>.tmp" path, and an env default from docker-compose silently
overrode a persisted value on reload.
"""
import threading

import pytest


@pytest.fixture()
def config_file(tmp_path, monkeypatch):
    monkeypatch.delenv("SPEAKER_THRESHOLD", raising=False)
    monkeypatch.delenv("CONTEXT_PADDING", raising=False)
    monkeypatch.delenv("SILENCE_DURATION", raising=False)
    monkeypatch.delenv("FILTER_HALLUCINATIONS", raising=False)
    monkeypatch.delenv("EMOTION_THRESHOLD", raising=False)
    return str(tmp_path / "config.json")


def test_persisted_value_survives_reload_without_env_override(config_file):
    from app.config import ConfigManager
    cm = ConfigManager(config_file=config_file)
    cm.update_settings({"speaker_threshold": 0.55})
    reloaded = cm.reload_settings()
    assert reloaded.speaker_threshold == 0.55, (
        "persisted settings must survive reload when no env override is set"
    )


def test_env_override_still_wins_when_set(config_file, monkeypatch):
    from app.config import ConfigManager
    monkeypatch.setenv("SPEAKER_THRESHOLD", "0.4")
    cm = ConfigManager(config_file=config_file)
    assert cm.get_settings().speaker_threshold == 0.4


def test_concurrent_updates_no_errors_no_corruption(config_file):
    from app.config import ConfigManager
    cm = ConfigManager(config_file=config_file)
    errors = []
    barrier = threading.Barrier(8)

    def updater(i):
        barrier.wait()
        try:
            for _ in range(25):
                cm.update_settings({"speaker_threshold": 0.3 + (i % 5) * 0.01})
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=updater, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"concurrent settings updates failed: {errors[:3]}"
    final = cm.get_settings()
    assert 0.29 <= final.speaker_threshold <= 0.35

    # Persisted file must be valid JSON
    import json
    with open(config_file) as f:
        data = json.load(f)
    assert "speaker_threshold" in data


def test_failed_validation_does_not_persist(config_file):
    from pydantic import ValidationError

    from app.config import ConfigManager
    cm = ConfigManager(config_file=config_file)
    with pytest.raises(ValidationError):
        cm.update_settings({"speaker_threshold": 99.0})  # out of range (le=0.9)
    # In-memory state unchanged and file not written with garbage
    assert cm.get_settings().speaker_threshold == 0.30


def test_failed_disk_write_does_not_publish_candidate_in_memory(config_file, monkeypatch):
    """A failed atomic save must leave both observed and persisted state old."""
    from app.config import ConfigManager

    cm = ConfigManager(config_file=config_file)
    cm.update_settings({"speaker_threshold": 0.42})

    def fail_save(_settings):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(cm, "_save_settings", fail_save)
    with pytest.raises(OSError, match="read-only"):
        cm.update_settings({"speaker_threshold": 0.66})

    assert cm.get_settings().speaker_threshold == 0.42
    reloaded = ConfigManager(config_file=config_file)
    assert reloaded.get_settings().speaker_threshold == 0.42
