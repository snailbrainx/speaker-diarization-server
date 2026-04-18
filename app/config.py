"""
Configuration management for voice settings.
Supports runtime updates and persistence.
"""
import os
import json
from typing import Dict, Any, Callable, Tuple
from pydantic import BaseModel, Field


def _as_bool(value: str) -> bool:
    return value.lower() == "true"


# (setting_name, env_var, parser). Declarative so adding a new tunable
# requires one line instead of a copy-paste if-block.
_ENV_OVERRIDES: Tuple[Tuple[str, str, Callable[[str], Any]], ...] = (
    ("speaker_threshold", "SPEAKER_THRESHOLD", float),
    ("context_padding", "CONTEXT_PADDING", float),
    ("silence_duration", "SILENCE_DURATION", float),
    ("filter_hallucinations", "FILTER_HALLUCINATIONS", _as_bool),
    ("emotion_threshold", "EMOTION_THRESHOLD", float),
)


class VoiceSettings(BaseModel):
    """Voice processing settings"""
    speaker_threshold: float = Field(default=0.30, ge=0.1, le=0.9, description="Speaker similarity threshold (0.1-0.9)")
    context_padding: float = Field(default=0.15, ge=0.05, le=2.0, description="Context padding for embeddings (seconds)")
    silence_duration: float = Field(default=0.5, ge=0.1, le=5.0, description="Silence duration for streaming (seconds)")
    filter_hallucinations: bool = Field(default=True, description="Filter common Whisper hallucinations")
    emotion_threshold: float = Field(default=0.6, ge=0.3, le=1.0, description="Global emotion matching threshold (0.3-1.0)")


class ConfigManager:
    """
    Manages application configuration with runtime updates.
    Settings are loaded from:
    1. Config file (if exists)
    2. Environment variables (override file values)
    3. VoiceSettings defaults (fallback)
    """

    def __init__(self, config_file: str = "data/config.json"):
        self.config_file = config_file
        self._settings: VoiceSettings = self._load_settings()

    def _load_settings(self) -> VoiceSettings:
        """Load settings from file, apply env overrides, fall back to defaults."""
        settings_dict: Dict[str, Any] = {}

        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    settings_dict = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")

        for name, env_var, parser in _ENV_OVERRIDES:
            raw = os.getenv(env_var)
            if raw:
                settings_dict[name] = parser(raw)

        return VoiceSettings(**settings_dict)

    def get_settings(self) -> VoiceSettings:
        """Get current settings"""
        return self._settings

    def reload_settings(self) -> VoiceSettings:
        """Reload settings from config file (call after external updates)"""
        self._settings = self._load_settings()
        return self._settings

    def update_settings(self, updates: Dict[str, Any]) -> VoiceSettings:
        """
        Update settings at runtime and persist to file.
        Returns updated settings.
        """
        # Update settings object
        current = self._settings.model_dump()
        current.update(updates)
        self._settings = VoiceSettings(**current)

        # Persist to file
        self._save_settings()

        return self._settings

    def _save_settings(self):
        """Save settings to config file atomically (tempfile + os.replace)."""
        target_dir = os.path.dirname(self.config_file) or "."
        os.makedirs(target_dir, exist_ok=True)
        tmp_path = f"{self.config_file}.tmp"
        with open(tmp_path, 'w') as f:
            json.dump(self._settings.model_dump(), f, indent=2)
        os.replace(tmp_path, self.config_file)


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global config manager instance"""
    return config_manager
