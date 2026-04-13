"""
Configuration management for voice settings.
Supports runtime updates and persistence.
"""
import os
import json
from typing import Dict, Any
from pydantic import BaseModel, Field


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
    1. Environment variables (highest priority)
    2. Config file (if exists)
    3. Defaults
    """

    def __init__(self, config_file: str = "data/config.json"):
        self.config_file = config_file
        self._settings: VoiceSettings = self._load_settings()

    def _load_settings(self) -> VoiceSettings:
        """Load settings from env vars, file, or defaults"""
        settings_dict = {}

        # Try to load from config file first
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    settings_dict = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")

        # Override with environment variables if set
        if os.getenv("SPEAKER_THRESHOLD"):
            settings_dict["speaker_threshold"] = float(os.getenv("SPEAKER_THRESHOLD"))
        if os.getenv("CONTEXT_PADDING"):
            settings_dict["context_padding"] = float(os.getenv("CONTEXT_PADDING"))
        if os.getenv("SILENCE_DURATION"):
            settings_dict["silence_duration"] = float(os.getenv("SILENCE_DURATION"))
        if os.getenv("FILTER_HALLUCINATIONS"):
            settings_dict["filter_hallucinations"] = os.getenv("FILTER_HALLUCINATIONS").lower() == "true"
        if os.getenv("EMOTION_THRESHOLD"):
            settings_dict["emotion_threshold"] = float(os.getenv("EMOTION_THRESHOLD"))

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
        """Save settings to config file"""
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self._settings.model_dump(), f, indent=2)


# Global config manager instance
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """Get the global config manager instance"""
    return config_manager
