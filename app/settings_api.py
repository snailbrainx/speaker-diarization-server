"""
API endpoints for settings management
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from .config import get_config, VoiceSettings

router = APIRouter(prefix="/settings", tags=["Settings"])


class SettingsUpdateRequest(BaseModel):
    """Request model for updating settings"""
    speaker_threshold: Optional[float] = None
    emotion_threshold: Optional[float] = None
    context_padding: Optional[float] = None
    silence_duration: Optional[float] = None
    filter_hallucinations: Optional[bool] = None


@router.get("/voice", response_model=VoiceSettings)
async def get_voice_settings():
    """
    Get current voice processing settings.

    Returns the current configuration for speaker threshold, context padding,
    silence duration, and hallucination filtering.
    """
    config = get_config()
    return config.get_settings()


@router.post("/voice", response_model=VoiceSettings)
async def update_voice_settings(updates: SettingsUpdateRequest):
    """
    Update voice processing settings at runtime.

    Settings are persisted to config file and take effect immediately.
    Note: Some settings may require restarting active streams or reprocessing.

    **Parameters:**
    - **speaker_threshold**: Speaker similarity threshold (0.0-1.0). Lower = stricter matching.
      - 0.30: Normal home usage (default)
      - 0.20: Noisy environments (movies, background noise)
    - **emotion_threshold**: Global emotion matching threshold (0.3-1.0). Higher = stricter matching.
      - 0.60: Balanced (default)
      - 0.70-0.90: Fewer false positives, require stronger emotion matches
      - 0.95-1.00: Very strict, require near-perfect matches
      - 0.40-0.50: More lenient, applies personalization more often
    - **context_padding**: Padding before/after segments for embedding extraction (seconds)
      - 0.15: Default, helps with background music
    - **silence_duration**: Silence duration before processing streaming segment (seconds)
      - 0.5: Default, balance between responsiveness and completeness
    - **filter_hallucinations**: Filter common Whisper hallucinations (true/false)
      - true: Removes "thank you.", "thanks for watching", etc.
    """
    config = get_config()

    # Only include non-None values in update
    update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}

    if not update_dict:
        raise HTTPException(status_code=400, detail="No settings provided to update")

    try:
        return config.update_settings(update_dict)
    except ValueError as e:
        # Pydantic validation errors — the message is user-facing data, safe to surface
        raise HTTPException(status_code=400, detail=f"Invalid settings: {e}")
    except Exception:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Failed to update settings")


@router.post("/voice/reset", response_model=VoiceSettings)
async def reset_voice_settings():
    """
    Reset voice settings to defaults.

    Resets all settings to their default values and persists the changes.
    """
    config = get_config()
    try:
        default_settings = VoiceSettings()
        return config.update_settings(default_settings.model_dump())
    except Exception:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error resetting settings")
