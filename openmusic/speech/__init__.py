"""
Speech Processing Module

This module provides speech recognition and text-to-speech synthesis capabilities.
"""

from .recognition import recognize_speech, detect_voice_activity
from .synthesis import synthesize_text, list_voices

__all__ = [
    "recognize_speech",
    "detect_voice_activity",
    "synthesize_text", 
    "list_voices",
]