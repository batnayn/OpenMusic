"""
OpenMusic - Comprehensive Audio Processing Library

A powerful Python library for audio processing, featuring over 1000 audio processing 
capabilities including feature extraction, speech processing, audio enhancement, 
and music analysis.
"""

__version__ = "1.0.0"
__author__ = "OpenMusic Team"
__email__ = "team@openmusic.org"

# Core imports
from . import features
from . import speech
from . import enhancement
from . import effects
from . import analysis
from . import generation

# Convenience functions
from .core import load_audio, save_audio
from .generation import generate_music_from_prompt, generate_song

__all__ = [
    "features",
    "speech", 
    "enhancement",
    "effects",
    "analysis",
    "generation",
    "load_audio",
    "save_audio",
    "generate_music_from_prompt",
    "generate_song",
]