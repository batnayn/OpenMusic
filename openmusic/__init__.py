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

# AI imports - The massive AI enhancement
from . import ai

# Convenience functions
from .core import load_audio, save_audio

__all__ = [
    "features",
    "speech", 
    "enhancement",
    "effects",
    "analysis",
    "ai",  # Massive AI capabilities
    "load_audio",
    "save_audio",
]