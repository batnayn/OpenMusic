"""
Audio Effects Module

This module provides comprehensive audio effects processing including
time-based effects, pitch effects, modulation, and dynamic effects.
"""

from .time_based import add_reverb, add_delay, add_echo
from .pitch_effects import pitch_shift, time_stretch, auto_tune
from .modulation import add_chorus, add_flanger, add_phaser
from .distortion import add_distortion, add_saturation, add_overdrive

__all__ = [
    "add_reverb",
    "add_delay",
    "add_echo",
    "pitch_shift",
    "time_stretch", 
    "auto_tune",
    "add_chorus",
    "add_flanger",
    "add_phaser",
    "add_distortion",
    "add_saturation",
    "add_overdrive",
]