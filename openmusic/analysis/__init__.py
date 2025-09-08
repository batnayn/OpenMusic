"""
Music Analysis Module

This module provides comprehensive music analysis capabilities including
tempo detection, key detection, chord recognition, and structural analysis.
"""

from .tempo import detect_tempo, track_beats
from .harmony import detect_key, recognize_chords, analyze_harmony
from .structure import analyze_structure, detect_segments

__all__ = [
    "detect_tempo",
    "track_beats", 
    "detect_key",
    "recognize_chords",
    "analyze_harmony",
    "analyze_structure",
    "detect_segments",
]