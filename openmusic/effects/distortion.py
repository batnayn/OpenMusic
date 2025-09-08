"""Distortion and saturation effects."""

import numpy as np
from typing import Optional


def add_distortion(audio: np.ndarray, gain: float = 5.0,
                  threshold: float = 0.3) -> np.ndarray:
    """
    Add hard clipping distortion to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    gain : float, default=5.0
        Input gain before clipping
    threshold : float, default=0.3
        Clipping threshold (0.0 to 1.0)
        
    Returns:
    --------
    np.ndarray
        Distorted audio
    """
    # Apply gain
    gained_audio = audio * gain
    
    # Hard clipping
    clipped_audio = np.clip(gained_audio, -threshold, threshold)
    
    # Normalize
    return clipped_audio / np.max(np.abs(clipped_audio)) if np.max(np.abs(clipped_audio)) > 0 else clipped_audio


def add_saturation(audio: np.ndarray, drive: float = 2.0) -> np.ndarray:
    """
    Add soft saturation to audio using tanh function.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    drive : float, default=2.0
        Saturation drive amount
        
    Returns:
    --------
    np.ndarray
        Saturated audio
    """
    # Apply drive and soft saturation
    driven_audio = audio * drive
    saturated_audio = np.tanh(driven_audio)
    
    # Compensate for level
    return saturated_audio / drive


def add_overdrive(audio: np.ndarray, gain: float = 3.0,
                 tone: float = 0.5) -> np.ndarray:
    """
    Add overdrive effect with tone control.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    gain : float, default=3.0
        Overdrive gain
    tone : float, default=0.5
        Tone control (0.0 = dark, 1.0 = bright)
        
    Returns:
    --------
    np.ndarray
        Overdriven audio
    """
    # Apply gain
    gained_audio = audio * gain
    
    # Asymmetric soft clipping
    overdriven = np.where(gained_audio > 0, 
                         np.tanh(gained_audio * 2), 
                         np.tanh(gained_audio * 1.5))
    
    # Simple tone control (high-frequency emphasis)
    if tone > 0.5:
        # Emphasize high frequencies
        diff = np.diff(overdriven, prepend=overdriven[0])
        overdriven = overdriven + (tone - 0.5) * 2 * diff
    elif tone < 0.5:
        # Smooth/filter high frequencies
        alpha = 1 - (0.5 - tone) * 2
        for i in range(1, len(overdriven)):
            overdriven[i] = alpha * overdriven[i] + (1 - alpha) * overdriven[i-1]
    
    # Normalize
    peak = np.max(np.abs(overdriven))
    return overdriven / peak if peak > 0 else overdriven