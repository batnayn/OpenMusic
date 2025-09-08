"""
Audio filtering functions for frequency domain processing.
"""

import numpy as np
from scipy import signal
from typing import Optional, Union


def apply_lowpass(audio: np.ndarray, sr: int, cutoff: float, 
                 order: int = 5) -> np.ndarray:
    """
    Apply low-pass filter to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    cutoff : float
        Cutoff frequency in Hz
    order : int, default=5
        Filter order
        
    Returns:
    --------
    np.ndarray
        Filtered audio
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    
    if normalized_cutoff >= 1.0:
        return audio
    
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def apply_highpass(audio: np.ndarray, sr: int, cutoff: float, 
                  order: int = 5) -> np.ndarray:
    """
    Apply high-pass filter to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    cutoff : float
        Cutoff frequency in Hz
    order : int, default=5
        Filter order
        
    Returns:
    --------
    np.ndarray
        Filtered audio
    """
    nyquist = sr / 2
    normalized_cutoff = cutoff / nyquist
    
    if normalized_cutoff <= 0.0:
        return audio
    
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def apply_bandpass(audio: np.ndarray, sr: int, low_cutoff: float, 
                  high_cutoff: float, order: int = 5) -> np.ndarray:
    """
    Apply band-pass filter to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    low_cutoff : float
        Low cutoff frequency in Hz
    high_cutoff : float
        High cutoff frequency in Hz
    order : int, default=5
        Filter order
        
    Returns:
    --------
    np.ndarray
        Filtered audio
    """
    nyquist = sr / 2
    low_normalized = low_cutoff / nyquist
    high_normalized = high_cutoff / nyquist
    
    if low_normalized >= high_normalized:
        raise ValueError("Low cutoff must be less than high cutoff")
    
    b, a = signal.butter(order, [low_normalized, high_normalized], btype='band')
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def apply_bandstop(audio: np.ndarray, sr: int, low_cutoff: float, 
                  high_cutoff: float, order: int = 5) -> np.ndarray:
    """
    Apply band-stop (notch) filter to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    low_cutoff : float
        Low cutoff frequency in Hz
    high_cutoff : float
        High cutoff frequency in Hz
    order : int, default=5
        Filter order
        
    Returns:
    --------
    np.ndarray
        Filtered audio
    """
    nyquist = sr / 2
    low_normalized = low_cutoff / nyquist
    high_normalized = high_cutoff / nyquist
    
    if low_normalized >= high_normalized:
        raise ValueError("Low cutoff must be less than high cutoff")
    
    b, a = signal.butter(order, [low_normalized, high_normalized], btype='bandstop')
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def apply_notch(audio: np.ndarray, sr: int, frequency: float, 
               quality: float = 30.0) -> np.ndarray:
    """
    Apply notch filter to remove specific frequency.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    frequency : float
        Frequency to notch out in Hz
    quality : float, default=30.0
        Quality factor (higher = narrower notch)
        
    Returns:
    --------
    np.ndarray
        Filtered audio
    """
    nyquist = sr / 2
    normalized_freq = frequency / nyquist
    
    # Calculate notch width
    bandwidth = normalized_freq / quality
    low_cutoff = normalized_freq - bandwidth/2
    high_cutoff = normalized_freq + bandwidth/2
    
    # Ensure valid range
    low_cutoff = max(0.001, low_cutoff)
    high_cutoff = min(0.999, high_cutoff)
    
    b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='bandstop')
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def apply_parametric_eq(audio: np.ndarray, sr: int, frequency: float,
                       gain_db: float, quality: float = 1.0) -> np.ndarray:
    """
    Apply parametric equalizer at specific frequency.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    frequency : float
        Center frequency in Hz
    gain_db : float
        Gain in dB (positive = boost, negative = cut)
    quality : float, default=1.0
        Quality factor (bandwidth control)
        
    Returns:
    --------
    np.ndarray
        Processed audio
    """
    # Convert gain from dB to linear
    gain_linear = 10 ** (gain_db / 20)
    
    # Calculate filter coefficients for peaking EQ
    w = 2 * np.pi * frequency / sr
    cos_w = np.cos(w)
    sin_w = np.sin(w)
    alpha = sin_w / (2 * quality)
    
    # Peaking EQ coefficients
    A = gain_linear
    b0 = 1 + alpha * A
    b1 = -2 * cos_w
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cos_w
    a2 = 1 - alpha / A
    
    # Normalize coefficients
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1, a2]) / a0
    
    # Apply filter
    filtered_audio = signal.filtfilt(b, a, audio)
    
    return filtered_audio


def apply_graphic_eq(audio: np.ndarray, sr: int, 
                    bands: list, gains_db: list) -> np.ndarray:
    """
    Apply multi-band graphic equalizer.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    bands : list
        List of center frequencies in Hz
    gains_db : list
        List of gains in dB for each band
        
    Returns:
    --------
    np.ndarray
        Processed audio
    """
    if len(bands) != len(gains_db):
        raise ValueError("Number of bands must equal number of gains")
    
    processed_audio = audio.copy()
    
    for freq, gain in zip(bands, gains_db):
        if gain != 0:  # Skip processing if no gain change
            processed_audio = apply_parametric_eq(
                processed_audio, sr, freq, gain
            )
    
    return processed_audio