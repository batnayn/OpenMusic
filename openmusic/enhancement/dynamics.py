"""
Dynamic range processing functions for audio.
"""

import numpy as np
from typing import Optional


def compress_audio(audio: np.ndarray, sr: int, threshold: float = -20.0,
                  ratio: float = 4.0, attack: float = 0.01, 
                  release: float = 0.1) -> np.ndarray:
    """
    Apply dynamic range compression to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    threshold : float, default=-20.0
        Compression threshold in dB
    ratio : float, default=4.0
        Compression ratio (4:1 means 4dB input -> 1dB output above threshold)
    attack : float, default=0.01
        Attack time in seconds
    release : float, default=0.1
        Release time in seconds
        
    Returns:
    --------
    np.ndarray
        Compressed audio
    """
    # Convert to dB
    audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
    
    # Calculate gain reduction
    gain_reduction = np.zeros_like(audio_db)
    above_threshold = audio_db > threshold
    gain_reduction[above_threshold] = (audio_db[above_threshold] - threshold) * (1 - 1/ratio)
    
    # Apply envelope following for attack/release
    envelope = np.zeros_like(gain_reduction)
    attack_coeff = np.exp(-1 / (attack * sr))
    release_coeff = np.exp(-1 / (release * sr))
    
    for i in range(1, len(envelope)):
        if gain_reduction[i] > envelope[i-1]:
            # Attack
            envelope[i] = attack_coeff * envelope[i-1] + (1 - attack_coeff) * gain_reduction[i]
        else:
            # Release
            envelope[i] = release_coeff * envelope[i-1] + (1 - release_coeff) * gain_reduction[i]
    
    # Apply compression
    gain_linear = 10 ** (-envelope / 20)
    compressed_audio = audio * gain_linear
    
    return compressed_audio


def normalize_audio(audio: np.ndarray, target_lufs: float = -23.0,
                   peak_limit: float = -1.0) -> np.ndarray:
    """
    Normalize audio to target loudness (LUFS) with peak limiting.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    target_lufs : float, default=-23.0
        Target loudness in LUFS
    peak_limit : float, default=-1.0
        Peak limit in dBFS
        
    Returns:
    --------
    np.ndarray
        Normalized audio
    """
    # Simple RMS-based normalization (approximation of LUFS)
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
    
    # Convert target LUFS to linear gain (rough approximation)
    target_rms = 10**(target_lufs / 20)
    gain = target_rms / rms
    
    # Apply gain
    normalized_audio = audio * gain
    
    # Apply peak limiting
    peak_limit_linear = 10**(peak_limit / 20)
    peak = np.max(np.abs(normalized_audio))
    
    if peak > peak_limit_linear:
        normalized_audio = normalized_audio * (peak_limit_linear / peak)
    
    return normalized_audio


def gate_audio(audio: np.ndarray, sr: int, threshold: float = -40.0,
              attack: float = 0.001, release: float = 0.1,
              hold: float = 0.01) -> np.ndarray:
    """
    Apply noise gate to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    threshold : float, default=-40.0
        Gate threshold in dB
    attack : float, default=0.001
        Attack time in seconds
    release : float, default=0.1
        Release time in seconds
    hold : float, default=0.01
        Hold time in seconds
        
    Returns:
    --------
    np.ndarray
        Gated audio
    """
    # Convert threshold to linear
    threshold_linear = 10**(threshold / 20)
    
    # Calculate envelope
    envelope = np.abs(audio)
    
    # Smooth envelope (simple low-pass filter)
    alpha = 1 - np.exp(-1 / (0.01 * sr))  # 10ms smoothing
    for i in range(1, len(envelope)):
        envelope[i] = alpha * envelope[i] + (1 - alpha) * envelope[i-1]
    
    # Gate logic
    gate_open = envelope > threshold_linear
    gate_state = np.zeros_like(audio, dtype=bool)
    
    # Apply attack, hold, and release
    attack_samples = int(attack * sr)
    release_samples = int(release * sr)
    hold_samples = int(hold * sr)
    
    hold_counter = 0
    for i in range(len(gate_open)):
        if gate_open[i]:
            gate_state[i] = True
            hold_counter = hold_samples
        elif hold_counter > 0:
            gate_state[i] = True
            hold_counter -= 1
        else:
            gate_state[i] = False
    
    # Apply smooth transitions
    gain = np.zeros_like(audio)
    attack_coeff = np.exp(-1 / (attack * sr))
    release_coeff = np.exp(-1 / (release * sr))
    
    for i in range(1, len(gain)):
        target_gain = 1.0 if gate_state[i] else 0.0
        
        if target_gain > gain[i-1]:
            # Attack
            gain[i] = attack_coeff * gain[i-1] + (1 - attack_coeff) * target_gain
        else:
            # Release
            gain[i] = release_coeff * gain[i-1] + (1 - release_coeff) * target_gain
    
    return audio * gain


def limit_audio(audio: np.ndarray, sr: int, threshold: float = -1.0,
               release: float = 0.05) -> np.ndarray:
    """
    Apply hard limiting to prevent clipping.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    threshold : float, default=-1.0
        Limiting threshold in dBFS
    release : float, default=0.05
        Release time in seconds
        
    Returns:
    --------
    np.ndarray
        Limited audio
    """
    threshold_linear = 10**(threshold / 20)
    
    # Calculate instantaneous amplitude
    amplitude = np.abs(audio)
    
    # Calculate gain reduction
    gain_reduction = np.ones_like(audio)
    over_threshold = amplitude > threshold_linear
    gain_reduction[over_threshold] = threshold_linear / amplitude[over_threshold]
    
    # Apply release envelope
    smoothed_gain = np.zeros_like(gain_reduction)
    release_coeff = np.exp(-1 / (release * sr))
    
    smoothed_gain[0] = gain_reduction[0]
    for i in range(1, len(smoothed_gain)):
        if gain_reduction[i] < smoothed_gain[i-1]:
            # Instant attack for limiting
            smoothed_gain[i] = gain_reduction[i]
        else:
            # Smooth release
            smoothed_gain[i] = release_coeff * smoothed_gain[i-1] + \
                              (1 - release_coeff) * gain_reduction[i]
    
    return audio * smoothed_gain


def expand_audio(audio: np.ndarray, sr: int, threshold: float = -30.0,
                ratio: float = 2.0, attack: float = 0.001,
                release: float = 0.1) -> np.ndarray:
    """
    Apply dynamic range expansion to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    threshold : float, default=-30.0
        Expansion threshold in dB
    ratio : float, default=2.0
        Expansion ratio
    attack : float, default=0.001
        Attack time in seconds
    release : float, default=0.1
        Release time in seconds
        
    Returns:
    --------
    np.ndarray
        Expanded audio
    """
    # Convert to dB
    audio_db = 20 * np.log10(np.abs(audio) + 1e-10)
    
    # Calculate gain change
    gain_change = np.zeros_like(audio_db)
    below_threshold = audio_db < threshold
    gain_change[below_threshold] = (audio_db[below_threshold] - threshold) * (ratio - 1)
    
    # Apply envelope following
    envelope = np.zeros_like(gain_change)
    attack_coeff = np.exp(-1 / (attack * sr))
    release_coeff = np.exp(-1 / (release * sr))
    
    for i in range(1, len(envelope)):
        if np.abs(gain_change[i]) > np.abs(envelope[i-1]):
            # Attack
            envelope[i] = attack_coeff * envelope[i-1] + (1 - attack_coeff) * gain_change[i]
        else:
            # Release
            envelope[i] = release_coeff * envelope[i-1] + (1 - release_coeff) * gain_change[i]
    
    # Apply expansion
    gain_linear = 10 ** (envelope / 20)
    expanded_audio = audio * gain_linear
    
    return expanded_audio