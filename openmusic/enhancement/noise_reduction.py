"""
Noise reduction and audio enhancement functions.
"""

import numpy as np
import librosa
import noisereduce as nr
from scipy import signal
from typing import Optional, Tuple


def reduce_noise(audio: np.ndarray, sr: int, 
                noise_duration: Optional[float] = None,
                stationary: bool = True, prop_decrease: float = 1.0) -> np.ndarray:
    """
    Reduce noise in audio using spectral subtraction.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    noise_duration : float, optional
        Duration of noise sample at beginning of audio (in seconds)
        If None, estimates noise from entire signal
    stationary : bool, default=True
        Whether noise is stationary
    prop_decrease : float, default=1.0
        Proportion to decrease noise (0.0 to 1.0)
        
    Returns:
    --------
    np.ndarray
        Noise-reduced audio
    """
    try:
        # Use noisereduce library for advanced noise reduction
        reduced_audio = nr.reduce_noise(
            y=audio, 
            sr=sr,
            stationary=stationary,
            prop_decrease=prop_decrease
        )
        return reduced_audio
    except Exception:
        # Fallback to simple spectral subtraction
        return spectral_subtraction(audio, sr, noise_duration)


def spectral_subtraction(audio: np.ndarray, sr: int, 
                        noise_duration: Optional[float] = None,
                        alpha: float = 2.0, beta: float = 0.001) -> np.ndarray:
    """
    Apply spectral subtraction for noise reduction.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    noise_duration : float, optional
        Duration of noise sample at beginning (in seconds)
    alpha : float, default=2.0
        Over-subtraction factor
    beta : float, default=0.001
        Spectral floor factor
        
    Returns:
    --------
    np.ndarray
        Processed audio
    """
    # Compute STFT
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise spectrum
    if noise_duration is not None:
        noise_frames = int(noise_duration * sr / 512)  # Assuming hop_length=512
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    else:
        # Use minimum statistics for noise estimation
        noise_spectrum = np.percentile(magnitude, 10, axis=1, keepdims=True)
    
    # Spectral subtraction
    magnitude_clean = magnitude - alpha * noise_spectrum
    
    # Apply spectral floor
    magnitude_clean = np.maximum(magnitude_clean, beta * magnitude)
    
    # Reconstruct audio
    stft_clean = magnitude_clean * np.exp(1j * phase)
    audio_clean = librosa.istft(stft_clean)
    
    return audio_clean


def wiener_filter(audio: np.ndarray, sr: int, 
                 noise_power: Optional[float] = None) -> np.ndarray:
    """
    Apply Wiener filtering for noise reduction.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    noise_power : float, optional
        Estimated noise power. If None, estimated automatically
        
    Returns:
    --------
    np.ndarray
        Filtered audio
    """
    # Compute STFT
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate signal and noise power
    signal_power = magnitude ** 2
    
    if noise_power is None:
        # Estimate noise power from quiet segments
        noise_power = np.percentile(signal_power, 10)
    
    # Wiener filter
    wiener_gain = signal_power / (signal_power + noise_power)
    magnitude_filtered = magnitude * wiener_gain
    
    # Reconstruct audio
    stft_filtered = magnitude_filtered * np.exp(1j * phase)
    audio_filtered = librosa.istft(stft_filtered)
    
    return audio_filtered


def adaptive_noise_reduction(audio: np.ndarray, sr: int,
                           frame_length: int = 2048,
                           hop_length: int = 512) -> np.ndarray:
    """
    Apply adaptive noise reduction based on signal characteristics.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    frame_length : int, default=2048
        Frame length for analysis
    hop_length : int, default=512
        Hop length for analysis
        
    Returns:
    --------
    np.ndarray
        Processed audio
    """
    # Compute STFT
    stft = librosa.stft(audio, n_fft=frame_length, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Adaptive noise estimation
    noise_estimate = np.zeros_like(magnitude)
    alpha = 0.95  # Smoothing factor
    
    for i in range(magnitude.shape[1]):
        if i == 0:
            noise_estimate[:, i] = magnitude[:, i]
        else:
            # Update noise estimate based on minimum statistics
            noise_estimate[:, i] = alpha * noise_estimate[:, i-1] + \
                                 (1 - alpha) * np.minimum(magnitude[:, i], 
                                                        noise_estimate[:, i-1])
    
    # Adaptive Wiener filtering
    snr = magnitude / (noise_estimate + 1e-10)
    wiener_gain = snr / (snr + 1)
    
    # Apply gain
    magnitude_filtered = magnitude * wiener_gain
    
    # Reconstruct audio
    stft_filtered = magnitude_filtered * np.exp(1j * phase)
    audio_filtered = librosa.istft(stft_filtered, hop_length=hop_length)
    
    return audio_filtered


def remove_dc_offset(audio: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from audio signal.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
        
    Returns:
    --------
    np.ndarray
        Audio with DC offset removed
    """
    return audio - np.mean(audio)


def remove_clipping(audio: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Detect and attempt to reduce clipping artifacts.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    threshold : float, default=0.95
        Clipping detection threshold
        
    Returns:
    --------
    np.ndarray
        Audio with reduced clipping
    """
    # Detect clipped samples
    clipped = np.abs(audio) >= threshold
    
    if not np.any(clipped):
        return audio
    
    # Simple declipping using interpolation
    audio_declipped = audio.copy()
    clipped_indices = np.where(clipped)[0]
    
    for idx in clipped_indices:
        # Find nearest non-clipped samples
        left_idx = idx - 1
        right_idx = idx + 1
        
        while left_idx >= 0 and clipped[left_idx]:
            left_idx -= 1
        while right_idx < len(audio) and clipped[right_idx]:
            right_idx += 1
        
        # Interpolate if we have valid neighbors
        if left_idx >= 0 and right_idx < len(audio):
            audio_declipped[idx] = (audio[left_idx] + audio[right_idx]) / 2
    
    return audio_declipped