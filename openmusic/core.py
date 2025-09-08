"""
Core audio processing utilities for OpenMusic library.
"""

import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional, Union


def load_audio(file_path: str, sr: Optional[int] = None, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load an audio file.
    
    Parameters:
    -----------
    file_path : str
        Path to the audio file
    sr : int, optional
        Target sampling rate. If None, uses the file's native sample rate
    mono : bool, default=True
        Convert to mono if True
        
    Returns:
    --------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=mono)
        return audio, sample_rate
    except Exception as e:
        raise ValueError(f"Error loading audio file {file_path}: {str(e)}")


def save_audio(file_path: str, audio: np.ndarray, sr: int) -> None:
    """
    Save an audio array to file.
    
    Parameters:
    -----------
    file_path : str
        Output file path
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    """
    try:
        sf.write(file_path, audio, sr)
    except Exception as e:
        raise ValueError(f"Error saving audio file {file_path}: {str(e)}")


def normalize_audio(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target RMS level in dB.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio
    target_level : float, default=-20.0
        Target RMS level in dB
        
    Returns:
    --------
    np.ndarray
        Normalized audio
    """
    rms = np.sqrt(np.mean(audio**2))
    if rms == 0:
        return audio
        
    target_rms = 10**(target_level / 20.0)
    scaling_factor = target_rms / rms
    return audio * scaling_factor


def convert_sample_rate(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Convert audio to different sample rate.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio
    orig_sr : int
        Original sample rate
    target_sr : int
        Target sample rate
        
    Returns:
    --------
    np.ndarray
        Resampled audio
    """
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)