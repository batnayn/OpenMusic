"""
Audio feature extraction functions for comprehensive audio analysis.
"""

import librosa
import numpy as np
from typing import Dict, Tuple, Optional
import warnings


def extract_mfccs(audio: np.ndarray, sr: int, n_mfcc: int = 13, 
                  n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCCs).
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    n_mfcc : int, default=13
        Number of MFCCs to extract
    n_fft : int, default=2048
        FFT window size
    hop_length : int, default=512
        Hop length for STFT
        
    Returns:
    --------
    np.ndarray
        MFCC features with shape (n_mfcc, n_frames)
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sr, n_mfcc=n_mfcc, 
            n_fft=n_fft, hop_length=hop_length
        )
    return mfccs


def extract_spectral_features(audio: np.ndarray, sr: int, 
                            n_fft: int = 2048, hop_length: int = 512) -> Dict[str, np.ndarray]:
    """
    Extract comprehensive spectral features.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    n_fft : int, default=2048
        FFT window size
    hop_length : int, default=512
        Hop length for STFT
        
    Returns:
    --------
    dict
        Dictionary containing spectral features:
        - spectral_centroid: Spectral centroid
        - spectral_rolloff: Spectral rolloff point
        - spectral_bandwidth: Spectral bandwidth
        - spectral_contrast: Spectral contrast
        - zero_crossing_rate: Zero crossing rate
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Compute STFT for efficiency
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        
        features = {}
        
        # Spectral centroid
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            S=magnitude, sr=sr, hop_length=hop_length
        )[0]
        
        # Spectral rolloff
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            S=magnitude, sr=sr, hop_length=hop_length
        )[0]
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            S=magnitude, sr=sr, hop_length=hop_length
        )[0]
        
        # Spectral contrast
        features['spectral_contrast'] = librosa.feature.spectral_contrast(
            S=magnitude, sr=sr, hop_length=hop_length
        )
        
        # Zero crossing rate
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(
            audio, hop_length=hop_length
        )[0]
        
    return features


def extract_temporal_features(audio: np.ndarray, sr: int, 
                            hop_length: int = 512) -> Dict[str, np.ndarray]:
    """
    Extract temporal features from audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    hop_length : int, default=512
        Hop length for analysis
        
    Returns:
    --------
    dict
        Dictionary containing temporal features:
        - rms_energy: Root mean square energy
        - tempo: Tempo estimation
        - beat_frames: Beat frame positions
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        features = {}
        
        # RMS Energy
        features['rms_energy'] = librosa.feature.rms(
            y=audio, hop_length=hop_length
        )[0]
        
        # Tempo and beat tracking
        tempo, beat_frames = librosa.beat.beat_track(
            y=audio, sr=sr, hop_length=hop_length
        )
        features['tempo'] = tempo
        features['beat_frames'] = beat_frames
        
    return features


def extract_harmonic_features(audio: np.ndarray, sr: int, 
                            n_fft: int = 2048, hop_length: int = 512) -> Dict[str, np.ndarray]:
    """
    Extract harmonic and tonal features.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    n_fft : int, default=2048
        FFT window size
    hop_length : int, default=512
        Hop length for STFT
        
    Returns:
    --------
    dict
        Dictionary containing harmonic features:
        - chroma: Chroma features (12-dimensional)
        - tonnetz: Tonal centroid features
        - harmonic: Harmonic component
        - percussive: Percussive component
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        features = {}
        
        # Harmonic-percussive separation
        harmonic, percussive = librosa.effects.hpss(audio)
        features['harmonic'] = harmonic
        features['percussive'] = percussive
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        
        # Tonnetz (tonal centroid features)
        features['tonnetz'] = librosa.feature.tonnetz(
            y=harmonic, sr=sr
        )
        
    return features


def extract_all_features(audio: np.ndarray, sr: int, 
                        n_mfcc: int = 13, n_fft: int = 2048, 
                        hop_length: int = 512) -> Dict[str, np.ndarray]:
    """
    Extract all available features from audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    n_mfcc : int, default=13
        Number of MFCCs to extract
    n_fft : int, default=2048
        FFT window size
    hop_length : int, default=512
        Hop length for analysis
        
    Returns:
    --------
    dict
        Dictionary containing all extracted features
    """
    all_features = {}
    
    # MFCCs
    all_features['mfccs'] = extract_mfccs(
        audio, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    
    # Spectral features
    spectral_features = extract_spectral_features(
        audio, sr, n_fft=n_fft, hop_length=hop_length
    )
    all_features.update(spectral_features)
    
    # Temporal features
    temporal_features = extract_temporal_features(
        audio, sr, hop_length=hop_length
    )
    all_features.update(temporal_features)
    
    # Harmonic features
    harmonic_features = extract_harmonic_features(
        audio, sr, n_fft=n_fft, hop_length=hop_length
    )
    all_features.update(harmonic_features)
    
    return all_features