"""
Tempo detection and beat tracking functions.
"""

import librosa
import numpy as np
from typing import Tuple, Optional, List


def detect_tempo(audio: np.ndarray, sr: int, 
                hop_length: int = 512) -> float:
    """
    Detect the tempo (BPM) of audio.
    
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
    float
        Detected tempo in BPM
    """
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
    return float(tempo)


def track_beats(audio: np.ndarray, sr: int, 
               hop_length: int = 512) -> Tuple[float, np.ndarray]:
    """
    Track beats in audio.
    
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
    tuple
        (tempo, beat_times) where beat_times are in seconds
    """
    tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
    
    return float(tempo), beat_times


def detect_onset(audio: np.ndarray, sr: int, 
                hop_length: int = 512, 
                onset_envelope: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Detect onset times in audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    hop_length : int, default=512
        Hop length for analysis
    onset_envelope : np.ndarray, optional
        Pre-computed onset strength envelope
        
    Returns:
    --------
    np.ndarray
        Onset times in seconds
    """
    if onset_envelope is None:
        onset_envelope = librosa.onset.onset_strength(
            y=audio, sr=sr, hop_length=hop_length
        )
    
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_envelope, sr=sr, hop_length=hop_length
    )
    
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    return onset_times


def analyze_rhythm(audio: np.ndarray, sr: int) -> dict:
    """
    Comprehensive rhythm analysis.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
        
    Returns:
    --------
    dict
        Dictionary containing rhythm analysis results:
        - tempo: Detected tempo in BPM
        - beat_times: Beat positions in seconds
        - onset_times: Onset positions in seconds
        - tempo_stability: Measure of tempo consistency
        - rhythmic_complexity: Measure of rhythmic complexity
    """
    hop_length = 512
    
    # Basic tempo and beat tracking
    tempo, beat_times = track_beats(audio, sr, hop_length)
    
    # Onset detection
    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    onset_times = detect_onset(audio, sr, hop_length, onset_envelope)
    
    # Tempo stability analysis
    if len(beat_times) > 2:
        beat_intervals = np.diff(beat_times)
        tempo_stability = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
        tempo_stability = np.clip(tempo_stability, 0.0, 1.0)
    else:
        tempo_stability = 0.0
    
    # Rhythmic complexity (based on onset density)
    if len(onset_times) > 0 and len(beat_times) > 0:
        audio_duration = len(audio) / sr
        onset_density = len(onset_times) / audio_duration
        beat_density = len(beat_times) / audio_duration
        rhythmic_complexity = onset_density / (beat_density + 1e-10)
    else:
        rhythmic_complexity = 0.0
    
    return {
        'tempo': tempo,
        'beat_times': beat_times,
        'onset_times': onset_times,
        'tempo_stability': tempo_stability,
        'rhythmic_complexity': rhythmic_complexity
    }


def detect_meter(audio: np.ndarray, sr: int) -> Tuple[int, int]:
    """
    Detect time signature (meter) of audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
        
    Returns:
    --------
    tuple
        (beats_per_measure, beat_unit) representing time signature
    """
    # This is a simplified meter detection
    tempo, beat_times = track_beats(audio, sr)
    
    if len(beat_times) < 8:
        return (4, 4)  # Default to 4/4
    
    # Analyze beat strength pattern
    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sr)
    
    # Simple pattern recognition for common meters
    beat_frames = librosa.time_to_frames(beat_times, sr=sr)
    beat_strengths = []
    
    for frame in beat_frames:
        if frame < len(onset_envelope):
            beat_strengths.append(onset_envelope[frame])
        else:
            beat_strengths.append(0.0)
    
    beat_strengths = np.array(beat_strengths)
    
    # Check for common patterns
    if len(beat_strengths) >= 8:
        # Test for 4/4 (strong-weak-medium-weak pattern)
        pattern_4_4 = []
        for i in range(0, len(beat_strengths) - 3, 4):
            if i + 3 < len(beat_strengths):
                pattern = beat_strengths[i:i+4]
                if len(pattern) == 4:
                    pattern_4_4.append(np.var(pattern))
        
        # Test for 3/4 (strong-weak-weak pattern)
        pattern_3_4 = []
        for i in range(0, len(beat_strengths) - 2, 3):
            if i + 2 < len(beat_strengths):
                pattern = beat_strengths[i:i+3]
                if len(pattern) == 3:
                    pattern_3_4.append(np.var(pattern))
        
        # Simple heuristic: higher variance suggests stronger patterns
        if pattern_4_4 and pattern_3_4:
            avg_var_4_4 = np.mean(pattern_4_4)
            avg_var_3_4 = np.mean(pattern_3_4)
            
            if avg_var_3_4 > avg_var_4_4 * 1.2:
                return (3, 4)
    
    return (4, 4)  # Default to 4/4


def extract_rhythmic_features(audio: np.ndarray, sr: int) -> dict:
    """
    Extract comprehensive rhythmic features.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
        
    Returns:
    --------
    dict
        Dictionary containing rhythmic features
    """
    rhythm_analysis = analyze_rhythm(audio, sr)
    meter = detect_meter(audio, sr)
    
    # Additional features
    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sr)
    
    features = {
        **rhythm_analysis,
        'time_signature': meter,
        'onset_strength_mean': np.mean(onset_envelope),
        'onset_strength_std': np.std(onset_envelope),
        'onset_strength_max': np.max(onset_envelope),
    }
    
    return features