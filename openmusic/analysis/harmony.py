"""
Harmonic analysis functions including key detection and chord recognition.
"""

import librosa
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy import signal


def detect_key(audio: np.ndarray, sr: int) -> Tuple[str, str]:
    """
    Detect the musical key of audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
        
    Returns:
    --------
    tuple
        (key, mode) where key is the tonic note and mode is 'major' or 'minor'
    """
    # Extract chroma features
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    
    # Average chroma over time
    chroma_mean = np.mean(chroma, axis=1)
    
    # Key profiles (Krumhansl-Schmuckler)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    
    # Normalize profiles
    major_profile = major_profile / np.sum(major_profile)
    minor_profile = minor_profile / np.sum(minor_profile)
    
    # Normalize chroma
    chroma_norm = chroma_mean / np.sum(chroma_mean)
    
    # Calculate correlation with each key
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    best_correlation = -1
    best_key = 'C'
    best_mode = 'major'
    
    for i in range(12):
        # Rotate profile to match key
        major_rotated = np.roll(major_profile, i)
        minor_rotated = np.roll(minor_profile, i)
        
        # Calculate correlations
        major_corr = np.corrcoef(chroma_norm, major_rotated)[0, 1]
        minor_corr = np.corrcoef(chroma_norm, minor_rotated)[0, 1]
        
        if major_corr > best_correlation:
            best_correlation = major_corr
            best_key = note_names[i]
            best_mode = 'major'
        
        if minor_corr > best_correlation:
            best_correlation = minor_corr
            best_key = note_names[i]
            best_mode = 'minor'
    
    return best_key, best_mode


def recognize_chords(audio: np.ndarray, sr: int, 
                    hop_length: int = 512) -> List[Dict]:
    """
    Recognize chord progressions in audio.
    
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
    list
        List of chord dictionaries with 'time', 'chord', 'confidence' keys
    """
    # Extract chroma features
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    
    # Time axis
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    
    # Chord templates (simplified)
    chord_templates = _get_chord_templates()
    
    chords = []
    
    for i, time in enumerate(times):
        frame_chroma = chroma[:, i]
        frame_chroma = frame_chroma / (np.sum(frame_chroma) + 1e-10)
        
        best_match = None
        best_confidence = 0
        
        for chord_name, template in chord_templates.items():
            # Calculate similarity (cosine similarity)
            similarity = np.dot(frame_chroma, template) / (
                np.linalg.norm(frame_chroma) * np.linalg.norm(template) + 1e-10
            )
            
            if similarity > best_confidence:
                best_confidence = similarity
                best_match = chord_name
        
        chords.append({
            'time': time,
            'chord': best_match,
            'confidence': best_confidence
        })
    
    # Post-process to reduce chord flickering
    chords = _smooth_chord_sequence(chords)
    
    return chords


def analyze_harmony(audio: np.ndarray, sr: int) -> Dict:
    """
    Comprehensive harmonic analysis.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
        
    Returns:
    --------
    dict
        Dictionary containing harmonic analysis results
    """
    # Key detection
    key, mode = detect_key(audio, sr)
    
    # Chord recognition
    chords = recognize_chords(audio, sr)
    
    # Extract harmonic features
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
    
    # Harmonic complexity measures
    chroma_var = np.var(chroma, axis=1)
    harmonic_complexity = np.mean(chroma_var)
    
    # Tonal stability
    chroma_std = np.std(chroma, axis=1)
    tonal_stability = 1.0 - np.mean(chroma_std)
    
    # Chord change rate
    unique_chords = len(set(chord['chord'] for chord in chords if chord['chord']))
    audio_duration = len(audio) / sr
    chord_change_rate = unique_chords / audio_duration if audio_duration > 0 else 0
    
    return {
        'key': key,
        'mode': mode,
        'chords': chords,
        'harmonic_complexity': harmonic_complexity,
        'tonal_stability': tonal_stability,
        'chord_change_rate': chord_change_rate,
        'chroma_features': chroma,
        'tonnetz_features': tonnetz
    }


def _get_chord_templates() -> Dict[str, np.ndarray]:
    """
    Get chord templates for chord recognition.
    
    Returns:
    --------
    dict
        Dictionary mapping chord names to chroma templates
    """
    templates = {}
    
    # Major triads
    major_pattern = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    for i, note in enumerate(note_names):
        templates[f"{note}"] = np.roll(major_pattern, i)
    
    # Minor triads
    minor_pattern = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    for i, note in enumerate(note_names):
        templates[f"{note}m"] = np.roll(minor_pattern, i)
    
    # Seventh chords (simplified)
    dom7_pattern = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    for i, note in enumerate(note_names):
        templates[f"{note}7"] = np.roll(dom7_pattern, i)
    
    # Normalize templates
    for chord in templates:
        templates[chord] = np.array(templates[chord], dtype=float)
        templates[chord] = templates[chord] / (np.sum(templates[chord]) + 1e-10)
    
    return templates


def _smooth_chord_sequence(chords: List[Dict], min_duration: float = 0.5) -> List[Dict]:
    """
    Smooth chord sequence to reduce flickering.
    
    Parameters:
    -----------
    chords : list
        Raw chord sequence
    min_duration : float
        Minimum chord duration in seconds
        
    Returns:
    --------
    list
        Smoothed chord sequence
    """
    if len(chords) < 2:
        return chords
    
    smoothed = []
    current_chord = chords[0]['chord']
    current_start = chords[0]['time']
    confidence_sum = chords[0]['confidence']
    count = 1
    
    for i in range(1, len(chords)):
        if (chords[i]['chord'] == current_chord or 
            chords[i]['time'] - current_start < min_duration):
            confidence_sum += chords[i]['confidence']
            count += 1
        else:
            # Add previous chord
            smoothed.append({
                'time': current_start,
                'chord': current_chord,
                'confidence': confidence_sum / count
            })
            
            # Start new chord
            current_chord = chords[i]['chord']
            current_start = chords[i]['time']
            confidence_sum = chords[i]['confidence']
            count = 1
    
    # Add final chord
    smoothed.append({
        'time': current_start,
        'chord': current_chord,
        'confidence': confidence_sum / count
    })
    
    return smoothed


def extract_harmonic_features(audio: np.ndarray, sr: int) -> Dict:
    """
    Extract comprehensive harmonic features for analysis.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
        
    Returns:
    --------
    dict
        Dictionary containing harmonic features
    """
    harmony_analysis = analyze_harmony(audio, sr)
    
    # Additional features
    chroma = harmony_analysis['chroma_features']
    tonnetz = harmony_analysis['tonnetz_features']
    
    features = {
        'key': harmony_analysis['key'],
        'mode': harmony_analysis['mode'],
        'harmonic_complexity': harmony_analysis['harmonic_complexity'],
        'tonal_stability': harmony_analysis['tonal_stability'],
        'chord_change_rate': harmony_analysis['chord_change_rate'],
        'chroma_mean': np.mean(chroma, axis=1),
        'chroma_std': np.std(chroma, axis=1),
        'tonnetz_mean': np.mean(tonnetz, axis=1),
        'tonnetz_std': np.std(tonnetz, axis=1),
    }
    
    return features