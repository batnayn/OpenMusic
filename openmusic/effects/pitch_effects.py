"""
Pitch-based audio effects including pitch shifting, time stretching, and auto-tune.
"""

import librosa
import numpy as np
from typing import Optional


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """
    Shift the pitch of audio by a given number of semitones.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    n_steps : float
        Number of semitones to shift (positive = higher, negative = lower)
        
    Returns:
    --------
    np.ndarray
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def time_stretch(audio: np.ndarray, rate: float) -> np.ndarray:
    """
    Change the speed of audio without changing pitch.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    rate : float
        Stretch factor (>1.0 = faster, <1.0 = slower)
        
    Returns:
    --------
    np.ndarray
        Time-stretched audio
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def auto_tune(audio: np.ndarray, sr: int, target_notes: Optional[list] = None,
             correction_strength: float = 0.8) -> np.ndarray:
    """
    Apply auto-tune effect to correct pitch to target notes.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    target_notes : list, optional
        List of target note frequencies. If None, uses chromatic scale
    correction_strength : float, default=0.8
        Strength of pitch correction (0.0 to 1.0)
        
    Returns:
    --------
    np.ndarray
        Auto-tuned audio
    """
    if target_notes is None:
        # Use chromatic scale starting from C4 (261.63 Hz)
        base_freq = 261.63
        target_notes = [base_freq * (2 ** (i/12)) for i in range(12)]
    
    # Extract pitch using piptrack
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, fmin=80, fmax=800)
    
    # Get fundamental frequency for each frame
    f0 = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        f0.append(pitch if pitch > 0 else None)
    
    # Apply pitch correction
    corrected_audio = audio.copy()
    hop_length = 512
    
    for i, detected_pitch in enumerate(f0):
        if detected_pitch is not None and detected_pitch > 0:
            # Find closest target note
            closest_note = min(target_notes, key=lambda x: abs(x - detected_pitch))
            
            # Calculate correction needed
            correction_ratio = closest_note / detected_pitch
            correction_semitones = 12 * np.log2(correction_ratio)
            
            # Apply partial correction based on strength
            actual_correction = correction_semitones * correction_strength
            
            # Extract frame
            start_sample = i * hop_length
            end_sample = min(start_sample + hop_length, len(audio))
            
            if end_sample > start_sample:
                frame = audio[start_sample:end_sample]
                if len(frame) > 0:
                    corrected_frame = librosa.effects.pitch_shift(
                        frame, sr=sr, n_steps=actual_correction
                    )
                    corrected_audio[start_sample:end_sample] = corrected_frame
    
    return corrected_audio


def harmonize(audio: np.ndarray, sr: int, intervals: list,
             mix_levels: Optional[list] = None) -> np.ndarray:
    """
    Add harmonized voices to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    intervals : list
        List of intervals in semitones to add
    mix_levels : list, optional
        Mix levels for each harmony. If None, uses equal levels
        
    Returns:
    --------
    np.ndarray
        Audio with harmonies added
    """
    if mix_levels is None:
        mix_levels = [0.5] * len(intervals)
    
    if len(intervals) != len(mix_levels):
        raise ValueError("Number of intervals must match number of mix levels")
    
    # Start with original audio
    harmonized = audio.copy()
    
    # Add each harmony
    for interval, level in zip(intervals, mix_levels):
        harmony = pitch_shift(audio, sr, interval)
        harmonized += level * harmony
    
    # Normalize to prevent clipping
    peak = np.max(np.abs(harmonized))
    if peak > 1.0:
        harmonized = harmonized / peak
    
    return harmonized


def formant_shift(audio: np.ndarray, sr: int, shift_factor: float) -> np.ndarray:
    """
    Shift formants while preserving pitch (simplified implementation).
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    shift_factor : float
        Formant shift factor (>1.0 = higher formants, <1.0 = lower formants)
        
    Returns:
    --------
    np.ndarray
        Audio with shifted formants
    """
    # Use STFT for frequency domain processing
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Frequency bins
    freqs = librosa.fft_frequencies(sr=sr, n_fft=stft.shape[0]*2-1)
    
    # Create shifted magnitude spectrum
    shifted_magnitude = np.zeros_like(magnitude)
    
    for i in range(magnitude.shape[1]):  # For each time frame
        for j in range(magnitude.shape[0]):  # For each frequency bin
            # Calculate new frequency bin for this formant shift
            new_freq = freqs[j] * shift_factor
            
            # Find closest bin in original spectrum
            if new_freq < freqs[-1]:
                new_bin = np.argmin(np.abs(freqs - new_freq))
                if new_bin < magnitude.shape[0]:
                    shifted_magnitude[j, i] = magnitude[new_bin, i]
    
    # Reconstruct audio
    stft_shifted = shifted_magnitude * np.exp(1j * phase)
    shifted_audio = librosa.istft(stft_shifted)
    
    return shifted_audio


def vocoder_effect(carrier: np.ndarray, modulator: np.ndarray, sr: int,
                  num_bands: int = 16) -> np.ndarray:
    """
    Apply vocoder effect using carrier and modulator signals.
    
    Parameters:
    -----------
    carrier : np.ndarray
        Carrier signal (typically synthesizer)
    modulator : np.ndarray
        Modulator signal (typically voice)
    sr : int
        Sampling rate
    num_bands : int, default=16
        Number of frequency bands
        
    Returns:
    --------
    np.ndarray
        Vocoded audio
    """
    from scipy import signal
    
    # Make signals same length
    min_length = min(len(carrier), len(modulator))
    carrier = carrier[:min_length]
    modulator = modulator[:min_length]
    
    # Define frequency bands
    nyquist = sr / 2
    band_edges = np.logspace(np.log10(80), np.log10(nyquist), num_bands + 1)
    
    vocoded_signal = np.zeros_like(carrier)
    
    for i in range(num_bands):
        # Band-pass filter both signals
        low_freq = band_edges[i] / nyquist
        high_freq = band_edges[i + 1] / nyquist
        
        # Ensure valid frequency range
        low_freq = max(0.01, min(0.99, low_freq))
        high_freq = max(0.01, min(0.99, high_freq))
        
        if high_freq > low_freq:
            b, a = signal.butter(4, [low_freq, high_freq], btype='band')
            
            carrier_band = signal.filtfilt(b, a, carrier)
            modulator_band = signal.filtfilt(b, a, modulator)
            
            # Extract envelope from modulator
            envelope = np.abs(signal.hilbert(modulator_band))
            
            # Apply envelope to carrier
            vocoded_band = carrier_band * envelope
            
            vocoded_signal += vocoded_band
    
    # Normalize
    peak = np.max(np.abs(vocoded_signal))
    if peak > 0:
        vocoded_signal = vocoded_signal / peak
    
    return vocoded_signal