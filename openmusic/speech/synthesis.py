"""
Text-to-speech synthesis functionality for OpenMusic.
"""

import pyttsx3
import numpy as np
import tempfile
import os
import librosa
from typing import List, Dict, Optional


def synthesize_text(text: str, voice: Optional[str] = None, 
                   rate: int = 200, volume: float = 0.9,
                   output_sr: int = 22050) -> tuple[np.ndarray, int]:
    """
    Convert text to speech audio.
    
    Parameters:
    -----------
    text : str
        Text to synthesize
    voice : str, optional
        Voice ID to use. If None, uses default voice
    rate : int, default=200
        Speech rate (words per minute)
    volume : float, default=0.9
        Speech volume (0.0 to 1.0)
    output_sr : int, default=22050
        Output sample rate
        
    Returns:
    --------
    tuple
        (audio_array, sample_rate)
    """
    engine = pyttsx3.init()
    
    # Set properties
    engine.setProperty('rate', rate)
    engine.setProperty('volume', volume)
    
    if voice:
        voices = engine.getProperty('voices')
        for v in voices:
            if voice in v.id or voice in v.name:
                engine.setProperty('voice', v.id)
                break
    
    # Create temporary file for audio output
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Save speech to file
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        
        # Load audio file
        audio, sr = librosa.load(temp_path, sr=output_sr)
        
        return audio, sr
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        engine.stop()


def list_voices() -> List[Dict[str, str]]:
    """
    Get list of available voices.
    
    Returns:
    --------
    list
        List of dictionaries with voice information
    """
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    voice_list = []
    for voice in voices:
        voice_info = {
            'id': voice.id,
            'name': voice.name,
            'age': getattr(voice, 'age', 'Unknown'),
            'gender': getattr(voice, 'gender', 'Unknown'),
            'languages': getattr(voice, 'languages', [])
        }
        voice_list.append(voice_info)
    
    engine.stop()
    return voice_list


def synthesize_with_effects(text: str, voice: Optional[str] = None,
                          rate: int = 200, volume: float = 0.9,
                          pitch_shift: float = 0.0, 
                          time_stretch: float = 1.0,
                          output_sr: int = 22050) -> tuple[np.ndarray, int]:
    """
    Synthesize text with audio effects applied.
    
    Parameters:
    -----------
    text : str
        Text to synthesize
    voice : str, optional
        Voice ID to use
    rate : int, default=200
        Speech rate (words per minute)
    volume : float, default=0.9
        Speech volume (0.0 to 1.0)
    pitch_shift : float, default=0.0
        Pitch shift in semitones
    time_stretch : float, default=1.0
        Time stretch factor (>1.0 = slower, <1.0 = faster)
    output_sr : int, default=22050
        Output sample rate
        
    Returns:
    --------
    tuple
        (processed_audio_array, sample_rate)
    """
    # Generate base speech
    audio, sr = synthesize_text(text, voice, rate, volume, output_sr)
    
    # Apply pitch shift if specified
    if pitch_shift != 0.0:
        audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_shift)
    
    # Apply time stretch if specified
    if time_stretch != 1.0:
        audio = librosa.effects.time_stretch(audio, rate=time_stretch)
    
    return audio, sr


def synthesize_phonemes(phonemes: List[str], voice: Optional[str] = None,
                       rate: int = 200, volume: float = 0.9,
                       output_sr: int = 22050) -> tuple[np.ndarray, int]:
    """
    Synthesize speech from phoneme sequence.
    
    Parameters:
    -----------
    phonemes : list
        List of phoneme strings
    voice : str, optional
        Voice ID to use
    rate : int, default=200
        Speech rate
    volume : float, default=0.9
        Speech volume
    output_sr : int, default=22050
        Output sample rate
        
    Returns:
    --------
    tuple
        (audio_array, sample_rate)
    """
    # Convert phonemes to approximate text representation
    # This is a simplified approach - a full implementation would use
    # phoneme-to-speech synthesis
    phoneme_text = " ".join(phonemes)
    
    return synthesize_text(phoneme_text, voice, rate, volume, output_sr)


def batch_synthesize(texts: List[str], voice: Optional[str] = None,
                    rate: int = 200, volume: float = 0.9,
                    output_sr: int = 22050) -> List[tuple[np.ndarray, int]]:
    """
    Synthesize multiple texts in batch.
    
    Parameters:
    -----------
    texts : list
        List of texts to synthesize
    voice : str, optional
        Voice ID to use
    rate : int, default=200
        Speech rate
    volume : float, default=0.9
        Speech volume
    output_sr : int, default=22050
        Output sample rate
        
    Returns:
    --------
    list
        List of (audio_array, sample_rate) tuples
    """
    results = []
    for text in texts:
        audio, sr = synthesize_text(text, voice, rate, volume, output_sr)
        results.append((audio, sr))
    
    return results