"""
Text-to-Music Generation Module for OpenMusic

This module provides AI-powered music generation capabilities that allow users 
to create original songs from text prompts, implementing the core functionality
described in the OpenMusic platform concept.
"""

from .prompt_processor import PromptProcessor
from .music_generator import MusicGenerator
from .generation_engine import GenerationEngine

__all__ = [
    "PromptProcessor",
    "MusicGenerator", 
    "GenerationEngine",
    "generate_music_from_prompt",
    "generate_song",
]

# Convenience function for direct text-to-music generation
def generate_music_from_prompt(prompt: str, duration: float = 30.0, **kwargs):
    """
    Generate music from a text prompt.
    
    Parameters:
    -----------
    prompt : str
        Text description of the desired music
    duration : float, default=30.0
        Length of generated music in seconds
    **kwargs : dict
        Additional generation parameters
        
    Returns:
    --------
    audio : np.ndarray
        Generated audio data
    sr : int
        Sample rate
    metadata : dict
        Generation metadata and parameters
    """
    engine = GenerationEngine()
    return engine.generate_from_prompt(prompt, duration=duration, **kwargs)

def generate_song(prompt: str, structure: str = "verse-chorus-verse-chorus-bridge-chorus", **kwargs):
    """
    Generate a complete song from a text prompt with structure.
    
    Parameters:
    -----------
    prompt : str
        Text description of the desired song
    structure : str, default="verse-chorus-verse-chorus-bridge-chorus"
        Song structure specification
    **kwargs : dict
        Additional generation parameters
        
    Returns:
    --------
    audio : np.ndarray
        Generated song audio
    sr : int
        Sample rate
    metadata : dict
        Song metadata including structure, lyrics, etc.
    """
    engine = GenerationEngine()
    return engine.generate_song(prompt, structure=structure, **kwargs)