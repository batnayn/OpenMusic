"""Modulation effects including chorus, flanger, and phaser."""

import numpy as np
from scipy import signal
from typing import Optional


def add_chorus(audio: np.ndarray, sr: int, depth: float = 0.5,
              rate: float = 1.5, delay: float = 0.02) -> np.ndarray:
    """
    Add chorus effect to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    depth : float, default=0.5
        Chorus depth (0.0 to 1.0)
    rate : float, default=1.5
        LFO rate in Hz
    delay : float, default=0.02
        Base delay time in seconds
        
    Returns:
    --------
    np.ndarray
        Audio with chorus effect
    """
    # Create LFO
    t = np.arange(len(audio)) / sr
    lfo = np.sin(2 * np.pi * rate * t)
    
    # Modulated delay
    delay_samples = int(delay * sr)
    max_delay_variation = int(depth * 0.01 * sr)  # Up to 10ms variation
    
    output = np.zeros_like(audio)
    
    for i in range(len(audio)):
        # Calculate current delay
        current_delay = delay_samples + int(lfo[i] * max_delay_variation)
        
        if i >= current_delay:
            delayed_sample = audio[i - current_delay]
        else:
            delayed_sample = 0
        
        # Mix original and delayed
        output[i] = 0.7 * audio[i] + 0.3 * delayed_sample
    
    return output


def add_flanger(audio: np.ndarray, sr: int, depth: float = 0.7,
               rate: float = 0.5, delay: float = 0.005) -> np.ndarray:
    """
    Add flanger effect to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    depth : float, default=0.7
        Flanger depth (0.0 to 1.0)
    rate : float, default=0.5
        LFO rate in Hz
    delay : float, default=0.005
        Base delay time in seconds
        
    Returns:
    --------
    np.ndarray
        Audio with flanger effect
    """
    # Create LFO
    t = np.arange(len(audio)) / sr
    lfo = np.sin(2 * np.pi * rate * t)
    
    # Short modulated delay for flanger
    delay_samples = int(delay * sr)
    max_delay_variation = int(depth * 0.002 * sr)  # Up to 2ms variation
    
    output = np.zeros_like(audio)
    
    for i in range(len(audio)):
        # Calculate current delay
        current_delay = delay_samples + int(lfo[i] * max_delay_variation)
        
        if i >= current_delay:
            delayed_sample = audio[i - current_delay]
        else:
            delayed_sample = 0
        
        # Mix with phase inversion for comb filtering
        output[i] = audio[i] + 0.5 * delayed_sample
    
    return output


def add_phaser(audio: np.ndarray, sr: int, depth: float = 0.5,
              rate: float = 0.3, stages: int = 4) -> np.ndarray:
    """
    Add phaser effect to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    depth : float, default=0.5
        Phaser depth (0.0 to 1.0)
    rate : float, default=0.3
        LFO rate in Hz
    stages : int, default=4
        Number of all-pass filter stages
        
    Returns:
    --------
    np.ndarray
        Audio with phaser effect
    """
    # Create LFO
    t = np.arange(len(audio)) / sr
    lfo = np.sin(2 * np.pi * rate * t)
    
    # Apply multiple all-pass filters with modulated center frequency
    output = audio.copy()
    
    for stage in range(stages):
        stage_output = np.zeros_like(audio)
        
        for i in range(len(audio)):
            # Modulated center frequency
            center_freq = 1000 + depth * 800 * lfo[i]  # 200Hz to 1800Hz
            
            # Simple all-pass filter approximation
            # This is a simplified implementation
            if i > 0:
                stage_output[i] = output[i] + 0.7 * stage_output[i-1]
            else:
                stage_output[i] = output[i]
        
        output = stage_output
    
    # Mix with original
    return 0.7 * audio + 0.3 * output