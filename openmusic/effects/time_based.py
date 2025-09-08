"""
Time-based audio effects including reverb, delay, and echo.
"""

import numpy as np
from scipy import signal
from typing import Optional


def add_reverb(audio: np.ndarray, sr: int, room_size: float = 0.5,
              damping: float = 0.5, wet_level: float = 0.3,
              dry_level: float = 0.7) -> np.ndarray:
    """
    Add reverb effect to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    room_size : float, default=0.5
        Room size (0.0 to 1.0)
    damping : float, default=0.5
        High frequency damping (0.0 to 1.0)
    wet_level : float, default=0.3
        Wet signal level (0.0 to 1.0)
    dry_level : float, default=0.7
        Dry signal level (0.0 to 1.0)
        
    Returns:
    --------
    np.ndarray
        Audio with reverb effect
    """
    # Create impulse response for reverb
    reverb_time = room_size * 2.0 + 0.5  # 0.5 to 2.5 seconds
    decay_samples = int(reverb_time * sr)
    
    # Generate exponentially decaying noise as impulse response
    t = np.arange(decay_samples) / sr
    decay_envelope = np.exp(-t / reverb_time)
    
    # Create filtered noise for natural reverb character
    noise = np.random.normal(0, 1, decay_samples)
    
    # Apply damping (low-pass filter)
    cutoff_freq = 8000 * (1 - damping) + 1000  # 1kHz to 8kHz
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(2, normalized_cutoff, btype='low')
    filtered_noise = signal.filtfilt(b, a, noise)
    
    # Create impulse response
    impulse_response = filtered_noise * decay_envelope
    impulse_response = impulse_response / np.max(np.abs(impulse_response))
    
    # Convolve audio with impulse response
    reverb_signal = signal.convolve(audio, impulse_response, mode='same')
    
    # Mix dry and wet signals
    output = dry_level * audio + wet_level * reverb_signal
    
    return output


def add_delay(audio: np.ndarray, sr: int, delay_time: float = 0.3,
             feedback: float = 0.3, wet_level: float = 0.3,
             dry_level: float = 0.7) -> np.ndarray:
    """
    Add delay effect to audio.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    delay_time : float, default=0.3
        Delay time in seconds
    feedback : float, default=0.3
        Feedback amount (0.0 to 0.95)
    wet_level : float, default=0.3
        Wet signal level (0.0 to 1.0)
    dry_level : float, default=0.7
        Dry signal level (0.0 to 1.0)
        
    Returns:
    --------
    np.ndarray
        Audio with delay effect
    """
    delay_samples = int(delay_time * sr)
    
    # Create delay buffer
    delay_buffer = np.zeros(len(audio) + delay_samples)
    output = np.zeros_like(delay_buffer)
    
    # Apply delay with feedback
    for i in range(len(audio)):
        # Current input sample
        input_sample = audio[i]
        
        # Delayed sample (with feedback)
        if i >= delay_samples:
            delayed_sample = delay_buffer[i - delay_samples]
        else:
            delayed_sample = 0
        
        # Mix input with delayed feedback
        delay_buffer[i] = input_sample + feedback * delayed_sample
        
        # Output mix
        output[i] = dry_level * input_sample + wet_level * delayed_sample
    
    # Trim to original length
    return output[:len(audio)]


def add_echo(audio: np.ndarray, sr: int, delay_time: float = 0.5,
            decay: float = 0.5, num_echoes: int = 3) -> np.ndarray:
    """
    Add echo effect with multiple repetitions.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    delay_time : float, default=0.5
        Time between echoes in seconds
    decay : float, default=0.5
        Decay factor for each echo (0.0 to 1.0)
    num_echoes : int, default=3
        Number of echo repetitions
        
    Returns:
    --------
    np.ndarray
        Audio with echo effect
    """
    delay_samples = int(delay_time * sr)
    total_length = len(audio) + delay_samples * num_echoes
    
    output = np.zeros(total_length)
    
    # Add original signal
    output[:len(audio)] = audio
    
    # Add echoes
    for echo_num in range(1, num_echoes + 1):
        echo_gain = decay ** echo_num
        echo_delay = delay_samples * echo_num
        
        start_idx = echo_delay
        end_idx = start_idx + len(audio)
        
        if end_idx <= total_length:
            output[start_idx:end_idx] += echo_gain * audio
    
    # Trim to reasonable length
    return output[:len(audio) + delay_samples * 2]


def add_hall_reverb(audio: np.ndarray, sr: int, 
                   pre_delay: float = 0.02,
                   decay_time: float = 2.0,
                   high_freq_decay: float = 0.5) -> np.ndarray:
    """
    Add hall-style reverb with pre-delay.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    pre_delay : float, default=0.02
        Pre-delay time in seconds
    decay_time : float, default=2.0
        Reverb decay time in seconds
    high_freq_decay : float, default=0.5
        High frequency decay factor
        
    Returns:
    --------
    np.ndarray
        Audio with hall reverb effect
    """
    # Apply pre-delay
    pre_delay_samples = int(pre_delay * sr)
    pre_delayed = np.concatenate([np.zeros(pre_delay_samples), audio])
    
    # Create complex impulse response
    decay_samples = int(decay_time * sr)
    t = np.arange(decay_samples) / sr
    
    # Multiple decay envelopes for complexity
    early_reflections = np.exp(-t / 0.1) * np.random.normal(0, 1, decay_samples)
    late_reverb = np.exp(-t / decay_time) * np.random.normal(0, 1, decay_samples)
    
    # Apply high frequency decay
    nyquist = sr / 2
    hf_cutoff = 4000 * high_freq_decay + 1000
    normalized_cutoff = hf_cutoff / nyquist
    b, a = signal.butter(2, normalized_cutoff, btype='low')
    
    early_reflections = signal.filtfilt(b, a, early_reflections)
    late_reverb = signal.filtfilt(b, a, late_reverb)
    
    # Combine early and late components
    impulse_response = 0.3 * early_reflections + 0.7 * late_reverb
    impulse_response = impulse_response / np.max(np.abs(impulse_response))
    
    # Convolve and mix
    reverb_signal = signal.convolve(pre_delayed, impulse_response, mode='same')
    output = 0.7 * pre_delayed + 0.3 * reverb_signal
    
    return output[:len(audio)]


def add_spring_reverb(audio: np.ndarray, sr: int,
                     tank_size: float = 0.5,
                     dispersion: float = 0.3) -> np.ndarray:
    """
    Simulate spring reverb tank effect.
    
    Parameters:
    -----------
    audio : np.ndarray
        Input audio signal
    sr : int
        Sampling rate
    tank_size : float, default=0.5
        Spring tank size simulation (0.0 to 1.0)
    dispersion : float, default=0.3
        Frequency dispersion amount
        
    Returns:
    --------
    np.ndarray
        Audio with spring reverb effect
    """
    # Create spring-like impulse response
    decay_time = tank_size * 1.5 + 0.2  # 0.2 to 1.7 seconds
    decay_samples = int(decay_time * sr)
    
    t = np.arange(decay_samples) / sr
    
    # Spring characteristics: boingy resonances
    fundamental_freq = 100 * (1 - tank_size) + 50
    resonance_freqs = [fundamental_freq * (i + 1) for i in range(5)]
    
    impulse_response = np.zeros(decay_samples)
    
    for freq in resonance_freqs:
        # Decaying sinusoid for each resonance
        resonance = np.sin(2 * np.pi * freq * t) * np.exp(-t / decay_time)
        impulse_response += resonance / len(resonance_freqs)
    
    # Add dispersive delay
    if dispersion > 0:
        # High frequencies arrive slightly later
        delay_samples = int(dispersion * 0.01 * sr)  # Up to 10ms
        if delay_samples > 0:
            impulse_response = np.concatenate([
                np.zeros(delay_samples), 
                impulse_response[:-delay_samples]
            ])
    
    # Normalize
    impulse_response = impulse_response / np.max(np.abs(impulse_response))
    
    # Convolve and mix
    spring_signal = signal.convolve(audio, impulse_response, mode='same')
    output = 0.8 * audio + 0.2 * spring_signal
    
    return output