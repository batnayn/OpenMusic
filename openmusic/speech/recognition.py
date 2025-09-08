"""
Speech recognition functionality for OpenMusic.
"""

import speech_recognition as sr
import numpy as np
import io
import wave
import librosa
from typing import Optional, Dict, List


def recognize_speech(audio: np.ndarray, sr: int, 
                    engine: str = "google", language: str = "en-US") -> Optional[str]:
    """
    Recognize speech from audio using various recognition engines.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    engine : str, default="google"
        Recognition engine ("google", "sphinx", "wit", "bing", "houndify", "ibm")
    language : str, default="en-US"
        Language code for recognition
        
    Returns:
    --------
    str or None
        Recognized text, or None if recognition failed
    """
    recognizer = sr.Recognizer()
    
    # Convert numpy array to AudioData
    audio_data = _numpy_to_audiodata(audio, sr)
    
    try:
        if engine == "google":
            text = recognizer.recognize_google(audio_data, language=language)
        elif engine == "sphinx":
            text = recognizer.recognize_sphinx(audio_data, language=language)
        elif engine == "wit":
            # Note: Requires API key
            text = recognizer.recognize_wit(audio_data)
        elif engine == "bing":
            # Note: Requires API key
            text = recognizer.recognize_bing(audio_data, language=language)
        elif engine == "houndify":
            # Note: Requires API key
            text = recognizer.recognize_houndify(audio_data)
        elif engine == "ibm":
            # Note: Requires API key
            text = recognizer.recognize_ibm(audio_data)
        else:
            raise ValueError(f"Unsupported recognition engine: {engine}")
            
        return text
        
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Could not request results from {engine} service; {e}")
        return None


def detect_voice_activity(audio: np.ndarray, sr: int, 
                         frame_length: int = 2048, hop_length: int = 512,
                         energy_threshold: float = 0.01) -> np.ndarray:
    """
    Detect voice activity in audio using energy-based approach.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    frame_length : int, default=2048
        Frame length for analysis
    hop_length : int, default=512
        Hop length for analysis
    energy_threshold : float, default=0.01
        Energy threshold for voice detection
        
    Returns:
    --------
    np.ndarray
        Boolean array indicating voice activity frames
    """
    # Calculate RMS energy
    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    
    # Normalize energy
    rms_normalized = rms / np.max(rms) if np.max(rms) > 0 else rms
    
    # Voice activity detection
    voice_activity = rms_normalized > energy_threshold
    
    return voice_activity


def segment_speech(audio: np.ndarray, sr: int, 
                  min_segment_length: float = 0.5) -> List[Dict]:
    """
    Segment audio into speech segments.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
    min_segment_length : float, default=0.5
        Minimum segment length in seconds
        
    Returns:
    --------
    list
        List of dictionaries with 'start', 'end', 'audio' keys
    """
    vad = detect_voice_activity(audio, sr)
    hop_length = 512
    
    # Convert frame indices to time
    times = librosa.frames_to_time(np.arange(len(vad)), sr=sr, hop_length=hop_length)
    
    segments = []
    start_time = None
    
    for i, (is_voice, time) in enumerate(zip(vad, times)):
        if is_voice and start_time is None:
            start_time = time
        elif not is_voice and start_time is not None:
            # End of segment
            duration = time - start_time
            if duration >= min_segment_length:
                start_sample = int(start_time * sr)
                end_sample = int(time * sr)
                segment_audio = audio[start_sample:end_sample]
                
                segments.append({
                    'start': start_time,
                    'end': time,
                    'audio': segment_audio
                })
            start_time = None
    
    # Handle case where audio ends with voice
    if start_time is not None:
        duration = times[-1] - start_time
        if duration >= min_segment_length:
            start_sample = int(start_time * sr)
            segment_audio = audio[start_sample:]
            
            segments.append({
                'start': start_time,
                'end': times[-1],
                'audio': segment_audio
            })
    
    return segments


def _numpy_to_audiodata(audio: np.ndarray, sr: int) -> sr.AudioData:
    """
    Convert numpy array to speech_recognition AudioData format.
    
    Parameters:
    -----------
    audio : np.ndarray
        Audio time series
    sr : int
        Sampling rate
        
    Returns:
    --------
    AudioData
        Audio data in speech_recognition format
    """
    # Ensure audio is in the right format
    audio = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sr)
        wav_file.writeframes(audio_int16.tobytes())
    
    wav_data = wav_buffer.getvalue()
    
    # Create AudioData object
    audio_data = sr.AudioData(wav_data, sr, 2)  # 2 bytes per sample
    
    return audio_data