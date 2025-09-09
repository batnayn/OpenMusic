"""
Music Generator for creating audio from musical parameters.

This module generates actual audio content based on the musical parameters
extracted from text prompts, creating melodies, harmonies, rhythms, and
applying appropriate effects.
"""

import numpy as np
import librosa
from typing import Dict, List, Optional, Tuple, Any
from ..core import normalize_audio
from ..effects import add_reverb, add_chorus
from ..enhancement import apply_lowpass, compress_audio

class MusicGenerator:
    """Generate audio content from musical parameters."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the music generator.
        
        Parameters:
        -----------
        sample_rate : int, default=22050
            Audio sample rate for generation
        """
        self.sample_rate = sample_rate
        
        # Musical note frequencies (equal temperament)
        self.note_frequencies = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }
        
        # Chord progressions for different moods/genres
        self.chord_progressions = {
            'happy': ['I', 'V', 'vi', 'IV'],  # Classic happy progression
            'sad': ['vi', 'IV', 'I', 'V'],    # Minor to major resolution
            'energetic': ['I', 'bVII', 'IV', 'I'], # Rock progression
            'calm': ['I', 'vi', 'ii', 'V'],   # Gentle progression
            'mysterious': ['i', 'bVI', 'bVII', 'i'], # Minor modal
            'romantic': ['I', 'vi', 'IV', 'V'], # Classic love song
        }
        
        # Scale degrees for different keys
        self.major_scale_degrees = [0, 2, 4, 5, 7, 9, 11]  # Semitones from root
        self.minor_scale_degrees = [0, 2, 3, 5, 7, 8, 10]  # Natural minor
        
        # Drum patterns for different genres
        self.drum_patterns = {
            'pop': [1, 0, 0, 1, 1, 0, 0, 1],      # Simple 4/4 pattern
            'rock': [1, 0, 1, 0, 1, 0, 1, 0],     # Rock beat
            'jazz': [1, 0, 0.5, 0, 0.7, 0, 0.5, 0], # Swing feel
            'electronic': [1, 0.3, 0.6, 0.3, 1, 0.3, 0.6, 0.3], # Electronic pattern
        }
    
    def generate_music(self, parameters: Dict[str, Any], duration: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate music from extracted parameters.
        
        Parameters:
        -----------
        parameters : dict
            Musical parameters from prompt processing
        duration : float
            Duration of music to generate in seconds
            
        Returns:
        --------
        audio : np.ndarray
            Generated audio data
        metadata : dict
            Generation metadata
        """
        # Create base musical elements
        melody = self._generate_melody(parameters, duration)
        harmony = self._generate_harmony(parameters, duration)
        rhythm = self._generate_rhythm(parameters, duration)
        bass = self._generate_bass(parameters, duration)
        
        # Mix the elements
        mix = self._mix_elements(melody, harmony, rhythm, bass, parameters)
        
        # Apply effects based on genre and mood
        processed = self._apply_effects(mix, parameters)
        
        # Final processing
        final_audio = self._finalize_audio(processed, parameters)
        
        metadata = {
            'duration': duration,
            'sample_rate': self.sample_rate,
            'parameters': parameters,
            'elements': ['melody', 'harmony', 'rhythm', 'bass']
        }
        
        return final_audio, metadata
    
    def _generate_melody(self, parameters: Dict[str, Any], duration: float) -> np.ndarray:
        """Generate a melody line based on parameters."""
        key = parameters.get('key', 'C')
        mood = parameters.get('mood', 'happy')
        tempo_bpm = parameters.get('tempo_bpm', 120)
        
        # Determine scale
        is_minor = 'm' in key
        root_note = key.replace('m', '')
        root_freq = self.note_frequencies.get(root_note, 261.63)
        
        scale_degrees = self.minor_scale_degrees if is_minor else self.major_scale_degrees
        
        # Generate note sequence
        note_duration = 60.0 / tempo_bpm  # Duration of each beat
        num_notes = int(duration / note_duration)
        
        melody_audio = np.zeros(int(duration * self.sample_rate))
        
        for i in range(num_notes):
            # Choose scale degree based on mood and musical logic
            if mood == 'happy':
                # Favor higher notes and major intervals
                degree_weights = [0.2, 0.1, 0.2, 0.15, 0.25, 0.05, 0.05]
            elif mood == 'sad':
                # Favor lower notes and minor intervals
                degree_weights = [0.25, 0.05, 0.2, 0.1, 0.15, 0.1, 0.15]
            else:
                # Balanced approach
                degree_weights = [0.15, 0.1, 0.15, 0.15, 0.2, 0.15, 0.1]
            
            # Select note
            degree_idx = np.random.choice(len(scale_degrees), p=degree_weights)
            semitone_offset = scale_degrees[degree_idx]
            
            # Add octave variation
            octave_shift = np.random.choice([-12, 0, 12], p=[0.1, 0.7, 0.2])
            note_freq = root_freq * (2 ** ((semitone_offset + octave_shift) / 12))
            
            # Generate note
            start_sample = int(i * note_duration * self.sample_rate)
            end_sample = int((i + 1) * note_duration * self.sample_rate)
            
            if end_sample > len(melody_audio):
                end_sample = len(melody_audio)
            
            note_length = end_sample - start_sample
            if note_length > 0:
                t = np.linspace(0, note_duration, note_length, False)
                
                # Create note with envelope
                note_signal = np.sin(2 * np.pi * note_freq * t)
                
                # Add subtle harmonics
                note_signal += 0.3 * np.sin(2 * np.pi * note_freq * 2 * t)
                note_signal += 0.1 * np.sin(2 * np.pi * note_freq * 3 * t)
                
                # Apply envelope
                envelope = np.exp(-t * 3) * (1 - np.exp(-t * 20))
                note_signal *= envelope
                
                melody_audio[start_sample:end_sample] = note_signal * 0.3
        
        return melody_audio
    
    def _generate_harmony(self, parameters: Dict[str, Any], duration: float) -> np.ndarray:
        """Generate harmonic accompaniment."""
        key = parameters.get('key', 'C')
        mood = parameters.get('mood', 'happy')
        tempo_bpm = parameters.get('tempo_bpm', 120)
        
        # Get chord progression
        progression = self.chord_progressions.get(mood, self.chord_progressions['happy'])
        
        # Generate harmony
        chord_duration = 60.0 / tempo_bpm * 4  # Each chord lasts 4 beats
        num_chords = int(duration / chord_duration) + 1
        
        harmony_audio = np.zeros(int(duration * self.sample_rate))
        
        root_freq = self.note_frequencies.get(key.replace('m', ''), 261.63)
        
        for i in range(num_chords):
            chord_type = progression[i % len(progression)]
            
            start_sample = int(i * chord_duration * self.sample_rate)
            end_sample = int((i + 1) * chord_duration * self.sample_rate)
            
            if start_sample >= len(harmony_audio):
                break
                
            if end_sample > len(harmony_audio):
                end_sample = len(harmony_audio)
            
            chord_length = end_sample - start_sample
            if chord_length > 0:
                t = np.linspace(0, chord_duration, chord_length, False)
                
                # Generate chord tones
                chord_signal = self._generate_chord(chord_type, root_freq, t)
                harmony_audio[start_sample:end_sample] = chord_signal * 0.2
        
        return harmony_audio
    
    def _generate_chord(self, chord_type: str, root_freq: float, t: np.ndarray) -> np.ndarray:
        """Generate a chord signal."""
        # Define chord intervals (in semitones from root)
        chord_intervals = {
            'I': [0, 4, 7],      # Major triad
            'ii': [2, 5, 9],     # Minor ii
            'iii': [4, 7, 11],   # Minor iii
            'IV': [5, 9, 0],     # Major IV
            'V': [7, 11, 2],     # Major V
            'vi': [9, 0, 4],     # Minor vi
            'vii': [11, 2, 5],   # Diminished vii
            'i': [0, 3, 7],      # Minor i
            'bVI': [8, 0, 3],    # Flat VI
            'bVII': [10, 2, 5],  # Flat VII
        }
        
        intervals = chord_intervals.get(chord_type, [0, 4, 7])
        
        chord_signal = np.zeros_like(t)
        for interval in intervals:
            freq = root_freq * (2 ** (interval / 12))
            chord_signal += np.sin(2 * np.pi * freq * t)
        
        # Apply gentle envelope
        envelope = 1 - np.exp(-t * 2)
        return chord_signal * envelope / len(intervals)
    
    def _generate_rhythm(self, parameters: Dict[str, Any], duration: float) -> np.ndarray:
        """Generate rhythmic elements (drums)."""
        genre = parameters.get('genre', 'pop')
        tempo_bpm = parameters.get('tempo_bpm', 120)
        
        # Get drum pattern
        pattern = self.drum_patterns.get(genre, self.drum_patterns['pop'])
        
        beat_duration = 60.0 / tempo_bpm
        pattern_duration = beat_duration * len(pattern)
        
        rhythm_audio = np.zeros(int(duration * self.sample_rate))
        
        # Generate drum sounds
        for i in range(int(duration / pattern_duration) + 1):
            for j, intensity in enumerate(pattern):
                if intensity > 0:
                    hit_time = i * pattern_duration + j * beat_duration
                    hit_sample = int(hit_time * self.sample_rate)
                    
                    if hit_sample < len(rhythm_audio):
                        # Generate kick/snare hit
                        drum_hit = self._generate_drum_hit(intensity, beat_duration)
                        hit_end = min(hit_sample + len(drum_hit), len(rhythm_audio))
                        actual_length = hit_end - hit_sample
                        
                        if actual_length > 0:
                            rhythm_audio[hit_sample:hit_end] += drum_hit[:actual_length]
        
        return rhythm_audio
    
    def _generate_drum_hit(self, intensity: float, duration: float) -> np.ndarray:
        """Generate a single drum hit."""
        hit_duration = min(duration * 0.2, 0.1)  # Short hit
        samples = int(hit_duration * self.sample_rate)
        t = np.linspace(0, hit_duration, samples, False)
        
        # Generate synthetic drum sound
        if intensity >= 1.0:  # Kick drum
            freq = 60  # Low frequency
            hit = np.sin(2 * np.pi * freq * t) * np.exp(-t * 30)
        else:  # Snare/hi-hat
            freq = 200  # Higher frequency
            hit = (np.random.normal(0, 1, samples) * np.exp(-t * 20) + 
                   np.sin(2 * np.pi * freq * t) * np.exp(-t * 25))
        
        return hit * intensity * 0.3
    
    def _generate_bass(self, parameters: Dict[str, Any], duration: float) -> np.ndarray:
        """Generate bass line."""
        key = parameters.get('key', 'C')
        tempo_bpm = parameters.get('tempo_bpm', 120)
        
        root_freq = self.note_frequencies.get(key.replace('m', ''), 261.63) / 4  # Bass octave
        
        beat_duration = 60.0 / tempo_bpm
        bass_audio = np.zeros(int(duration * self.sample_rate))
        
        # Simple bass pattern - root notes on beats
        num_beats = int(duration / beat_duration)
        
        for i in range(num_beats):
            start_sample = int(i * beat_duration * self.sample_rate)
            end_sample = int((i + 1) * beat_duration * self.sample_rate)
            
            if end_sample > len(bass_audio):
                end_sample = len(bass_audio)
            
            note_length = end_sample - start_sample
            if note_length > 0:
                t = np.linspace(0, beat_duration, note_length, False)
                
                # Simple bass note
                bass_note = np.sin(2 * np.pi * root_freq * t)
                bass_note += 0.3 * np.sin(2 * np.pi * root_freq * 2 * t)  # Octave harmonic
                
                # Apply envelope
                envelope = np.exp(-t * 5) * (1 - np.exp(-t * 50))
                bass_audio[start_sample:end_sample] = bass_note * envelope * 0.4
        
        return bass_audio
    
    def _mix_elements(self, melody: np.ndarray, harmony: np.ndarray, 
                     rhythm: np.ndarray, bass: np.ndarray, 
                     parameters: Dict[str, Any]) -> np.ndarray:
        """Mix all musical elements together."""
        # Ensure all elements are the same length
        max_length = max(len(melody), len(harmony), len(rhythm), len(bass))
        
        # Pad shorter elements
        melody_padded = np.pad(melody, (0, max_length - len(melody)))
        harmony_padded = np.pad(harmony, (0, max_length - len(harmony)))
        rhythm_padded = np.pad(rhythm, (0, max_length - len(rhythm)))
        bass_padded = np.pad(bass, (0, max_length - len(bass)))
        
        # Mix with appropriate levels
        genre = parameters.get('genre', 'pop')
        
        if genre == 'electronic':
            # Emphasize rhythm and bass
            mix = (melody_padded * 0.6 + harmony_padded * 0.4 + 
                   rhythm_padded * 0.8 + bass_padded * 0.7)
        elif genre == 'rock':
            # Emphasize melody and rhythm
            mix = (melody_padded * 0.8 + harmony_padded * 0.6 + 
                   rhythm_padded * 0.7 + bass_padded * 0.6)
        else:
            # Balanced mix
            mix = (melody_padded * 0.7 + harmony_padded * 0.5 + 
                   rhythm_padded * 0.6 + bass_padded * 0.5)
        
        return mix
    
    def _apply_effects(self, audio: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Apply effects based on genre and mood."""
        genre = parameters.get('genre', 'pop')
        mood = parameters.get('mood', 'happy')
        
        processed = audio.copy()
        
        # Apply reverb for spaciousness
        if genre in ['rock', 'pop']:
            processed = add_reverb(processed, self.sample_rate, room_size=0.3, wet_level=0.2)
        elif mood == 'mysterious':
            processed = add_reverb(processed, self.sample_rate, room_size=0.7, wet_level=0.4)
        
        # Apply chorus for richness
        if genre in ['pop', 'electronic']:
            processed = add_chorus(processed, self.sample_rate, depth=0.3, rate=0.5)
        
        # Low-pass filtering for warmth
        if mood in ['calm', 'romantic']:
            processed = apply_lowpass(processed, self.sample_rate, cutoff=8000)
        
        return processed
    
    def _finalize_audio(self, audio: np.ndarray, parameters: Dict[str, Any]) -> np.ndarray:
        """Final processing and normalization."""
        # Gentle compression
        compressed = compress_audio(audio, self.sample_rate, threshold=-15, ratio=2.0)
        
        # Normalize to appropriate level
        normalized = normalize_audio(compressed, target_level=-16.0)
        
        # Ensure we don't clip
        normalized = np.clip(normalized, -1.0, 1.0)
        
        return normalized