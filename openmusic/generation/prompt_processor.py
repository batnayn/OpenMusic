"""
Prompt Processor for text-to-music generation.

This module handles the interpretation of text prompts and extracts musical
parameters such as genre, mood, tempo, key, instrumentation, and other
musical characteristics.
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

class PromptProcessor:
    """Process text prompts to extract musical parameters."""
    
    def __init__(self):
        """Initialize the prompt processor with musical knowledge."""
        self.genre_keywords = {
            'pop': ['pop', 'catchy', 'mainstream', 'radio-friendly'],
            'rock': ['rock', 'guitar', 'electric', 'powerful', 'driving'],
            'jazz': ['jazz', 'swing', 'bebop', 'smooth', 'improvisation'],
            'classical': ['classical', 'orchestral', 'symphony', 'piano', 'violin'],
            'electronic': ['electronic', 'synth', 'edm', 'techno', 'house', 'digital'],
            'folk': ['folk', 'acoustic', 'traditional', 'simple', 'storytelling'],
            'country': ['country', 'twang', 'banjo', 'rural', 'southern'],
            'hip-hop': ['hip-hop', 'rap', 'beats', 'urban', 'rhythmic'],
            'r&b': ['r&b', 'soul', 'groove', 'smooth', 'vocal'],
            'blues': ['blues', 'melancholy', 'twelve-bar', 'emotional'],
            'reggae': ['reggae', 'island', 'caribbean', 'relaxed', 'rhythm'],
            'metal': ['metal', 'heavy', 'intense', 'aggressive', 'distorted']
        }
        
        self.mood_keywords = {
            'happy': ['happy', 'joyful', 'upbeat', 'cheerful', 'bright', 'positive', 'energetic'],
            'sad': ['sad', 'melancholy', 'sorrowful', 'depressing', 'blue', 'tragic'],
            'energetic': ['energetic', 'lively', 'dynamic', 'powerful', 'driving', 'intense'],
            'calm': ['calm', 'peaceful', 'relaxing', 'serene', 'gentle', 'soothing'],
            'romantic': ['romantic', 'love', 'intimate', 'tender', 'passionate'],
            'mysterious': ['mysterious', 'dark', 'enigmatic', 'haunting', 'ethereal'],
            'aggressive': ['aggressive', 'angry', 'fierce', 'hostile', 'violent'],
            'nostalgic': ['nostalgic', 'wistful', 'reminiscent', 'longing', 'memories']
        }
        
        self.tempo_keywords = {
            'very_slow': ['very slow', 'extremely slow', 'glacial', 'funeral'],
            'slow': ['slow', 'ballad', 'gentle', 'relaxed', 'moderate'],
            'medium': ['medium', 'moderate', 'walking', 'steady'],
            'fast': ['fast', 'quick', 'lively', 'upbeat', 'brisk'],
            'very_fast': ['very fast', 'extremely fast', 'blazing', 'frantic']
        }
        
        self.key_keywords = {
            'C': ['C major', 'C'], 'Am': ['A minor', 'Am'],
            'G': ['G major', 'G'], 'Em': ['E minor', 'Em'],
            'D': ['D major', 'D'], 'Bm': ['B minor', 'Bm'],
            'A': ['A major', 'A'], 'F#m': ['F# minor', 'F#m'],
            'E': ['E major', 'E'], 'C#m': ['C# minor', 'C#m'],
            'B': ['B major', 'B'], 'G#m': ['G# minor', 'G#m'],
            'F': ['F major', 'F'], 'Dm': ['D minor', 'Dm'],
            'Bb': ['Bb major', 'Bb'], 'Gm': ['G minor', 'Gm'],
            'Eb': ['Eb major', 'Eb'], 'Cm': ['C minor', 'Cm'],
            'Ab': ['Ab major', 'Ab'], 'Fm': ['F minor', 'Fm']
        }
        
        self.instrument_keywords = {
            'piano': ['piano', 'keyboard', 'keys'],
            'guitar': ['guitar', 'acoustic guitar', 'electric guitar'],
            'drums': ['drums', 'percussion', 'beat', 'rhythm section'],
            'bass': ['bass', 'bass guitar', 'upright bass'],
            'violin': ['violin', 'strings', 'orchestral strings'],
            'trumpet': ['trumpet', 'brass', 'horn'],
            'saxophone': ['saxophone', 'sax'],
            'synth': ['synthesizer', 'synth', 'electronic'],
            'vocals': ['vocals', 'voice', 'singing', 'singer']
        }
        
        self.tempo_bpm_mapping = {
            'very_slow': (40, 60),
            'slow': (60, 80),
            'medium': (80, 120),
            'fast': (120, 160),
            'very_fast': (160, 200)
        }
    
    def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Process a text prompt and extract musical parameters.
        
        Parameters:
        -----------
        prompt : str
            The input text prompt describing the desired music
            
        Returns:
        --------
        dict
            Dictionary containing extracted musical parameters
        """
        prompt_lower = prompt.lower()
        
        # Extract basic parameters
        genre = self._extract_genre(prompt_lower)
        mood = self._extract_mood(prompt_lower)
        tempo_category = self._extract_tempo_category(prompt_lower)
        key = self._extract_key(prompt_lower)
        instruments = self._extract_instruments(prompt_lower)
        
        # Generate specific tempo BPM
        tempo_bpm = self._generate_tempo_bpm(tempo_category, mood, genre)
        
        # Extract additional musical elements
        structure = self._extract_structure(prompt_lower)
        vocals = self._detect_vocals(prompt_lower)
        
        return {
            'genre': genre,
            'mood': mood,
            'tempo_category': tempo_category,
            'tempo_bpm': tempo_bpm,
            'key': key,
            'instruments': instruments,
            'structure': structure,
            'vocals': vocals,
            'original_prompt': prompt
        }
    
    def _extract_genre(self, prompt: str) -> str:
        """Extract genre from prompt text."""
        genre_scores = {}
        
        for genre, keywords in self.genre_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt)
            if score > 0:
                genre_scores[genre] = score
        
        if genre_scores:
            return max(genre_scores, key=genre_scores.get)
        else:
            return 'pop'  # Default genre
    
    def _extract_mood(self, prompt: str) -> str:
        """Extract mood from prompt text."""
        mood_scores = {}
        
        for mood, keywords in self.mood_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt)
            if score > 0:
                mood_scores[mood] = score
        
        if mood_scores:
            return max(mood_scores, key=mood_scores.get)
        else:
            return 'happy'  # Default mood
    
    def _extract_tempo_category(self, prompt: str) -> str:
        """Extract tempo category from prompt text."""
        for tempo_cat, keywords in self.tempo_keywords.items():
            if any(keyword in prompt for keyword in keywords):
                return tempo_cat
        
        return 'medium'  # Default tempo
    
    def _extract_key(self, prompt: str) -> str:
        """Extract musical key from prompt text."""
        for key, keywords in self.key_keywords.items():
            if any(keyword.lower() in prompt for keyword in keywords):
                return key
        
        # Default key based on mood
        mood = self._extract_mood(prompt)
        if mood in ['sad', 'melancholy', 'mysterious']:
            return 'Am'  # Minor key for sad moods
        else:
            return 'C'   # Major key for happy moods
    
    def _extract_instruments(self, prompt: str) -> List[str]:
        """Extract mentioned instruments from prompt text."""
        found_instruments = []
        
        for instrument, keywords in self.instrument_keywords.items():
            if any(keyword in prompt for keyword in keywords):
                found_instruments.append(instrument)
        
        # Default instruments based on genre if none specified
        if not found_instruments:
            genre = self._extract_genre(prompt)
            found_instruments = self._get_default_instruments(genre)
        
        return found_instruments
    
    def _get_default_instruments(self, genre: str) -> List[str]:
        """Get default instruments for a genre."""
        defaults = {
            'pop': ['piano', 'guitar', 'drums', 'bass', 'vocals'],
            'rock': ['guitar', 'drums', 'bass', 'vocals'],
            'jazz': ['piano', 'bass', 'drums', 'saxophone'],
            'classical': ['piano', 'violin'],
            'electronic': ['synth', 'drums'],
            'folk': ['guitar', 'vocals'],
            'country': ['guitar', 'vocals'],
            'hip-hop': ['drums', 'bass', 'vocals'],
            'r&b': ['piano', 'bass', 'drums', 'vocals'],
            'blues': ['guitar', 'piano', 'vocals'],
            'reggae': ['guitar', 'bass', 'drums'],
            'metal': ['guitar', 'drums', 'bass']
        }
        return defaults.get(genre, ['piano', 'guitar', 'drums'])
    
    def _generate_tempo_bpm(self, tempo_category: str, mood: str, genre: str) -> int:
        """Generate specific BPM based on tempo category, mood, and genre."""
        bpm_range = self.tempo_bpm_mapping.get(tempo_category, (80, 120))
        base_bpm = np.random.randint(bpm_range[0], bpm_range[1] + 1)
        
        # Adjust based on mood
        if mood == 'energetic':
            base_bpm = min(base_bpm + 20, 180)
        elif mood == 'calm':
            base_bpm = max(base_bpm - 15, 40)
        
        # Adjust based on genre
        if genre == 'metal':
            base_bpm = max(base_bpm, 120)
        elif genre == 'classical':
            base_bpm = min(base_bpm, 140)
        
        return base_bpm
    
    def _extract_structure(self, prompt: str) -> str:
        """Extract or infer song structure from prompt."""
        if 'verse' in prompt and 'chorus' in prompt:
            return 'verse-chorus-verse-chorus-bridge-chorus'
        elif 'instrumental' in prompt:
            return 'intro-theme-variation-theme-outro'
        else:
            return 'intro-main-variation-main-outro'
    
    def _detect_vocals(self, prompt: str) -> Dict[str, Any]:
        """Detect vocal requirements from prompt."""
        has_vocals = any(word in prompt for word in ['vocals', 'singing', 'voice', 'singer', 'lyrics'])
        
        vocal_style = 'normal'
        if 'powerful' in prompt or 'strong' in prompt:
            vocal_style = 'powerful'
        elif 'soft' in prompt or 'gentle' in prompt:
            vocal_style = 'soft'
        elif 'rap' in prompt or 'hip-hop' in prompt:
            vocal_style = 'rap'
        
        return {
            'has_vocals': has_vocals,
            'style': vocal_style,
            'gender': 'mixed'  # Default to mixed
        }