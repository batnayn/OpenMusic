"""
Generation Engine for OpenMusic text-to-music system.

This is the main orchestrator that combines prompt processing and music generation
to create complete songs from text descriptions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from .prompt_processor import PromptProcessor
from .music_generator import MusicGenerator
from ..speech import synthesize_text
from ..core import save_audio

class GenerationEngine:
    """Main engine for text-to-music generation."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the generation engine.
        
        Parameters:
        -----------
        sample_rate : int, default=22050
            Audio sample rate for generation
        """
        self.sample_rate = sample_rate
        self.prompt_processor = PromptProcessor()
        self.music_generator = MusicGenerator(sample_rate)
        
        # Song structure templates
        self.structure_templates = {
            'verse-chorus-verse-chorus-bridge-chorus': {
                'sections': ['verse', 'chorus', 'verse', 'chorus', 'bridge', 'chorus'],
                'durations': [16, 12, 16, 12, 8, 12]  # seconds
            },
            'intro-verse-chorus-verse-chorus-outro': {
                'sections': ['intro', 'verse', 'chorus', 'verse', 'chorus', 'outro'],
                'durations': [8, 16, 12, 16, 12, 8]
            },
            'intro-theme-variation-theme-outro': {
                'sections': ['intro', 'theme', 'variation', 'theme', 'outro'],
                'durations': [8, 20, 16, 20, 8]
            },
            'intro-main-variation-main-outro': {
                'sections': ['intro', 'main', 'variation', 'main', 'outro'],
                'durations': [6, 18, 12, 18, 6]
            }
        }
    
    def generate_from_prompt(self, prompt: str, duration: float = 30.0, **kwargs) -> Tuple[np.ndarray, int, Dict[str, Any]]:
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
        # Process the prompt
        parameters = self.prompt_processor.process_prompt(prompt)
        
        # Override parameters with any provided kwargs
        parameters.update(kwargs)
        
        # Generate the music
        audio, generation_metadata = self.music_generator.generate_music(parameters, duration)
        
        # Compile final metadata
        metadata = {
            'prompt': prompt,
            'duration': duration,
            'sample_rate': self.sample_rate,
            'extracted_parameters': parameters,
            'generation_info': generation_metadata,
            'type': 'music_from_prompt'
        }
        
        return audio, self.sample_rate, metadata
    
    def generate_song(self, prompt: str, structure: str = "verse-chorus-verse-chorus-bridge-chorus", 
                     **kwargs) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Generate a complete song with structure from a text prompt.
        
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
            Song metadata including structure, sections, etc.
        """
        # Process the prompt
        base_parameters = self.prompt_processor.process_prompt(prompt)
        base_parameters.update(kwargs)
        
        # Get structure template
        template = self.structure_templates.get(structure, 
                                               self.structure_templates['verse-chorus-verse-chorus-bridge-chorus'])
        
        sections = template['sections']
        durations = template['durations']
        
        # Generate each section
        song_sections = []
        section_metadata = []
        
        for i, (section_type, section_duration) in enumerate(zip(sections, durations)):
            # Modify parameters for each section type
            section_params = self._adjust_parameters_for_section(base_parameters.copy(), section_type)
            
            # Generate section audio
            section_audio, section_meta = self.music_generator.generate_music(section_params, section_duration)
            
            song_sections.append(section_audio)
            section_metadata.append({
                'type': section_type,
                'duration': section_duration,
                'parameters': section_params,
                'index': i
            })
        
        # Combine sections with smooth transitions
        full_song = self._combine_sections_with_transitions(song_sections)
        
        # Calculate total duration
        total_duration = sum(durations)
        
        # Compile song metadata
        metadata = {
            'prompt': prompt,
            'structure': structure,
            'total_duration': total_duration,
            'sample_rate': self.sample_rate,
            'base_parameters': base_parameters,
            'sections': section_metadata,
            'type': 'complete_song'
        }
        
        return full_song, self.sample_rate, metadata
    
    def generate_with_lyrics(self, prompt: str, lyrics: str, **kwargs) -> Tuple[np.ndarray, int, Dict[str, Any]]:
        """
        Generate music with synthesized vocals from lyrics.
        
        Parameters:
        -----------
        prompt : str
            Text description of the musical style
        lyrics : str
            Lyrics to be sung
        **kwargs : dict
            Additional generation parameters
            
        Returns:
        --------
        audio : np.ndarray
            Generated audio with vocals
        sr : int
            Sample rate
        metadata : dict
            Generation metadata
        """
        # Generate instrumental track
        instrumental, sr, music_metadata = self.generate_from_prompt(prompt, **kwargs)
        
        # Generate vocals from lyrics
        vocal_params = music_metadata['extracted_parameters']['vocals']
        
        try:
            # Synthesize speech/vocals
            vocal_audio, vocal_sr = synthesize_text(lyrics, rate=120, volume=0.7)
            
            # Resample vocal audio to match instrumental
            if vocal_sr != sr:
                import librosa
                vocal_audio = librosa.resample(vocal_audio, orig_sr=vocal_sr, target_sr=sr)
            
            # Match lengths (loop or trim vocals to fit instrumental)
            if len(vocal_audio) < len(instrumental):
                # Loop vocals if too short
                repeats = int(np.ceil(len(instrumental) / len(vocal_audio)))
                vocal_audio = np.tile(vocal_audio, repeats)[:len(instrumental)]
            else:
                # Trim vocals if too long
                vocal_audio = vocal_audio[:len(instrumental)]
            
            # Mix vocals with instrumental
            mixed_audio = instrumental * 0.7 + vocal_audio * 0.3
            
            # Ensure no clipping
            mixed_audio = np.clip(mixed_audio, -1.0, 1.0)
            
            # Update metadata
            music_metadata['has_vocals'] = True
            music_metadata['lyrics'] = lyrics
            music_metadata['vocal_parameters'] = vocal_params
            
        except Exception as e:
            print(f"Warning: Could not generate vocals: {e}")
            # Return instrumental only
            mixed_audio = instrumental
            music_metadata['has_vocals'] = False
            music_metadata['vocal_generation_error'] = str(e)
        
        return mixed_audio, sr, music_metadata
    
    def _adjust_parameters_for_section(self, params: Dict[str, Any], section_type: str) -> Dict[str, Any]:
        """Adjust musical parameters for different song sections."""
        adjusted = params.copy()
        
        if section_type == 'intro':
            # Intro: Simpler, build-up feel
            adjusted['tempo_bpm'] = int(params['tempo_bpm'] * 0.9)  # Slightly slower
            
        elif section_type == 'verse':
            # Verse: Standard parameters, perhaps more subdued
            pass  # Keep base parameters
            
        elif section_type == 'chorus':
            # Chorus: More energetic, fuller sound
            adjusted['tempo_bpm'] = int(params['tempo_bpm'] * 1.05)  # Slightly faster
            if params['mood'] != 'calm':  # Don't make calm songs too energetic
                adjusted['mood'] = 'energetic'
                
        elif section_type == 'bridge':
            # Bridge: Different feel, perhaps different key
            # Modulate to relative minor/major
            if 'm' in params['key']:
                # If minor, go to relative major
                adjusted['key'] = params['key'].replace('m', '')
            else:
                # If major, go to relative minor
                adjusted['key'] = params['key'] + 'm'
                
        elif section_type == 'outro':
            # Outro: Wind down, perhaps slower
            adjusted['tempo_bpm'] = int(params['tempo_bpm'] * 0.85)
            adjusted['mood'] = 'calm'
            
        elif section_type == 'variation':
            # Variation: Change some element
            adjusted['tempo_bpm'] = int(params['tempo_bpm'] * 1.1)
            
        return adjusted
    
    def _combine_sections_with_transitions(self, sections: List[np.ndarray]) -> np.ndarray:
        """Combine song sections with smooth transitions."""
        if not sections:
            return np.array([])
        
        if len(sections) == 1:
            return sections[0]
        
        # Calculate transition length (0.5 seconds)
        transition_samples = int(0.5 * self.sample_rate)
        
        combined = []
        
        for i, section in enumerate(sections):
            if i == 0:
                # First section: add as-is
                combined.append(section)
            else:
                # Create smooth transition from previous section
                prev_section = combined[-1]
                
                # Extract end of previous section and beginning of current
                if len(prev_section) >= transition_samples:
                    prev_end = prev_section[-transition_samples:]
                else:
                    prev_end = prev_section
                
                if len(section) >= transition_samples:
                    curr_start = section[:transition_samples]
                else:
                    curr_start = section
                
                # Create crossfade
                fade_length = min(len(prev_end), len(curr_start))
                if fade_length > 0:
                    fade_out = np.linspace(1, 0, fade_length)
                    fade_in = np.linspace(0, 1, fade_length)
                    
                    # Apply crossfade
                    transition = (prev_end[:fade_length] * fade_out + 
                                curr_start[:fade_length] * fade_in)
                    
                    # Remove overlapped portions and add transition
                    combined[-1] = combined[-1][:-fade_length]
                    combined.append(transition)
                    combined.append(section[fade_length:])
                else:
                    # No transition possible, just append
                    combined.append(section)
        
        return np.concatenate(combined)
    
    def save_generated_music(self, audio: np.ndarray, sr: int, filename: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save generated music to file with metadata.
        
        Parameters:
        -----------
        audio : np.ndarray
            Audio data to save
        sr : int
            Sample rate
        filename : str
            Output filename
        metadata : dict, optional
            Metadata to include in file or separate file
        """
        try:
            save_audio(filename, audio, sr)
            print(f"Generated music saved to: {filename}")
            
            # Save metadata if provided
            if metadata:
                import json
                metadata_filename = filename.replace('.wav', '_metadata.json')
                
                # Convert numpy types to JSON serializable
                serializable_metadata = self._make_json_serializable(metadata)
                
                with open(metadata_filename, 'w') as f:
                    json.dump(serializable_metadata, f, indent=2)
                print(f"Metadata saved to: {metadata_filename}")
                
        except Exception as e:
            print(f"Error saving generated music: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        else:
            return obj