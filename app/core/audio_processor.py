"""
Audio processing utilities for mixing, combining, and manipulating audio files
"""

import os
import time
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional

from app.core.config import get_settings

settings = get_settings()


class AudioProcessor:
    """Audio processing utilities"""
    
    def __init__(self):
        self.sample_rate = settings.sample_rate
    
    async def process(
        self,
        audio_files: List[str],
        operation: str,
        parameters: Dict[str, Any]
    ) -> str:
        """Process audio files with specified operation"""
        
        print(f"🎚️ Processing audio: operation='{operation}', files={len(audio_files)}")
        
        # Create output filename
        timestamp = int(time.time())
        filename = f"processed_{operation}_{timestamp}.{settings.audio_format}"
        output_path = os.path.join(settings.outputs_dir, filename)
        
        try:
            if operation == "mix":
                await self._mix_audio(audio_files, output_path, parameters)
            elif operation == "concat":
                await self._concatenate_audio(audio_files, output_path, parameters)
            elif operation == "layer":
                await self._layer_audio(audio_files, output_path, parameters)
            elif operation == "fade":
                await self._apply_fade(audio_files[0], output_path, parameters)
            elif operation == "normalize":
                await self._normalize_audio(audio_files[0], output_path, parameters)
            elif operation == "trim":
                await self._trim_audio(audio_files[0], output_path, parameters)
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            print(f"✅ Audio processed: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Audio processing failed: {e}")
            raise
    
    async def _mix_audio(
        self,
        audio_files: List[str],
        output_path: str,
        parameters: Dict[str, Any]
    ):
        """Mix multiple audio files together"""
        def _process():
            import soundfile as sf
            
            # Load all audio files
            audio_data = []
            max_length = 0
            
            for file_path in audio_files:
                if not os.path.exists(file_path):
                    # Try relative to outputs directory
                    file_path = os.path.join(settings.outputs_dir, os.path.basename(file_path))
                
                if os.path.exists(file_path):
                    data, sr = sf.read(file_path)
                    
                    # Resample if necessary
                    if sr != self.sample_rate:
                        data = self._resample(data, sr, self.sample_rate)
                    
                    # Ensure mono for simplicity
                    if len(data.shape) > 1:
                        data = np.mean(data, axis=1)
                    
                    audio_data.append(data)
                    max_length = max(max_length, len(data))
                else:
                    print(f"⚠️ Audio file not found: {file_path}")
            
            if not audio_data:
                raise ValueError("No valid audio files found")
            
            # Pad all audio to same length
            padded_audio = []
            for data in audio_data:
                if len(data) < max_length:
                    padding = np.zeros(max_length - len(data))
                    data = np.concatenate([data, padding])
                padded_audio.append(data)
            
            # Mix audio with optional weights
            weights = parameters.get("weights", [1.0] * len(padded_audio))
            if len(weights) != len(padded_audio):
                weights = [1.0] * len(padded_audio)
            
            mixed_audio = np.zeros(max_length)
            for i, (data, weight) in enumerate(zip(padded_audio, weights)):
                mixed_audio += data * weight
            
            # Normalize to prevent clipping
            max_val = np.max(np.abs(mixed_audio))
            if max_val > 0:
                mixed_audio = mixed_audio / max_val * 0.95
            
            # Save mixed audio
            sf.write(output_path, mixed_audio, self.sample_rate)
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _process)
    
    async def _concatenate_audio(
        self,
        audio_files: List[str],
        output_path: str,
        parameters: Dict[str, Any]
    ):
        """Concatenate audio files end-to-end"""
        def _process():
            import soundfile as sf
            
            concatenated_audio = []
            
            for file_path in audio_files:
                if not os.path.exists(file_path):
                    file_path = os.path.join(settings.outputs_dir, os.path.basename(file_path))
                
                if os.path.exists(file_path):
                    data, sr = sf.read(file_path)
                    
                    # Resample if necessary
                    if sr != self.sample_rate:
                        data = self._resample(data, sr, self.sample_rate)
                    
                    # Ensure mono
                    if len(data.shape) > 1:
                        data = np.mean(data, axis=1)
                    
                    concatenated_audio.append(data)
                    
                    # Add silence between files if specified
                    silence_duration = parameters.get("silence_between", 0.0)
                    if silence_duration > 0 and file_path != audio_files[-1]:
                        silence_samples = int(silence_duration * self.sample_rate)
                        silence = np.zeros(silence_samples)
                        concatenated_audio.append(silence)
                else:
                    print(f"⚠️ Audio file not found: {file_path}")
            
            if not concatenated_audio:
                raise ValueError("No valid audio files found")
            
            # Concatenate all audio
            final_audio = np.concatenate(concatenated_audio)
            
            # Save concatenated audio
            sf.write(output_path, final_audio, self.sample_rate)
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _process)
    
    async def _layer_audio(
        self,
        audio_files: List[str],
        output_path: str,
        parameters: Dict[str, Any]
    ):
        """Layer audio files with time offsets"""
        def _process():
            import soundfile as sf
            
            # Get offsets for each file
            offsets = parameters.get("offsets", [0.0] * len(audio_files))
            if len(offsets) != len(audio_files):
                offsets = [i * 2.0 for i in range(len(audio_files))]  # Default 2s offsets
            
            # Calculate total length needed
            max_end_time = 0
            audio_segments = []
            
            for i, file_path in enumerate(audio_files):
                if not os.path.exists(file_path):
                    file_path = os.path.join(settings.outputs_dir, os.path.basename(file_path))
                
                if os.path.exists(file_path):
                    data, sr = sf.read(file_path)
                    
                    # Resample if necessary
                    if sr != self.sample_rate:
                        data = self._resample(data, sr, self.sample_rate)
                    
                    # Ensure mono
                    if len(data.shape) > 1:
                        data = np.mean(data, axis=1)
                    
                    offset_samples = int(offsets[i] * self.sample_rate)
                    end_time = offset_samples + len(data)
                    max_end_time = max(max_end_time, end_time)
                    
                    audio_segments.append((data, offset_samples))
            
            # Create output array
            layered_audio = np.zeros(max_end_time)
            
            # Layer all audio segments
            for data, offset in audio_segments:
                layered_audio[offset:offset + len(data)] += data
            
            # Normalize
            max_val = np.max(np.abs(layered_audio))
            if max_val > 0:
                layered_audio = layered_audio / max_val * 0.95
            
            # Save layered audio
            sf.write(output_path, layered_audio, self.sample_rate)
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _process)
    
    async def _apply_fade(
        self,
        input_file: str,
        output_path: str,
        parameters: Dict[str, Any]
    ):
        """Apply fade in/out to audio"""
        def _process():
            import soundfile as sf
            
            if not os.path.exists(input_file):
                input_file_path = os.path.join(settings.outputs_dir, os.path.basename(input_file))
            else:
                input_file_path = input_file
            
            data, sr = sf.read(input_file_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                data = self._resample(data, sr, self.sample_rate)
            
            # Ensure mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            # Apply fade in
            fade_in_duration = parameters.get("fade_in", 1.0)
            fade_in_samples = int(fade_in_duration * self.sample_rate)
            if fade_in_samples > 0 and fade_in_samples < len(data):
                fade_in_curve = np.linspace(0, 1, fade_in_samples)
                data[:fade_in_samples] *= fade_in_curve
            
            # Apply fade out
            fade_out_duration = parameters.get("fade_out", 1.0)
            fade_out_samples = int(fade_out_duration * self.sample_rate)
            if fade_out_samples > 0 and fade_out_samples < len(data):
                fade_out_curve = np.linspace(1, 0, fade_out_samples)
                data[-fade_out_samples:] *= fade_out_curve
            
            # Save processed audio
            sf.write(output_path, data, self.sample_rate)
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _process)
    
    async def _normalize_audio(
        self,
        input_file: str,
        output_path: str,
        parameters: Dict[str, Any]
    ):
        """Normalize audio amplitude"""
        def _process():
            import soundfile as sf
            
            if not os.path.exists(input_file):
                input_file_path = os.path.join(settings.outputs_dir, os.path.basename(input_file))
            else:
                input_file_path = input_file
            
            data, sr = sf.read(input_file_path)
            
            # Normalize to specified level
            target_level = parameters.get("target_level", 0.95)
            max_val = np.max(np.abs(data))
            
            if max_val > 0:
                data = data / max_val * target_level
            
            # Save normalized audio
            sf.write(output_path, data, sr)
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _process)
    
    async def _trim_audio(
        self,
        input_file: str,
        output_path: str,
        parameters: Dict[str, Any]
    ):
        """Trim audio to specified start and end times"""
        def _process():
            import soundfile as sf
            
            if not os.path.exists(input_file):
                input_file_path = os.path.join(settings.outputs_dir, os.path.basename(input_file))
            else:
                input_file_path = input_file
            
            data, sr = sf.read(input_file_path)
            
            # Get trim parameters
            start_time = parameters.get("start", 0.0)
            end_time = parameters.get("end", len(data) / sr)
            
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # Trim audio
            trimmed_data = data[start_sample:end_sample]
            
            # Save trimmed audio
            sf.write(output_path, trimmed_data, sr)
        
        # Run in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _process)
    
    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling (basic linear interpolation)"""
        if orig_sr == target_sr:
            return audio
        
        # Calculate resampling ratio
        ratio = target_sr / orig_sr
        new_length = int(len(audio) * ratio)
        
        # Create new time indices
        old_indices = np.arange(len(audio))
        new_indices = np.linspace(0, len(audio) - 1, new_length)
        
        # Interpolate
        resampled = np.interp(new_indices, old_indices, audio)
        
        return resampled