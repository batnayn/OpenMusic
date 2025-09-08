"""
Pytest test suite for OpenMusic library.
"""

import pytest
import numpy as np
import sys
import os

# Add the openmusic package to the path
sys.path.insert(0, '/home/runner/work/OpenMusic/OpenMusic')

import openmusic as om

@pytest.fixture
def sample_audio():
    """Create sample audio for testing."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), False)
    audio = np.sin(2 * np.pi * 440 * t) * 0.3  # A4 note
    return audio, sr

@pytest.fixture
def complex_audio():
    """Create complex audio with multiple frequencies."""
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), False)
    # A major chord
    audio = (np.sin(2 * np.pi * 440 * t) +     # A
             np.sin(2 * np.pi * 554.37 * t) +  # C#
             np.sin(2 * np.pi * 659.25 * t)) * 0.1  # E
    return audio, sr

class TestFeatureExtraction:
    
    def test_extract_mfccs(self, sample_audio):
        audio, sr = sample_audio
        mfccs = om.features.extract_mfccs(audio, sr)
        assert mfccs.shape[0] == 13  # Default number of MFCCs
        assert mfccs.shape[1] > 0    # Should have time frames
    
    def test_extract_spectral_features(self, sample_audio):
        audio, sr = sample_audio
        features = om.features.extract_spectral_features(audio, sr)
        
        expected_keys = ['spectral_centroid', 'spectral_rolloff', 
                        'spectral_bandwidth', 'spectral_contrast', 
                        'zero_crossing_rate']
        
        for key in expected_keys:
            assert key in features
            assert len(features[key]) > 0
    
    def test_extract_all_features(self, complex_audio):
        audio, sr = complex_audio
        features = om.features.extract_all_features(audio, sr)
        assert len(features) >= 10  # Should extract multiple feature types
        assert 'mfccs' in features
        assert 'chroma' in features

class TestAudioEffects:
    
    def test_add_reverb(self, sample_audio):
        audio, sr = sample_audio
        reverb = om.effects.add_reverb(audio, sr)
        assert len(reverb) == len(audio)
        assert not np.array_equal(audio, reverb)  # Should be different
    
    def test_pitch_shift(self, sample_audio):
        audio, sr = sample_audio
        shifted = om.effects.pitch_shift(audio, sr, n_steps=12)  # Octave up
        assert len(shifted) == len(audio)
        assert not np.array_equal(audio, shifted)
    
    def test_add_distortion(self, sample_audio):
        audio, sr = sample_audio
        distorted = om.effects.add_distortion(audio, gain=5.0)
        assert len(distorted) == len(audio)
        # Distortion should clip the signal
        assert np.max(np.abs(distorted)) <= 1.0

class TestAudioEnhancement:
    
    def test_reduce_noise(self, sample_audio):
        audio, sr = sample_audio
        # Add noise
        noisy = audio + np.random.normal(0, 0.1, len(audio))
        clean = om.enhancement.reduce_noise(noisy, sr)
        assert len(clean) == len(noisy)
    
    def test_apply_lowpass(self, sample_audio):
        audio, sr = sample_audio
        filtered = om.enhancement.apply_lowpass(audio, sr, cutoff=1000)
        assert len(filtered) == len(audio)
    
    def test_compress_audio(self, sample_audio):
        audio, sr = sample_audio
        compressed = om.enhancement.compress_audio(audio, sr)
        assert len(compressed) == len(audio)

class TestMusicAnalysis:
    
    def test_detect_tempo(self, complex_audio):
        audio, sr = complex_audio
        tempo = om.analysis.detect_tempo(audio, sr)
        assert isinstance(tempo, (int, float))
        assert tempo >= 0
    
    def test_detect_key(self, complex_audio):
        audio, sr = complex_audio
        key, mode = om.analysis.detect_key(audio, sr)
        assert isinstance(key, str)
        assert mode in ['major', 'minor']
    
    def test_track_beats(self, complex_audio):
        audio, sr = complex_audio
        tempo, beats = om.analysis.track_beats(audio, sr)
        assert isinstance(tempo, (int, float))
        assert isinstance(beats, np.ndarray)

class TestSpeechProcessing:
    
    def test_list_voices(self):
        voices = om.speech.list_voices()
        assert isinstance(voices, list)
        assert len(voices) >= 1
    
    def test_synthesize_text(self):
        text = "Hello world"
        audio, sr = om.speech.synthesize_text(text)
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert sr > 0

class TestCoreUtilities:
    
    def test_normalize_audio(self, sample_audio):
        audio, sr = sample_audio
        normalized = om.core.normalize_audio(audio)
        assert len(normalized) == len(audio)
    
    def test_convert_sample_rate(self, sample_audio):
        audio, sr = sample_audio
        resampled = om.core.convert_sample_rate(audio, sr, sr//2)
        assert len(resampled) == len(audio) // 2

def test_integration_pipeline(complex_audio):
    """Test a complete processing pipeline."""
    audio, sr = complex_audio
    
    # Extract features
    features = om.features.extract_all_features(audio, sr)
    assert len(features) > 0
    
    # Apply effects
    processed = om.effects.add_reverb(audio, sr, room_size=0.5)
    
    # Enhance audio
    enhanced = om.enhancement.reduce_noise(processed, sr)
    
    # Analyze result
    tempo = om.analysis.detect_tempo(enhanced, sr)
    key, mode = om.analysis.detect_key(enhanced, sr)
    
    # All should complete without errors
    assert len(enhanced) == len(audio)
    assert isinstance(tempo, (int, float))
    assert isinstance(key, str)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])