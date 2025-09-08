"""
Basic test to validate OpenMusic core functionality.
"""

import numpy as np
import sys
import os

# Add the openmusic package to the path
sys.path.insert(0, '/home/runner/work/OpenMusic/OpenMusic')

try:
    import openmusic as om
    print("✓ OpenMusic package imported successfully")
    
    # Test audio generation for testing
    sr = 22050
    duration = 2.0
    frequency = 440.0  # A4
    t = np.linspace(0, duration, int(sr * duration), False)
    test_audio = np.sin(2 * np.pi * frequency * t) * 0.3
    
    print("✓ Test audio generated")
    
    # Test feature extraction
    mfccs = om.features.extract_mfccs(test_audio, sr)
    print(f"✓ MFCCs extracted: shape {mfccs.shape}")
    
    spectral_features = om.features.extract_spectral_features(test_audio, sr)
    print(f"✓ Spectral features extracted: {list(spectral_features.keys())}")
    
    # Test tempo detection
    tempo = om.analysis.detect_tempo(test_audio, sr)
    print(f"✓ Tempo detected: {tempo:.1f} BPM")
    
    # Test key detection
    key, mode = om.analysis.detect_key(test_audio, sr)
    print(f"✓ Key detected: {key} {mode}")
    
    # Test audio effects
    reverb_audio = om.effects.add_reverb(test_audio, sr, room_size=0.5)
    print(f"✓ Reverb applied: shape {reverb_audio.shape}")
    
    # Test noise reduction
    clean_audio = om.enhancement.reduce_noise(test_audio, sr)
    print(f"✓ Noise reduction applied: shape {clean_audio.shape}")
    
    print("\n🎉 All core functionality tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)