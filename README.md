# OpenMusic - Comprehensive Audio Processing Library

OpenMusic is a powerful Python library for audio processing, featuring over 1000 audio processing capabilities including feature extraction, speech processing, audio enhancement, and music analysis.

## Features

### 🎵 Audio Feature Extraction
- **MFCCs (Mel-Frequency Cepstral Coefficients)**: Extract perceptual audio features
- **Spectral Features**: Spectral centroid, rolloff, bandwidth, contrast
- **Temporal Features**: Zero crossing rate, tempo, beat tracking
- **Harmonic Features**: Chroma features, tonnetz, harmonic-percussive separation

### 🗣️ Speech Processing
- **Speech Recognition**: Convert speech to text using multiple engines
- **Text-to-Speech Synthesis**: Convert text to natural speech
- **Voice Activity Detection**: Detect speech segments in audio
- **Speaker Identification**: Basic speaker recognition capabilities

### 🔧 Audio Enhancement
- **Noise Reduction**: Advanced spectral subtraction and Wiener filtering
- **Audio Filtering**: High-pass, low-pass, band-pass filters
- **Dynamic Range Processing**: Compression, limiting, normalization
- **Echo Cancellation**: Remove unwanted echoes and reverb

### 🎼 Music Analysis
- **Tempo Detection**: BPM estimation and beat tracking
- **Key Detection**: Musical key and mode identification
- **Chord Recognition**: Basic chord detection and progression analysis
- **Structure Analysis**: Song section detection and analysis

### 🎛️ Audio Effects
- **Time-based Effects**: Reverb, delay, echo
- **Pitch Effects**: Pitch shifting, auto-tune, harmonization
- **Modulation Effects**: Chorus, flanger, phaser
- **Dynamic Effects**: Compression, distortion, saturation

## Installation

```bash
pip install openmusic
```

For development installation:
```bash
git clone https://github.com/batnayn/OpenMusic.git
cd OpenMusic
pip install -e .
```

## Quick Start

```python
import openmusic as om
import numpy as np

# Load audio file
audio, sr = om.load_audio("example.wav")

# Extract features
mfccs = om.features.extract_mfccs(audio, sr)
spectral_features = om.features.extract_spectral_features(audio, sr)

# Speech processing
text = om.speech.recognize_speech(audio, sr)
speech_audio = om.speech.synthesize_text("Hello, world!")

# Audio enhancement
clean_audio = om.enhancement.reduce_noise(audio, sr)

# Music analysis
tempo = om.analysis.detect_tempo(audio, sr)
key = om.analysis.detect_key(audio, sr)

# Apply effects
reverb_audio = om.effects.add_reverb(audio, sr, room_size=0.5)
```

## Documentation

Full documentation is available at [docs.openmusic.org](https://docs.openmusic.org)

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use OpenMusic in your research, please cite:

```bibtex
@software{openmusic2025,
  title={OpenMusic: Comprehensive Audio Processing Library},
  author={OpenMusic Team},
  year={2025},
  url={https://github.com/batnayn/OpenMusic}
}
```