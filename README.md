# OpenMusic - AI-Powered Music Generation & Comprehensive Audio Processing Library

OpenMusic is a revolutionary Python library that combines AI-powered music generation with over 1000 audio processing capabilities. Create original songs from simple text prompts or leverage advanced audio processing features including feature extraction, speech processing, audio enhancement, and music analysis.

## 🤖 NEW: AI-Powered Music Generation

### Text-to-Music Generation
Transform text descriptions into original musical compositions:

```python
import openmusic as om

# Generate music from a text prompt
audio, sr, metadata = om.generate_music_from_prompt(
    "Create an upbeat pop song with catchy melody and driving drums",
    duration=30.0
)

# Generate a complete song with structure
song_audio, sr, song_metadata = om.generate_song(
    "Create an emotional rock ballad about overcoming challenges",
    structure="verse-chorus-verse-chorus-bridge-chorus"
)

# Save the generated music
om.save_audio("my_generated_song.wav", song_audio, sr)
```

### How It Works
1. **Prompt Processing**: Advanced text analysis extracts musical parameters (genre, mood, tempo, key, instruments)
2. **AI Music Generation**: Intelligent composition engine creates melodies, harmonies, rhythms, and bass lines
3. **Professional Effects**: Automatic application of reverb, compression, and other effects based on style
4. **Song Structure**: Support for complete songs with verses, choruses, bridges, and smooth transitions

### Example Prompts
- "Create an upbeat, anthemic pop song with powerful vocals and soaring chorus"
- "Generate a melancholy piano ballad in A minor with gentle strings"  
- "Make energetic electronic dance music with heavy bass and synthesizers"
- "Compose a relaxing jazz piece with saxophone and soft piano"
- "Create mysterious ambient music with dark, ethereal sounds"

## Features

### 🤖 AI-Powered Music Generation
- **Text-to-Music**: Generate original songs from text descriptions
- **Prompt Processing**: Intelligent extraction of musical parameters from natural language
- **Multi-Genre Support**: Pop, rock, jazz, electronic, classical, folk, and more
- **Mood Recognition**: Happy, sad, energetic, calm, mysterious, romantic moods
- **Song Structure**: Complete songs with verses, choruses, bridges, and transitions
- **Smart Instrumentation**: Automatic selection of appropriate instruments for each genre

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

### AI Music Generation
```python
import openmusic as om

# Simple text-to-music generation
audio, sr, metadata = om.generate_music_from_prompt(
    "Create a happy pop song with piano and drums"
)

# Generate a complete structured song
song, sr, metadata = om.generate_song(
    "Rock ballad about overcoming challenges",
    structure="intro-verse-chorus-verse-chorus-bridge-chorus"
)

# Save your creation
om.save_audio("my_song.wav", song, sr)
```

### Audio Processing
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