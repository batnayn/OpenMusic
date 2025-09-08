# OpenMusic - AI Music Studio

An open-source AI Music Studio that creates music and vocals using Large Language Models (LLMs) and other open-source AI models.

## Features

- 🎵 **Music Generation**: Create instrumental music using state-of-the-art AI models
- 🎤 **Vocal Synthesis**: Generate vocals and speech using advanced TTS models
- ✍️ **Lyric Generation**: AI-powered lyric writing using LLMs
- 🎚️ **Audio Processing**: Mix, combine, and process generated audio
- 🌐 **Web Interface**: User-friendly web interface for music creation
- 🔧 **API Access**: RESTful API for programmatic access

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/batnayn/OpenMusic.git
cd OpenMusic

# Install dependencies
pip install -r requirements.txt

# Run the application
python -m app.main
```

### Using Docker

```bash
# Build and run with Docker
docker-compose up --build
```

## Technology Stack

- **Backend**: FastAPI (Python)
- **Frontend**: Gradio
- **Music Generation**: MusicGen, AudioCraft
- **Vocal Synthesis**: Bark TTS, Tortoise TTS
- **LLM Integration**: Transformers, Ollama
- **Audio Processing**: librosa, pydub

## API Usage

### Generate Music

```bash
curl -X POST "http://localhost:8000/api/generate-music" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "upbeat jazz melody", "duration": 30}'
```

### Generate Vocals

```bash
curl -X POST "http://localhost:8000/api/generate-vocals" \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world", "voice": "default"}'
```

### Generate Lyrics

```bash
curl -X POST "http://localhost:8000/api/generate-lyrics" \
     -H "Content-Type: application/json" \
     -d '{"theme": "love song", "style": "pop"}'
```

## Models Used

### Music Generation
- **MusicGen**: Facebook's music generation model
- **AudioCraft**: Advanced audio generation toolkit

### Vocal Synthesis
- **Bark**: High-quality TTS model by Suno AI
- **Tortoise TTS**: Expressive text-to-speech

### Language Models
- **Local LLMs**: Various models via Transformers
- **Ollama**: Local LLM inference

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Facebook AI Research for MusicGen
- Suno AI for Bark TTS
- Hugging Face for Transformers
- All open-source contributors