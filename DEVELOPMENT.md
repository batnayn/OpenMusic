# OpenMusic - Development Guide

## Project Structure

```
OpenMusic/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── models/                 # AI model implementations
│   │   ├── music_generator.py  # Music generation (MusicGen)
│   │   ├── vocal_generator.py  # Vocal synthesis (Bark TTS)
│   │   └── llm_client.py      # Language model for lyrics
│   ├── api/                   # REST API routes
│   │   ├── routes.py          # API endpoints
│   │   └── schemas.py         # Pydantic models
│   ├── core/                  # Core utilities
│   │   ├── config.py          # Configuration management
│   │   └── audio_processor.py # Audio processing utilities
│   └── ui/                    # User interface
│       └── interface.py       # Gradio web interface
├── examples/                  # Example scripts
├── tests/                     # Test suite
├── outputs/                   # Generated files
├── models/                    # Downloaded AI models
├── requirements.txt           # Python dependencies
├── setup.py                  # Package configuration
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose setup
└── README.md                 # Project documentation
```

## Development Setup

### Prerequisites

- Python 3.8+
- Git
- Optional: Docker for containerized deployment

### Installation

1. Clone the repository:
```bash
git clone https://github.com/batnayn/OpenMusic.git
cd OpenMusic
```

2. Install dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

3. For advanced models (optional):
```bash
pip install "openmusic[advanced]"
```

### Running the Application

#### API Server
```bash
python -m app.main
# or
openmusic
```

#### Web Interface
```bash
python -m app.ui.interface
```

#### Docker
```bash
docker-compose up --build
```

## API Endpoints

### Core Endpoints

- `GET /health` - Health check
- `GET /api/models/status` - Model status
- `GET /api/files` - List generated files

### Generation Endpoints

- `POST /api/generate-music` - Generate music
- `POST /api/generate-vocals` - Generate vocals
- `POST /api/generate-lyrics` - Generate lyrics
- `POST /api/process-audio` - Process audio files

### File Access

- `GET /api/download/{filename}` - Download generated files

## Model Integration

### Music Generation

The system supports multiple music generation models:

1. **MusicGen** (Facebook) - Primary model
2. **AudioCraft** - Advanced audio generation
3. **Fallback** - Simple synthetic audio for testing

### Vocal Synthesis

Supported TTS models:

1. **Bark** (Suno AI) - High-quality neural TTS
2. **Tortoise TTS** - Expressive speech synthesis
3. **Fallback** - Basic synthetic speech

### Language Models

For lyric generation:

1. **GPT-2/GPT-3.5** - Via Transformers
2. **Local LLMs** - Via Ollama
3. **Fallback** - Template-based generation

## Configuration

Configuration is managed through environment variables and `app/core/config.py`:

```python
# Key settings
DEVICE = "auto"  # auto, cpu, cuda
MUSICGEN_MODEL = "facebook/musicgen-small"
BARK_MODEL = "suno/bark"
LLM_MODEL = "gpt2"
```

Environment variables:
```bash
export DEVICE="cuda"
export DEBUG="true"
export CORS_ORIGINS="*"
```

## Testing

Run tests with:
```bash
pytest tests/ -v
```

Run specific test categories:
```bash
pytest tests/test_basic.py::TestMusicGenerator -v
```

## Deployment

### Local Development
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
docker-compose -f docker-compose.yml up -d
```

### Environment Variables for Production
```bash
DEVICE=cpu
DEBUG=false
CORS_ORIGINS=https://yourdomain.com
```

## Adding New Models

### Music Generation Model

1. Create new class in `app/models/music_generator.py`
2. Implement `generate()` method
3. Add model configuration to `config.py`
4. Update API schemas if needed

### TTS Model

1. Add to `app/models/vocal_generator.py`
2. Implement voice generation interface
3. Update voice preset options
4. Test with different text inputs

### LLM Model

1. Extend `app/models/llm_client.py`
2. Add model loading logic
3. Implement text generation methods
4. Update prompt templates

## Performance Optimization

### GPU Usage
```python
# Enable CUDA if available
DEVICE = "cuda"

# Use mixed precision
torch.backends.cudnn.benchmark = True
```

### Memory Management
```python
# Clear GPU cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Model quantization for smaller footprint
model = model.half()  # FP16
```

### Caching
- Models are lazy-loaded on first use
- Generated files are cached in `outputs/`
- Consider implementing Redis for session management

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce model sizes
   - Use CPU instead of GPU
   - Lower generation parameters

2. **Slow Generation**
   - Use smaller models
   - Reduce duration/length parameters
   - Enable GPU acceleration

3. **Model Loading Errors**
   - Check internet connection
   - Verify model paths
   - Use fallback implementations

### Debug Mode

Enable debug logging:
```bash
export DEBUG=true
python -m app.main
```

### Model Fallbacks

The system includes fallback implementations that work without external models:
- Synthetic audio generation
- Template-based lyrics
- Simple audio processing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to functions
- Write tests for new features

### Commit Messages

Use conventional commits:
```
feat: add new music generation model
fix: resolve audio processing bug
docs: update API documentation
```