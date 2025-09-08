"""
API request/response schemas
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    app: str
    version: str


class MusicGenerationRequest(BaseModel):
    """Request for music generation"""
    prompt: str = Field(..., description="Text description of the music to generate")
    duration: int = Field(30, ge=5, le=120, description="Duration in seconds")
    model: Optional[str] = Field(None, description="Model to use for generation")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Creativity/randomness")
    top_k: int = Field(250, ge=1, le=1000, description="Top-k sampling parameter")
    top_p: float = Field(0.0, ge=0.0, le=1.0, description="Top-p sampling parameter")


class VocalGenerationRequest(BaseModel):
    """Request for vocal generation"""
    text: str = Field(..., description="Text to convert to speech")
    voice_preset: Optional[str] = Field(None, description="Voice preset to use")
    model: Optional[str] = Field(None, description="TTS model to use")
    temperature: float = Field(0.7, ge=0.1, le=1.0, description="Voice expressiveness")


class LyricGenerationRequest(BaseModel):
    """Request for lyric generation"""
    theme: str = Field(..., description="Theme or topic for the lyrics")
    style: Optional[str] = Field("pop", description="Musical style (pop, rock, jazz, etc.)")
    mood: Optional[str] = Field("upbeat", description="Mood (happy, sad, energetic, etc.)")
    length: int = Field(4, ge=1, le=20, description="Number of verses")
    model: Optional[str] = Field(None, description="LLM model to use")


class AudioProcessingRequest(BaseModel):
    """Request for audio processing"""
    audio_files: List[str] = Field(..., description="List of audio file paths to process")
    operation: str = Field(..., description="Processing operation (mix, concat, etc.)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")


class GenerationResponse(BaseModel):
    """Generic generation response"""
    success: bool
    message: str
    file_path: Optional[str] = None
    file_url: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration: Optional[float] = None
    generation_time: Optional[float] = None


class ModelStatus(BaseModel):
    """Model status information"""
    name: str
    loaded: bool
    device: str
    memory_usage: Optional[str] = None
    last_used: Optional[str] = None


class ModelsStatusResponse(BaseModel):
    """Response with all model statuses"""
    models: List[ModelStatus]
    total_models: int
    loaded_models: int


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: str
    code: int = 500