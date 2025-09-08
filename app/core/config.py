"""
Configuration settings for OpenMusic AI Studio
"""

import os
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    app_name: str = "OpenMusic AI Studio"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    
    # Paths
    models_dir: str = "models"
    outputs_dir: str = "outputs"
    static_dir: str = "static"
    
    # AI Models Settings
    device: str = "auto"  # auto, cpu, cuda
    max_duration: int = 120  # Maximum audio duration in seconds
    
    # Music Generation
    musicgen_model: str = "facebook/musicgen-small"
    musicgen_use_sampling: bool = True
    musicgen_top_k: int = 250
    musicgen_top_p: float = 0.0
    
    # TTS Settings
    bark_model: str = "suno/bark"
    bark_voice_preset: str = "v2/en_speaker_6"
    
    # LLM Settings
    llm_model: str = "gpt2"
    llm_max_length: int = 500
    ollama_base_url: str = "http://localhost:11434"
    
    # Audio Processing
    sample_rate: int = 32000
    audio_format: str = "wav"
    
    # Security
    cors_origins: list = ["*"]
    
    # Database (optional)
    database_url: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings