"""
API routes for OpenMusic AI Studio
"""

import os
import time
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from app.api.schemas import (
    HealthResponse,
    MusicGenerationRequest,
    VocalGenerationRequest, 
    LyricGenerationRequest,
    AudioProcessingRequest,
    GenerationResponse,
    ModelsStatusResponse,
    ErrorResponse
)
from app.core.config import get_settings

settings = get_settings()
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        app=settings.app_name,
        version=settings.app_version
    )


@router.get("/models/status", response_model=ModelsStatusResponse)
async def get_models_status():
    """Get status of all AI models"""
    try:
        # Import here to avoid startup delays
        from app.models.music_generator import MusicGenerator
        from app.models.vocal_generator import VocalGenerator
        from app.models.llm_client import LLMClient
        
        models = []
        
        # Check music generator
        try:
            music_gen = MusicGenerator()
            models.append({
                "name": "MusicGen",
                "loaded": music_gen.is_loaded,
                "device": str(music_gen.device) if hasattr(music_gen, 'device') else "unknown",
                "memory_usage": None,
                "last_used": None
            })
        except Exception as e:
            models.append({
                "name": "MusicGen",
                "loaded": False,
                "device": "unknown",
                "memory_usage": None,
                "last_used": None
            })
        
        # Check vocal generator
        try:
            vocal_gen = VocalGenerator()
            models.append({
                "name": "Bark TTS",
                "loaded": vocal_gen.is_loaded,
                "device": str(vocal_gen.device) if hasattr(vocal_gen, 'device') else "unknown",
                "memory_usage": None,
                "last_used": None
            })
        except Exception as e:
            models.append({
                "name": "Bark TTS",
                "loaded": False,
                "device": "unknown",
                "memory_usage": None,
                "last_used": None
            })
        
        # Check LLM
        try:
            llm = LLMClient()
            models.append({
                "name": "LLM",
                "loaded": llm.is_loaded,
                "device": str(llm.device) if hasattr(llm, 'device') else "unknown",
                "memory_usage": None,
                "last_used": None
            })
        except Exception as e:
            models.append({
                "name": "LLM",
                "loaded": False,
                "device": "unknown",
                "memory_usage": None,
                "last_used": None
            })
        
        loaded_count = sum(1 for m in models if m["loaded"])
        
        return ModelsStatusResponse(
            models=models,
            total_models=len(models),
            loaded_models=loaded_count
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking model status: {str(e)}")


@router.post("/generate-music", response_model=GenerationResponse)
async def generate_music(request: MusicGenerationRequest, background_tasks: BackgroundTasks):
    """Generate music from text prompt"""
    try:
        start_time = time.time()
        
        # Import here to avoid startup delays
        from app.models.music_generator import MusicGenerator
        
        music_generator = MusicGenerator()
        
        # Generate music
        output_path = await music_generator.generate(
            prompt=request.prompt,
            duration=request.duration,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p
        )
        
        generation_time = time.time() - start_time
        
        # Get file URL
        filename = os.path.basename(output_path)
        file_url = f"/api/download/{filename}"
        
        return GenerationResponse(
            success=True,
            message="Music generated successfully",
            file_path=output_path,
            file_url=file_url,
            metadata={
                "prompt": request.prompt,
                "duration": request.duration,
                "model": request.model or settings.musicgen_model
            },
            duration=request.duration,
            generation_time=generation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Music generation failed: {str(e)}")


@router.post("/generate-vocals", response_model=GenerationResponse)
async def generate_vocals(request: VocalGenerationRequest, background_tasks: BackgroundTasks):
    """Generate vocals from text"""
    try:
        start_time = time.time()
        
        # Import here to avoid startup delays
        from app.models.vocal_generator import VocalGenerator
        
        vocal_generator = VocalGenerator()
        
        # Generate vocals
        output_path = await vocal_generator.generate(
            text=request.text,
            voice_preset=request.voice_preset,
            temperature=request.temperature
        )
        
        generation_time = time.time() - start_time
        
        # Get file URL
        filename = os.path.basename(output_path)
        file_url = f"/api/download/{filename}"
        
        return GenerationResponse(
            success=True,
            message="Vocals generated successfully",
            file_path=output_path,
            file_url=file_url,
            metadata={
                "text": request.text,
                "voice_preset": request.voice_preset or settings.bark_voice_preset,
                "model": request.model or settings.bark_model
            },
            generation_time=generation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vocal generation failed: {str(e)}")


@router.post("/generate-lyrics", response_model=GenerationResponse)
async def generate_lyrics(request: LyricGenerationRequest, background_tasks: BackgroundTasks):
    """Generate lyrics using LLM"""
    try:
        start_time = time.time()
        
        # Import here to avoid startup delays
        from app.models.llm_client import LLMClient
        
        llm = LLMClient()
        
        # Generate lyrics
        lyrics = await llm.generate_lyrics(
            theme=request.theme,
            style=request.style,
            mood=request.mood,
            length=request.length
        )
        
        generation_time = time.time() - start_time
        
        # Save lyrics to file
        timestamp = int(time.time())
        filename = f"lyrics_{timestamp}.txt"
        output_path = os.path.join(settings.outputs_dir, filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(lyrics)
        
        file_url = f"/api/download/{filename}"
        
        return GenerationResponse(
            success=True,
            message="Lyrics generated successfully",
            file_path=output_path,
            file_url=file_url,
            metadata={
                "theme": request.theme,
                "style": request.style,
                "mood": request.mood,
                "length": request.length,
                "lyrics": lyrics
            },
            generation_time=generation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lyric generation failed: {str(e)}")


@router.post("/process-audio", response_model=GenerationResponse)
async def process_audio(request: AudioProcessingRequest, background_tasks: BackgroundTasks):
    """Process audio files (mix, concat, etc.)"""
    try:
        start_time = time.time()
        
        # Import here to avoid startup delays
        from app.core.audio_processor import AudioProcessor
        
        audio_processor = AudioProcessor()
        
        # Process audio
        output_path = await audio_processor.process(
            audio_files=request.audio_files,
            operation=request.operation,
            parameters=request.parameters
        )
        
        generation_time = time.time() - start_time
        
        # Get file URL
        filename = os.path.basename(output_path)
        file_url = f"/api/download/{filename}"
        
        return GenerationResponse(
            success=True,
            message="Audio processed successfully",
            file_path=output_path,
            file_url=file_url,
            metadata={
                "operation": request.operation,
                "input_files": request.audio_files,
                "parameters": request.parameters
            },
            generation_time=generation_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")


@router.get("/download/{filename}")
async def download_file(filename: str):
    """Download generated files"""
    file_path = os.path.join(settings.outputs_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Determine media type based on file extension
    media_type = "application/octet-stream"
    if filename.endswith('.wav'):
        media_type = "audio/wav"
    elif filename.endswith('.mp3'):
        media_type = "audio/mpeg"
    elif filename.endswith('.txt'):
        media_type = "text/plain"
    
    return FileResponse(
        file_path,
        media_type=media_type,
        filename=filename
    )


@router.get("/files")
async def list_files():
    """List all generated files"""
    try:
        files = []
        for filename in os.listdir(settings.outputs_dir):
            file_path = os.path.join(settings.outputs_dir, filename)
            if os.path.isfile(file_path):
                stat = os.stat(file_path)
                files.append({
                    "filename": filename,
                    "size": stat.st_size,
                    "created": stat.st_ctime,
                    "download_url": f"/api/download/{filename}"
                })
        
        return {"files": files, "total": len(files)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")