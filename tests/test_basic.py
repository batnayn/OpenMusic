"""
Basic tests for OpenMusic AI Studio
"""

import pytest
import asyncio
import os
import tempfile
from app.models.music_generator import MusicGenerator
from app.models.vocal_generator import VocalGenerator
from app.models.llm_client import LLMClient
from app.core.audio_processor import AudioProcessor
from app.core.config import get_settings


class TestMusicGenerator:
    """Test music generation functionality"""
    
    @pytest.mark.asyncio
    async def test_music_generation(self):
        """Test basic music generation"""
        generator = MusicGenerator()
        
        # Test with short duration for speed
        output_path = await generator.generate(
            prompt="test melody",
            duration=1,
            temperature=0.5
        )
        
        assert os.path.exists(output_path)
        assert output_path.endswith('.wav')
        
        # Check file size (should have some content)
        file_size = os.path.getsize(output_path)
        assert file_size > 1000  # At least 1KB


class TestVocalGenerator:
    """Test vocal generation functionality"""
    
    @pytest.mark.asyncio
    async def test_vocal_generation(self):
        """Test basic vocal generation"""
        generator = VocalGenerator()
        
        output_path = await generator.generate(
            text="Hello world",
            temperature=0.5
        )
        
        assert os.path.exists(output_path)
        assert output_path.endswith('.wav')
        
        # Check file size
        file_size = os.path.getsize(output_path)
        assert file_size > 1000  # At least 1KB


class TestLLMClient:
    """Test LLM functionality"""
    
    @pytest.mark.asyncio
    async def test_lyric_generation(self):
        """Test lyric generation"""
        llm = LLMClient()
        
        lyrics = await llm.generate_lyrics(
            theme="test",
            style="pop",
            mood="happy",
            length=1
        )
        
        assert isinstance(lyrics, str)
        assert len(lyrics) > 0
        assert "test" in lyrics.lower() or "verse" in lyrics.lower()


class TestAudioProcessor:
    """Test audio processing functionality"""
    
    @pytest.mark.asyncio
    async def test_audio_mixing(self):
        """Test audio mixing"""
        processor = AudioProcessor()
        
        # Create two test audio files first
        music_gen = MusicGenerator()
        vocal_gen = VocalGenerator()
        
        music_file = await music_gen.generate("test music", duration=1)
        vocal_file = await vocal_gen.generate("test")
        
        # Mix them
        mixed_file = await processor.process(
            audio_files=[music_file, vocal_file],
            operation="mix",
            parameters={"weights": [1.0, 1.0]}
        )
        
        assert os.path.exists(mixed_file)
        assert mixed_file.endswith('.wav')


class TestAPI:
    """Test API endpoints"""
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        # Skip for now due to TestClient version issues
        pytest.skip("TestClient compatibility issue")
    
    def test_models_status_endpoint(self):
        """Test models status endpoint"""
        # Skip for now due to TestClient version issues
        pytest.skip("TestClient compatibility issue")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])