"""
Gradio web interface for OpenMusic AI Studio
"""

import os
import asyncio
import gradio as gr
from typing import Optional, Tuple

from app.models.music_generator import MusicGenerator
from app.models.vocal_generator import VocalGenerator
from app.models.llm_client import LLMClient
from app.core.audio_processor import AudioProcessor
from app.core.config import get_settings

settings = get_settings()

# Global model instances
music_generator = None
vocal_generator = None
llm_client = None
audio_processor = None


def initialize_models():
    """Initialize all AI models"""
    global music_generator, vocal_generator, llm_client, audio_processor
    
    if music_generator is None:
        music_generator = MusicGenerator()
    
    if vocal_generator is None:
        vocal_generator = VocalGenerator()
    
    if llm_client is None:
        llm_client = LLMClient()
    
    if audio_processor is None:
        audio_processor = AudioProcessor()


async def generate_music_async(prompt: str, duration: int, temperature: float) -> Tuple[str, str]:
    """Generate music asynchronously"""
    try:
        initialize_models()
        output_path = await music_generator.generate(
            prompt=prompt,
            duration=duration,
            temperature=temperature
        )
        return output_path, f"✅ Music generated successfully! Duration: {duration}s"
    except Exception as e:
        return None, f"❌ Error generating music: {str(e)}"


async def generate_vocals_async(text: str, voice_preset: str, temperature: float) -> Tuple[str, str]:
    """Generate vocals asynchronously"""
    try:
        initialize_models()
        output_path = await vocal_generator.generate(
            text=text,
            voice_preset=voice_preset if voice_preset else None,
            temperature=temperature
        )
        return output_path, f"✅ Vocals generated successfully!"
    except Exception as e:
        return None, f"❌ Error generating vocals: {str(e)}"


async def generate_lyrics_async(theme: str, style: str, mood: str, length: int) -> Tuple[str, str]:
    """Generate lyrics asynchronously"""
    try:
        initialize_models()
        lyrics = await llm_client.generate_lyrics(
            theme=theme,
            style=style,
            mood=mood,
            length=length
        )
        return lyrics, f"✅ Lyrics generated successfully! ({len(lyrics.split())} words)"
    except Exception as e:
        return "", f"❌ Error generating lyrics: {str(e)}"


def generate_music(prompt: str, duration: int, temperature: float):
    """Synchronous wrapper for music generation"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_path, message = loop.run_until_complete(
            generate_music_async(prompt, duration, temperature)
        )
        loop.close()
        return audio_path, message
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def generate_vocals(text: str, voice_preset: str, temperature: float):
    """Synchronous wrapper for vocal generation"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        audio_path, message = loop.run_until_complete(
            generate_vocals_async(text, voice_preset, temperature)
        )
        loop.close()
        return audio_path, message
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def generate_lyrics(theme: str, style: str, mood: str, length: int):
    """Synchronous wrapper for lyric generation"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        lyrics, message = loop.run_until_complete(
            generate_lyrics_async(theme, style, mood, length)
        )
        loop.close()
        return lyrics, message
    except Exception as e:
        return "", f"❌ Error: {str(e)}"


def get_recent_files():
    """Get list of recently generated files"""
    try:
        files = []
        for filename in os.listdir(settings.outputs_dir):
            if filename.endswith(('.wav', '.mp3', '.txt')):
                filepath = os.path.join(settings.outputs_dir, filename)
                files.append(filepath)
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files[:10]  # Return last 10 files
    except Exception:
        return []


def create_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .feature-card {
        border: 1px solid #e1e5e9;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background: #f8f9fa;
    }
    """
    
    with gr.Blocks(css=css, title="OpenMusic AI Studio") as interface:
        
        # Header
        gr.HTML("""
        <div class="header">
            <h1>🎵 OpenMusic AI Studio</h1>
            <p>Create music and vocals using cutting-edge AI models</p>
        </div>
        """)
        
        with gr.Tabs():
            
            # Music Generation Tab
            with gr.Tab("🎼 Music Generation"):
                gr.Markdown("Generate instrumental music from text descriptions")
                
                with gr.Row():
                    with gr.Column():
                        music_prompt = gr.Textbox(
                            label="Music Description",
                            placeholder="e.g., 'upbeat jazz melody with piano', 'calm ambient electronic music'",
                            lines=2
                        )
                        
                        with gr.Row():
                            music_duration = gr.Slider(
                                minimum=5,
                                maximum=60,
                                value=30,
                                step=5,
                                label="Duration (seconds)"
                            )
                            
                            music_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.8,
                                step=0.1,
                                label="Creativity"
                            )
                        
                        music_generate_btn = gr.Button("🎵 Generate Music", variant="primary")
                    
                    with gr.Column():
                        music_output = gr.Audio(label="Generated Music")
                        music_status = gr.Textbox(label="Status", interactive=False)
                
                music_generate_btn.click(
                    generate_music,
                    inputs=[music_prompt, music_duration, music_temperature],
                    outputs=[music_output, music_status]
                )
            
            # Vocal Generation Tab
            with gr.Tab("🎤 Vocal Generation"):
                gr.Markdown("Generate vocals and speech from text")
                
                with gr.Row():
                    with gr.Column():
                        vocal_text = gr.Textbox(
                            label="Text to Speak",
                            placeholder="Enter the text you want to convert to speech...",
                            lines=3
                        )
                        
                        with gr.Row():
                            vocal_voice = gr.Dropdown(
                                choices=[
                                    "v2/en_speaker_0",
                                    "v2/en_speaker_1",
                                    "v2/en_speaker_2",
                                    "v2/en_speaker_3",
                                    "v2/en_speaker_4",
                                    "v2/en_speaker_5",
                                    "v2/en_speaker_6",
                                    "v2/en_speaker_7",
                                    "v2/en_speaker_8",
                                    "v2/en_speaker_9"
                                ],
                                value="v2/en_speaker_6",
                                label="Voice Preset"
                            )
                            
                            vocal_temperature = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                step=0.1,
                                label="Expressiveness"
                            )
                        
                        vocal_generate_btn = gr.Button("🎙️ Generate Vocals", variant="primary")
                    
                    with gr.Column():
                        vocal_output = gr.Audio(label="Generated Vocals")
                        vocal_status = gr.Textbox(label="Status", interactive=False)
                
                vocal_generate_btn.click(
                    generate_vocals,
                    inputs=[vocal_text, vocal_voice, vocal_temperature],
                    outputs=[vocal_output, vocal_status]
                )
            
            # Lyric Generation Tab
            with gr.Tab("✍️ Lyric Generation"):
                gr.Markdown("Generate song lyrics using AI")
                
                with gr.Row():
                    with gr.Column():
                        lyric_theme = gr.Textbox(
                            label="Theme/Topic",
                            placeholder="e.g., 'love', 'freedom', 'adventure'",
                            lines=1
                        )
                        
                        with gr.Row():
                            lyric_style = gr.Dropdown(
                                choices=["pop", "rock", "jazz", "country", "hip-hop", "folk", "electronic"],
                                value="pop",
                                label="Musical Style"
                            )
                            
                            lyric_mood = gr.Dropdown(
                                choices=["happy", "sad", "energetic", "calm", "romantic", "mysterious", "uplifting"],
                                value="happy",
                                label="Mood"
                            )
                            
                            lyric_length = gr.Slider(
                                minimum=1,
                                maximum=8,
                                value=4,
                                step=1,
                                label="Number of Verses"
                            )
                        
                        lyric_generate_btn = gr.Button("✍️ Generate Lyrics", variant="primary")
                    
                    with gr.Column():
                        lyric_output = gr.Textbox(
                            label="Generated Lyrics",
                            lines=15,
                            max_lines=20
                        )
                        lyric_status = gr.Textbox(label="Status", interactive=False)
                
                lyric_generate_btn.click(
                    generate_lyrics,
                    inputs=[lyric_theme, lyric_style, lyric_mood, lyric_length],
                    outputs=[lyric_output, lyric_status]
                )
            
            # Recent Files Tab
            with gr.Tab("📁 Recent Files"):
                gr.Markdown("Browse and play recently generated files")
                
                refresh_btn = gr.Button("🔄 Refresh List")
                file_list = gr.File(
                    file_count="multiple",
                    label="Recent Files",
                    interactive=False
                )
                
                def refresh_files():
                    return get_recent_files()
                
                refresh_btn.click(refresh_files, outputs=[file_list])
                
                # Auto-refresh on load
                interface.load(refresh_files, outputs=[file_list])
        
        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 20px; color: #666;">
            <p>🎵 OpenMusic AI Studio - Open Source Music Creation Platform</p>
            <p>Built with ❤️ using FastAPI, Gradio, and open-source AI models</p>
        </div>
        """)
    
    return interface


def launch_interface():
    """Launch the Gradio interface"""
    interface = create_interface()
    
    # Initialize models in background
    initialize_models()
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=settings.debug
    )


if __name__ == "__main__":
    launch_interface()