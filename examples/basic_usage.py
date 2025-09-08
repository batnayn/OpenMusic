"""
Example script demonstrating OpenMusic AI Studio usage
"""

import asyncio
import os
from app.models.music_generator import MusicGenerator
from app.models.vocal_generator import VocalGenerator
from app.models.llm_client import LLMClient
from app.core.audio_processor import AudioProcessor


async def main():
    """Run example generations"""
    
    print("🎵 OpenMusic AI Studio - Example Usage\n")
    
    # Initialize generators
    print("📥 Initializing AI models...")
    music_gen = MusicGenerator()
    vocal_gen = VocalGenerator()
    llm = LLMClient()
    audio_proc = AudioProcessor()
    
    print("✅ Models initialized!\n")
    
    # Generate lyrics
    print("1️⃣ Generating lyrics...")
    lyrics = await llm.generate_lyrics(
        theme="space exploration",
        style="electronic",
        mood="mysterious",
        length=3
    )
    print(f"Generated lyrics:\n{lyrics}\n")
    
    # Generate music
    print("2️⃣ Generating background music...")
    music_file = await music_gen.generate(
        prompt="mysterious electronic ambient space music",
        duration=20,
        temperature=0.8
    )
    print(f"Generated music: {music_file}\n")
    
    # Generate vocals
    print("3️⃣ Generating vocals...")
    # Extract first verse for vocal generation
    first_verse = lyrics.split('\n\n')[0].replace('[Verse 1]', '').strip()
    first_lines = '\n'.join(first_verse.split('\n')[:2])  # First 2 lines
    
    vocal_file = await vocal_gen.generate(
        text=first_lines,
        voice_preset="v2/en_speaker_3",
        temperature=0.7
    )
    print(f"Generated vocals: {vocal_file}\n")
    
    # Mix music and vocals
    print("4️⃣ Mixing audio...")
    mixed_file = await audio_proc.process(
        audio_files=[music_file, vocal_file],
        operation="mix",
        parameters={"weights": [0.7, 1.0]}  # Music quieter, vocals prominent
    )
    print(f"Mixed audio: {mixed_file}\n")
    
    print("🎉 Example complete! Check the outputs/ directory for generated files.")
    
    # List all generated files
    print("\n📁 Generated files:")
    for filename in os.listdir("outputs"):
        filepath = os.path.join("outputs", filename)
        size = os.path.getsize(filepath)
        print(f"  - {filename} ({size} bytes)")


if __name__ == "__main__":
    asyncio.run(main())