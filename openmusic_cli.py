#!/usr/bin/env python3
"""
OpenMusic CLI - Command Line Interface for AI-Powered Music Generation

A simple command-line tool that demonstrates the text-to-music generation
capabilities of OpenMusic. Users can create songs by providing text prompts.
"""

import sys
import os
import argparse

# Add the openmusic package to the path
sys.path.insert(0, '/home/runner/work/OpenMusic/OpenMusic')

import openmusic as om

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Generate music from text prompts using OpenMusic AI",
        epilog="""
Examples:
  python openmusic_cli.py "Create a happy pop song with piano"
  python openmusic_cli.py "Sad ballad in minor key" --duration 20 --output my_song.wav
  python openmusic_cli.py "Electronic dance music" --song --structure verse-chorus-verse-chorus
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        'prompt', 
        help='Text description of the music to generate'
    )
    
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=15.0,
        help='Duration of generated music in seconds (default: 15.0)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='generated_music.wav',
        help='Output filename (default: generated_music.wav)'
    )
    
    parser.add_argument(
        '--song', '-s',
        action='store_true',
        help='Generate a complete song with structure instead of a short clip'
    )
    
    parser.add_argument(
        '--structure',
        type=str,
        default='verse-chorus-verse-chorus-bridge-chorus',
        help='Song structure for complete songs (default: verse-chorus-verse-chorus-bridge-chorus)'
    )
    
    parser.add_argument(
        '--show-params',
        action='store_true',
        help='Show extracted musical parameters'
    )
    
    args = parser.parse_args()
    
    print("🎵 OpenMusic AI - Text-to-Music Generation")
    print("=" * 50)
    print(f"Prompt: \"{args.prompt}\"")
    
    try:
        if args.song:
            print(f"Generating complete song with structure: {args.structure}")
            audio, sr, metadata = om.generate_song(
                args.prompt,
                structure=args.structure
            )
            print(f"✅ Generated song: {len(audio)/sr:.1f} seconds")
        else:
            print(f"Generating music clip: {args.duration} seconds")
            audio, sr, metadata = om.generate_music_from_prompt(
                args.prompt,
                duration=args.duration
            )
            print(f"✅ Generated music: {len(audio)/sr:.1f} seconds")
        
        # Show parameters if requested
        if args.show_params:
            params = metadata.get('extracted_parameters') or metadata.get('base_parameters', {})
            if params:
                print(f"\n📊 Extracted Parameters:")
                print(f"   Genre: {params.get('genre', 'Unknown')}")
                print(f"   Mood: {params.get('mood', 'Unknown')}")
                print(f"   Tempo: {params.get('tempo_bpm', 'Unknown')} BPM")
                print(f"   Key: {params.get('key', 'Unknown')}")
                print(f"   Instruments: {', '.join(params.get('instruments', []))}")
                print(f"   Vocals: {params.get('vocals', {}).get('has_vocals', 'Unknown')}")
            else:
                print("\n📊 Parameters not available for this generation type")
        
        # Save the generated music
        om.save_audio(args.output, audio, sr)
        print(f"🎼 Saved to: {args.output}")
        
        print("\n✨ Music generation complete!")
        
    except Exception as e:
        print(f"❌ Error generating music: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()