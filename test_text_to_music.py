#!/usr/bin/env python3
"""
Test script for OpenMusic text-to-music generation functionality.

This script tests the new AI-powered music generation capabilities
that allow users to create original songs from text prompts.
"""

import sys
import os

# Add the openmusic package to the path
sys.path.insert(0, '/home/runner/work/OpenMusic/OpenMusic')

import openmusic as om
import numpy as np

def test_prompt_processing():
    """Test the prompt processing functionality."""
    print("🔍 Testing Prompt Processing...")
    
    test_prompts = [
        "Create an upbeat pop song with energetic drums and catchy melody",
        "Generate a sad piano ballad in A minor with gentle strings",
        "Make a powerful rock anthem with electric guitars and strong vocals",
        "Compose a relaxing jazz piece with saxophone and soft piano",
        "Create mysterious electronic music with dark ambient sounds"
    ]
    
    processor = om.generation.PromptProcessor()
    
    for i, prompt in enumerate(test_prompts):
        print(f"\n  Test {i+1}: '{prompt[:50]}...'")
        params = processor.process_prompt(prompt)
        
        print(f"    Genre: {params['genre']}")
        print(f"    Mood: {params['mood']}")
        print(f"    Tempo: {params['tempo_bpm']} BPM")
        print(f"    Key: {params['key']}")
        print(f"    Instruments: {', '.join(params['instruments'])}")
        print(f"    Vocals: {params['vocals']['has_vocals']}")
    
    print("✅ Prompt processing tests completed!")
    return True

def test_music_generation():
    """Test basic music generation."""
    print("\n🎵 Testing Music Generation...")
    
    # Test simple generation
    try:
        audio, sr, metadata = om.generate_music_from_prompt(
            "Create a happy pop song with piano and drums", 
            duration=10.0
        )
        
        print(f"  ✅ Generated audio: {len(audio)} samples at {sr} Hz")
        print(f"  ✅ Duration: {len(audio)/sr:.2f} seconds")
        print(f"  ✅ Genre: {metadata['extracted_parameters']['genre']}")
        print(f"  ✅ Tempo: {metadata['extracted_parameters']['tempo_bpm']} BPM")
        
        # Save the generated music
        output_file = "/tmp/test_generated_music.wav"
        om.save_audio(output_file, audio, sr)
        print(f"  ✅ Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Music generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_song_generation():
    """Test complete song generation with structure."""
    print("\n🎼 Testing Song Generation...")
    
    try:
        audio, sr, metadata = om.generate_song(
            "Create an emotional rock ballad about overcoming challenges",
            structure="intro-verse-chorus-verse-chorus-bridge-chorus",
            duration=45.0
        )
        
        print(f"  ✅ Generated song: {len(audio)} samples at {sr} Hz")
        print(f"  ✅ Duration: {len(audio)/sr:.2f} seconds")
        print(f"  ✅ Structure: {metadata['structure']}")
        print(f"  ✅ Sections: {len(metadata['sections'])}")
        
        for section in metadata['sections']:
            print(f"    - {section['type']}: {section['duration']}s")
        
        # Save the generated song
        output_file = "/tmp/test_generated_song.wav"
        om.save_audio(output_file, audio, sr)
        print(f"  ✅ Saved to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Song generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_various_genres_and_moods():
    """Test generation with different genres and moods."""
    print("\n🎨 Testing Various Genres and Moods...")
    
    test_cases = [
        ("jazz", "relaxed jazz with piano and saxophone"),
        ("rock", "powerful rock song with electric guitars"),
        ("electronic", "energetic electronic dance music"),
        ("classical", "peaceful classical piece with strings"),
        ("folk", "acoustic folk song with storytelling feel")
    ]
    
    results = {}
    
    for genre, prompt in test_cases:
        try:
            print(f"  Testing {genre}...")
            audio, sr, metadata = om.generate_music_from_prompt(prompt, duration=8.0)
            
            extracted_genre = metadata['extracted_parameters']['genre']
            extracted_mood = metadata['extracted_parameters']['mood']
            
            print(f"    ✅ {genre}: Genre={extracted_genre}, Mood={extracted_mood}")
            results[genre] = True
            
        except Exception as e:
            print(f"    ❌ {genre}: Failed - {e}")
            results[genre] = False
    
    success_rate = sum(results.values()) / len(results) * 100
    print(f"  Success rate: {success_rate:.1f}%")
    
    return success_rate >= 80  # 80% success rate threshold

def test_integration_with_existing_features():
    """Test integration with existing OpenMusic features."""
    print("\n🔗 Testing Integration with Existing Features...")
    
    try:
        # Generate music
        audio, sr, metadata = om.generate_music_from_prompt(
            "Create upbeat electronic music", duration=8.0
        )
        
        # Test integration with existing features
        print("  Testing feature extraction on generated music...")
        features = om.features.extract_all_features(audio, sr)
        print(f"    ✅ Extracted {len(features)} feature types")
        
        print("  Testing music analysis on generated music...")
        tempo = om.analysis.detect_tempo(audio, sr)
        key, mode = om.analysis.detect_key(audio, sr)
        print(f"    ✅ Detected tempo: {tempo:.1f} BPM")
        print(f"    ✅ Detected key: {key} {mode}")
        
        print("  Testing effects on generated music...")
        reverb_audio = om.effects.add_reverb(audio, sr, room_size=0.5)
        print(f"    ✅ Applied reverb: shape {reverb_audio.shape}")
        
        print("  Testing enhancement on generated music...")
        enhanced = om.enhancement.reduce_noise(audio, sr)
        print(f"    ✅ Applied noise reduction: shape {enhanced.shape}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_text_to_music_examples():
    """Demonstrate various text-to-music examples."""
    print("\n🎯 Demonstrating Text-to-Music Examples...")
    
    examples = [
        {
            'prompt': "Create an upbeat, anthemic pop song for a female vocalist. The lyrics should be about overcoming challenges and finding inner strength. I want a driving beat, prominent synthesizers, and a powerful, soaring chorus. Focus on a positive and inspiring mood.",
            'name': "inspirational_pop"
        },
        {
            'prompt': "Generate a reflective, hopeful indie pop song with acoustic guitar, light drums, and subtle piano. The mood should be contemplative but optimistic.",
            'name': "indie_reflection"
        },
        {
            'prompt': "Create a dark, mysterious electronic track with deep bass, ethereal pads, and subtle percussion. Make it atmospheric and haunting.",
            'name': "dark_electronic"
        }
    ]
    
    for i, example in enumerate(examples):
        print(f"\n  Example {i+1}: {example['name']}")
        print(f"  Prompt: '{example['prompt'][:80]}...'")
        
        try:
            audio, sr, metadata = om.generate_music_from_prompt(
                example['prompt'], duration=12.0
            )
            
            params = metadata['extracted_parameters']
            print(f"    ✅ Generated: {params['genre']} in {params['key']}")
            print(f"    ✅ Tempo: {params['tempo_bpm']} BPM, Mood: {params['mood']}")
            print(f"    ✅ Instruments: {', '.join(params['instruments'][:3])}")
            
            # Save example
            output_file = f"/tmp/{example['name']}_example.wav"
            om.save_audio(output_file, audio, sr)
            print(f"    ✅ Saved to: {output_file}")
            
        except Exception as e:
            print(f"    ❌ Failed: {e}")
    
    print("\n✨ Text-to-music examples completed!")

def main():
    """Run all text-to-music generation tests."""
    print("🎵" * 20)
    print("🎵 OPENMUSIC TEXT-TO-MUSIC TESTS 🎵")
    print("🎵 AI-Powered Music Generation 🎵")
    print("🎵" * 20)
    
    # Create temp directory for outputs
    os.makedirs("/tmp", exist_ok=True)
    
    try:
        # Run tests
        tests_passed = 0
        total_tests = 5
        
        if test_prompt_processing():
            tests_passed += 1
        
        if test_music_generation():
            tests_passed += 1
            
        if test_song_generation():
            tests_passed += 1
            
        if test_various_genres_and_moods():
            tests_passed += 1
            
        if test_integration_with_existing_features():
            tests_passed += 1
        
        # Demonstrate examples
        demonstrate_text_to_music_examples()
        
        print(f"\n🌟 TEST SUMMARY")
        print("=" * 50)
        print(f"✅ Tests passed: {tests_passed}/{total_tests}")
        print(f"📊 Success rate: {tests_passed/total_tests*100:.1f}%")
        
        if tests_passed == total_tests:
            print("🎉 All text-to-music generation tests passed!")
            print("🚀 OpenMusic AI-powered music generation is ready!")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        print("\n✨ Key Features Implemented:")
        print("  • Text prompt interpretation and parameter extraction")
        print("  • AI-powered music generation from text descriptions")
        print("  • Complete song generation with customizable structure") 
        print("  • Genre, mood, tempo, and key detection from prompts")
        print("  • Integration with existing audio processing features")
        print("  • Support for various musical styles and moods")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()