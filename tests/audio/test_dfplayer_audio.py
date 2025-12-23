#!/usr/bin/env python3
"""
Test DFPlayer Pro audio functionality
Tests the AT command implementation
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.hardware.audio_controller import AudioController
from config.settings import AudioFiles

def main():
    """Test DFPlayer Pro with AT commands"""
    print("🔊 DFPlayer Pro Audio Test")
    print("=" * 50)

    # Initialize audio controller
    audio = AudioController()

    if not audio.is_initialized():
        print("❌ Audio controller failed to initialize")
        print("Check:")
        print("1. DFPlayer Pro connected to /dev/ttyAMA0")
        print("2. Audio files loaded on SD card")
        print("3. Serial permissions correct")
        return

    print("\n✅ Audio controller initialized")
    print("Testing AT command audio playback...")

    try:
        # Test volume setting
        print("\n1. Setting volume to 20...")
        audio.set_volume(20)
        time.sleep(1)

        # Test specific sounds using settings.py mappings
        test_sounds = [
            ("ELSA", "Elsa's name"),
            ("SIT", "Sit command"),
            ("GOOD_DOG", "Good dog praise")
        ]

        for sound_name, description in test_sounds:
            print(f"\n2. Playing {description}...")
            audio.play_sound(sound_name)
            time.sleep(3)  # Wait for audio to finish

        # Test direct file path
        print("\n3. Playing direct path /talks/0003.mp3...")
        audio.play_file_by_path("/talks/0003.mp3")
        time.sleep(3)

        # Test file by number (using AT+PLAYNUM)
        print("\n4. Playing file #15 (Sit command)...")
        audio.play_file_by_number(15)
        time.sleep(3)

        print("\n✅ All audio tests completed!")
        print("\nAudio system status:")
        status = audio.get_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
    finally:
        audio.cleanup()
        print("\n✅ Audio controller cleaned up")

if __name__ == "__main__":
    main()