#!/usr/bin/env python3
"""Test Xbox controller audio functionality"""

import requests
import time

def test_audio():
    """Test audio playback via API"""
    print("Testing audio API endpoints...")

    # Test play track 8 (Good Dog)
    print("\n1. Playing track 8 (Good Dog)...")
    response = requests.post("http://localhost:8000/audio/play",
                            json={"track": 8, "name": "Good Dog"})
    print(f"Response: {response.json()}")
    time.sleep(3)

    # Test play track 13 (Treat)
    print("\n2. Playing track 13 (Treat)...")
    response = requests.post("http://localhost:8000/audio/play",
                            json={"track": 13, "name": "Treat"})
    print(f"Response: {response.json()}")
    time.sleep(3)

    # Test pause
    print("\n3. Testing pause...")
    response = requests.post("http://localhost:8000/audio/pause")
    print(f"Response: {response.json()}")
    time.sleep(2)

    # Test status
    print("\n4. Getting audio status...")
    response = requests.get("http://localhost:8000/audio/status")
    print(f"Response: {response.json()}")

    print("\n✅ Audio API test complete!")
    print("\nIf you heard audio, the API is working.")
    print("If not, check:")
    print("- DFPlayer serial connection (/dev/ttyAMA0)")
    print("- Speaker connections")
    print("- SD card with audio files")

if __name__ == "__main__":
    test_audio()