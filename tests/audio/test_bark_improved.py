#!/usr/bin/env python3
"""
Test improved bark detection with better null rejection
"""

import sys
import os
import time
import subprocess
import tempfile
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/home/morgan/dogbot')

# Import bark detection components
from services.perception.bark_detector import BarkDetectorService
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_direct_detection():
    """Test bark detection with various sounds"""

    # Initialize detector with test config
    config = {
        'enabled': True,
        'model_path': '/home/morgan/dogbot/ai/models/dog_bark_classifier.tflite',
        'emotion_mapping_path': '/home/morgan/dogbot/ai/models/emotion_mapping.json',
        'sample_rate': 22050,
        'duration': 3.0,
        'audio_gain': 30.0,
        'confidence_threshold': 0.55,
        'reward_emotions': ['alert', 'attention'],
        'check_interval': 0.5,
        'cooldown_period': 5.0
    }

    detector = BarkDetectorService(config)

    if not detector.initialize():
        print("Failed to initialize detector")
        return

    print("\n=== Testing Improved Bark Detection ===")
    print("Will test with different sounds to verify null rejection")
    print("Press Ctrl+C to stop\n")

    try:
        # Start the detector
        detector.start()

        print("Listening for barks... Make different sounds:")
        print("1. Try fake barking sounds")
        print("2. Try talking/speaking")
        print("3. Try clapping or tapping")
        print("4. Try silence")
        print("5. Try playing music\n")

        # Monitor for 30 seconds
        for i in range(30):
            time.sleep(1)
            status = detector.get_status()

            # Show status every 5 seconds
            if i % 5 == 0:
                print(f"\nTime: {i}s")
                print(f"Total barks detected: {status['statistics']['total_barks']}")
                print(f"Emotions: {status['statistics']['emotions_detected']}")

                if status['statistics']['total_barks'] > 0:
                    print("✓ Bark detection is working!")
                else:
                    print("No barks detected yet...")

        print("\n=== Final Statistics ===")
        final_status = detector.get_status()
        print(f"Total barks: {final_status['statistics']['total_barks']}")
        print(f"Emotions detected: {final_status['statistics']['emotions_detected']}")
        print(f"Rewarded barks: {final_status['statistics']['rewarded_barks']}")

    except KeyboardInterrupt:
        print("\n\nStopping test...")
    finally:
        detector.stop()
        print("Test completed")

def test_wav_files():
    """Test with WAV files if available"""

    import glob
    wav_files = glob.glob('/home/morgan/dogbot/*.wav')

    if not wav_files:
        print("No WAV files found for testing")
        return

    print(f"\n=== Testing with {len(wav_files)} WAV files ===")

    # Import classifier directly
    from ai.bark_classifier import BarkClassifier

    classifier = BarkClassifier(
        model_path='/home/morgan/dogbot/ai/models/dog_bark_classifier.tflite',
        emotion_mapping_path='/home/morgan/dogbot/ai/models/emotion_mapping.json'
    )

    for wav_file in wav_files:
        print(f"\nTesting: {Path(wav_file).name}")
        try:
            result = classifier.predict_from_file(wav_file, confidence_threshold=0.55)

            print(f"  Emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
            print(f"  All probabilities:")
            for emotion, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
                print(f"    {emotion}: {prob:.3f}")

            # Check our new logic
            notbark_conf = result['all_probabilities'].get('notbark', 0.0)
            is_bark = (result['emotion'] != 'notbark' and
                      result['is_confident'] and
                      notbark_conf < 0.5)
            print(f"  Would trigger bark detection: {is_bark}")

        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    print("Testing improved bark detection with better null rejection\n")

    # Test with WAV files first if available
    test_wav_files()

    # Then test live detection
    print("\n" + "="*50)
    test_direct_detection()