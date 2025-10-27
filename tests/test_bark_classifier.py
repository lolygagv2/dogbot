#!/usr/bin/env python3
"""
Test script for bark emotion classifier
Tests model loading, inference, and audio capture
"""

import sys
import os
import time
import numpy as np
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.bark_classifier import BarkClassifier
from audio.bark_buffer import BarkAudioBuffer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_loading():
    """Test that the TFLite model loads correctly"""
    print("\n" + "="*60)
    print("TEST 1: Model Loading")
    print("="*60)

    try:
        classifier = BarkClassifier()
        print("✅ Model loaded successfully")
        print(f"   - Input shape: {classifier.input_details[0]['shape']}")
        print(f"   - Output shape: {classifier.output_details[0]['shape']}")
        print(f"   - Emotions: {list(classifier.emotion_mapping.keys())}")
        return True
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False


def test_dummy_inference():
    """Test inference on random audio data"""
    print("\n" + "="*60)
    print("TEST 2: Dummy Inference")
    print("="*60)

    try:
        classifier = BarkClassifier()

        # Create dummy audio (3 seconds of random data)
        sample_rate = 22050
        duration = 3.0
        dummy_audio = np.random.randn(int(sample_rate * duration)).astype(np.float32) * 0.1

        # Run inference
        result = classifier.predict(dummy_audio, confidence_threshold=0.3)

        print("✅ Inference successful")
        print(f"   - Predicted emotion: {result['emotion']}")
        print(f"   - Confidence: {result['confidence']:.3f}")
        print(f"   - Is confident: {result['is_confident']}")
        print("\n   All probabilities:")
        for emotion, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 30)
            print(f"     {emotion:12s}: {prob:.3f} {bar}")

        return True
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return False


def test_audio_file_inference(audio_path: str):
    """Test inference on an actual audio file"""
    print("\n" + "="*60)
    print(f"TEST 3: Audio File Inference")
    print(f"File: {audio_path}")
    print("="*60)

    if not os.path.exists(audio_path):
        print(f"⚠️  File not found: {audio_path}")
        return False

    try:
        classifier = BarkClassifier()

        # Run inference
        result = classifier.predict_from_file(audio_path, confidence_threshold=0.5)

        print("✅ File inference successful")
        print(f"   - Predicted emotion: {result['emotion']}")
        print(f"   - Confidence: {result['confidence']:.3f}")
        print(f"   - Is confident: {result['is_confident']}")
        print("\n   All probabilities:")
        for emotion, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
            bar = '█' * int(prob * 30)
            print(f"     {emotion:12s}: {prob:.3f} {bar}")

        return True
    except Exception as e:
        print(f"❌ File inference failed: {e}")
        return False


def test_audio_capture():
    """Test real-time audio capture"""
    print("\n" + "="*60)
    print("TEST 4: Audio Capture (5 seconds)")
    print("="*60)

    try:
        buffer = BarkAudioBuffer()
        buffer.start()

        print("🎤 Recording audio for 5 seconds...")
        print("   Make some noise or play dog barks!")

        for i in range(5):
            time.sleep(1)
            print(f"   {5-i} seconds remaining...")

        # Get captured audio
        audio_chunk = buffer.get_audio_chunk(timeout=1.0)
        buffer.stop()

        if audio_chunk is not None:
            print(f"✅ Audio captured successfully")
            print(f"   - Shape: {audio_chunk.shape}")
            print(f"   - Min value: {audio_chunk.min():.3f}")
            print(f"   - Max value: {audio_chunk.max():.3f}")
            print(f"   - Mean value: {audio_chunk.mean():.3f}")
            return True
        else:
            print("❌ No audio captured")
            return False

    except Exception as e:
        print(f"❌ Audio capture failed: {e}")
        return False


def test_live_detection(duration: int = 10):
    """Test live bark detection"""
    print("\n" + "="*60)
    print(f"TEST 5: Live Bark Detection ({duration} seconds)")
    print("="*60)

    try:
        classifier = BarkClassifier()
        buffer = BarkAudioBuffer()

        buffer.start()
        print(f"🎤 Listening for dog barks for {duration} seconds...")
        print("   Play dog bark sounds or make barking noises!")
        print("")

        start_time = time.time()
        detection_count = 0

        while time.time() - start_time < duration:
            # Get audio chunk
            audio_chunk = buffer.get_audio_chunk(timeout=0.5)

            if audio_chunk is not None:
                # Check audio level (simple VAD)
                audio_level = np.abs(audio_chunk).mean()

                if audio_level > 0.01:  # Threshold for "sound detected"
                    # Run classification
                    result = classifier.predict(audio_chunk, confidence_threshold=0.4)

                    if result['emotion'] != 'notbark' and result['is_confident']:
                        detection_count += 1
                        print(f"🐕 BARK DETECTED #{detection_count}: {result['emotion']} (conf: {result['confidence']:.2f})")
                    elif audio_level > 0.05:
                        print(f"   Sound detected (level: {audio_level:.3f}) - classified as: {result['emotion']}")

            # Show progress
            remaining = duration - (time.time() - start_time)
            if remaining > 0 and int(remaining) % 2 == 0:
                print(f"   {int(remaining)} seconds remaining...", end='\r')

        buffer.stop()
        print(f"\n✅ Live detection complete")
        print(f"   Total barks detected: {detection_count}")

        return True

    except Exception as e:
        print(f"❌ Live detection failed: {e}")
        if 'buffer' in locals():
            buffer.stop()
        return False


def test_batch_files(directory: str = "/home/morgan/dogbot/audio/dogbarktest"):
    """Test on multiple audio files in a directory"""
    print("\n" + "="*60)
    print(f"TEST 6: Batch File Testing")
    print(f"Directory: {directory}")
    print("="*60)

    if not os.path.exists(directory):
        print(f"⚠️  Directory not found: {directory}")
        print("   Creating directory for test audio files...")
        os.makedirs(directory, exist_ok=True)
        return False

    # Find audio files
    audio_files = []
    for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
        audio_files.extend(Path(directory).glob(ext))

    if not audio_files:
        print(f"⚠️  No audio files found in {directory}")
        print("   Add .wav, .mp3, .m4a, or .flac files to test")
        return False

    print(f"Found {len(audio_files)} audio files")
    print("")

    try:
        classifier = BarkClassifier()
        results = []

        for audio_file in audio_files:
            print(f"Processing: {audio_file.name}")

            try:
                result = classifier.predict_from_file(str(audio_file), confidence_threshold=0.4)

                # Store result
                results.append({
                    'file': audio_file.name,
                    'emotion': result['emotion'],
                    'confidence': result['confidence'],
                    'is_bark': result['emotion'] != 'notbark'
                })

                # Display result
                if result['emotion'] != 'notbark':
                    print(f"  🐕 BARK: {result['emotion']} (conf: {result['confidence']:.2f})")
                else:
                    print(f"  ❌ Not a bark (conf: {result['confidence']:.2f})")

            except Exception as e:
                print(f"  ⚠️  Failed: {e}")
                results.append({
                    'file': audio_file.name,
                    'error': str(e)
                })

        # Summary
        print("\n" + "-"*60)
        print("SUMMARY:")
        bark_files = [r for r in results if r.get('is_bark', False)]
        print(f"  Total files: {len(audio_files)}")
        print(f"  Barks detected: {len(bark_files)}")

        if bark_files:
            print("\n  Detected emotions:")
            emotion_counts = {}
            for r in bark_files:
                emotion = r['emotion']
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    {emotion}: {count}")

        # Save results
        results_file = Path(directory) / "test_results.txt"
        with open(results_file, 'w') as f:
            f.write("Bark Classification Test Results\n")
            f.write("="*60 + "\n")
            f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for r in results:
                if 'error' in r:
                    f.write(f"{r['file']}: ERROR - {r['error']}\n")
                else:
                    f.write(f"{r['file']}: {r['emotion']} (conf: {r['confidence']:.3f})\n")

        print(f"\n✅ Results saved to: {results_file}")
        return True

    except Exception as e:
        print(f"❌ Batch testing failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n🐕 TreatBot Bark Classifier Test Suite")
    print("="*60)

    # Track results
    results = {}

    # Test 1: Model loading
    results['model_loading'] = test_model_loading()

    # Test 2: Dummy inference
    results['dummy_inference'] = test_dummy_inference()

    # Test 3: File inference (if test file exists)
    test_audio = "/home/morgan/dogbot/audio/dogbarktest/sample.wav"
    if os.path.exists(test_audio):
        results['file_inference'] = test_audio_file_inference(test_audio)
    else:
        print(f"\n⚠️  Skipping file inference test (no test file at {test_audio})")

    # Test 4: Audio capture
    print("\n⚠️  Audio capture test requires microphone access")
    response = input("Run audio capture test? (y/n): ").lower()
    if response == 'y':
        results['audio_capture'] = test_audio_capture()

    # Test 5: Live detection
    response = input("\nRun live detection test? (y/n): ").lower()
    if response == 'y':
        results['live_detection'] = test_live_detection(duration=10)

    # Test 6: Batch files
    response = input("\nRun batch file test? (y/n): ").lower()
    if response == 'y':
        results['batch_files'] = test_batch_files()

    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")

    total = len(results)
    passed = sum(1 for p in results.values() if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    elif passed > 0:
        print("\n⚠️  Some tests failed")
    else:
        print("\n❌ All tests failed")


if __name__ == "__main__":
    main()