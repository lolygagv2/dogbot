#!/usr/bin/env python3
"""
Record audio and save it so we can check what's actually being captured
"""

import subprocess
import time

print("="*60)
print("AUDIO RECORDING TEST")
print("="*60)
print("\nWill record 3 separate 3-second chunks")
print("PLAY BARK SOUNDS from your phone/speaker during recording!\n")

for i in range(1, 4):
    filename = f"/home/morgan/dogbot/test_recording_{i}.wav"
    print(f"Recording #{i} - MAKE NOISE NOW! (3 seconds)")

    cmd = ['arecord', '-D', 'hw:0,0', '-f', 'S16_LE', '-r', '44100', '-c', '1', '-d', '3', filename]
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode == 0:
        print(f"  ✓ Saved to {filename}")
    else:
        print(f"  ✗ Failed: {result.stderr.decode()}")

    if i < 3:
        print("  Waiting 2 seconds before next recording...\n")
        time.sleep(2)

print("\n" + "="*60)
print("Now testing these recordings with the bark classifier...")
print("="*60 + "\n")

import sys
import numpy as np
import wave
sys.path.insert(0, '/home/morgan/dogbot')
from ai.bark_classifier import BarkClassifier

classifier = BarkClassifier()

for i in range(1, 4):
    filename = f"/home/morgan/dogbot/test_recording_{i}.wav"
    print(f"\nTesting recording #{i}: {filename}")

    try:
        # Read the WAV file
        with wave.open(filename, 'rb') as w:
            frames = w.readframes(w.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

            # Show audio stats
            print(f"  Raw audio stats:")
            print(f"    Range: {audio.min():.6f} to {audio.max():.6f}")
            print(f"    RMS energy: {np.sqrt(np.mean(audio**2)):.6f}")

            # Test with different gain levels
            for gain in [1, 10, 30, 50]:
                audio_gained = np.clip(audio * gain, -1.0, 1.0)

                # Resample to 22050 Hz for model
                from scipy import signal
                audio_resampled = signal.resample(audio_gained, int(len(audio_gained) * 22050 / 44100))

                # Classify
                result = classifier.predict(audio_resampled, confidence_threshold=0.55)

                print(f"\n  With {gain}x gain:")
                print(f"    Result: {result['emotion']} (conf: {result['confidence']:.3f})")
                print(f"    NotBark: {result['all_probabilities'].get('notbark', 0):.3f}")

                # Only show full probabilities for one gain level
                if gain == 30:
                    print(f"    All probabilities:")
                    for emotion, prob in sorted(result['all_probabilities'].items(),
                                               key=lambda x: x[1], reverse=True)[:3]:
                        print(f"      {emotion}: {prob:.3f}")

    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*60)
print("ANALYSIS:")
print("- If all show 'notbark' even when you played barks, the mic is the problem")
print("- If gain changes the result, we need to tune the gain level")
print("- The WAV files are saved for manual inspection")
print("="*60)