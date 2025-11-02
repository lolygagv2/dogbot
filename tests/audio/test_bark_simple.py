#!/usr/bin/env python3
import sys
import time
import os
import numpy as np
sys.path.append('/home/morgan/dogbot')

from ai.bark_classifier import BarkClassifier
from audio.bark_buffer import BarkAudioBuffer

print("Initializing bark detection...")
classifier = BarkClassifier(
    model_path='/home/morgan/dogbot/ai/models/dog_bark_classifier.tflite',
    emotion_mapping_path='/home/morgan/dogbot/ai/models/emotion_mapping.json'
)

# Don't use the broken BarkAudioBuffer - use arecord instead
print("\n🎤 LISTENING FOR BARKS - Using arecord (won't freeze)\n")

import subprocess
import tempfile

def record_chunk(duration=1.0):
    """Record using arecord to avoid freezing"""
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_path = tmp.name
    tmp.close()

    cmd = ['arecord', '-D', 'hw:2,0', '-f', 'S16_LE', '-r', '44100',
           '-c', '1', '-d', str(int(duration)), tmp_path]

    try:
        # Short timeout - if it freezes, we'll know
        result = subprocess.run(cmd, timeout=duration+0.2,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode == 0 and os.path.exists(tmp_path):
            import wave
            with wave.open(tmp_path, 'rb') as w:
                frames = w.readframes(w.getnframes())
                audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            os.unlink(tmp_path)
            return audio * 30  # Apply gain for quiet mic
        else:
            print(f"    arecord failed: {result.stderr.decode()[:100]}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {duration+0.2}s - USB frozen!")
        # Kill the stuck process
        subprocess.run(['pkill', 'arecord'], capture_output=True)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None
    except Exception as e:
        print(f"    Error: {e}")
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None

# Test direct 3-second recordings - what the model actually needs
print("\nRecording 3-second chunks (what the model expects)...")

for i in range(10):  # 10 attempts
    print(f"\n[{i+1}/10] Recording 3 seconds...")
    audio = record_chunk(3.0)

    if audio is not None:
        energy = np.sqrt(np.mean(audio**2))
        print(f"  Energy: {energy:.4f}")

        if energy > 0.05:  # Reasonable threshold for amplified audio
            # Resample to 22050Hz for classifier
            from scipy import signal
            audio_22k = signal.resample(audio, int(len(audio) * 22050 / 44100))

            result = classifier.predict(audio_22k)

            # Show all results with confidence
            if result['emotion'] != 'notbark':
                if result['confidence'] > 0.5:
                    print(f"  🐕 BARK DETECTED: {result['emotion'].upper()} ({result['confidence']:.1%})")
                else:
                    print(f"  Possible bark: {result['emotion']} ({result['confidence']:.1%})")
            else:
                print(f"  Not a bark ({result['confidence']:.1%} sure)")

            # Show top 3 predictions
            sorted_probs = sorted(result['all_probabilities'].items(),
                                key=lambda x: x[1], reverse=True)[:3]
            print("  Top predictions:", ", ".join([f"{e}:{p:.0%}" for e,p in sorted_probs]))
        else:
            print("  Too quiet to analyze")
    else:
        print("  Recording failed")

    time.sleep(0.5)  # Brief pause between recordings

print("Done")
