#!/usr/bin/env python3
"""
Test bark detection on actual WAV files to verify the model works
"""

import sys
import os
import numpy as np
import wave
import json

sys.path.append('/home/morgan/dogbot')

def load_wav_file(filepath):
    """Load a WAV file and return audio data"""
    with wave.open(filepath, 'rb') as wav:
        frames = wav.readframes(wav.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        sample_rate = wav.getframerate()
        return audio, sample_rate

def test_wav_files():
    """Test bark detection on actual dog barking WAV files"""

    print("="*60)
    print("BARK DETECTION WAV FILE TEST")
    print("="*60)

    # Load the actual bark classifier to see how it processes audio
    from ai.bark_classifier import BarkClassifier

    print("\nLoading bark classifier...")
    classifier = BarkClassifier(
        model_path='/home/morgan/dogbot/ai/models/dog_bark_classifier.tflite',
        emotion_mapping_path='/home/morgan/dogbot/ai/models/emotion_mapping.json',
        sample_rate=22050,
        duration=3.0,
        n_mels=128
    )
    print("✅ Classifier loaded")

    # Test on some WAV files
    test_dir = '/home/morgan/dogbot/audio/dogbarktest/'
    wav_files = [f for f in os.listdir(test_dir) if f.endswith('.wav')][:5]  # Test first 5

    print(f"\nTesting {len(wav_files)} WAV files from {test_dir}")
    print("-" * 60)

    for wav_file in wav_files:
        filepath = os.path.join(test_dir, wav_file)
        print(f"\nTesting: {wav_file}")

        try:
            # Load audio
            audio, orig_sr = load_wav_file(filepath)
            print(f"  Original: {len(audio)} samples @ {orig_sr}Hz")

            # Resample to 22050 if needed
            if orig_sr != 22050:
                from scipy import signal
                num_samples = int(len(audio) * 22050 / orig_sr)
                audio = signal.resample(audio, num_samples)
                print(f"  Resampled to: {len(audio)} samples @ 22050Hz")

            # Ensure 3 seconds duration
            target_length = int(22050 * 3.0)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]

            # Get prediction
            result = classifier.predict(audio)

            print(f"  ✅ Result: {result['emotion']} ({result['confidence']:.1%})")

            # Show top 3
            sorted_probs = sorted(result['all_probabilities'].items(),
                                key=lambda x: x[1], reverse=True)
            print("  Top 3:")
            for emotion, prob in sorted_probs[:3]:
                print(f"    - {emotion}: {prob:.1%}")

        except Exception as e:
            print(f"  ❌ Error: {e}")

    print("\n" + "="*60)

def inspect_model_input():
    """Inspect what the model actually expects"""

    print("\n=== MODEL INSPECTION ===")

    try:
        import tflite_runtime.interpreter as tflite
    except:
        import tensorflow.lite as tflite

    model_path = '/home/morgan/dogbot/ai/models/dog_bark_classifier.tflite'
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    print(f"Input shape: {input_details['shape']}")
    print(f"Input dtype: {input_details['dtype']}")
    print(f"Output shape: {output_details['shape']}")

    # Create dummy input with correct shape
    dummy_input = np.zeros(input_details['shape'], dtype=np.float32)
    print(f"Dummy input shape: {dummy_input.shape}")

    # Test inference
    interpreter.set_tensor(input_details['index'], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])
    print(f"Output shape: {output.shape}")
    print(f"Output values (zeros input): {output}")

if __name__ == "__main__":
    inspect_model_input()
    print()
    test_wav_files()