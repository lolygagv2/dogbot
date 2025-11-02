#!/usr/bin/env python3
"""
Test the bark model directly to see if it's working correctly
"""

import sys
import numpy as np
sys.path.append('/home/morgan/dogbot')

from ai.bark_classifier import BarkClassifier

# Create the classifier
classifier = BarkClassifier()

print("Testing bark classifier with different inputs:")
print("="*60)

# Test 1: Complete silence
print("\n1. Testing with SILENCE (should be 'notbark'):")
silent_audio = np.zeros(int(22050 * 3), dtype=np.float32)
result = classifier.predict(silent_audio)
print(f"   Result: {result['emotion']} ({result['confidence']:.2%})")
print(f"   All probabilities:")
for emotion, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
    print(f"     - {emotion}: {prob:.2%}")

# Test 2: Random noise
print("\n2. Testing with RANDOM NOISE (should be 'notbark'):")
noise_audio = np.random.randn(int(22050 * 3)) * 0.1
result = classifier.predict(noise_audio)
print(f"   Result: {result['emotion']} ({result['confidence']:.2%})")
print(f"   All probabilities:")
for emotion, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
    print(f"     - {emotion}: {prob:.2%}")

# Test 3: Sine wave (pure tone)
print("\n3. Testing with SINE WAVE 1000Hz (should be 'notbark'):")
t = np.linspace(0, 3, int(22050 * 3))
sine_audio = np.sin(2 * np.pi * 1000 * t) * 0.5
result = classifier.predict(sine_audio)
print(f"   Result: {result['emotion']} ({result['confidence']:.2%})")
print(f"   All probabilities:")
for emotion, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
    print(f"     - {emotion}: {prob:.2%}")

# Test 4: Clicking sound (impulses)
print("\n4. Testing with CLICKS (should be 'notbark'):")
click_audio = np.zeros(int(22050 * 3), dtype=np.float32)
# Add clicks every 0.5 seconds
for i in range(6):
    idx = int(i * 22050 * 0.5)
    if idx < len(click_audio):
        click_audio[idx:idx+100] = 0.8
result = classifier.predict(click_audio)
print(f"   Result: {result['emotion']} ({result['confidence']:.2%})")
print(f"   All probabilities:")
for emotion, prob in sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
    print(f"     - {emotion}: {prob:.2%}")

print("\n" + "="*60)
print("ANALYSIS:")
print("If most of these are NOT returning 'notbark' with high confidence,")
print("then the model or preprocessing is broken.")