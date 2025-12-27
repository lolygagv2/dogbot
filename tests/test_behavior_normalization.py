#!/usr/bin/env python3
"""
Quick test to verify behavior model normalization is working correctly
"""

import sys
sys.path.insert(0, '/home/morgan/dogbot')

import numpy as np
import torch

# Load behavior model
behavior_model_path = "/home/morgan/dogbot/ai/models/behavior_14.ts"
behavior_model = torch.jit.load(behavior_model_path, map_location="cpu").eval()

behaviors = ["stand", "sit", "lie", "cross", "spin"]

print("=== Testing Behavior Model with Different Normalization ===\n")

# Test 1: Random normalized data (0-1) - this worked before
print("Test 1: Random normalized data (0-1)")
random_input = torch.rand(1, 16, 48)  # (batch=1, T=16, features=48)
with torch.no_grad():
    output = behavior_model(random_input)
    probs = torch.softmax(output, dim=-1).numpy()[0]

for b, p in zip(behaviors, probs):
    print(f"  {b}: {p*100:.2f}%")
print()

# Test 2: Simulated 4K coordinates (BAD - what was happening before)
print("Test 2: 4K pixel coords divided by 640 (WRONG)")
# Simulate 4K keypoints (e.g., 0-3840 for x, 0-2160 for y)
fake_4k_kpts = np.random.rand(16, 24, 2) * np.array([3840, 2160])
# Old buggy normalization - divide by 640
bad_normalized = fake_4k_kpts / 640.0  # Values go to 0-6 !
bad_input = torch.tensor(bad_normalized.reshape(1, 16, 48), dtype=torch.float32)
with torch.no_grad():
    output = behavior_model(bad_input)
    probs = torch.softmax(output, dim=-1).numpy()[0]

print(f"  Input range: {bad_normalized.min():.2f} - {bad_normalized.max():.2f}")
for b, p in zip(behaviors, probs):
    print(f"  {b}: {p*100:.2f}%")
print()

# Test 3: Proper normalization (CORRECT)
print("Test 3: 4K pixel coords properly normalized (CORRECT)")
# Same 4K keypoints
proper_normalized = fake_4k_kpts.copy()
proper_normalized[:, :, 0] = proper_normalized[:, :, 0] / 3840  # x / width
proper_normalized[:, :, 1] = proper_normalized[:, :, 1] / 2160  # y / height
good_input = torch.tensor(proper_normalized.reshape(1, 16, 48), dtype=torch.float32)
with torch.no_grad():
    output = behavior_model(good_input)
    probs = torch.softmax(output, dim=-1).numpy()[0]

print(f"  Input range: {proper_normalized.min():.2f} - {proper_normalized.max():.2f}")
for b, p in zip(behaviors, probs):
    print(f"  {b}: {p*100:.2f}%")
print()

# Test 4: Simulated SITTING pose (keypoints clustered lower with hips down)
print("Test 4: Simulated sitting pose (0-1 normalized)")
# Create a sitting pose pattern across 16 frames
sitting_sequence = np.zeros((16, 24, 2))
for t in range(16):
    # Place body keypoints in a "sitting" configuration
    # Front legs up, back legs down (y increases = lower in image)
    for k in range(24):
        if k < 12:  # Front body keypoints
            sitting_sequence[t, k, 0] = 0.4 + np.random.rand() * 0.2  # x centered
            sitting_sequence[t, k, 1] = 0.5 + np.random.rand() * 0.1  # y middle
        else:  # Back body keypoints (should be lower/higher y for sitting)
            sitting_sequence[t, k, 0] = 0.4 + np.random.rand() * 0.2  # x centered
            sitting_sequence[t, k, 1] = 0.65 + np.random.rand() * 0.1  # y lower (sitting)

sit_input = torch.tensor(sitting_sequence.reshape(1, 16, 48), dtype=torch.float32)
with torch.no_grad():
    output = behavior_model(sit_input)
    probs = torch.softmax(output, dim=-1).numpy()[0]

for b, p in zip(behaviors, probs):
    marker = " <-- Expected" if b == "sit" else ""
    print(f"  {b}: {p*100:.2f}%{marker}")
print()

print("=== Summary ===")
print("If Test 2 shows 0% for most classes, that's the bug we fixed")
print("Test 3 should show proper distribution like Test 1")
