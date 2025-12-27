#!/usr/bin/env python3
"""
Explore what keypoint patterns trigger different behaviors
"""

import sys
sys.path.insert(0, '/home/morgan/dogbot')

import numpy as np
import torch

# Load behavior model
behavior_model_path = "/home/morgan/dogbot/ai/models/behavior_14.ts"
behavior_model = torch.jit.load(behavior_model_path, map_location="cpu").eval()

behaviors = ["stand", "sit", "lie", "cross", "spin"]

def test_model(input_tensor, label):
    """Test model and print results"""
    with torch.no_grad():
        output = behavior_model(input_tensor)
        probs = torch.softmax(output, dim=-1).numpy()[0]

    winner = behaviors[np.argmax(probs)]
    print(f"{label}: {winner} ({max(probs)*100:.1f}%)")
    return probs

print("=== Exploring Behavior Model Activation Patterns ===\n")

# Try random patterns to find what triggers "sit"
print("Testing 50 random patterns to find sit activation...")
best_sit_prob = 0
best_sit_input = None
for i in range(50):
    random_input = torch.rand(1, 16, 48)
    probs = test_model(random_input, f"Random {i+1}")
    if probs[1] > best_sit_prob:  # sit is index 1
        best_sit_prob = probs[1]
        best_sit_input = random_input.clone()

print(f"\nBest sit probability: {best_sit_prob*100:.2f}%")
if best_sit_prob > 0.1:
    print("Found pattern that triggers sit!")
    print(f"Input stats: min={best_sit_input.min():.3f}, max={best_sit_input.max():.3f}, mean={best_sit_input.mean():.3f}")

# Test extreme patterns
print("\n=== Testing Extreme Patterns ===")

# All zeros
zeros = torch.zeros(1, 16, 48)
test_model(zeros, "All zeros")

# All ones
ones = torch.ones(1, 16, 48)
test_model(ones, "All ones")

# All 0.5 (center)
center = torch.ones(1, 16, 48) * 0.5
test_model(center, "All 0.5 (center)")

# Negative values
negative = torch.ones(1, 16, 48) * -1
test_model(negative, "All negative (-1)")

# Large positive values
large = torch.ones(1, 16, 48) * 10
test_model(large, "All large (10)")

# Check model input expectations
print("\n=== Model Input Analysis ===")
print(f"Model expects: (batch, T=16, features=48)")
print("Features = 24 keypoints × 2 (x, y) = 48")
print("\nKeypoint layout (assuming standard dog pose):")
print("Keypoints 0-11: Front body (head, neck, shoulders, front legs)")
print("Keypoints 12-23: Back body (spine, hips, back legs, tail)")

# Try patterns with specific keypoint relationships
print("\n=== Testing Keypoint Relationship Patterns ===")

# Sitting: back legs (y values high = low in image) relative to front
def create_sitting_pattern():
    kpts = np.zeros((16, 48))
    for t in range(16):
        # Front body - centered x, upper y
        for k in range(0, 12):
            kpts[t, k*2] = 0.5 + np.random.randn() * 0.1  # x centered
            kpts[t, k*2+1] = 0.3 + np.random.randn() * 0.05  # y upper
        # Back body - centered x, lower y (sitting = hips down)
        for k in range(12, 24):
            kpts[t, k*2] = 0.5 + np.random.randn() * 0.1  # x centered
            kpts[t, k*2+1] = 0.7 + np.random.randn() * 0.05  # y lower
    return torch.tensor(kpts, dtype=torch.float32).unsqueeze(0)

# Lying: all keypoints at bottom
def create_lying_pattern():
    kpts = np.zeros((16, 48))
    for t in range(16):
        for k in range(24):
            kpts[t, k*2] = 0.5 + np.random.randn() * 0.2  # x spread
            kpts[t, k*2+1] = 0.85 + np.random.randn() * 0.05  # y very low
    return torch.tensor(kpts, dtype=torch.float32).unsqueeze(0)

# Standing: legs extended, body level
def create_standing_pattern():
    kpts = np.zeros((16, 48))
    for t in range(16):
        for k in range(24):
            kpts[t, k*2] = 0.5 + np.random.randn() * 0.15  # x
            kpts[t, k*2+1] = 0.5 + np.random.randn() * 0.1  # y level
    return torch.tensor(kpts, dtype=torch.float32).unsqueeze(0)

# Spin: circular motion over time
def create_spin_pattern():
    kpts = np.zeros((16, 48))
    for t in range(16):
        angle = (t / 16) * 2 * np.pi
        for k in range(24):
            kpts[t, k*2] = 0.5 + 0.3 * np.cos(angle + k*0.1)  # x rotates
            kpts[t, k*2+1] = 0.5 + 0.3 * np.sin(angle + k*0.1)  # y rotates
    return torch.tensor(kpts, dtype=torch.float32).unsqueeze(0)

print("\nTesting synthetic poses:")
test_model(create_sitting_pattern(), "Synthetic sitting")
test_model(create_lying_pattern(), "Synthetic lying")
test_model(create_standing_pattern(), "Synthetic standing")
test_model(create_spin_pattern(), "Synthetic spin")

# Try VERY extreme sitting pattern
print("\n=== Extreme Sitting Pattern ===")
extreme_sit = np.zeros((16, 48))
for t in range(16):
    # Front at y=0.1, back at y=0.9
    for k in range(12):
        extreme_sit[t, k*2] = 0.5
        extreme_sit[t, k*2+1] = 0.1
    for k in range(12, 24):
        extreme_sit[t, k*2] = 0.5
        extreme_sit[t, k*2+1] = 0.9
extreme_input = torch.tensor(extreme_sit, dtype=torch.float32).unsqueeze(0)
test_model(extreme_input, "Extreme sit (y: 0.1 front, 0.9 back)")
