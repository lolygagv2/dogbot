#!/usr/bin/env python3
"""
Systematically find what keypoint patterns trigger "sit" in the behavior model.
Test various input ranges and patterns.
"""

import numpy as np
import torch

# Load behavior model
model = torch.jit.load("/home/morgan/dogbot/ai/models/behavior_14.ts", map_location="cpu").eval()
behaviors = ["stand", "sit", "lie", "cross", "spin"]
T = 16

def test_pattern(tensor_input, label):
    """Test and print results"""
    with torch.no_grad():
        output = model(tensor_input)
        probs = torch.softmax(output, dim=-1).numpy()[0]

    winner = behaviors[np.argmax(probs)]
    sit_prob = probs[1]  # sit is index 1
    return winner, sit_prob, probs

print("=" * 70)
print("FINDING WHAT TRIGGERS 'SIT' IN BEHAVIOR MODEL")
print("=" * 70)
print()

# Test 1: Different value ranges
print("=== TEST 1: Different input value ranges ===")
ranges_to_test = [
    (0.0, 1.0, "[0, 1] - training range"),
    (-0.5, 0.5, "[-0.5, 0.5] - mean centered"),
    (-1.0, 1.0, "[-1, 1] - doubled"),
    (0.0, 0.5, "[0, 0.5] - half scale"),
    (0.25, 0.75, "[0.25, 0.75] - centered subset"),
]

for lo, hi, desc in ranges_to_test:
    tensor = torch.rand(1, T, 48) * (hi - lo) + lo
    winner, sit_prob, probs = test_pattern(tensor, desc)
    print(f"  {desc}: winner={winner}, sit={sit_prob:.3f}")

print()
print("=== TEST 2: Y values (vertical position) variations ===")
# Keypoints: 48 features = 24 points * 2 coords
# Odd indices (1, 3, 5...) are Y coordinates

y_positions = [
    (0.1, "Y=0.1 - all keypoints at TOP of bbox"),
    (0.3, "Y=0.3 - upper third"),
    (0.5, "Y=0.5 - middle"),
    (0.7, "Y=0.7 - lower third"),
    (0.9, "Y=0.9 - all keypoints at BOTTOM of bbox"),
]

for y_val, desc in y_positions:
    data = np.zeros((1, T, 48), dtype=np.float32)
    data[:, :, 0::2] = 0.5  # X = 0.5 (centered)
    data[:, :, 1::2] = y_val  # Y = test value
    tensor = torch.tensor(data)
    winner, sit_prob, probs = test_pattern(tensor, desc)
    print(f"  {desc}: winner={winner}, sit={sit_prob:.3f}")

print()
print("=== TEST 3: Split Y - front high, rear low (classic sit) ===")
# Assume keypoints 0-11 are front body, 12-23 are rear

split_configs = [
    (0.2, 0.8, "Front Y=0.2 (high), Rear Y=0.8 (low)"),
    (0.3, 0.7, "Front Y=0.3, Rear Y=0.7"),
    (0.4, 0.6, "Front Y=0.4, Rear Y=0.6"),
    (0.8, 0.2, "Front Y=0.8 (low), Rear Y=0.2 (high) - INVERTED"),
]

for front_y, rear_y, desc in split_configs:
    data = np.zeros((1, T, 48), dtype=np.float32)
    data[:, :, 0::2] = 0.5  # X centered
    # Front keypoints (0-11): Y coords at indices 1,3,5,...23
    for i in range(12):
        data[:, :, i*2+1] = front_y
    # Rear keypoints (12-23): Y coords at indices 25,27,...47
    for i in range(12, 24):
        data[:, :, i*2+1] = rear_y

    tensor = torch.tensor(data)
    winner, sit_prob, probs = test_pattern(tensor, desc)
    print(f"  {desc}: winner={winner}, sit={sit_prob:.3f}")

print()
print("=== TEST 4: Add noise to patterns ===")
for front_y, rear_y, desc in split_configs[:2]:
    data = np.zeros((1, T, 48), dtype=np.float32)
    data[:, :, 0::2] = 0.5 + np.random.randn(1, T, 24) * 0.1  # X with noise
    for i in range(12):
        data[:, :, i*2+1] = front_y + np.random.randn(1, T) * 0.05
    for i in range(12, 24):
        data[:, :, i*2+1] = rear_y + np.random.randn(1, T) * 0.05

    data = np.clip(data, 0, 1)
    tensor = torch.tensor(data, dtype=torch.float32)
    winner, sit_prob, probs = test_pattern(tensor, desc + " +noise")
    print(f"  {desc} +noise: winner={winner}, sit={sit_prob:.3f}")

print()
print("=== TEST 5: Time-varying patterns (motion) ===")
# Movement patterns

# Sit motion: front stays, rear drops
data = np.zeros((1, T, 48), dtype=np.float32)
data[:, :, 0::2] = 0.5  # X centered
# Front Y: constant at 0.3
for i in range(12):
    data[:, :, i*2+1] = 0.3
# Rear Y: drops from 0.3 to 0.8 over time
for i in range(12, 24):
    for t in range(T):
        data[:, t, i*2+1] = 0.3 + (0.5 * t / T)

tensor = torch.tensor(data)
winner, sit_prob, probs = test_pattern(tensor, "Sit motion (rear dropping)")
print(f"  Sit motion (rear drops): winner={winner}, sit={sit_prob:.3f}")

# Stand motion: rear rises
data = np.zeros((1, T, 48), dtype=np.float32)
data[:, :, 0::2] = 0.5
for i in range(12):
    data[:, :, i*2+1] = 0.3
for i in range(12, 24):
    for t in range(T):
        data[:, t, i*2+1] = 0.8 - (0.5 * t / T)

tensor = torch.tensor(data)
winner, sit_prob, probs = test_pattern(tensor, "Stand motion (rear rising)")
print(f"  Stand motion (rear rises): winner={winner}, sit={sit_prob:.3f}")

print()
print("=== TEST 6: Mean-centered versions of above ===")
# Repeat TEST 3 split configs but mean-centered

for front_y, rear_y, desc in split_configs[:2]:
    data = np.zeros((1, T, 48), dtype=np.float32)
    data[:, :, 0::2] = 0.5
    for i in range(12):
        data[:, :, i*2+1] = front_y
    for i in range(12, 24):
        data[:, :, i*2+1] = rear_y

    # Mean center
    data = data - 0.5

    tensor = torch.tensor(data, dtype=torch.float32)
    winner, sit_prob, probs = test_pattern(tensor, desc + " (CENTERED)")
    all_probs = " | ".join([f"{b}:{p:.2f}" for b, p in zip(behaviors, probs)])
    print(f"  {desc} CENTERED: winner={winner}, sit={sit_prob:.3f}")
    print(f"    All: {all_probs}")

print()
print("=" * 70)
print("DONE - Look for patterns that give high sit probability")
print("=" * 70)
