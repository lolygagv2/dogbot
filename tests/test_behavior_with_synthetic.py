#!/usr/bin/env python3
"""
Test behavior model with synthetic keypoints that simulate sitting pose
This verifies the behavior model and normalization logic without needing live dogs
"""

import sys
sys.path.insert(0, '/home/morgan/dogbot')

import numpy as np
import torch

# Load behavior model
behavior_model_path = "/home/morgan/dogbot/ai/models/behavior_14.ts"
model = torch.jit.load(behavior_model_path, map_location="cpu").eval()
behaviors = ["stand", "sit", "lie", "cross", "spin"]

def test_input(tensor_input, label):
    """Test behavior model with given input and return probabilities"""
    with torch.no_grad():
        output = model(tensor_input)
        probs = torch.softmax(output, dim=-1).numpy()[0]

    winner = behaviors[np.argmax(probs)]
    prob_str = " | ".join([f"{b}:{p:.2f}" for b, p in zip(behaviors, probs)])
    print(f"{label}: {winner} | {prob_str}")
    return probs

def simulate_sitting_dog():
    """
    Create synthetic keypoints for a sitting dog.

    Sitting dog characteristics:
    - Front legs extended (keypoints 0-11 have mid-range Y values)
    - Rear body low (keypoints 12-23 have high Y values = bottom of bbox)
    - All normalized to [0,1] within bounding box, then centered to [-0.5, 0.5]
    """
    T = 16
    kpts = np.zeros((T, 24, 2))

    for t in range(T):
        # Front body (keypoints 0-11): upper portion of bbox
        for k in range(12):
            kpts[t, k, 0] = 0.5 + np.random.randn() * 0.1  # x centered
            kpts[t, k, 1] = 0.3 + np.random.randn() * 0.05  # y in upper-mid

        # Rear body (keypoints 12-23): lower portion of bbox
        for k in range(12, 24):
            kpts[t, k, 0] = 0.5 + np.random.randn() * 0.1  # x centered
            kpts[t, k, 1] = 0.8 + np.random.randn() * 0.05  # y near bottom

    # Center around 0 (as the fixed code does)
    kpts_centered = kpts - 0.5

    # Reshape to (1, T, 48)
    tensor = torch.tensor(kpts_centered.reshape(1, T, 48), dtype=torch.float32)
    return tensor

def simulate_standing_dog():
    """Standing dog: body level, legs extended"""
    T = 16
    kpts = np.zeros((T, 24, 2))

    for t in range(T):
        for k in range(24):
            kpts[t, k, 0] = 0.5 + np.random.randn() * 0.15  # x spread
            kpts[t, k, 1] = 0.5 + np.random.randn() * 0.1   # y centered/level

    kpts_centered = kpts - 0.5
    tensor = torch.tensor(kpts_centered.reshape(1, T, 48), dtype=torch.float32)
    return tensor

def simulate_lying_dog():
    """Lying dog: all keypoints at bottom of bbox"""
    T = 16
    kpts = np.zeros((T, 24, 2))

    for t in range(T):
        for k in range(24):
            kpts[t, k, 0] = 0.5 + np.random.randn() * 0.2  # x wide spread
            kpts[t, k, 1] = 0.85 + np.random.randn() * 0.05  # y at bottom

    kpts_centered = kpts - 0.5
    tensor = torch.tensor(kpts_centered.reshape(1, T, 48), dtype=torch.float32)
    return tensor

def simulate_cross_legs():
    """Cross: front legs crossed (overlapping x positions)"""
    T = 16
    kpts = np.zeros((T, 24, 2))

    for t in range(T):
        # Front legs (assume indices 4-7 left, 8-11 right) with overlapping X
        for k in range(12):
            kpts[t, k, 0] = 0.5 + np.random.randn() * 0.05  # x very clustered (crossed)
            kpts[t, k, 1] = 0.4 + np.random.randn() * 0.05
        for k in range(12, 24):
            kpts[t, k, 0] = 0.5 + np.random.randn() * 0.1
            kpts[t, k, 1] = 0.6 + np.random.randn() * 0.05

    kpts_centered = kpts - 0.5
    tensor = torch.tensor(kpts_centered.reshape(1, T, 48), dtype=torch.float32)
    return tensor

print("=" * 70)
print("BEHAVIOR MODEL TEST WITH SYNTHETIC DATA")
print("=" * 70)
print("Testing with keypoints that simulate different poses")
print("Keypoints are normalized [0,1] within bbox, then centered to [-0.5, 0.5]")
print("-" * 70)
print()

print("=== Expected: sit should have highest probability ===")
sit_tensor = simulate_sitting_dog()
print(f"Input stats: min={sit_tensor.min():.3f}, max={sit_tensor.max():.3f}, mean={sit_tensor.mean():.3f}")
test_input(sit_tensor, "Sitting dog (front high, rear low)")

print()
print("=== Expected: stand should have highest probability ===")
stand_tensor = simulate_standing_dog()
print(f"Input stats: min={stand_tensor.min():.3f}, max={stand_tensor.max():.3f}, mean={stand_tensor.mean():.3f}")
test_input(stand_tensor, "Standing dog (body level)")

print()
print("=== Expected: lie should have highest probability ===")
lie_tensor = simulate_lying_dog()
print(f"Input stats: min={lie_tensor.min():.3f}, max={lie_tensor.max():.3f}, mean={lie_tensor.mean():.3f}")
test_input(lie_tensor, "Lying dog (all at bottom)")

print()
print("=== Testing: cross should have highest probability ===")
cross_tensor = simulate_cross_legs()
print(f"Input stats: min={cross_tensor.min():.3f}, max={cross_tensor.max():.3f}, mean={cross_tensor.mean():.3f}")
test_input(cross_tensor, "Cross (front legs clustered)")

print()
print("=" * 70)
print("Test what happens with OLD broken normalization (no centering)")
print("=" * 70)

# Old broken: [0,1] range instead of [-0.5, 0.5]
old_sit = simulate_sitting_dog() + 0.5  # Undo the centering
print(f"OLD Input stats: min={old_sit.min():.3f}, max={old_sit.max():.3f}, mean={old_sit.mean():.3f}")
test_input(old_sit, "OLD (no centering) sitting")

old_stand = simulate_standing_dog() + 0.5
print(f"OLD Input stats: min={old_stand.min():.3f}, max={old_stand.max():.3f}, mean={old_stand.mean():.3f}")
test_input(old_stand, "OLD (no centering) standing")

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("If the model correctly detects different poses with synthetic data,")
print("the issue is likely in keypoint decoding from Hailo, not the behavior model.")
print()
