#!/usr/bin/env python3
"""
Test to verify bbox-relative keypoint normalization for behavior model
"""

import sys
sys.path.insert(0, '/home/morgan/dogbot')

import numpy as np
import torch

# Load behavior model
behavior_model_path = "/home/morgan/dogbot/ai/models/behavior_14.ts"
behavior_model = torch.jit.load(behavior_model_path, map_location="cpu").eval()

behaviors = ["stand", "sit", "lie", "cross", "spin"]

print("=== Testing BBOX-Relative Keypoint Normalization ===\n")

def normalize_bbox_relative(keypoints, bbox):
    """Normalize keypoints relative to bounding box (how model was trained)"""
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)

    normalized = keypoints.copy()
    normalized[:, 0] = np.clip((keypoints[:, 0] - x1) / w, 0, 1)
    normalized[:, 1] = np.clip((keypoints[:, 1] - y1) / h, 0, 1)
    return normalized

def test_model(input_tensor, label):
    """Test model and print results"""
    with torch.no_grad():
        output = behavior_model(input_tensor)
        probs = torch.softmax(output, dim=-1).numpy()[0]

    print(f"{label}:")
    for b, p in zip(behaviors, probs):
        marker = " <--" if p == max(probs) else ""
        print(f"  {b}: {p*100:.2f}%{marker}")
    print()

# Test 1: Frame-relative normalization (WRONG - what was happening before)
print("Test 1: Frame-relative normalization (WRONG)")
# Simulate 4K frame keypoints (0-3840, 0-2160)
fake_kpts = np.random.rand(16, 24, 2) * np.array([3840, 2160])
# Wrong: normalize by frame size
wrong_normalized = fake_kpts / np.array([3840, 2160])
wrong_input = torch.tensor(wrong_normalized.reshape(1, 16, 48), dtype=torch.float32)
test_model(wrong_input, "Frame-relative (wrong)")

# Test 2: Bbox-relative normalization (CORRECT)
print("Test 2: Bbox-relative normalization (CORRECT)")
# Create dog in a bounding box at position (1000,500) with size 800x600
bbox = (1000, 500, 1800, 1100)  # x1, y1, x2, y2
# Keypoints scattered within the bbox
kpts_in_bbox = np.zeros((16, 24, 2))
for t in range(16):
    for k in range(24):
        kpts_in_bbox[t, k, 0] = 1000 + np.random.rand() * 800  # x within bbox
        kpts_in_bbox[t, k, 1] = 500 + np.random.rand() * 600   # y within bbox

# Correct: normalize relative to bbox
correct_normalized = np.zeros_like(kpts_in_bbox)
for t in range(16):
    correct_normalized[t] = normalize_bbox_relative(kpts_in_bbox[t], bbox)

correct_input = torch.tensor(correct_normalized.reshape(1, 16, 48), dtype=torch.float32)
print(f"Normalized range: {correct_normalized.min():.3f} - {correct_normalized.max():.3f}")
test_model(correct_input, "Bbox-relative (correct)")

# Test 3: Simulated SITTING pose (hips down, front paws up)
print("Test 3: Simulated sitting pose with bbox normalization")
sitting_kpts = np.zeros((16, 24, 2))
for t in range(16):
    # Front body - higher in frame (smaller y when normalized)
    for k in range(12):
        sitting_kpts[t, k, 0] = 0.3 + np.random.rand() * 0.4  # x: 0.3-0.7
        sitting_kpts[t, k, 1] = 0.2 + np.random.rand() * 0.2  # y: 0.2-0.4 (upper)
    # Back body - lower (larger y = sitting position)
    for k in range(12, 24):
        sitting_kpts[t, k, 0] = 0.3 + np.random.rand() * 0.4  # x: 0.3-0.7
        sitting_kpts[t, k, 1] = 0.6 + np.random.rand() * 0.25  # y: 0.6-0.85 (lower)

sit_input = torch.tensor(sitting_kpts.reshape(1, 16, 48), dtype=torch.float32)
test_model(sit_input, "Sitting pose (bbox-normalized)")

# Test 4: Simulated LYING pose (all keypoints low in frame)
print("Test 4: Simulated lying pose with bbox normalization")
lying_kpts = np.zeros((16, 24, 2))
for t in range(16):
    # All body points low and spread wide
    for k in range(24):
        lying_kpts[t, k, 0] = 0.1 + np.random.rand() * 0.8  # x: 0.1-0.9 (wide spread)
        lying_kpts[t, k, 1] = 0.7 + np.random.rand() * 0.2  # y: 0.7-0.9 (low in bbox)

lie_input = torch.tensor(lying_kpts.reshape(1, 16, 48), dtype=torch.float32)
test_model(lie_input, "Lying pose (bbox-normalized)")

# Test 5: Simulated STANDING pose (all keypoints mid-level)
print("Test 5: Simulated standing pose with bbox normalization")
standing_kpts = np.zeros((16, 24, 2))
for t in range(16):
    for k in range(24):
        standing_kpts[t, k, 0] = 0.2 + np.random.rand() * 0.6  # x: 0.2-0.8
        standing_kpts[t, k, 1] = 0.4 + np.random.rand() * 0.3  # y: 0.4-0.7 (mid)

stand_input = torch.tensor(standing_kpts.reshape(1, 16, 48), dtype=torch.float32)
test_model(stand_input, "Standing pose (bbox-normalized)")

print("=== Summary ===")
print("The behavior model was trained with bbox-relative normalized keypoints")
print("Keypoints should be normalized to 0-1 relative to the dog's bounding box,")
print("NOT relative to the entire frame!")
