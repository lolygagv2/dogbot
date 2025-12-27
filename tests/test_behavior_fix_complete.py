#!/usr/bin/env python3
"""
Complete test of behavior model with all fixes:
1. Keypoints normalized relative to bounding box (not frame)
2. Input centered around 0 (shifted from [0,1] to [-0.5,0.5])
"""

import sys
sys.path.insert(0, '/home/morgan/dogbot')

import numpy as np
import torch

behavior_model_path = "/home/morgan/dogbot/ai/models/behavior_14.ts"
model = torch.jit.load(behavior_model_path, map_location="cpu").eval()
behaviors = ["stand", "sit", "lie", "cross", "spin"]

def test_model(input_tensor, label):
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=-1).numpy()[0]

    winner = behaviors[np.argmax(probs)]
    print(f"{label}:")
    for b, p in zip(behaviors, probs):
        marker = " <-- WINNER" if b == winner else ""
        print(f"  {b}: {p*100:.2f}%{marker}")
    print()
    return probs

def simulate_real_detection(pose_type="stand"):
    """Simulate what real detection output looks like"""
    T = 16

    # Simulate a bounding box in 4K frame
    frame_w, frame_h = 3840, 2160
    # Dog bbox somewhere in frame
    bbox = (800, 400, 1600, 1200)  # x1, y1, x2, y2
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    # Generate keypoints in 4K pixel coordinates within the bbox
    keypoints_4k = np.zeros((T, 24, 2))

    for t in range(T):
        if pose_type == "stand":
            # Standing: keypoints spread evenly
            for k in range(24):
                keypoints_4k[t, k, 0] = x1 + w * (0.2 + np.random.rand() * 0.6)
                keypoints_4k[t, k, 1] = y1 + h * (0.3 + np.random.rand() * 0.4)

        elif pose_type == "sit":
            # Sitting: front high, back low
            for k in range(12):  # Front body
                keypoints_4k[t, k, 0] = x1 + w * (0.3 + np.random.rand() * 0.4)
                keypoints_4k[t, k, 1] = y1 + h * (0.2 + np.random.rand() * 0.2)  # Higher up
            for k in range(12, 24):  # Back body
                keypoints_4k[t, k, 0] = x1 + w * (0.3 + np.random.rand() * 0.4)
                keypoints_4k[t, k, 1] = y1 + h * (0.6 + np.random.rand() * 0.25)  # Lower down

        elif pose_type == "lie":
            # Lying: all keypoints at bottom of bbox
            for k in range(24):
                keypoints_4k[t, k, 0] = x1 + w * (0.1 + np.random.rand() * 0.8)  # Wide spread
                keypoints_4k[t, k, 1] = y1 + h * (0.7 + np.random.rand() * 0.2)  # Very low

    # Step 1: Normalize relative to bounding box (as model was trained)
    normalized = np.zeros((T, 24, 2))
    for t in range(T):
        normalized[t, :, 0] = np.clip((keypoints_4k[t, :, 0] - x1) / w, 0, 1)
        normalized[t, :, 1] = np.clip((keypoints_4k[t, :, 1] - y1) / h, 0, 1)

    # Step 2: Center around 0 (shift from [0,1] to [-0.5, 0.5])
    centered = normalized - 0.5

    # Step 3: Flatten to (1, T, 48)
    flat = centered.reshape(1, T, 48)

    return torch.tensor(flat, dtype=torch.float32), normalized, centered

print("=" * 60)
print("BEHAVIOR MODEL FIX VALIDATION")
print("=" * 60)
print()

print("Testing OLD method (frame-relative normalization, no centering):")
print("-" * 40)
old_input = torch.rand(1, 16, 48)  # Random 0-1 values
test_model(old_input, "OLD method - random 0-1")

print("Testing NEW method (bbox-relative + centered):")
print("-" * 40)

# Test standing
stand_input, norm, cent = simulate_real_detection("stand")
print(f"Standing - normalized range: [{norm.min():.3f}, {norm.max():.3f}]")
print(f"Standing - centered range: [{cent.min():.3f}, {cent.max():.3f}]")
test_model(stand_input, "NEW method - standing pose")

# Test sitting
sit_input, norm, cent = simulate_real_detection("sit")
print(f"Sitting - normalized range: [{norm.min():.3f}, {norm.max():.3f}]")
print(f"Sitting - centered range: [{cent.min():.3f}, {cent.max():.3f}]")
test_model(sit_input, "NEW method - sitting pose")

# Test lying
lie_input, norm, cent = simulate_real_detection("lie")
print(f"Lying - normalized range: [{norm.min():.3f}, {norm.max():.3f}]")
print(f"Lying - centered range: [{cent.min():.3f}, {cent.max():.3f}]")
test_model(lie_input, "NEW method - lying pose")

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print("Fixes applied:")
print("1. Normalize keypoints RELATIVE TO BOUNDING BOX (not frame)")
print("2. Center input around 0: shift [0,1] to [-0.5, 0.5]")
print()
print("Expected behavior:")
print("- Standing pose should give high 'stand' probability")
print("- Sitting pose should give high 'sit' probability")
print("- Lying pose should give high 'lie' probability")
