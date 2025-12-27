#!/usr/bin/env python3
"""Debug: See exactly what keypoints look like after normalization"""

import sys
sys.path.insert(0, '/home/morgan/dogbot')

import numpy as np
import torch
import cv2
import time

from core.ai_controller_3stage_fixed import AI3StageControllerFixed

try:
    from picamera2 import Picamera2
except ImportError:
    print("Need picamera2")
    exit(1)

# Load behavior model for testing
behavior_model = torch.jit.load("/home/morgan/dogbot/ai/models/behavior_14.ts", map_location="cpu").eval()
behaviors = ["stand", "sit", "lie", "cross", "spin"]

print("Initializing AI controller...")
ai = AI3StageControllerFixed()
ai.initialize()

print("Initializing camera...")
camera = Picamera2()
config = camera.create_preview_configuration(main={"size": (640, 640), "format": "RGB888"})
camera.configure(config)
camera.start()
time.sleep(1)

print("Collecting 20 frames with dogs...")
print("-" * 60)

pose_count = 0
for frame_num in range(100):
    frame = camera.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    detections, poses, behaviors_out = ai.process_frame(frame)

    if poses:
        pose_count += 1
        pose = poses[0]
        det = pose.detection
        kpts = pose.keypoints

        # Calculate bbox-normalized keypoints (as we're doing in the code)
        x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)

        normalized = np.zeros((24, 2))
        normalized[:, 0] = np.clip((kpts[:, 0] - x1) / w, 0, 1)
        normalized[:, 1] = np.clip((kpts[:, 1] - y1) / h, 0, 1)

        # Center around 0
        centered = normalized - 0.5

        print(f"\nFrame {frame_num} - Dog at ({det.center[0]}, {det.center[1]})")
        print(f"  Bbox: x1={x1:.0f}, y1={y1:.0f}, x2={x2:.0f}, y2={y2:.0f}")
        print(f"  Keypoint stats BEFORE centering:")
        print(f"    X: min={normalized[:,0].min():.3f}, max={normalized[:,0].max():.3f}, mean={normalized[:,0].mean():.3f}")
        print(f"    Y: min={normalized[:,1].min():.3f}, max={normalized[:,1].max():.3f}, mean={normalized[:,1].mean():.3f}")
        print(f"  Keypoint stats AFTER centering:")
        print(f"    X: min={centered[:,0].min():.3f}, max={centered[:,0].max():.3f}, mean={centered[:,0].mean():.3f}")
        print(f"    Y: min={centered[:,1].min():.3f}, max={centered[:,1].max():.3f}, mean={centered[:,1].mean():.3f}")

        # Test behavior model with this single frame (repeated 16 times)
        test_input = np.tile(centered.flatten(), (16, 1)).reshape(1, 16, 48)
        test_tensor = torch.tensor(test_input, dtype=torch.float32)

        with torch.no_grad():
            output = behavior_model(test_tensor)
            probs = torch.softmax(output, dim=-1).numpy()[0]

        print(f"  Model output: ", end="")
        for b, p in zip(behaviors, probs):
            if p > 0.1:
                print(f"{b}={p:.2f} ", end="")
        print()

        if pose_count >= 5:
            break

    time.sleep(0.1)

camera.stop()
camera.close()
ai.cleanup()
print("\nDone!")
