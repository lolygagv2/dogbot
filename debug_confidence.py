#!/usr/bin/env python3
"""
Debug confidence values to understand why no detections
"""

import sys
import numpy as np

sys.path.insert(0, '/home/morgan/dogbot')

from run_pi_1024x768 import PoseDetectionApp
from test_pose_headless import HeadlessPoseApp
import cv2

def debug_confidence_values():
    """Test with a lower confidence threshold and debug output"""

    app = HeadlessPoseApp()

    if not app.initialize():
        print("Failed to initialize")
        return

    # Try Picamera2
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        config = camera.create_preview_configuration(
            main={"size": (1024, 768), "format": "XBGR8888"}
        )
        camera.configure(config)
        camera.start()
        use_picamera = True
    except:
        camera = cv2.VideoCapture(0)
        use_picamera = False

    print("Capturing frame for debug...")

    # Capture one frame
    if use_picamera:
        frame = camera.capture_array()
    else:
        ret, frame = camera.read()
        if not ret:
            print("Failed to capture frame")
            return

    # Process frame
    results = app.process_frame(frame)

    print(f"\nResults: {len(results['detections'])} detections")

    # Now let's modify the decoder to show confidence values
    print("\nTesting with raw confidence analysis...")

    # Get Hailo outputs directly
    from run_pi_1024x768 import rotate_image, letterbox, CAM_ROT_DEG, IMGSZ_W, IMGSZ_H

    if CAM_ROT_DEG:
        frame = rotate_image(frame, CAM_ROT_DEG)

    preprocessed, (dx, dy, scale_inv) = letterbox(frame, IMGSZ_W, IMGSZ_H)
    outputs = app.hailo.infer(preprocessed)

    if outputs:
        print("\nAnalyzing confidence values:")

        # Find confidence outputs
        conf_outputs = []
        for name, output in outputs.items():
            if output.shape[3] == 1:  # Confidence outputs have 1 channel
                conf_outputs.append((name, output))

        for name, conf_output in conf_outputs:
            print(f"\n{name}: shape={conf_output.shape}")

            # Get raw confidence values
            raw_conf = conf_output[0, :, :, 0]

            # Try different activations
            sigmoid_conf = 1.0 / (1.0 + np.exp(-raw_conf))

            print(f"  Raw range: {raw_conf.min():.3f} to {raw_conf.max():.3f}")
            print(f"  Sigmoid range: {sigmoid_conf.min():.6f} to {sigmoid_conf.max():.6f}")
            print(f"  Sigmoid > 0.01: {(sigmoid_conf > 0.01).sum()}")
            print(f"  Sigmoid > 0.1: {(sigmoid_conf > 0.1).sum()}")
            print(f"  Sigmoid > 0.25: {(sigmoid_conf > 0.25).sum()}")

            # Show top confidence locations
            flat_sigmoid = sigmoid_conf.flatten()
            top_indices = np.argsort(flat_sigmoid)[-5:]  # Top 5

            print(f"  Top 5 confidence values:")
            for i, idx in enumerate(top_indices):
                y, x = np.unravel_index(idx, sigmoid_conf.shape)
                conf_val = flat_sigmoid[idx]
                print(f"    #{i+1}: ({x},{y}) = {conf_val:.6f}")

    # Cleanup
    if use_picamera:
        camera.stop()
    else:
        camera.release()

    app.hailo.cleanup()

if __name__ == "__main__":
    debug_confidence_values()