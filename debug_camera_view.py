#!/usr/bin/env python3
"""
Debug Camera View - Save images to see what the camera is actually seeing
This will help determine if the issue is no dogs or detection problems
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

def save_debug_images():
    """Save images to see what camera is seeing"""
    print("📷 Camera Debug View")
    print("=" * 40)

    if not PICAMERA2_AVAILABLE:
        print("❌ Picamera2 not available")
        return

    # Create debug directory
    debug_dir = Path("debug_images")
    debug_dir.mkdir(exist_ok=True)

    # Initialize camera
    camera = Picamera2()
    config = camera.create_still_configuration(main={"size": (1920, 1080)})
    camera.configure(config)
    camera.start()
    time.sleep(1)
    print("✅ Camera ready")

    print("\n📸 Capturing 5 debug images...")

    for i in range(5):
        # Capture frame
        frame = camera.capture_array()

        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame

        # Save full resolution image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = debug_dir / f"camera_view_{timestamp}_{i+1}.jpg"
        cv2.imwrite(str(filename), frame_bgr)

        # Create 640x640 version (what AI sees)
        frame_640 = cv2.resize(frame_bgr, (640, 640))
        filename_640 = debug_dir / f"ai_input_{timestamp}_{i+1}.jpg"
        cv2.imwrite(str(filename_640), frame_640)

        # Print stats
        frame_min = frame_640.min()
        frame_max = frame_640.max()
        frame_mean = frame_640.mean()

        print(f"  Image {i+1}: {filename.name}")
        print(f"    Stats: min={frame_min}, max={frame_max}, mean={frame_mean:.1f}")
        print(f"    AI input: {filename_640.name}")

        time.sleep(1)

    camera.stop()

    print(f"\n✅ Debug images saved to: {debug_dir}")
    print("\n🔍 Check these images to see:")
    print("  1. Is there a dog visible?")
    print("  2. Is the camera oriented correctly?")
    print("  3. Is lighting adequate?")
    print("  4. Are there any obvious issues?")
    print("\n📋 If you see dogs in the images but AI finds 0,")
    print("    then it's a detection model issue.")
    print("    If no dogs visible, move camera or bring dog into view.")

if __name__ == "__main__":
    save_debug_images()