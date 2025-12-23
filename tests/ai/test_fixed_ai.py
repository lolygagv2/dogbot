#!/usr/bin/env python3
"""
Test the FIXED AI3StageControllerFixed system
This should work properly without the 21,504 detection bug
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the CORRECT AI system
from core.ai_controller_3stage_fixed import AI3StageControllerFixed

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

def main():
    """Test the fixed AI system"""
    print("🔧 Testing FIXED AI3StageControllerFixed System")
    print("=" * 60)

    # Initialize AI
    ai = AI3StageControllerFixed()

    if not ai.initialize():
        print("❌ AI initialization failed")
        return

    print("✅ AI system initialized successfully")

    # Initialize camera
    if not PICAMERA2_AVAILABLE:
        print("❌ Picamera2 not available")
        return

    camera = Picamera2()
    config = camera.create_still_configuration(main={"size": (1920, 1080)})
    camera.configure(config)
    camera.start()
    time.sleep(1)
    print("✅ Camera ready")

    # Test detection
    print("\n🎯 Testing Detection (10 frames)")
    print("-" * 40)

    for i in range(10):
        # Capture frame
        frame = camera.capture_array()

        # Process with AI
        start_time = time.time()
        detections, poses, behaviors = ai.process_frame(frame)
        inference_time = (time.time() - start_time) * 1000

        print(f"Frame {i+1:2d}: {len(detections)} dogs, {len(poses)} poses, {len(behaviors)} behaviors - {inference_time:.1f}ms")

        if detections:
            for j, det in enumerate(detections):
                print(f"  Dog {j+1}: confidence={det.confidence:.3f}, box=({det.x1},{det.y1},{det.x2},{det.y2})")

        if behaviors:
            for j, beh in enumerate(behaviors):
                print(f"  Behavior {j+1}: {beh.behavior} (confidence={beh.confidence:.3f})")

        time.sleep(0.5)

    camera.stop()
    print("\n✅ Test completed successfully!")
    print("\nThis should show:")
    print("- Proper rotation (90° counter-clockwise)")
    print("- Real dog detections (not 21,504 fake ones)")
    print("- Working pose estimation")
    print("- Reasonable inference times")

if __name__ == "__main__":
    main()