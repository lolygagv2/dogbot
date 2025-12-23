#!/usr/bin/env python3
"""
Test script optimized for white Pomeranian detection
Runs with improved thresholds for better pose detection
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.ai_controller_3stage_fixed import AI3StageControllerFixed
from missions import MissionController

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("Warning: Picamera2 not available")

def main():
    print("\n🐕 White Pomeranian Detection Test")
    print("=" * 50)
    print("Optimizations applied:")
    print("- Lowered pose confidence threshold to 0.15")
    print("- Added debug output for pose confidence values")
    print("- Detection threshold remains at 0.1")
    print("=" * 50)

    # Initialize camera
    if not PICAMERA2_AVAILABLE:
        print("❌ Picamera2 not available!")
        return

    # Initialize AI with optimized settings FIRST (like working script)
    print("🔧 Initializing AI system...")
    ai = AI3StageControllerFixed()
    print("✅ AI system ready")

    # Then start camera
    camera = Picamera2()
    config = camera.create_still_configuration(main={"size": (1920, 1080)})
    camera.configure(config)
    camera.start()
    time.sleep(1)  # Let camera stabilize

    # Create test mission
    mission = MissionController("white_pomeranian_test")
    mission_id = mission.start()

    # Track detection statistics
    frame_count = 0
    dogs_detected = 0
    poses_detected = 0
    detection_counts = []

    # Test frame capture
    test_frame = camera.capture_array()
    print(f"📷 Camera test: frame shape = {test_frame.shape}, dtype = {test_frame.dtype}")

    print("\n📊 Starting detection test...")
    print("Position your white Pomeranians in view")
    print("Press Ctrl+C to stop\n")

    try:
        start_time = time.time()
        while frame_count < 30:  # 30 frames = ~10 seconds
            try:
                # Capture frame (EXACT copy from working script)
                frame = camera.capture_array()

                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Process with CORRECT AI system
                start_time_frame = time.time()
                detections, poses, behaviors = ai.process_frame(frame)
                inference_time = (time.time() - start_time_frame) * 1000

                frame_count += 1

                # Debug frame info first time
                if frame_count == 1:
                    print(f"🔍 First frame: shape={frame.shape}, dtype={frame.dtype}, min={frame.min()}, max={frame.max()}")

                # Log results (should be MUCH better than 21,504 fake detections!)
                print(f"Frame {frame_count:2d}: {len(detections)} dogs detected - {inference_time:.1f}ms")

                if detections:
                    dogs_detected += 1
                    for i, det in enumerate(detections):
                        print(f"  Dog {i+1}: confidence={det.confidence:.3f}")
                        # Update mission
                        mission.log_event("dog_detected", {
                            "confidence": det.confidence,
                            "box": [det.x1, det.y1, det.x2, det.y2]
                        })

                if behaviors:
                    for i, beh in enumerate(behaviors):
                        print(f"  Behavior {i+1}: {beh.behavior} (conf={beh.confidence:.3f})")
                        mission.set_current_pose(beh.behavior, beh.confidence)

                time.sleep(0.3)  # ~3 FPS for testing

            except Exception as e:
                print(f"❌ Detection error: {e}")
                break

    except KeyboardInterrupt:
        print("\n\n🛑 Test stopped by user")

    finally:
        mission.end(success=True)
        camera.stop()

        # Print summary
        print("\n" + "=" * 50)
        print("📊 DETECTION SUMMARY")
        print("=" * 50)
        print(f"Total frames: {frame_count}")
        print(f"Frames with dogs: {dogs_detected} ({(dogs_detected/frame_count)*100:.1f}%)")
        print(f"Poses detected: {poses_detected}")

        if detection_counts:
            max_dogs = max(detection_counts)
            avg_dogs = sum(detection_counts) / len(detection_counts)
            print(f"Max dogs in single frame: {max_dogs}")
            print(f"Average dogs per frame: {avg_dogs:.2f}")

        print("\n💡 RECOMMENDATIONS:")
        if poses_detected == 0:
            print("- No poses detected. Check if dogs are fully visible")
            print("- Try positioning dogs closer to camera")
            print("- Ensure good lighting on white fur")
        elif poses_detected < dogs_detected / 2:
            print("- Low pose detection rate")
            print("- Consider further threshold adjustments")
        else:
            print("- Detection working well!")

        print("\nCheck logs for detailed pose confidence values")

if __name__ == "__main__":
    main()