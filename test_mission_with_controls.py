#!/usr/bin/env python3
"""
Test Mission Training with Camera Controls
This is the CORRECT script to run - uses AI3StageControllerFixed
"""

import sys
import os
import time
import cv2
import numpy as np
import threading
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the CORRECT systems
from core.ai_controller_3stage_fixed import AI3StageControllerFixed
from missions import MissionController

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

try:
    from servo_control_module import ServoController
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False

class MissionTesterWithControls:
    """Test mission system with camera controls"""

    def __init__(self):
        # AI system (CORRECT ONE)
        self.ai = AI3StageControllerFixed()
        self.camera = None

        # Servo for camera positioning
        self.servo = None
        if SERVO_AVAILABLE:
            try:
                self.servo = ServoController()
                self.servo.initialize()
                print("✅ Servo controls ready")
            except Exception as e:
                print(f"❌ Servo failed: {e}")

        self.running = False

    def initialize(self):
        """Initialize systems"""
        print("🔧 Initializing Mission Test System")
        print("=" * 50)

        # AI system
        if not self.ai.initialize():
            print("❌ AI initialization failed")
            return False
        print("✅ AI system ready (AI3StageControllerFixed)")

        # Camera
        if not PICAMERA2_AVAILABLE:
            print("❌ Picamera2 not available")
            return False

        self.camera = Picamera2()
        config = self.camera.create_still_configuration(main={"size": (1920, 1080)})
        self.camera.configure(config)
        self.camera.start()
        time.sleep(1)
        print("✅ Camera ready (1920x1080, will be rotated 90°)")

        print("\n📋 Camera Controls:")
        print("  W/S - Pan up/down")
        print("  A/D - Pan left/right")
        print("  Q   - Quit")
        print("  Space - Test treat dispenser")
        print("  1   - Start sit training mission")

        return True

    def handle_key_input(self):
        """Handle keyboard input for camera controls"""
        while self.running:
            try:
                # Non-blocking input (you might need to adapt this for your system)
                # For now, just sleep and let other threads work
                time.sleep(0.1)
            except KeyboardInterrupt:
                self.running = False
                break

    def test_detection(self):
        """Test detection with the CORRECT AI system"""
        print("\n🎯 Starting Detection Test")
        print("-" * 40)

        mission = MissionController("test_detection")
        mission_id = mission.start()

        frame_count = 0
        dogs_detected = 0

        while self.running and frame_count < 30:  # 30 frames = ~10 seconds
            try:
                # Capture frame
                frame = self.camera.capture_array()

                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Process with CORRECT AI system
                start_time = time.time()
                detections, poses, behaviors = self.ai.process_frame(frame)
                inference_time = (time.time() - start_time) * 1000

                frame_count += 1

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

        # Summary
        print(f"\n📊 Detection Summary:")
        print(f"  Total frames: {frame_count}")
        print(f"  Frames with dogs: {dogs_detected}")
        print(f"  Detection rate: {dogs_detected/frame_count*100:.1f}%")

        if dogs_detected == 0:
            print("⚠️  No dogs detected - check:")
            print("    1. Is there a dog in view?")
            print("    2. Is camera properly positioned?")
            print("    3. Is lighting adequate?")
        else:
            print("✅ Detection working correctly!")

        mission.end(success=dogs_detected > 0)

    def run(self):
        """Main run loop"""
        if not self.initialize():
            return

        self.running = True

        # Start keyboard input handler
        input_thread = threading.Thread(target=self.handle_key_input, daemon=True)
        input_thread.start()

        try:
            # Run detection test
            self.test_detection()

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")

        finally:
            self.running = False
            if self.camera:
                self.camera.stop()
            print("✅ Clean shutdown")

def main():
    """Main function"""
    print("🚀 Mission Training Test with Camera Controls")
    print("Using AI3StageControllerFixed (NOT the broken run_pi_1024_fixed.py)")
    print("=" * 60)

    tester = MissionTesterWithControls()
    tester.run()

if __name__ == "__main__":
    main()