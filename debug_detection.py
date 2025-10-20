#!/usr/bin/env python3
"""
Debug Detection System
Test if camera and detection are working at all
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.treat_dispenser_robot import TreatDispenserRobot, RobotMode
import time

def main():
    print("🔍 Debug Detection System")
    print("="*40)

    try:
        # Initialize robot
        print("Initializing robot...")
        robot = TreatDispenserRobot()

        if not robot.initialization_successful:
            print("❌ Robot initialization failed!")
            print("Continuing anyway to test what we can...")

        # Check vision system
        if not robot.vision:
            print("❌ Vision system is None!")
            return

        print(f"✅ Vision system exists: {type(robot.vision)}")

        # Get initial status
        vision_status = robot.vision.get_status()
        print(f"📊 Vision Status: {vision_status}")

        # Set up event monitoring
        def on_detection(event_data):
            print(f"🐕 DOG DETECTED: {event_data}")

        def on_no_detection(event_data):
            print("🔍 No detections")

        if hasattr(robot, 'event_bus'):
            robot.event_bus.subscribe('dog_detected', on_detection)
            robot.event_bus.subscribe('no_detections', on_no_detection)
            print("✅ Event handlers registered")

        # Start tracking mode
        print("Setting tracking mode...")
        robot.set_mode(RobotMode.TRACKING)

        # Check if detection started
        time.sleep(2)  # Wait for startup
        vision_status = robot.vision.get_status()
        print(f"📊 After tracking mode: {vision_status}")

        # Monitor for 30 seconds
        print("\n👀 Monitoring for detections (30 seconds)...")
        print("Point camera at dogs now!")

        for i in range(30):
            if robot.vision:
                frame = robot.vision.get_latest_frame()
                if frame is not None:
                    print(f"📹 Frame {i+1}/30: {frame.shape}")
                else:
                    print(f"❌ Frame {i+1}/30: No frame")

            time.sleep(1)

        print("\n✅ Detection test complete")

    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'robot' in locals():
            print("🔧 Cleaning up...")
            robot.cleanup()

if __name__ == "__main__":
    main()