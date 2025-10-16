#!/usr/bin/env python3
"""
Quick camera test to verify camera functionality
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from camera_viewer import CameraViewer
from core.treat_dispenser_robot import TreatDispenserRobot

def main():
    print("🎥 Testing Camera Viewer...")
    print("This will start the camera viewer GUI")
    print("Make sure you have a camera connected!")
    print()

    try:
        # Initialize robot
        robot = TreatDispenserRobot()

        # Start camera viewer
        viewer = CameraViewer(robot)
        viewer.run()

    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check that:")
        print("  - Camera is connected")
        print("  - You're running with display (not headless)")
        print("  - All dependencies are installed")

    finally:
        if 'robot' in locals():
            robot.cleanup()

if __name__ == "__main__":
    main()