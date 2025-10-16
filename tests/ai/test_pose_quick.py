#!/usr/bin/env python3
"""
Quick test of pose detection system
Minimal output for HDMI viewing
"""

import sys
sys.path.insert(0, '/home/morgan/dogbot')

from run_pi_1024x768 import PoseDetectionApp
import time

def main():
    print("=" * 50)
    print("POSE DETECTION QUICK TEST")
    print("=" * 50)
    print("Initializing system...")

    app = PoseDetectionApp()

    # Initialize components
    if not app.initialize():
        print("ERROR: Failed to initialize")
        return 1

    print("\nSystem ready!")
    print("\nStarting camera feed...")
    print("Press Ctrl+C to stop")
    print("-" * 50)

    try:
        # Run the camera
        app.run_camera()
    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    return 0

if __name__ == "__main__":
    exit(main())