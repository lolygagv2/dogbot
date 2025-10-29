#!/usr/bin/env python3
"""
Complete test of GUI with camera and servo controls
"""

import time
import subprocess
import sys

def test_camera_servo_integration():
    """Test complete GUI with camera and servos"""

    print("=" * 60)
    print("WIM-Z GUI COMPLETE TEST")
    print("=" * 60)

    # Kill any existing Python processes
    print("\n1. Cleaning up existing processes...")
    subprocess.run(["killall", "python3"], capture_output=True)
    time.sleep(2)

    # Test servo controller
    print("\n2. Testing servo controller...")
    try:
        from servo_control_module import ServoController
        controller = ServoController()
        if controller.initialize():
            print("   ✅ Servo controller initialized")

            # Test movements
            print("   Testing pan...")
            controller.set_pan_angle(-45, smooth=False)
            time.sleep(0.5)
            controller.set_pan_angle(45, smooth=False)
            time.sleep(0.5)
            controller.set_pan_angle(0, smooth=False)

            print("   Testing tilt...")
            controller.set_tilt_angle(-30, smooth=False)
            time.sleep(0.5)
            controller.set_tilt_angle(30, smooth=False)
            time.sleep(0.5)
            controller.set_tilt_angle(0, smooth=False)

            print("   ✅ Servo tests passed")
            controller.cleanup()
        else:
            print("   ❌ Servo initialization failed")
    except Exception as e:
        print(f"   ❌ Servo test error: {e}")

    # Test camera
    print("\n3. Testing camera...")
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_preview_configuration(
            main={"size": (1920, 1080), "format": "BGR888"}
        )
        cam.configure(config)
        cam.start()
        print("   ✅ Camera initialized")

        # Capture a test frame
        frame = cam.capture_array()
        print(f"   ✅ Captured frame: {frame.shape}")

        cam.stop()
        cam.close()
        print("   ✅ Camera test passed")
    except Exception as e:
        print(f"   ❌ Camera test error: {e}")

    # Test AI controller
    print("\n4. Testing AI controller...")
    try:
        from core.ai_controller_3stage_fixed import AI3StageControllerFixed
        ai = AI3StageControllerFixed()
        print("   ✅ AI controller loaded")
    except Exception as e:
        print(f"   ❌ AI controller error: {e}")

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("\nTo run the complete GUI:")
    print("  - Local (with display): python3 live_gui_with_aruco.py")
    print("  - Remote (headless): python3 live_gui_with_aruco.py")
    print("\nKeyboard controls when GUI is running:")
    print("  W/S - Tilt camera up/down")
    print("  A/D - Pan camera left/right")
    print("  H   - Home position (center)")
    print("  Q   - Quit")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_camera_servo_integration()