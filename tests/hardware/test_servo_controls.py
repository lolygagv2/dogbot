#!/usr/bin/env python3
"""
Test servo controls for pan/tilt
"""

import time
from servo_control_module import ServoController

def test_servo_movement():
    """Test servo pan and tilt controls"""
    print("Initializing servo controller...")
    controller = ServoController()

    if not controller.initialize():
        print("Failed to initialize servo controller!")
        return

    print("✅ Servo controller initialized")
    time.sleep(1)

    # Test pan servo
    print("\n=== Testing PAN Servo ===")
    print("Moving to left (-90°)...")
    controller.set_pan_angle(-90, smooth=False)
    time.sleep(1)

    print("Moving to center (0°)...")
    controller.set_pan_angle(0, smooth=False)
    time.sleep(1)

    print("Moving to right (90°)...")
    controller.set_pan_angle(90, smooth=False)
    time.sleep(1)

    print("Returning to center...")
    controller.set_pan_angle(0, smooth=False)
    time.sleep(1)

    # Test tilt servo
    print("\n=== Testing TILT Servo ===")
    print("Moving up (45°)...")
    controller.set_tilt_angle(45, smooth=False)
    time.sleep(1)

    print("Moving to center (0°)...")
    controller.set_tilt_angle(0, smooth=False)
    time.sleep(1)

    print("Moving down (-45°)...")
    controller.set_tilt_angle(-45, smooth=False)
    time.sleep(1)

    print("Returning to center...")
    controller.set_tilt_angle(0, smooth=False)
    time.sleep(1)

    # Test combined movements
    print("\n=== Testing Combined Movement ===")
    print("Look up-left...")
    controller.set_pan_angle(-45, smooth=False)
    controller.set_tilt_angle(30, smooth=False)
    time.sleep(1)

    print("Look down-right...")
    controller.set_pan_angle(45, smooth=False)
    controller.set_tilt_angle(-30, smooth=False)
    time.sleep(1)

    # Return to center
    print("\n=== Centering all servos ===")
    controller.center_all()
    time.sleep(1)

    print("\n✅ Test complete!")
    controller.cleanup()

if __name__ == "__main__":
    test_servo_movement()