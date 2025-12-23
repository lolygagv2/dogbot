#!/usr/bin/env python3
"""
Test motor direction consistency
Goal: Forward motion should produce POSITIVE encoder counts
If negative, direction logic is inverted
"""

import sys
import time
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

def test_direction_consistency():
    """Test if forward motion gives positive encoder counts"""

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    print("🧭 DIRECTION LOGIC TEST")
    print("======================")
    print("Testing: Forward motion should = POSITIVE encoder counts")
    print()

    try:
        controller = MotorControllerPolling()
        time.sleep(1)

        # Test LEFT motor directions
        print("📋 LEFT MOTOR DIRECTION TEST")

        # Forward test
        controller.reset_encoder_counts()
        print("LEFT motor FORWARD for 2 seconds...")

        start_time = time.time()
        while time.time() - start_time < 2.0:
            controller.set_motor_speed('left', 50, 'forward')
            time.sleep(0.1)

        controller.set_motor_speed('left', 0, 'stop')
        left_forward, _ = controller.get_encoder_counts()
        print(f"LEFT FORWARD result: {left_forward} counts")

        time.sleep(0.5)

        # Backward test
        controller.reset_encoder_counts()
        print("LEFT motor BACKWARD for 2 seconds...")

        start_time = time.time()
        while time.time() - start_time < 2.0:
            controller.set_motor_speed('left', 50, 'backward')
            time.sleep(0.1)

        controller.set_motor_speed('left', 0, 'stop')
        left_backward, _ = controller.get_encoder_counts()
        print(f"LEFT BACKWARD result: {left_backward} counts")

        # Test RIGHT motor directions
        print("\n📋 RIGHT MOTOR DIRECTION TEST")

        # Forward test
        controller.reset_encoder_counts()
        print("RIGHT motor FORWARD for 2 seconds...")

        start_time = time.time()
        while time.time() - start_time < 2.0:
            controller.set_motor_speed('right', 50, 'forward')
            time.sleep(0.1)

        controller.set_motor_speed('right', 0, 'stop')
        _, right_forward = controller.get_encoder_counts()
        print(f"RIGHT FORWARD result: {right_forward} counts")

        time.sleep(0.5)

        # Backward test
        controller.reset_encoder_counts()
        print("RIGHT motor BACKWARD for 2 seconds...")

        start_time = time.time()
        while time.time() - start_time < 2.0:
            controller.set_motor_speed('right', 50, 'backward')
            time.sleep(0.1)

        controller.set_motor_speed('right', 0, 'stop')
        _, right_backward = controller.get_encoder_counts()
        print(f"RIGHT BACKWARD result: {right_backward} counts")

        # Analyze results
        print(f"\n🎯 DIRECTION ANALYSIS:")
        print(f"Left:  Forward={left_forward:+d}, Backward={left_backward:+d}")
        print(f"Right: Forward={right_forward:+d}, Backward={right_backward:+d}")

        # Check if directions are consistent
        left_direction_correct = left_forward > 0 and left_backward < 0
        right_direction_correct = right_forward > 0 and right_backward < 0

        if left_direction_correct and right_direction_correct:
            print("✅ PERFECT: Forward=positive, Backward=negative for both motors")
        elif left_forward < 0 and left_backward > 0 and right_forward < 0 and right_backward > 0:
            print("❌ INVERTED: Forward=negative, Backward=positive for both motors")
            print("Direction logic needs to be flipped in motor controller or Xbox controller")
        else:
            print("⚠️ MIXED: Inconsistent direction behavior")
            if not left_direction_correct:
                print(f"  Left motor inverted: Forward={left_forward}, Backward={left_backward}")
            if not right_direction_correct:
                print(f"  Right motor inverted: Forward={right_forward}, Backward={right_backward}")

        print(f"\n💡 RECOMMENDATION:")
        if not (left_direction_correct and right_direction_correct):
            print("For Xbox joystick: Forward stick should produce POSITIVE encoder counts")
            print("If currently inverted, fix in Xbox controller joystick→motor mapping")
        else:
            print("Direction logic is correct - ready for Xbox testing")

        controller.cleanup()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direction_consistency()