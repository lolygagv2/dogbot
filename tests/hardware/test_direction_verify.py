#!/usr/bin/env python3
"""
Simple direction test - one motor at a time for physical verification
"""

import sys
import time
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

def test_direction_verify():
    """Test motor directions one at a time for physical verification"""

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    print("🔍 MOTOR DIRECTION VERIFICATION")
    print("===============================")
    print("Testing one motor at a time - WATCH THE PHYSICAL MOVEMENT")
    print()

    try:
        controller = MotorControllerPolling()
        time.sleep(1)

        # LEFT MOTOR FORWARD
        print("📋 LEFT MOTOR FORWARD TEST")
        print("Watch the LEFT motor - should spin in what you consider 'forward' direction")
        input("Press Enter to start LEFT motor FORWARD...")

        controller.reset_encoder_counts()
        start_time = time.time()
        while time.time() - start_time < 2.0:
            controller.set_motor_speed('left', 50, 'forward')
            time.sleep(0.1)

        controller.set_motor_speed('left', 0, 'stop')
        left_forward, _ = controller.get_encoder_counts()
        print(f"LEFT FORWARD: {left_forward} encoder counts")
        print("Was that the correct 'forward' direction for LEFT motor? (watch physical movement)")
        print()

        time.sleep(1)

        # LEFT MOTOR BACKWARD
        print("📋 LEFT MOTOR BACKWARD TEST")
        print("Watch the LEFT motor - should spin OPPOSITE to the previous direction")
        input("Press Enter to start LEFT motor BACKWARD...")

        controller.reset_encoder_counts()
        start_time = time.time()
        while time.time() - start_time < 2.0:
            controller.set_motor_speed('left', 50, 'backward')
            time.sleep(0.1)

        controller.set_motor_speed('left', 0, 'stop')
        left_backward, _ = controller.get_encoder_counts()
        print(f"LEFT BACKWARD: {left_backward} encoder counts")
        print("Did LEFT motor spin in OPPOSITE direction? (should be opposite to forward)")
        print()

        time.sleep(1)

        # RIGHT MOTOR FORWARD
        print("📋 RIGHT MOTOR FORWARD TEST")
        print("Watch the RIGHT motor - should spin in what you consider 'forward' direction")
        input("Press Enter to start RIGHT motor FORWARD...")

        controller.reset_encoder_counts()
        start_time = time.time()
        while time.time() - start_time < 2.0:
            controller.set_motor_speed('right', 50, 'forward')
            time.sleep(0.1)

        controller.set_motor_speed('right', 0, 'stop')
        _, right_forward = controller.get_encoder_counts()
        print(f"RIGHT FORWARD: {right_forward} encoder counts")
        print("Was that the correct 'forward' direction for RIGHT motor?")
        print()

        time.sleep(1)

        # RIGHT MOTOR BACKWARD
        print("📋 RIGHT MOTOR BACKWARD TEST")
        print("Watch the RIGHT motor - should spin OPPOSITE to the previous direction")
        input("Press Enter to start RIGHT motor BACKWARD...")

        controller.reset_encoder_counts()
        start_time = time.time()
        while time.time() - start_time < 2.0:
            controller.set_motor_speed('right', 50, 'backward')
            time.sleep(0.1)

        controller.set_motor_speed('right', 0, 'stop')
        _, right_backward = controller.get_encoder_counts()
        print(f"RIGHT BACKWARD: {right_backward} encoder counts")
        print("Did RIGHT motor spin in OPPOSITE direction?")
        print()

        # Summary
        print("🎯 ENCODER RESULTS:")
        print(f"Left:  Forward={left_forward:+d}, Backward={left_backward:+d}")
        print(f"Right: Forward={right_forward:+d}, Backward={right_backward:+d}")
        print()
        print("PHYSICAL VERIFICATION NEEDED:")
        print("- Did 'forward' commands make motors spin in the correct physical direction?")
        print("- Did 'backward' commands make motors spin opposite to 'forward'?")
        print("- For Xbox: 'forward' joystick should = 'forward' motor = POSITIVE encoder counts")

        controller.cleanup()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direction_verify()