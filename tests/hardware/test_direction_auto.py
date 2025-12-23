#!/usr/bin/env python3
"""
Auto direction test - no user input required
"""

import sys
import time
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

def test_direction_auto():
    """Test motor directions automatically"""

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    print("🔍 AUTOMATIC MOTOR DIRECTION TEST")
    print("================================")
    print("Watch the motors - will test each direction for 3 seconds")
    print()

    try:
        controller = MotorControllerPolling()
        time.sleep(1)

        # LEFT MOTOR FORWARD
        print("▶️ LEFT MOTOR FORWARD (3 seconds)")
        print("WATCH: Does LEFT motor spin in forward direction?")

        controller.reset_encoder_counts()
        start_time = time.time()
        while time.time() - start_time < 3.0:
            controller.set_motor_speed('left', 60, 'forward')
            time.sleep(0.1)

        controller.set_motor_speed('left', 0, 'stop')
        left_forward, _ = controller.get_encoder_counts()
        print(f"LEFT FORWARD: {left_forward} encoder counts")
        print()
        time.sleep(2)

        # LEFT MOTOR BACKWARD
        print("◀️ LEFT MOTOR BACKWARD (3 seconds)")
        print("WATCH: Does LEFT motor spin OPPOSITE to previous direction?")

        controller.reset_encoder_counts()
        start_time = time.time()
        while time.time() - start_time < 3.0:
            controller.set_motor_speed('left', 60, 'backward')
            time.sleep(0.1)

        controller.set_motor_speed('left', 0, 'stop')
        left_backward, _ = controller.get_encoder_counts()
        print(f"LEFT BACKWARD: {left_backward} encoder counts")
        print()
        time.sleep(2)

        # RIGHT MOTOR FORWARD
        print("▶️ RIGHT MOTOR FORWARD (3 seconds)")
        print("WATCH: Does RIGHT motor spin in forward direction?")

        controller.reset_encoder_counts()
        start_time = time.time()
        while time.time() - start_time < 3.0:
            controller.set_motor_speed('right', 60, 'forward')
            time.sleep(0.1)

        controller.set_motor_speed('right', 0, 'stop')
        _, right_forward = controller.get_encoder_counts()
        print(f"RIGHT FORWARD: {right_forward} encoder counts")
        print()
        time.sleep(2)

        # RIGHT MOTOR BACKWARD
        print("◀️ RIGHT MOTOR BACKWARD (3 seconds)")
        print("WATCH: Does RIGHT motor spin OPPOSITE to previous direction?")

        controller.reset_encoder_counts()
        start_time = time.time()
        while time.time() - start_time < 3.0:
            controller.set_motor_speed('right', 60, 'backward')
            time.sleep(0.1)

        controller.set_motor_speed('right', 0, 'stop')
        _, right_backward = controller.get_encoder_counts()
        print(f"RIGHT BACKWARD: {right_backward} encoder counts")
        print()

        # Summary
        print("🎯 DIRECTION TEST RESULTS:")
        print("=============================")
        print(f"Left:  Forward={left_forward:+d}, Backward={left_backward:+d}")
        print(f"Right: Forward={right_forward:+d}, Backward={right_backward:+d}")
        print()

        # Analysis
        left_correct = left_forward > 0 and left_backward < 0
        right_correct = right_forward > 0 and right_backward < 0

        if left_correct and right_correct:
            print("✅ PERFECT: Both motors have correct direction mapping")
            print("Forward = positive counts, Backward = negative counts")
        else:
            print("❌ DIRECTION ISSUES FOUND:")
            if not left_correct:
                print(f"  LEFT motor: Forward={left_forward}, Backward={left_backward}")
                if left_forward < 0:
                    print("    LEFT INVERTED: Forward should be positive")
            if not right_correct:
                print(f"  RIGHT motor: Forward={right_forward}, Backward={right_backward}")
                if right_forward < 0:
                    print("    RIGHT INVERTED: Forward should be positive")

        controller.cleanup()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_direction_auto()