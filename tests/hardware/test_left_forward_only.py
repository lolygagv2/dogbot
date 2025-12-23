#!/usr/bin/env python3
"""
Test ONLY left motor forward - verify direction is correct
"""

import sys
import time
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

def test_left_forward_only():
    """Test ONLY left motor forward direction"""

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    print("🔍 LEFT MOTOR FORWARD ONLY TEST")
    print("===============================")
    print("Testing ONLY left motor forward - watch physical direction")
    print()

    try:
        controller = MotorControllerPolling()
        time.sleep(1)

        print("▶️ LEFT MOTOR FORWARD (3 seconds)")
        print("WATCH: Does LEFT motor spin in the direction you consider 'forward'?")
        print("(Forward should be the direction that moves the robot forward)")
        print()

        controller.reset_encoder_counts()

        # Run for 3 seconds
        start_time = time.time()
        while time.time() - start_time < 3.0:
            controller.set_motor_speed('left', 60, 'forward')
            time.sleep(0.1)

        controller.set_motor_speed('left', 0, 'stop')
        left_forward, _ = controller.get_encoder_counts()

        print(f"LEFT MOTOR 'forward' command result: {left_forward} encoder counts")
        print()
        print("PHYSICAL VERIFICATION:")
        print("- Did the LEFT motor spin in the correct 'forward' direction?")
        print("- Forward = direction that would move robot forward if both motors did this")
        print()

        if left_forward > 0:
            print("✅ Encoder shows POSITIVE counts for 'forward' command")
        else:
            print("❌ Encoder shows NEGATIVE counts for 'forward' command")
            print("This means direction logic is INVERTED!")

        controller.cleanup()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_left_forward_only()