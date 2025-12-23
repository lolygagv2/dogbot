#!/usr/bin/env python3
"""
Test RIGHT motor encoder performance
Goal: See if right motor also gets 1000+ counts like left motor
"""

import sys
import time
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

def test_right_motor_encoder():
    """Test right motor encoder performance"""

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    print("➡️ RIGHT MOTOR ENCODER TEST")
    print("==========================")
    print("Testing if right motor encoder can also get 1000+ counts")
    print()

    try:
        controller = MotorControllerPolling()
        time.sleep(1)

        # Test right motor
        initial_left, initial_right = controller.get_encoder_counts()
        print(f"Initial counts: Left={initial_left}, Right={initial_right}")

        print("Running RIGHT motor continuously for 3 seconds...")

        # Start RIGHT motor and refresh command every 100ms
        start_time = time.time()
        while time.time() - start_time < 3.0:
            controller.set_motor_speed('right', 50, 'forward')
            time.sleep(0.1)  # 100ms refresh rate

        controller.set_motor_speed('right', 0, 'stop')

        # Check final accumulation
        final_left, final_right = controller.get_encoder_counts()
        print(f"Final counts: Left={final_left}, Right={final_right}")

        # Calculate actual change
        left_change = abs(final_left - initial_left)
        right_change = abs(final_right - initial_right)

        print(f"\n📊 RIGHT MOTOR ENCODER RESULTS:")
        print(f"Left motor: {left_change} counts (should be ~0)")
        print(f"Right motor: {right_change} counts in 3 seconds")

        # Compare to left motor performance (1,286 counts)
        if right_change > 1000:
            print(f"✅ EXCELLENT: Right encoder working! {right_change} counts")
            print("Both encoders are now functional")
        elif right_change > 100:
            print(f"⚠️ PARTIAL: Right encoder partially working: {right_change} counts")
            print("Better than before but still below left motor performance")
        else:
            print(f"❌ BROKEN: Right encoder still failed: {right_change} counts")
            print("Confirmed hardware issue on right encoder (GPIO5/6)")

        # Test both motors together
        print(f"\n🔄 TESTING BOTH MOTORS TOGETHER")

        # Reset for dual test
        initial_left, initial_right = controller.get_encoder_counts()
        print(f"Starting dual test from: Left={initial_left}, Right={initial_right}")

        print("Running BOTH motors forward for 3 seconds...")
        start_time = time.time()
        while time.time() - start_time < 3.0:
            controller.set_motor_speed('left', 50, 'forward')
            controller.set_motor_speed('right', 50, 'forward')
            time.sleep(0.1)

        controller.set_motor_speed('left', 0, 'stop')
        controller.set_motor_speed('right', 0, 'stop')

        final_left, final_right = controller.get_encoder_counts()
        left_dual_change = abs(final_left - initial_left)
        right_dual_change = abs(final_right - initial_right)

        print(f"Dual motor results:")
        print(f"Left: {left_dual_change} counts")
        print(f"Right: {right_dual_change} counts")

        controller.cleanup()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_right_motor_encoder()