#!/usr/bin/env python3
"""
Test the new direct lgpio polling motor controller
Should fix the encoder reading issues
"""

import sys
import time
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

def test_fixed_polling():
    """Test the fixed direct lgpio polling system"""

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    print("🔧 TESTING FIXED MOTOR CONTROLLER")
    print("===========================")
    print("Using direct lgpio.gpio_read() instead of subprocess calls")
    print("Expected: Encoder counts should work for both motors")
    print()

    try:
        controller = MotorControllerPolling()

        print("✅ Motor controller initialized")
        print("Waiting 2 seconds for polling thread to start...")
        time.sleep(2)

        # Test 1: Individual motors
        print("\n=== INDIVIDUAL MOTOR TESTS (50% speed, 3 seconds each) ===")

        tests = [
            ('left', 'forward', "Left motor forward"),
            ('right', 'forward', "Right motor forward"),
        ]

        for motor, direction, description in tests:
            print(f"\n🔍 {description}")
            controller.reset_encoder_counts()
            print("Encoder counts reset")

            print(f"Running {motor} motor {direction} for 3 seconds...")
            controller.set_motor_speed(motor, 50, direction)

            # Monitor encoder counts during motor operation
            for second in range(3):
                time.sleep(1)
                rpms = controller.get_motor_rpm()
                left_enc, right_enc = controller.get_encoder_counts()
                print(f"Second {second+1}: Encoders L={left_enc}, R={right_enc} | RPM L={rpms['left']:.1f}, R={rpms['right']:.1f}")

            controller.stop()
            time.sleep(0.5)

            # Final results
            final_rpms = controller.get_motor_rpm()
            final_left, final_right = controller.get_encoder_counts()
            print(f"Final: Encoders L={final_left}, R={final_right} | RPM L={final_rpms['left']:.1f}, R={final_rpms['right']:.1f}")

            if motor == 'left' and abs(final_left) > 10:
                print("✅ Left motor encoder working with direct polling!")
            elif motor == 'right' and abs(final_right) > 10:
                print("✅ Right motor encoder working with direct polling!")
            else:
                print("❌ Still low encoder counts - may need hardware check")

        # Test 2: Both motors forward
        print("\n🔍 BOTH MOTORS FORWARD TEST")
        controller.reset_encoder_counts()

        print("Running both motors forward for 3 seconds...")
        controller.set_motor_speed('left', 50, 'forward')
        controller.set_motor_speed('right', 50, 'forward')

        for second in range(3):
            time.sleep(1)
            rpms = controller.get_motor_rpm()
            left_enc, right_enc = controller.get_encoder_counts()
            print(f"Second {second+1}: Encoders L={left_enc}, R={right_enc} | RPM L={rpms['left']:.1f}, R={rpms['right']:.1f}")

        controller.stop()
        time.sleep(0.5)

        final_rpms = controller.get_motor_rpm()
        final_left, final_right = controller.get_encoder_counts()
        print(f"Final: Encoders L={final_left}, R={final_right} | RPM L={final_rpms['left']:.1f}, R={final_rpms['right']:.1f}")

        print("\n🎯 RESULTS:")
        if abs(final_left) > 10 and abs(final_right) > 10:
            print("✅ SUCCESS: Both encoders working with direct polling!")
            print("The subprocess call bug has been fixed!")
        elif abs(final_left) > 10:
            print("✅ Left encoder working, ❌ right encoder still broken")
        elif abs(final_right) > 10:
            print("✅ Right encoder working, ❌ left encoder still broken")
        else:
            print("❌ Both encoders still showing low counts")
            print("Issue may be deeper than subprocess calls")

        controller.cleanup()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_fixed_polling()