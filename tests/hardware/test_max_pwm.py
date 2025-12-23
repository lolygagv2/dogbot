#!/usr/bin/env python3
"""
Test motor controller with maximum PWM to maximize encoder counts
"""

import sys
import time
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

def test_max_pwm():
    """Test motors at maximum safe PWM to get maximum encoder counts"""

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    print("🚀 MAX PWM ENCODER TEST")
    print("======================")
    print("Testing motors at 100% speed (80% clamped PWM)")
    print("Goal: Get maximum encoder counts to verify system capability")
    print()

    try:
        controller = MotorControllerPolling()

        print("✅ Motor controller initialized")
        print("Waiting 2 seconds for polling thread to start...")
        time.sleep(2)

        # Test 1: Left motor at maximum speed
        print("\n🔥 LEFT MOTOR - 100% SPEED (80% clamped PWM)")
        controller.reset_encoder_counts()

        print("Running left motor at MAXIMUM speed for 5 seconds...")
        controller.set_motor_speed('left', 100, 'forward')  # Will clamp to 80% internally

        # Monitor every second for 5 seconds
        for second in range(5):
            time.sleep(1)
            left_enc, right_enc = controller.get_encoder_counts()
            rpms = controller.get_motor_rpm()
            print(f"Second {second+1}: Encoders L={left_enc}, R={right_enc} | RPM L={rpms['left']:.1f}")

        controller.stop()
        time.sleep(0.5)

        final_left, final_right = controller.get_encoder_counts()
        final_rpms = controller.get_motor_rpm()
        print(f"Final LEFT: {final_left} counts | RPM: {final_rpms['left']:.1f}")

        # Test 2: Right motor at maximum speed
        print("\n🔥 RIGHT MOTOR - 100% SPEED (80% clamped PWM)")
        controller.reset_encoder_counts()

        print("Running right motor at MAXIMUM speed for 5 seconds...")
        controller.set_motor_speed('right', 100, 'forward')  # Will clamp to 80% internally

        # Monitor every second for 5 seconds
        for second in range(5):
            time.sleep(1)
            left_enc, right_enc = controller.get_encoder_counts()
            rpms = controller.get_motor_rpm()
            print(f"Second {second+1}: Encoders L={left_enc}, R={right_enc} | RPM L={rpms['left']:.1f}, R={rpms['right']:.1f}")

        controller.stop()
        time.sleep(0.5)

        final_left, final_right = controller.get_encoder_counts()
        final_rpms = controller.get_motor_rpm()
        print(f"Final RIGHT: {final_right} counts | RPM: {final_rpms['right']:.1f}")

        print("\n🎯 MAX PWM RESULTS:")
        print(f"Left motor: {abs(final_left)} counts in 5 seconds")
        print(f"Right motor: {abs(final_right)} counts in 5 seconds")
        print()

        # Expected counts at maximum speed:
        # DFRobot motors at 6V: ~180 RPM
        # 298:1 gear ratio gives encoder: ~180 * 298 = 53,640 ticks/min
        # In 5 seconds: ~4,470 encoder ticks expected
        expected_counts = 4470

        print(f"Expected at max speed: ~{expected_counts} counts per motor")

        if abs(final_left) > 1000:
            print(f"✅ Left encoder getting good counts: {abs(final_left)}/{expected_counts} = {abs(final_left)/expected_counts*100:.1f}%")
        else:
            print(f"❌ Left encoder still low: {abs(final_left)}/{expected_counts} = {abs(final_left)/expected_counts*100:.1f}%")

        if abs(final_right) > 1000:
            print(f"✅ Right encoder getting good counts: {abs(final_right)}/{expected_counts} = {abs(final_right)/expected_counts*100:.1f}%")
        else:
            print(f"❌ Right encoder still low/broken: {abs(final_right)}/{expected_counts} = {abs(final_right)/expected_counts*100:.1f}%")

        if abs(final_left) < 100:
            print("\n🔍 LEFT MOTOR STILL LOW - Possible causes:")
            print("1. PWM still not high enough")
            print("2. Encoder polling frequency too low")
            print("3. Motor load/resistance too high")
            print("4. Quadrature decoding logic bug")

        controller.cleanup()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_max_pwm()