#!/usr/bin/env python3
"""
Test motor control directly to debug right motor issue
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hardware.motor_controller_polling import MotorControllerPolling

def test_individual_motors():
    print("Testing individual motor control...")

    try:
        controller = MotorControllerPolling()
        time.sleep(1)  # Let it initialize

        print("\n=== Testing LEFT motor (Motor A) ===")
        print("Forward 50% for 2 seconds...")
        controller.set_motor_speed('left', 50, 'forward')
        time.sleep(2)

        print("Backward 50% for 2 seconds...")
        controller.set_motor_speed('left', 50, 'backward')
        time.sleep(2)

        print("Stop left motor")
        controller.set_motor_speed('left', 0, 'stop')
        time.sleep(1)

        print("\n=== Testing RIGHT motor (Motor B) ===")
        print("Forward 50% for 2 seconds...")
        controller.set_motor_speed('right', 50, 'forward')
        time.sleep(2)

        print("Backward 50% for 2 seconds...")
        controller.set_motor_speed('right', 50, 'backward')
        time.sleep(2)

        print("Stop right motor")
        controller.set_motor_speed('right', 0, 'stop')
        time.sleep(1)

        print("\n=== Testing BOTH motors together ===")
        print("Both forward 50% for 2 seconds...")
        controller.set_motor_speed('left', 50, 'forward')
        controller.set_motor_speed('right', 50, 'forward')
        time.sleep(2)

        print("Stop all motors")
        controller.emergency_stop()

        # Show final status
        status = controller.get_status()
        print(f"\nFinal status:")
        print(f"Left motor: {status['motors']['left_speed']}%")
        print(f"Right motor: {status['motors']['right_speed']}%")
        print(f"PWM limits: {status['motors']['safety_limits']['min_pwm']}% - {status['motors']['safety_limits']['max_pwm']}%")

        controller.cleanup()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_individual_motors()