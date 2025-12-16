#!/usr/bin/env python3
"""
Debug PWM emulation directly
"""
import sys
import os
import time
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up verbose logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

from core.hardware.motor_controller_polling import MotorControllerPolling

def debug_pwm_directly():
    print("=== PWM EMULATION DEBUG TEST ===")

    controller = MotorControllerPolling()
    time.sleep(1)

    print("\nTesting RIGHT motor specifically...")

    # Test right motor with verbose logging
    print("1. Right motor forward 70% (should use PWM)...")
    controller.set_motor_speed('right', 70, 'forward')
    time.sleep(3)

    print("2. Right motor forward 30% (should use PWM)...")
    controller.set_motor_speed('right', 30, 'forward')
    time.sleep(3)

    print("3. Stop right motor...")
    controller.set_motor_speed('right', 0, 'stop')
    time.sleep(1)

    # Check PWM thread status
    print(f"\nPWM thread status:")
    print(f"Active PWM pins: {list(controller.pwm_running.keys())}")
    print(f"PWM threads: {list(controller.pwm_threads.keys())}")

    controller.cleanup()

if __name__ == "__main__":
    debug_pwm_directly()