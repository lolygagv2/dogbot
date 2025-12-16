#!/usr/bin/env python3
"""
Debug PWM power delivery - check actual duty cycles
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hardware.motor_controller_polling import MotorControllerPolling

def debug_pwm_calculation():
    """Check what PWM duty cycle we actually get"""
    controller = MotorControllerPolling()

    print("=== PWM CALCULATION DEBUG ===")
    print(f"PWM limits: {controller.MIN_SAFE_PWM}% - {controller.MAX_SAFE_PWM}%")

    # Test different speeds
    test_speeds = [10, 25, 50, 75, 100]

    for speed in test_speeds:
        # Calculate what the motor controller would do
        if speed > 0:
            safe_duty = int(controller.MIN_SAFE_PWM +
                           (speed * (controller.MAX_SAFE_PWM - controller.MIN_SAFE_PWM) / 100))
        else:
            safe_duty = 0

        safe_duty = max(0, min(controller.MAX_SAFE_PWM, safe_duty))
        effective_voltage = 12.6 * safe_duty / 100

        print(f"Speed {speed}% → PWM {safe_duty}% → {effective_voltage:.1f}V")

def test_raw_gpio_power():
    """Test if GPIO pins can deliver full power without PWM"""
    print(f"\n=== RAW GPIO POWER TEST ===")
    print("Testing full GPIO power (no PWM) on right motor...")

    import subprocess

    # Right motor pins: IN3=27, IN4=22, ENB=19
    print("Setting right motor to full forward power (no PWM)...")
    subprocess.run(['gpioset', 'gpiochip0', '27=1'], capture_output=True)  # IN3=HIGH
    subprocess.run(['gpioset', 'gpiochip0', '22=0'], capture_output=True)  # IN4=LOW
    subprocess.run(['gpioset', 'gpiochip0', '19=1'], capture_output=True)  # ENB=HIGH (full power)

    print("Right motor should spin at FULL POWER for 3 seconds...")
    time.sleep(3)

    # Stop
    subprocess.run(['gpioset', 'gpiochip0', '27=0'], capture_output=True)
    subprocess.run(['gpioset', 'gpiochip0', '22=0'], capture_output=True)
    subprocess.run(['gpioset', 'gpiochip0', '19=0'], capture_output=True)
    print("Stopped")

if __name__ == "__main__":
    debug_pwm_calculation()
    test_raw_gpio_power()