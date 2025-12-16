#!/usr/bin/env python3
"""
Test motors with simple on/off control - no PWM emulation
"""
import sys
import os
import time
import subprocess
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def set_gpio(pin, value):
    """Set GPIO pin directly"""
    try:
        subprocess.run(['gpioset', 'gpiochip0', f'{pin}={value}'],
                      capture_output=True, timeout=0.1)
        return True
    except:
        return False

def test_simple_motor_power():
    print("=== SIMPLE MOTOR POWER TEST ===")
    print("No PWM emulation - just direct on/off")

    # Left motor pins: IN1=17, IN2=18, ENA=13
    # Right motor pins: IN3=27, IN4=22, ENB=19

    # Clear all first
    for pin in [17, 18, 13, 27, 22, 19]:
        set_gpio(pin, 0)
    time.sleep(0.5)

    print("\n1. LEFT MOTOR - Full power forward for 2 seconds...")
    set_gpio(18, 1)  # IN2=1 (forward for left motor due to wiring)
    set_gpio(17, 0)  # IN1=0
    set_gpio(13, 1)  # ENA=1 (full power, no PWM)
    time.sleep(2)

    # Stop left
    set_gpio(13, 0)
    print("   Left motor stopped")

    print("\n2. RIGHT MOTOR - Full power forward for 2 seconds...")
    set_gpio(27, 1)  # IN3=1 (forward)
    set_gpio(22, 0)  # IN4=0
    set_gpio(19, 1)  # ENB=1 (full power, no PWM)
    time.sleep(2)

    # Stop right
    set_gpio(19, 0)
    print("   Right motor stopped")

    print("\n3. BOTH MOTORS - Full power forward for 2 seconds...")
    set_gpio(18, 1)  # Left forward
    set_gpio(17, 0)
    set_gpio(13, 1)  # Left enable
    set_gpio(27, 1)  # Right forward
    set_gpio(22, 0)
    set_gpio(19, 1)  # Right enable
    time.sleep(2)

    # Stop all
    for pin in [17, 18, 13, 27, 22, 19]:
        set_gpio(pin, 0)
    print("   Both motors stopped")

    print("\n=== SIMPLE POWER TEST COMPLETE ===")
    print("Did motors have FULL POWER with simple on/off?")

if __name__ == "__main__":
    test_simple_motor_power()