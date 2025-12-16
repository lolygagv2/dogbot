#!/usr/bin/env python3
"""
Simple GPIO direction test - just sets pins and tells you what to observe
"""
import subprocess
import time

def set_gpio(pin, value):
    """Set GPIO pin to value"""
    try:
        subprocess.run(['gpioset', 'gpiochip0', f'{pin}={value}'],
                      capture_output=True, timeout=0.1)
        print(f"GPIO{pin} = {value}")
        return True
    except Exception as e:
        print(f"ERROR setting GPIO{pin}: {e}")
        return False

def test_left_motor():
    """Test left motor GPIO directions"""
    print("=== LEFT MOTOR GPIO TEST ===")
    print("Left motor pins: IN1=GPIO17, IN2=GPIO18, ENA=GPIO13")

    # Clear all first
    print("\nClearing all pins...")
    set_gpio(17, 0)  # IN1
    set_gpio(18, 0)  # IN2
    set_gpio(13, 0)  # ENA
    time.sleep(1)

    print("\n1. Testing LEFT motor with IN1=1, IN2=0, ENA=1")
    print("   This should make left motor spin FORWARD...")
    set_gpio(17, 1)  # IN1=HIGH
    set_gpio(18, 0)  # IN2=LOW
    set_gpio(13, 1)  # ENA=HIGH (full power)

    print("   LEFT MOTOR should be spinning FORWARD now!")
    time.sleep(3)

    print("\n   Stopping left motor...")
    set_gpio(13, 0)  # ENA=LOW (stop)
    time.sleep(1)

    print("\n2. Testing LEFT motor with IN1=0, IN2=1, ENA=1")
    print("   This should make left motor spin BACKWARD...")
    set_gpio(17, 0)  # IN1=LOW
    set_gpio(18, 1)  # IN2=HIGH
    set_gpio(13, 1)  # ENA=HIGH (full power)

    print("   LEFT MOTOR should be spinning BACKWARD now!")
    time.sleep(3)

    print("\n   Stopping and clearing left motor...")
    set_gpio(17, 0)
    set_gpio(18, 0)
    set_gpio(13, 0)

def test_right_motor():
    """Test right motor GPIO directions"""
    print("\n=== RIGHT MOTOR GPIO TEST ===")
    print("Right motor pins: IN3=GPIO27, IN4=GPIO22, ENB=GPIO19")

    # Clear all first
    print("\nClearing all pins...")
    set_gpio(27, 0)  # IN3
    set_gpio(22, 0)  # IN4
    set_gpio(19, 0)  # ENB
    time.sleep(1)

    print("\n1. Testing RIGHT motor with IN3=1, IN4=0, ENB=1")
    print("   This should make right motor spin FORWARD...")
    set_gpio(27, 1)  # IN3=HIGH
    set_gpio(22, 0)  # IN4=LOW
    set_gpio(19, 1)  # ENB=HIGH (full power)

    print("   RIGHT MOTOR should be spinning FORWARD now!")
    time.sleep(3)

    print("\n   Stopping right motor...")
    set_gpio(19, 0)  # ENB=LOW (stop)
    time.sleep(1)

    print("\n2. Testing RIGHT motor with IN3=0, IN4=1, ENB=1")
    print("   This should make right motor spin BACKWARD...")
    set_gpio(27, 0)  # IN3=LOW
    set_gpio(22, 1)  # IN4=HIGH
    set_gpio(19, 1)  # ENB=HIGH (full power)

    print("   RIGHT MOTOR should be spinning BACKWARD now!")
    time.sleep(3)

    print("\n   Stopping and clearing right motor...")
    set_gpio(27, 0)
    set_gpio(22, 0)
    set_gpio(19, 0)

if __name__ == "__main__":
    print("GPIO MOTOR DIRECTION TEST")
    print("Watch the motors and note if they spin in the expected directions")

    test_left_motor()
    test_right_motor()

    print("\n=== TEST COMPLETE ===")
    print("Tell me what you observed:")
    print("1. Did LEFT motor spin FORWARD when IN1=1,IN2=0?")
    print("2. Did LEFT motor spin BACKWARD when IN1=0,IN2=1?")
    print("3. Did RIGHT motor spin FORWARD when IN3=1,IN4=0?")
    print("4. Did RIGHT motor spin BACKWARD when IN3=0,IN4=1?")