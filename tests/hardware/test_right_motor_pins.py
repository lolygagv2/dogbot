#!/usr/bin/env python3
"""
Test right motor pins specifically - GPIO19 might be broken
"""
import subprocess
import time

def test_gpio_pin(pin, name):
    """Test if a GPIO pin can output properly"""
    print(f"\nTesting GPIO{pin} ({name}):")

    try:
        # Test setting pin high
        print(f"  Setting GPIO{pin} HIGH...")
        result = subprocess.run(['gpioset', 'gpiochip0', f'{pin}=1'],
                               capture_output=True, text=True, timeout=1.0)
        if result.returncode != 0:
            print(f"  ERROR: Failed to set GPIO{pin} high: {result.stderr}")
            return False

        time.sleep(0.5)

        # Test setting pin low
        print(f"  Setting GPIO{pin} LOW...")
        result = subprocess.run(['gpioset', 'gpiochip0', f'{pin}=0'],
                               capture_output=True, text=True, timeout=1.0)
        if result.returncode != 0:
            print(f"  ERROR: Failed to set GPIO{pin} low: {result.stderr}")
            return False

        print(f"  GPIO{pin} ({name}) - OK")
        return True

    except Exception as e:
        print(f"  ERROR: GPIO{pin} ({name}) failed: {e}")
        return False

def test_right_motor_direct():
    """Test right motor with direct GPIO control"""
    print("=== TESTING RIGHT MOTOR PINS DIRECTLY ===")

    # Right motor pins
    IN3 = 27  # GPIO27 - Motor B Direction 1
    IN4 = 22  # GPIO22 - Motor B Direction 2
    ENB = 19  # GPIO19 - Motor B Enable/Speed PWM

    # Test each pin
    pin_tests = [
        (IN3, "MOTOR_IN3 (Direction 1)"),
        (IN4, "MOTOR_IN4 (Direction 2)"),
        (ENB, "MOTOR_ENB (PWM Enable)")
    ]

    failed_pins = []
    for pin, name in pin_tests:
        if not test_gpio_pin(pin, name):
            failed_pins.append((pin, name))

    if failed_pins:
        print(f"\nFAILED PINS: {failed_pins}")
        print("These pins may be hardware damaged or conflicting!")
    else:
        print("\nAll right motor pins test OK - issue is in PWM emulation logic")

    # Test manual motor control
    print(f"\n=== MANUAL RIGHT MOTOR TEST ===")
    print("Setting right motor to spin forward with full enable...")

    try:
        # Clear first
        subprocess.run(['gpioset', 'gpiochip0', f'{IN3}=0'], capture_output=True)
        subprocess.run(['gpioset', 'gpiochip0', f'{IN4}=0'], capture_output=True)
        subprocess.run(['gpioset', 'gpiochip0', f'{ENB}=0'], capture_output=True)
        time.sleep(0.5)

        # Set direction forward and enable high
        subprocess.run(['gpioset', 'gpiochip0', f'{IN3}=1'], capture_output=True)  # Forward
        subprocess.run(['gpioset', 'gpiochip0', f'{IN4}=0'], capture_output=True)
        subprocess.run(['gpioset', 'gpiochip0', f'{ENB}=1'], capture_output=True)  # Full power

        input("Right motor should be spinning forward at full power. Press Enter to stop...")

        # Stop motor
        subprocess.run(['gpioset', 'gpiochip0', f'{IN3}=0'], capture_output=True)
        subprocess.run(['gpioset', 'gpiochip0', f'{IN4}=0'], capture_output=True)
        subprocess.run(['gpioset', 'gpiochip0', f'{ENB}=0'], capture_output=True)

        print("Right motor stopped")

    except Exception as e:
        print(f"Manual test failed: {e}")

if __name__ == "__main__":
    test_right_motor_direct()