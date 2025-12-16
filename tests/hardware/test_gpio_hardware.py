#!/usr/bin/env python3
"""
Test if GPIO commands actually reach the hardware
"""
import subprocess
import time

def test_gpio_hardware():
    print("=== GPIO HARDWARE CONNECTION TEST ===")

    # Test GPIO19 (right motor enable pin)
    pin = 19

    print(f"Testing GPIO{pin} hardware connection...")

    # Set pin high
    print(f"Setting GPIO{pin} = HIGH")
    result1 = subprocess.run(['gpioset', 'gpiochip0', f'{pin}=1'],
                           capture_output=True, text=True)
    print(f"  Command result: {result1.returncode}")
    if result1.stderr:
        print(f"  Error: {result1.stderr}")

    time.sleep(0.5)

    # Read back the pin state
    print(f"Reading back GPIO{pin} state...")
    result2 = subprocess.run(['gpioget', 'gpiochip0', str(pin)],
                           capture_output=True, text=True)
    print(f"  Command result: {result2.returncode}")
    print(f"  GPIO{pin} state: {result2.stdout.strip()}")
    if result2.stderr:
        print(f"  Error: {result2.stderr}")

    # Set pin low
    print(f"Setting GPIO{pin} = LOW")
    result3 = subprocess.run(['gpioset', 'gpiochip0', f'{pin}=0'],
                           capture_output=True, text=True)
    print(f"  Command result: {result3.returncode}")
    if result3.stderr:
        print(f"  Error: {result3.stderr}")

    time.sleep(0.5)

    # Read back again
    print(f"Reading back GPIO{pin} state...")
    result4 = subprocess.run(['gpioget', 'gpiochip0', str(pin)],
                           capture_output=True, text=True)
    print(f"  Command result: {result4.returncode}")
    print(f"  GPIO{pin} state: {result4.stdout.strip()}")

    print(f"\n=== GPIO{pin} TEST COMPLETE ===")

    # Test all motor pins
    motor_pins = [17, 18, 13, 27, 22, 19]  # All motor control pins
    print(f"\n=== TESTING ALL MOTOR PINS ===")

    for pin in motor_pins:
        # Set high
        subprocess.run(['gpioset', 'gpiochip0', f'{pin}=1'], capture_output=True)
        time.sleep(0.1)

        # Read state
        result = subprocess.run(['gpioget', 'gpiochip0', str(pin)],
                              capture_output=True, text=True)
        state = result.stdout.strip() if result.returncode == 0 else "ERROR"

        # Set low
        subprocess.run(['gpioset', 'gpiochip0', f'{pin}=0'], capture_output=True)

        print(f"GPIO{pin}: {state}")

if __name__ == "__main__":
    test_gpio_hardware()