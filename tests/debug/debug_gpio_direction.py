#!/usr/bin/env python3
"""
Debug GPIO direction logic to understand actual motor behavior
"""
import subprocess
import time

def set_gpio(pin, value):
    """Set GPIO pin to value"""
    try:
        subprocess.run(['gpioset', 'gpiochip0', f'{pin}={value}'],
                      capture_output=True, timeout=0.1)
        print(f"  GPIO{pin} = {value}")
    except Exception as e:
        print(f"  ERROR setting GPIO{pin}: {e}")

def test_motor_direction(motor_name, in1_pin, in2_pin, ena_pin):
    """Test all direction combinations for a motor"""
    print(f"\n=== TESTING {motor_name} MOTOR ===")
    print(f"Pins: IN1=GPIO{in1_pin}, IN2=GPIO{in2_pin}, ENA=GPIO{ena_pin}")

    # Clear all pins first
    set_gpio(in1_pin, 0)
    set_gpio(in2_pin, 0)
    set_gpio(ena_pin, 0)
    time.sleep(0.5)

    print(f"\nTest 1: IN1=1, IN2=0, ENA=1 (should be forward)")
    set_gpio(in1_pin, 1)
    set_gpio(in2_pin, 0)
    set_gpio(ena_pin, 1)
    direction = input(f"Which direction did {motor_name} motor spin? (f=forward, b=backward, n=none): ").lower()

    # Stop
    set_gpio(ena_pin, 0)
    time.sleep(0.5)

    print(f"\nTest 2: IN1=0, IN2=1, ENA=1 (should be backward)")
    set_gpio(in1_pin, 0)
    set_gpio(in2_pin, 1)
    set_gpio(ena_pin, 1)
    direction2 = input(f"Which direction did {motor_name} motor spin? (f=forward, b=backward, n=none): ").lower()

    # Stop and clear
    set_gpio(in1_pin, 0)
    set_gpio(in2_pin, 0)
    set_gpio(ena_pin, 0)

    print(f"\n{motor_name} RESULTS:")
    print(f"  IN1=1,IN2=0: {direction}")
    print(f"  IN1=0,IN2=1: {direction2}")

    if direction == 'b':
        print(f"  ❌ {motor_name} MOTOR IS WIRED BACKWARDS!")
        print(f"     IN1=1,IN2=0 should be forward but goes backward")
    elif direction == 'f':
        print(f"  ✅ {motor_name} motor wired correctly")

    return direction, direction2

def main():
    print("=== GPIO DIRECTION DEBUG TEST ===")
    print("This will test actual motor directions vs expected GPIO logic")

    # Test left motor (Motor A)
    left_dir1, left_dir2 = test_motor_direction(
        "LEFT", 17, 18, 13  # MOTOR_IN1, MOTOR_IN2, MOTOR_ENA
    )

    # Test right motor (Motor B)
    right_dir1, right_dir2 = test_motor_direction(
        "RIGHT", 27, 22, 19  # MOTOR_IN3, MOTOR_IN4, MOTOR_ENB
    )

    print(f"\n=== SUMMARY ===")
    print(f"Left motor:  IN1=1,IN2=0 → {left_dir1}")
    print(f"Left motor:  IN1=0,IN2=1 → {left_dir2}")
    print(f"Right motor: IN1=1,IN2=0 → {right_dir1}")
    print(f"Right motor: IN1=0,IN2=1 → {right_dir2}")

    print(f"\n=== REQUIRED FIXES ===")
    if left_dir1 == 'b':
        print("LEFT MOTOR: Needs GPIO pin swap or code inversion")
    if right_dir1 == 'b':
        print("RIGHT MOTOR: Needs GPIO pin swap or code inversion")

if __name__ == "__main__":
    main()