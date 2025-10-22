#!/usr/bin/env python3
"""
Simple script to dispense treats on demand
Usage: python3 dispense_treat.py [number_of_treats]
"""

import sys
import time
import board
import busio
from adafruit_pca9685 import PCA9685

def pulse_to_duty(pulse_us):
    """Convert pulse width in microseconds to duty cycle"""
    return int((pulse_us / 20000.0) * 0xFFFF)

def dispense_treats(num_treats=1):
    """Dispense specified number of treats"""

    try:
        # Initialize PCA9685
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50
        carousel_servo = pca.channels[2]

        print(f"🍖 Dispensing {num_treats} treat(s)...")

        for i in range(num_treats):
            print(f"  Treat {i+1}/{num_treats}")

            # Rotate carousel forward
            carousel_servo.duty_cycle = pulse_to_duty(1700)
            time.sleep(0.12)  # Slightly longer rotation for reliable dispensing
            carousel_servo.duty_cycle = 0  # Stop

            # Pause between treats
            if i < num_treats - 1:
                time.sleep(0.5)

        print(f"✅ Successfully dispensed {num_treats} treat(s)!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False

    finally:
        try:
            carousel_servo.duty_cycle = 0
        except:
            pass

if __name__ == "__main__":
    # Get number of treats from command line or default to 1
    if len(sys.argv) > 1:
        try:
            num = int(sys.argv[1])
            if num < 1 or num > 5:
                print("⚠️  Number of treats should be between 1 and 5")
                num = 1
        except ValueError:
            print("⚠️  Invalid number, dispensing 1 treat")
            num = 1
    else:
        num = 1

    dispense_treats(num)