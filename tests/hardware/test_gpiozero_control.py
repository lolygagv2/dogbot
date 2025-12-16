#!/usr/bin/env python3
"""
Test motor control using gpiozero instead of gpioset subprocess calls
"""
import time
from gpiozero import OutputDevice

def test_gpiozero_motor():
    print("=== GPIOZERO MOTOR TEST ===")

    try:
        # Right motor pins
        in3 = OutputDevice(27)  # Motor B Direction 1
        in4 = OutputDevice(22)  # Motor B Direction 2
        enb = OutputDevice(19)  # Motor B Enable

        print("Testing right motor with gpiozero...")

        # Clear all first
        in3.off()
        in4.off()
        enb.off()
        time.sleep(0.5)

        print("Right motor FORWARD at full power for 3 seconds...")
        in3.on()   # Forward
        in4.off()
        enb.on()   # Full power
        time.sleep(3)

        print("Stopping right motor...")
        enb.off()
        in3.off()
        in4.off()

        print("Right motor BACKWARD at full power for 3 seconds...")
        in3.off()  # Backward
        in4.on()
        enb.on()   # Full power
        time.sleep(3)

        print("Final stop...")
        enb.off()
        in3.off()
        in4.off()

        print("=== GPIOZERO TEST COMPLETE ===")

        # Clean up
        in3.close()
        in4.close()
        enb.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gpiozero_motor()