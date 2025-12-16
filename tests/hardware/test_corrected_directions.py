#!/usr/bin/env python3
"""
Test with corrected motor directions
"""
import time
from gpiozero import OutputDevice, PWMOutputDevice

def test_corrected_directions():
    print("=== CORRECTED DIRECTION TEST ===")

    try:
        # Left motor pins (Motor A)
        left_in1 = OutputDevice(17)  # Motor A Direction 1
        left_in2 = OutputDevice(18)  # Motor A Direction 2
        left_ena = PWMOutputDevice(13)  # Motor A Enable with PWM

        # Right motor pins (Motor B)
        right_in3 = OutputDevice(27)  # Motor B Direction 1
        right_in4 = OutputDevice(22)  # Motor B Direction 2
        right_enb = PWMOutputDevice(19)  # Motor B Enable with PWM

        # Clear all first
        left_in1.off()
        left_in2.off()
        left_ena.off()
        right_in3.off()
        right_in4.off()
        right_enb.off()
        time.sleep(1)

        print("\n1. LEFT MOTOR TEST - try both directions")
        print("   Left motor direction 1...")
        left_in1.on()
        left_in2.off()
        left_ena.value = 0.7  # Higher power
        time.sleep(2)
        left_ena.off()

        print("   Left motor direction 2...")
        left_in1.off()
        left_in2.on()
        left_ena.value = 0.7  # Higher power
        time.sleep(2)
        left_ena.off()

        print("\n2. RIGHT MOTOR TEST - try both directions at higher power")
        print("   Right motor direction 1...")
        right_in3.on()
        right_in4.off()
        right_enb.value = 0.8  # Much higher power for right motor
        time.sleep(2)
        right_enb.off()

        print("   Right motor direction 2...")
        right_in3.off()
        right_in4.on()
        right_enb.value = 0.8  # Much higher power for right motor
        time.sleep(2)
        right_enb.off()

        print("\n3. BOTH MOTORS SAME DIRECTION TEST")
        print("   Both motors - same GPIO pattern...")
        # Try same GPIO pattern for both motors
        left_in1.on()
        left_in2.off()
        left_ena.value = 0.6
        right_in3.on()
        right_in4.off()
        right_enb.value = 0.8  # Right motor needs more power
        time.sleep(3)

        # Stop all
        left_in1.off()
        left_in2.off()
        left_ena.off()
        right_in3.off()
        right_in4.off()
        right_enb.off()

        print("\n=== TEST COMPLETE ===")
        print("Which directions worked correctly for each motor?")

        # Clean up
        left_in1.close()
        left_in2.close()
        left_ena.close()
        right_in3.close()
        right_in4.close()
        right_enb.close()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_corrected_directions()