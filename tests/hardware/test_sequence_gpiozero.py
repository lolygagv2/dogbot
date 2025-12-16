#!/usr/bin/env python3
"""
Test sequence: Forward, Back, Left, Right at 50% using gpiozero
"""
import time
from gpiozero import OutputDevice, PWMOutputDevice

def test_motor_sequence():
    print("=== MOTOR SEQUENCE TEST ===")
    print("Using gpiozero library for GPIO control")

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

        print("\n1. FORWARD at 50% for 3 seconds...")
        print("   WATCH: Both motors should spin forward at medium speed")
        # Left forward (inverted): IN1=0, IN2=1
        left_in1.off()
        left_in2.on()
        left_ena.value = 0.5  # 50% PWM
        # Right forward: IN3=1, IN4=0
        right_in3.on()
        right_in4.off()
        right_enb.value = 0.5  # 50% PWM
        time.sleep(3)

        # Stop
        left_ena.off()
        right_enb.off()
        print("   STOPPED")
        time.sleep(1)

        print("\n2. BACKWARD at 50% for 3 seconds...")
        print("   WATCH: Both motors should spin backward at medium speed")
        # Left backward (inverted): IN1=1, IN2=0
        left_in1.on()
        left_in2.off()
        left_ena.value = 0.5  # 50% PWM
        # Right backward: IN3=0, IN4=1
        right_in3.off()
        right_in4.on()
        right_enb.value = 0.5  # 50% PWM
        time.sleep(3)

        # Stop
        left_ena.off()
        right_enb.off()
        print("   STOPPED")
        time.sleep(1)

        print("\n3. LEFT TURN at 50% for 3 seconds...")
        print("   WATCH: Should turn left (left motor backward, right motor forward)")
        # Left backward (inverted): IN1=1, IN2=0
        left_in1.on()
        left_in2.off()
        left_ena.value = 0.5  # 50% PWM
        # Right forward: IN3=1, IN4=0
        right_in3.on()
        right_in4.off()
        right_enb.value = 0.5  # 50% PWM
        time.sleep(3)

        # Stop
        left_ena.off()
        right_enb.off()
        print("   STOPPED")
        time.sleep(1)

        print("\n4. RIGHT TURN at 50% for 3 seconds...")
        print("   WATCH: Should turn right (left motor forward, right motor backward)")
        # Left forward (inverted): IN1=0, IN2=1
        left_in1.off()
        left_in2.on()
        left_ena.value = 0.5  # 50% PWM
        # Right backward: IN3=0, IN4=1
        right_in3.off()
        right_in4.on()
        right_enb.value = 0.5  # 50% PWM
        time.sleep(3)

        # Final stop
        left_in1.off()
        left_in2.off()
        left_ena.off()
        right_in3.off()
        right_in4.off()
        right_enb.off()
        print("   FINAL STOP")

        print("\n=== TEST COMPLETE ===")
        print("Tell me what you observed for each movement!")

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
    test_motor_sequence()