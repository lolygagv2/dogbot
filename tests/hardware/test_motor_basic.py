#!/usr/bin/env python3
"""
Basic motor test - verify motors actually spin
"""
from gpiozero import OutputDevice, PWMOutputDevice
import time

# Motor pins
MOTOR_IN1 = 17  # Left direction 1
MOTOR_IN2 = 18  # Left direction 2
MOTOR_ENA = 13  # Left PWM

MOTOR_IN3 = 27  # Right direction 1
MOTOR_IN4 = 22  # Right direction 2
MOTOR_ENB = 19  # Right PWM

def test_motor_basic():
    print("🔧 BASIC MOTOR TEST")
    print("==================")
    print("Testing if motors actually spin at high PWM")
    print("Listen for motor sound and check for rotation")
    print()

    # Setup GPIO
    left_in1 = OutputDevice(MOTOR_IN1)
    left_in2 = OutputDevice(MOTOR_IN2)
    left_ena = PWMOutputDevice(MOTOR_ENA)

    right_in3 = OutputDevice(MOTOR_IN3)
    right_in4 = OutputDevice(MOTOR_IN4)
    right_enb = PWMOutputDevice(MOTOR_ENB)

    try:
        # Test left motor at 80% PWM
        print("🔍 LEFT MOTOR - 80% PWM for 3 seconds")
        print("Direction: Forward (IN1=0, IN2=1, ENA=0.8)")
        left_in1.off()  # 0
        left_in2.on()   # 1
        left_ena.value = 0.8  # 80% PWM

        print("Starting LEFT motor... Listen/watch for rotation!")
        time.sleep(3)

        # Stop left motor
        left_ena.value = 0
        left_in1.off()
        left_in2.off()
        print("LEFT motor stopped\n")

        time.sleep(1)

        # Test right motor at 80% PWM
        print("🔍 RIGHT MOTOR - 80% PWM for 3 seconds")
        print("Direction: Forward (IN3=1, IN4=0, ENB=0.8)")
        right_in3.on()   # 1
        right_in4.off()  # 0
        right_enb.value = 0.8  # 80% PWM

        print("Starting RIGHT motor... Listen/watch for rotation!")
        time.sleep(3)

        # Stop right motor
        right_enb.value = 0
        right_in3.off()
        right_in4.off()
        print("RIGHT motor stopped\n")

        print("🎯 RESULTS:")
        print("Did LEFT motor spin?  [Y/N]")
        print("Did RIGHT motor spin? [Y/N]")
        print("If NO rotation = PWM/driver issue")
        print("If rotation but low encoder counts = encoder reading issue")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        left_ena.value = 0
        right_enb.value = 0
        left_ena.close()
        right_enb.close()
        left_in1.close()
        left_in2.close()
        right_in3.close()
        right_in4.close()
        print("GPIO cleaned up")

if __name__ == "__main__":
    test_motor_basic()