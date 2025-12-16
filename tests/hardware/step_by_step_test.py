#!/usr/bin/env python3
"""
Step by step motor test with user confirmation
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hardware.motor_controller_polling import MotorControllerPolling

def step_by_step_test():
    print("=== STEP BY STEP MOTOR TEST ===")

    controller = MotorControllerPolling()
    time.sleep(1)

    try:
        print("\nSTEP 1: FORWARD TEST")
        print("Both motors forward at 50% for 3 seconds...")
        print("WATCH: Both motors should spin forward at good speed")
        input("Press Enter to start Step 1...")

        controller.set_motor_speed('left', 50, 'forward')
        controller.set_motor_speed('right', 50, 'forward')
        time.sleep(3)

        controller.set_motor_speed('left', 0, 'stop')
        controller.set_motor_speed('right', 0, 'stop')
        print("Step 1 complete - STOPPED")

        result1 = input("Did both motors go forward at good speed? (y/n): ").lower()
        if result1 != 'y':
            print("Step 1 failed - stopping test")
            controller.cleanup()
            return

        print("\nSTEP 2: BACKWARD TEST")
        print("Both motors backward at 50% for 3 seconds...")
        print("WATCH: Both motors should spin backward at good speed")
        input("Press Enter to start Step 2...")

        controller.set_motor_speed('left', 50, 'backward')
        controller.set_motor_speed('right', 50, 'backward')
        time.sleep(3)

        controller.set_motor_speed('left', 0, 'stop')
        controller.set_motor_speed('right', 0, 'stop')
        print("Step 2 complete - STOPPED")

        result2 = input("Did both motors go backward at good speed? (y/n): ").lower()
        if result2 != 'y':
            print("Step 2 failed - stopping test")
            controller.cleanup()
            return

        print("\nSTEP 3: LEFT TURN TEST")
        print("Left turn: left motor backward, right motor forward at 50%...")
        print("WATCH: Should turn LEFT (not right like before)")
        input("Press Enter to start Step 3...")

        controller.set_motor_speed('left', 50, 'backward')
        controller.set_motor_speed('right', 50, 'forward')
        time.sleep(3)

        controller.set_motor_speed('left', 0, 'stop')
        controller.set_motor_speed('right', 0, 'stop')
        print("Step 3 complete - STOPPED")

        result3 = input("Did it turn LEFT correctly? (y/n): ").lower()
        if result3 != 'y':
            print("Step 3 failed - stopping test")
            controller.cleanup()
            return

        print("\nSTEP 4: RIGHT TURN TEST")
        print("Right turn: left motor forward, right motor backward at 50%...")
        print("WATCH: Should turn RIGHT smoothly")
        input("Press Enter to start Step 4...")

        controller.set_motor_speed('left', 50, 'forward')
        controller.set_motor_speed('right', 50, 'backward')
        time.sleep(3)

        controller.set_motor_speed('left', 0, 'stop')
        controller.set_motor_speed('right', 0, 'stop')
        print("Step 4 complete - STOPPED")

        result4 = input("Did it turn RIGHT correctly? (y/n): ").lower()

        if result4 == 'y':
            print("\n✅ ALL TESTS PASSED!")
            print("Motor control is now working correctly!")
        else:
            print("Step 4 failed")

        controller.cleanup()

    except Exception as e:
        print(f"Error: {e}")
        controller.cleanup()

if __name__ == "__main__":
    step_by_step_test()