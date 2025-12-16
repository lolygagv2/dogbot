#!/usr/bin/env python3
"""
Step 3: Left turn test - CRITICAL TEST
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hardware.motor_controller_polling import MotorControllerPolling

def step3_left_turn():
    print("=== STEP 3: LEFT TURN TEST (CRITICAL) ===")
    print("Left motor backward, right motor forward at 50%...")
    print("WATCH: Should turn LEFT (not right like before!)")

    controller = MotorControllerPolling()
    time.sleep(1)

    controller.set_motor_speed('left', 50, 'backward')
    controller.set_motor_speed('right', 50, 'forward')
    time.sleep(3)

    controller.set_motor_speed('left', 0, 'stop')
    controller.set_motor_speed('right', 0, 'stop')
    print("STEP 3 COMPLETE - STOPPED")

    controller.cleanup()

if __name__ == "__main__":
    step3_left_turn()