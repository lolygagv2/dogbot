#!/usr/bin/env python3
"""
Step 2: Backward test
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hardware.motor_controller_polling import MotorControllerPolling

def step2_backward():
    print("=== STEP 2: BACKWARD TEST ===")
    print("Both motors backward at 50% for 3 seconds...")
    print("WATCH: Both motors should spin backward at good speed")

    controller = MotorControllerPolling()
    time.sleep(1)

    controller.set_motor_speed('left', 50, 'backward')
    controller.set_motor_speed('right', 50, 'backward')
    time.sleep(3)

    controller.set_motor_speed('left', 0, 'stop')
    controller.set_motor_speed('right', 0, 'stop')
    print("STEP 2 COMPLETE - STOPPED")

    controller.cleanup()

if __name__ == "__main__":
    step2_backward()