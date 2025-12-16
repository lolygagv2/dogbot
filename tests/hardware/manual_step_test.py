#!/usr/bin/env python3
"""
Manual step test - one step at a time
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hardware.motor_controller_polling import MotorControllerPolling

def step1_forward():
    print("=== STEP 1: FORWARD TEST ===")
    print("Both motors forward at 50% for 3 seconds...")
    print("WATCH: Both motors should spin forward at good speed")

    controller = MotorControllerPolling()
    time.sleep(1)

    controller.set_motor_speed('left', 50, 'forward')
    controller.set_motor_speed('right', 50, 'forward')
    time.sleep(3)

    controller.set_motor_speed('left', 0, 'stop')
    controller.set_motor_speed('right', 0, 'stop')
    print("STEP 1 COMPLETE - STOPPED")

    controller.cleanup()

if __name__ == "__main__":
    step1_forward()