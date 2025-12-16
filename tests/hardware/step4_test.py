#!/usr/bin/env python3
"""
Step 4: Right turn test - Final test
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hardware.motor_controller_polling import MotorControllerPolling

def step4_right_turn():
    print("=== STEP 4: RIGHT TURN TEST (FINAL) ===")
    print("Left motor forward, right motor backward at 50%...")
    print("WATCH: Should turn RIGHT smoothly")

    controller = MotorControllerPolling()
    time.sleep(1)

    controller.set_motor_speed('left', 50, 'forward')
    controller.set_motor_speed('right', 50, 'backward')
    time.sleep(3)

    controller.set_motor_speed('left', 0, 'stop')
    controller.set_motor_speed('right', 0, 'stop')
    print("STEP 4 COMPLETE - STOPPED")

    controller.cleanup()

if __name__ == "__main__":
    step4_right_turn()