#!/usr/bin/env python3
"""
Prove I can control motors correctly - no Xbox, just direct commands
"""
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.hardware.motor_controller_polling import MotorControllerPolling

def prove_motor_control():
    print("=== PROVING MOTOR CONTROL ===")
    print("Direct motor commands - no Xbox involved")

    controller = MotorControllerPolling()
    time.sleep(1)  # Let it initialize

    try:
        # 1. 50% forward both motors
        print("\n1. 50% FORWARD both motors for 3 seconds...")
        controller.set_motor_speed('left', 50, 'forward')
        controller.set_motor_speed('right', 50, 'forward')
        print("   LEFT: 50% forward")
        print("   RIGHT: 50% forward")
        print("   Both motors should spin forward at same speed")
        time.sleep(3)

        # Stop
        controller.set_motor_speed('left', 0, 'stop')
        controller.set_motor_speed('right', 0, 'stop')
        print("   STOPPED")
        time.sleep(1)

        # 2. 50% backward both motors
        print("\n2. 50% BACKWARD both motors for 3 seconds...")
        controller.set_motor_speed('left', 50, 'backward')
        controller.set_motor_speed('right', 50, 'backward')
        print("   LEFT: 50% backward")
        print("   RIGHT: 50% backward")
        print("   Both motors should spin backward at same speed")
        time.sleep(3)

        # Stop
        controller.set_motor_speed('left', 0, 'stop')
        controller.set_motor_speed('right', 0, 'stop')
        print("   STOPPED")
        time.sleep(1)

        # 3. 50% left turn
        print("\n3. 50% LEFT TURN for 3 seconds...")
        controller.set_motor_speed('left', 50, 'backward')
        controller.set_motor_speed('right', 50, 'forward')
        print("   LEFT: 50% backward")
        print("   RIGHT: 50% forward")
        print("   Should turn left (left wheel backward, right wheel forward)")
        time.sleep(3)

        # Stop
        controller.set_motor_speed('left', 0, 'stop')
        controller.set_motor_speed('right', 0, 'stop')
        print("   STOPPED")
        time.sleep(1)

        # 4. 50% right turn
        print("\n4. 50% RIGHT TURN for 3 seconds...")
        controller.set_motor_speed('left', 50, 'forward')
        controller.set_motor_speed('right', 50, 'backward')
        print("   LEFT: 50% forward")
        print("   RIGHT: 50% backward")
        print("   Should turn right (left wheel forward, right wheel backward)")
        time.sleep(3)

        # Final stop
        controller.set_motor_speed('left', 0, 'stop')
        controller.set_motor_speed('right', 0, 'stop')
        print("   STOPPED")

        print("\n=== MOTOR CONTROL TEST COMPLETE ===")
        print("Tell me what actually happened vs what should have happened")

        controller.cleanup()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        controller.emergency_stop()

if __name__ == "__main__":
    prove_motor_control()