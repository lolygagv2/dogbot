#!/usr/bin/env python3
"""
Exact Motor Test Sequence as Requested:
1. Left motor forward 4 seconds
2. Stop for 3 seconds
3. Right motor forward 4 seconds
4. Both backward together 5 seconds
"""

import time
from core.hardware.proper_pid_motor_controller import ProperPIDMotorController

def exact_sequence():
    print("🎯 EXACT MOTOR TEST SEQUENCE")
    print("=" * 50)
    print("1. Left motor forward 4 seconds")
    print("2. Stop for 3 seconds")
    print("3. Right motor forward 4 seconds")
    print("4. Both backward together 5 seconds")
    print("=" * 50)

    controller = ProperPIDMotorController()

    try:
        # Start controller
        controller.start()
        time.sleep(1)

        # Step 1: Left motor forward 4 seconds
        print("\n1️⃣ LEFT MOTOR FORWARD - 4 seconds")
        controller.set_motor_rpm(30, 0)  # Left 30 RPM, Right 0
        time.sleep(4)

        # Step 2: Stop for 3 seconds
        print("\n⏸️  STOP - 3 seconds")
        controller.set_motor_rpm(0, 0)
        time.sleep(3)

        # Step 3: Right motor forward 4 seconds
        print("\n2️⃣ RIGHT MOTOR FORWARD - 4 seconds")
        controller.set_motor_rpm(0, 30)  # Left 0, Right 30 RPM
        time.sleep(4)

        # Step 4: Both backward together 5 seconds
        print("\n3️⃣ BOTH BACKWARD - 5 seconds")
        controller.set_motor_rpm(-30, -30)  # Both -30 RPM
        time.sleep(5)

        # Final stop
        print("\n🛑 FINAL STOP")
        controller.set_motor_rpm(0, 0)
        time.sleep(1)

        # Show final encoder counts
        status = controller.get_status()
        print(f"\nFinal encoder counts: Left={status['encoders']['left_count']}, Right={status['encoders']['right_count']}")
        print("✅ Sequence complete")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        controller.set_motor_rpm(0, 0)
        controller.stop()
        controller.cleanup()

if __name__ == "__main__":
    exact_sequence()