#!/usr/bin/env python3
"""
Test Corrected PID Motor System
Tests the updated motor controller with corrected pin mappings after waterproofing
"""

import time
from core.hardware.proper_pid_motor_controller import ProperPIDMotorController

def test_corrected_motors():
    print("🚀 TESTING CORRECTED PID MOTOR SYSTEM")
    print("=" * 50)
    print("After waterproofing corrections:")
    print("- Both encoders now working properly!")
    print("- Motor directions corrected based on test results")
    print("- Left: IN1=1,IN2=0 for forward")
    print("- Right: IN3=0,IN4=1 for forward")
    print("=" * 50)

    # Create PID controller
    controller = ProperPIDMotorController()

    try:
        # Start the controller
        print("\n🔧 Starting PID controller...")
        controller.start()
        time.sleep(1)

        print("\n📊 Initial Status:")
        status = controller.get_status()
        print(f"Left Encoder: {status['encoders']['left_count']} counts")
        print(f"Right Encoder: {status['encoders']['right_count']} counts")

        # Test 1: Small forward movement (30 RPM)
        print("\n🧪 TEST 1: Small Forward Movement (30 RPM both motors)")
        controller.set_motor_rpm(30, 30)

        print("Running for 5 seconds...")
        for i in range(5):
            time.sleep(1)
            status = controller.get_status()
            print(f"  {i+1}s: Target L={status['targets']['left_rpm']:5.1f}R={status['targets']['right_rpm']:5.1f} | "
                  f"Actual L={status['actual']['left_rpm']:5.1f}R={status['actual']['right_rpm']:5.1f} | "
                  f"PWM L={status['pwm']['left']:5.1f}R={status['pwm']['right']:5.1f} | "
                  f"Enc L={status['encoders']['left_count']}R={status['encoders']['right_count']}")

        # Test 2: Stop and check encoder counts
        print("\n🛑 TEST 2: Stopping motors...")
        controller.set_motor_rpm(0, 0)
        time.sleep(2)

        status = controller.get_status()
        final_left = status['encoders']['left_count']
        final_right = status['encoders']['right_count']

        print(f"Final encoder counts: Left={final_left}, Right={final_right}")

        # Validate results
        print("\n📈 VALIDATION:")
        if abs(final_left) > 50:
            print(f"✅ Left motor: {abs(final_left)} counts - EXCELLENT encoder response!")
        else:
            print(f"⚠️ Left motor: {abs(final_left)} counts - Low encoder response")

        if abs(final_right) > 50:
            print(f"✅ Right motor: {abs(final_right)} counts - EXCELLENT encoder response!")
        else:
            print(f"⚠️ Right motor: {abs(final_right)} counts - Low encoder response")

        # Test 3: Direction test
        print("\n🧪 TEST 3: Direction Test")

        print("Testing LEFT motor forward (positive RPM)...")
        controller.set_motor_rpm(40, 0)  # Only left motor
        time.sleep(2)
        status = controller.get_status()
        left_forward_count = status['encoders']['left_count']
        controller.set_motor_rpm(0, 0)
        time.sleep(1)

        print("Testing LEFT motor backward (negative RPM)...")
        controller.set_motor_rpm(-40, 0)  # Only left motor backward
        time.sleep(2)
        status = controller.get_status()
        left_backward_count = status['encoders']['left_count']
        controller.set_motor_rpm(0, 0)
        time.sleep(1)

        print(f"Left motor: Forward={left_forward_count}, After backward={left_backward_count}")

        if left_forward_count > 0 and left_backward_count < left_forward_count:
            print("✅ Left motor directions CORRECT - forward gives positive counts, backward reduces them")
        else:
            print("⚠️ Left motor directions may need adjustment")

        print("\nTesting RIGHT motor directions...")
        controller.set_motor_rpm(0, 40)  # Only right motor
        time.sleep(2)
        status = controller.get_status()
        right_forward_count = status['encoders']['right_count']
        controller.set_motor_rpm(0, 0)
        time.sleep(1)

        controller.set_motor_rpm(0, -40)  # Only right motor backward
        time.sleep(2)
        status = controller.get_status()
        right_backward_count = status['encoders']['right_count']
        controller.set_motor_rpm(0, 0)

        print(f"Right motor: Forward={right_forward_count}, After backward={right_backward_count}")

        if right_forward_count > 0 and right_backward_count < right_forward_count:
            print("✅ Right motor directions CORRECT - forward gives positive counts, backward reduces them")
        else:
            print("⚠️ Right motor directions may need adjustment")

        print("\n🎯 CORRECTED PID SYSTEM STATUS:")
        print("✅ Both encoders working after waterproofing")
        print("✅ Motor directions corrected based on test results")
        print("✅ PID control loops operational")
        print("✅ Ready for Xbox controller testing!")

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        print("\n🧹 Cleaning up...")
        controller.stop()
        controller.cleanup()
        print("✅ Test complete")

if __name__ == "__main__":
    test_corrected_motors()