#!/usr/bin/env python3
"""
Test Fixed PID Motor Directions
Verify that motor directions and encoder directions are properly aligned
to prevent runaway motors in the PID system
"""

import time
from core.hardware.proper_pid_motor_controller import ProperPIDMotorController

def test_pid_directions():
    print("🔧 TESTING FIXED PID DIRECTIONS")
    print("=" * 50)
    print("This test verifies that:")
    print("1. Positive RPM target = forward motion = positive encoder counts")
    print("2. Negative RPM target = backward motion = negative encoder counts")
    print("3. No runaway motors due to direction mismatch")
    print("=" * 50)

    controller = ProperPIDMotorController()

    try:
        print("\n🚀 Starting PID controller...")
        controller.start()
        time.sleep(1)

        # Test 1: Very small forward command
        print("\n🧪 TEST 1: Small Forward Command (+10 RPM both motors)")
        print("   Should: Move forward slowly, positive encoder counts")

        controller.set_motor_rpm(10, 10)

        for i in range(5):
            time.sleep(1)
            status = controller.get_status()
            left_target = status['targets']['ramped_left']
            right_target = status['targets']['ramped_right']
            left_actual = status['actual']['left_rpm']
            right_actual = status['actual']['right_rpm']
            left_enc = status['encoders']['left_count']
            right_enc = status['encoders']['right_count']

            print(f"   {i+1}s: Target L={left_target:5.1f}R={right_target:5.1f} | "
                  f"Actual L={left_actual:5.1f}R={right_actual:5.1f} | "
                  f"Enc L={left_enc}R={right_enc}")

            # Safety check - if either motor is going wrong direction, stop immediately
            if (left_target > 0 and left_actual < -5) or (right_target > 0 and right_actual < -5):
                print("   ❌ DIRECTION MISMATCH DETECTED - STOPPING!")
                controller.set_motor_rpm(0, 0)
                break

        # Stop motors
        print("\n🛑 Stopping motors...")
        controller.set_motor_rpm(0, 0)
        time.sleep(2)

        # Check final encoder counts
        status = controller.get_status()
        final_left = status['encoders']['left_count']
        final_right = status['encoders']['right_count']

        print(f"\nFinal encoder counts: Left={final_left}, Right={final_right}")

        # Validate forward test
        forward_success = True
        if final_left < 5:
            print("   ⚠️ Left motor: Insufficient forward movement")
            forward_success = False
        if final_right < 5:
            print("   ⚠️ Right motor: Insufficient forward movement")
            forward_success = False

        if forward_success:
            print("   ✅ Forward test PASSED - both motors moved forward with positive encoder counts")
        else:
            print("   ❌ Forward test FAILED")
            return False

        # Test 2: Small backward command
        print("\n🧪 TEST 2: Small Backward Command (-10 RPM both motors)")
        print("   Should: Move backward slowly, encoder counts decrease")

        controller.set_motor_rpm(-10, -10)

        for i in range(5):
            time.sleep(1)
            status = controller.get_status()
            left_target = status['targets']['ramped_left']
            right_target = status['targets']['ramped_right']
            left_actual = status['actual']['left_rpm']
            right_actual = status['actual']['right_rpm']
            left_enc = status['encoders']['left_count']
            right_enc = status['encoders']['right_count']

            print(f"   {i+1}s: Target L={left_target:5.1f}R={right_target:5.1f} | "
                  f"Actual L={left_actual:5.1f}R={right_actual:5.1f} | "
                  f"Enc L={left_enc}R={right_enc}")

            # Safety check - if either motor is going wrong direction, stop immediately
            if (left_target < 0 and left_actual > 5) or (right_target < 0 and right_actual > 5):
                print("   ❌ DIRECTION MISMATCH DETECTED - STOPPING!")
                controller.set_motor_rpm(0, 0)
                break

        # Stop motors
        print("\n🛑 Final stop...")
        controller.set_motor_rpm(0, 0)
        time.sleep(1)

        # Check final encoder counts
        status = controller.get_status()
        backward_left = status['encoders']['left_count']
        backward_right = status['encoders']['right_count']

        print(f"Encoder counts after backward: Left={backward_left}, Right={backward_right}")

        # Validate backward test
        backward_success = True
        if backward_left >= final_left:
            print("   ⚠️ Left motor: Did not move backward (encoder count didn't decrease)")
            backward_success = False
        if backward_right >= final_right:
            print("   ⚠️ Right motor: Did not move backward (encoder count didn't decrease)")
            backward_success = False

        if backward_success:
            print("   ✅ Backward test PASSED - both motors moved backward, encoder counts decreased")
        else:
            print("   ❌ Backward test FAILED")
            return False

        print("\n🎉 ALL DIRECTION TESTS PASSED!")
        print("✅ PID motor system is safe for Xbox controller use")
        print("✅ No runaway motor risk - directions properly aligned")

        return True

    except Exception as e:
        print(f"\n❌ Test error: {e}")
        return False
    finally:
        print("\n🧹 Cleaning up...")
        controller.set_motor_rpm(0, 0)
        time.sleep(0.5)
        controller.stop()
        controller.cleanup()

if __name__ == "__main__":
    success = test_pid_directions()
    if success:
        print("\n✅ Safe to test Xbox controller!")
    else:
        print("\n❌ Fix direction issues before Xbox controller testing!")