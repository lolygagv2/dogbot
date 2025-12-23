#!/usr/bin/env python3
"""
Test PID System with Debug Logging
Simple test with debug logging to see exactly what pin configurations
the PID system sends and why we get negative encoder counts
"""

import time
import logging
from core.hardware.proper_pid_motor_controller import ProperPIDMotorController

def setup_logging():
    """Setup logging to see PID debug messages"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s:%(message)s'
    )

def test_pid_with_logging():
    print("🔍 PID SYSTEM WITH DEBUG LOGGING TEST")
    print("=" * 50)
    print("This test will show exactly what pin configurations")
    print("the PID system sends when we ask for forward motion")
    print("=" * 50)

    setup_logging()

    controller = ProperPIDMotorController()

    try:
        print("\n🚀 Starting PID controller...")
        controller.start()
        time.sleep(1)

        print(f"\n📊 Initial status:")
        status = controller.get_status()
        print(f"   Left encoder: {status['encoders']['left_count']} counts")
        print(f"   Right encoder: {status['encoders']['right_count']} counts")

        print(f"\n🧪 TEST: Requesting +10 RPM FORWARD for both motors")
        print(f"   This should trigger FORWARD pin configurations")
        print(f"   Expected: Left IN1=1,IN2=0 | Right IN3=0,IN4=1")
        print(f"   Running for 5 seconds...\n")

        controller.set_motor_rpm(10, 10)

        # Monitor for 5 seconds
        for i in range(10):
            time.sleep(0.5)
            status = controller.get_status()

            left_target = status['targets']['ramped_left']
            right_target = status['targets']['ramped_right']
            left_actual = status['actual']['left_rpm']
            right_actual = status['actual']['right_rpm']
            left_enc = status['encoders']['left_count']
            right_enc = status['encoders']['right_count']

            print(f"   {(i+1)*0.5:.1f}s: Target L={left_target:5.1f}R={right_target:5.1f} | "
                  f"Actual L={left_actual:6.1f}R={right_actual:6.1f} | "
                  f"Enc L={left_enc:4d}R={right_enc:4d}")

            # Safety check for direction mismatch
            if i >= 4:  # Give some time for ramping
                if left_target > 5 and left_actual < -10:
                    print("   🚨 LEFT MOTOR DIRECTION MISMATCH!")
                    break
                if right_target > 5 and right_actual < -10:
                    print("   🚨 RIGHT MOTOR DIRECTION MISMATCH!")
                    break

        print(f"\n🛑 Stopping motors...")
        controller.set_motor_rpm(0, 0)
        time.sleep(1)

        # Final analysis
        final_status = controller.get_status()
        final_left_enc = final_status['encoders']['left_count']
        final_right_enc = final_status['encoders']['right_count']

        print(f"\n📊 FINAL ANALYSIS:")
        print(f"   Final encoder counts: Left={final_left_enc}, Right={final_right_enc}")

        print(f"\n   Expected behavior for +10 RPM FORWARD request:")
        print(f"   ✅ Positive encoder counts (motor going forward)")
        print(f"   ✅ Debug logs showing FORWARD pin configurations")
        print(f"   ✅ Positive actual RPM values")

        print(f"\n   Actual results:")
        if final_left_enc > 0:
            print(f"   ✅ Left encoder: {final_left_enc} (positive - correct)")
        else:
            print(f"   ❌ Left encoder: {final_left_enc} (negative - wrong direction!)")

        if final_right_enc > 0:
            print(f"   ✅ Right encoder: {final_right_enc} (positive - correct)")
        else:
            print(f"   ❌ Right encoder: {final_right_enc} (negative - wrong direction!)")

        # Check debug logs in the output above for pin configurations

    except Exception as e:
        print(f"❌ Test error: {e}")
    finally:
        print(f"\n🧹 Cleanup...")
        controller.set_motor_rpm(0, 0)
        controller.stop()
        controller.cleanup()

if __name__ == "__main__":
    test_pid_with_logging()