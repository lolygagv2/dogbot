#!/usr/bin/env python3
"""
Test the SAFE motor controller with:
1. Watchdog timer (500ms timeout)
2. PID disabled (direct PWM only)
3. Working encoder code (should get 700+ counts like direct test)
"""

import sys
import time
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

def test_safe_motor_controller():
    """Test motor controller safety and encoder performance"""

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    print("🛡️ SAFE MOTOR CONTROLLER TEST")
    print("============================")
    print("Testing: Watchdog timer, PID disabled, working encoder code")
    print("Goal: Get 700+ counts like the direct test")
    print()

    try:
        controller = MotorControllerPolling()

        print("✅ Motor controller initialized")
        print("Waiting 2 seconds for threads to start...")
        time.sleep(2)

        # Test 1: Verify safety watchdog
        print("\n🛡️ SAFETY TEST: Watchdog timeout")
        print("Starting motor, then waiting 1 second (should timeout at 500ms)")

        controller.reset_encoder_counts()
        controller.set_motor_speed('left', 50, 'forward')

        # Wait 1 second without sending commands - watchdog should trigger
        time.sleep(1.0)

        left_count, right_count = controller.get_encoder_counts()
        print(f"After watchdog timeout: Left={left_count}, Right={right_count}")
        print("Watchdog should have stopped the motor\n")

        # Test 2: Normal operation with continuous commands
        print("🔧 ENCODER PERFORMANCE TEST")
        print("Running left motor with continuous commands (no timeout)")
        controller.reset_encoder_counts()

        # Run motor for 3 seconds with continuous commands (prevent timeout)
        print("Running left motor at 50% for 3 seconds...")
        for second in range(3):
            controller.set_motor_speed('left', 50, 'forward')  # Refresh command
            time.sleep(1)
            left_count, right_count = controller.get_encoder_counts()
            print(f"Second {second+1}: Left={abs(left_count)}, Right={abs(right_count)}")

        controller.set_motor_speed('left', 0, 'stop')
        time.sleep(0.5)

        final_left, final_right = controller.get_encoder_counts()
        print(f"\n🎯 FINAL RESULTS:")
        print(f"Left encoder: {abs(final_left)} counts in 3 seconds")
        print(f"Right encoder: {abs(final_right)} counts in 3 seconds")

        # Compare with direct test benchmark (740 counts)
        expected_counts = 700
        left_performance = abs(final_left) / expected_counts * 100

        if abs(final_left) >= expected_counts:
            print(f"✅ SUCCESS: Left encoder working at {left_performance:.1f}% of direct test")
            print("Motor controller encoder now matches direct test performance!")
        elif abs(final_left) > 100:
            print(f"⚠️ PARTIAL: Left encoder at {left_performance:.1f}% of direct test")
            print(f"Improved but still below target of {expected_counts}+ counts")
        else:
            print(f"❌ FAIL: Left encoder still broken at {left_performance:.1f}% of direct test")
            print("Encoder polling still has issues")

        # Test 3: Verify control mode
        print(f"\n🎛️ CONTROL MODE: {controller.control_mode}")
        if controller.control_mode == 'direct':
            print("✅ PID properly disabled - using direct PWM only")
        else:
            print("❌ PID still active - safety issue!")

        controller.cleanup()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_safe_motor_controller()