#!/usr/bin/env python3
"""
Test encoder without resets to see true accumulation like the direct test
"""

import sys
import time
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

def test_encoder_continuous():
    """Test encoder accumulation without resets"""

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    print("🔄 CONTINUOUS ENCODER TEST")
    print("=========================")
    print("Testing encoder accumulation like direct test (no resets)")
    print()

    try:
        controller = MotorControllerPolling()
        time.sleep(1)

        # NO RESET - let encoder accumulate from whatever current value
        initial_left, initial_right = controller.get_encoder_counts()
        print(f"Initial counts: Left={initial_left}, Right={initial_right}")

        print("Running left motor continuously for 3 seconds...")

        # Start motor and keep refreshing command every 100ms (well under 500ms watchdog)
        start_time = time.time()
        while time.time() - start_time < 3.0:
            controller.set_motor_speed('left', 50, 'forward')
            time.sleep(0.1)  # 100ms refresh rate

        controller.set_motor_speed('left', 0, 'stop')

        # Check final accumulation
        final_left, final_right = controller.get_encoder_counts()
        print(f"Final counts: Left={final_left}, Right={final_right}")

        # Calculate actual change
        left_change = abs(final_left - initial_left)
        right_change = abs(final_right - initial_right)

        print(f"\n📊 ENCODER ACCUMULATION:")
        print(f"Left motor: {left_change} counts in 3 seconds")
        print(f"Right motor: {right_change} counts in 3 seconds")

        # Compare to direct test (740 counts)
        if left_change > 500:
            print(f"✅ EXCELLENT: Encoder working at {left_change/740*100:.1f}% of direct test")
        elif left_change > 100:
            print(f"⚠️ PARTIAL: Encoder at {left_change/740*100:.1f}% of direct test - needs investigation")
        else:
            print(f"❌ BROKEN: Encoder at {left_change/740*100:.1f}% of direct test")
            print("Encoder polling still has fundamental issues")

        controller.cleanup()

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_encoder_continuous()