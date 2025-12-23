#!/usr/bin/env python3
"""
SIMPLE joystick to motor test - bypass complex Xbox controller
Direct joystick input → direct motor control
"""

import struct
import sys
import time
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

def test_simple_joystick_motor():
    """Simple direct joystick → motor control test"""

    logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')

    print("🎮🚗 SIMPLE JOYSTICK → MOTOR TEST")
    print("=================================")
    print("Direct joystick input → motor control")
    print("Move LEFT STICK to control motors")
    print("Press Ctrl+C to stop")
    print()

    try:
        # Initialize motor controller
        print("🔧 Initializing motor controller...")
        controller = MotorControllerPolling()
        time.sleep(1)

        print("✅ Motor controller ready")
        print("🎮 Reading joystick input from /dev/input/js0...")
        print("Move LEFT STICK to control motors!")
        print()

        with open('/dev/input/js0', 'rb') as js:
            left_x = 0.0
            left_y = 0.0

            while True:
                # Read joystick event
                event = js.read(8)
                if len(event) == 8:
                    time_stamp, value, event_type, number = struct.unpack('IhBB', event)

                    if event_type == 2:  # Axis event
                        # Normalize to -1.0 to 1.0
                        normalized = value / 32767.0

                        if number == 0:  # Left stick X
                            left_x = normalized
                            print(f"🎮 Left X: {left_x:.3f}")

                        elif number == 1:  # Left stick Y
                            left_y = -normalized  # Invert Y for forward
                            print(f"🎮 Left Y: {left_y:.3f}")

                        # Convert joystick to motor speeds
                        # Simple tank steering
                        forward = left_y * 30  # 30% max speed for safety
                        turn = -left_x * 20    # 20% turn rate (FIXED: negated for correct direction)

                        left_speed = int(forward - turn)
                        right_speed = int(forward + turn)

                        # Limit speeds
                        left_speed = max(-30, min(30, left_speed))
                        right_speed = max(-30, min(30, right_speed))

                        # Send motor commands
                        if abs(left_speed) > 2:  # Small deadzone
                            if left_speed > 0:
                                controller.set_motor_speed('left', abs(left_speed), 'forward')
                            else:
                                controller.set_motor_speed('left', abs(left_speed), 'backward')
                        else:
                            controller.set_motor_speed('left', 0, 'stop')

                        if abs(right_speed) > 2:  # Small deadzone
                            if right_speed > 0:
                                controller.set_motor_speed('right', abs(right_speed), 'forward')
                            else:
                                controller.set_motor_speed('right', abs(right_speed), 'backward')
                        else:
                            controller.set_motor_speed('right', 0, 'stop')

                        # Show motor commands
                        if abs(left_speed) > 2 or abs(right_speed) > 2:
                            print(f"🚗 Motors: L={left_speed:+3d}, R={right_speed:+3d}")

                    elif event_type == 1 and number == 1 and value == 1:  # B button pressed
                        print("🚨 B BUTTON - EMERGENCY STOP!")
                        controller.set_motor_speed('left', 0, 'stop')
                        controller.set_motor_speed('right', 0, 'stop')
                        break

    except KeyboardInterrupt:
        print("\n✅ Stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🛑 Stopping motors...")
        try:
            controller.set_motor_speed('left', 0, 'stop')
            controller.set_motor_speed('right', 0, 'stop')
            controller.cleanup()
        except:
            pass
        print("✅ Test complete")

if __name__ == "__main__":
    test_simple_joystick_motor()