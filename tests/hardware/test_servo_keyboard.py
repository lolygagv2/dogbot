#!/usr/bin/env python3
"""
Interactive servo control test with keyboard
"""

import sys
import time
import termios
import tty
from servo_control_module import ServoController

def get_key():
    """Get a single keypress"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def main():
    """Main interactive control loop"""
    print("Initializing servo controller...")
    controller = ServoController()

    if not controller.initialize():
        print("Failed to initialize servo controller!")
        return

    print("✅ Servo controller initialized")
    print("\n=== Keyboard Controls ===")
    print("W/S: Tilt camera up/down")
    print("A/D: Pan camera left/right")
    print("H: Home position (center)")
    print("Q: Quit")
    print("========================\n")

    # Start at center
    pan_angle = 0
    tilt_angle = 0
    angle_step = 10

    controller.set_pan_angle(pan_angle, smooth=False)
    controller.set_tilt_angle(tilt_angle, smooth=False)
    print(f"Position: Pan={pan_angle}°, Tilt={tilt_angle}°")

    try:
        while True:
            key = get_key().lower()

            moved = False

            if key == 'w':  # Tilt up
                tilt_angle = min(45, tilt_angle + angle_step)
                controller.set_tilt_angle(tilt_angle, smooth=False)
                moved = True

            elif key == 's':  # Tilt down
                tilt_angle = max(-45, tilt_angle - angle_step)
                controller.set_tilt_angle(tilt_angle, smooth=False)
                moved = True

            elif key == 'a':  # Pan left
                pan_angle = max(-90, pan_angle - angle_step)
                controller.set_pan_angle(pan_angle, smooth=False)
                moved = True

            elif key == 'd':  # Pan right
                pan_angle = min(90, pan_angle + angle_step)
                controller.set_pan_angle(pan_angle, smooth=False)
                moved = True

            elif key == 'h':  # Home
                pan_angle = 0
                tilt_angle = 0
                controller.set_pan_angle(pan_angle, smooth=False)
                controller.set_tilt_angle(tilt_angle, smooth=False)
                moved = True
                print("Centered!")

            elif key == 'q':  # Quit
                break

            if moved:
                print(f"Position: Pan={pan_angle}°, Tilt={tilt_angle}°")

    except KeyboardInterrupt:
        print("\n\nInterrupted")

    # Clean up
    print("\nCentering servos...")
    controller.center_all()
    controller.cleanup()
    print("Done!")

if __name__ == "__main__":
    main()