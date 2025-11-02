#!/usr/bin/env python3
"""
Test Xbox controller with evdev
"""

import sys
sys.path.append('/home/morgan/dogbot')

from services.control.xbox_controller import get_xbox_controller
import time
import logging

logging.basicConfig(level=logging.INFO)

print("="*50)
print("XBOX CONTROLLER TEST")
print("="*50)

# Get controller
controller = get_xbox_controller()

# Try to initialize
if controller.initialize():
    print("✅ Xbox controller found and connected!")
    print(f"   Device: {controller.controller.name}")
    print(f"   Path: {controller.controller.path}")

    print("\nControls:")
    print("  Left Stick: Movement")
    print("  Right Stick: Camera")
    print("  A: Treat")
    print("  B: Sound")
    print("  X: Toggle AI")
    print("  Y: Emergency Stop")
    print("  Bumpers: Speed adjust")

    print("\nPress buttons or move sticks (Ctrl+C to exit)...")

    try:
        # Read events directly for testing
        for event in controller.controller.read_loop():
            if event.type == 1:  # Button
                if event.value == 1:
                    print(f"Button {event.code} pressed")
                    if event.code == 304:
                        print("  -> A button!")
                    elif event.code == 305:
                        print("  -> B button!")
                    elif event.code == 307:
                        print("  -> X button!")
                    elif event.code == 308:
                        print("  -> Y button!")
            elif event.type == 3:  # Axis
                if event.code in [0, 1]:  # Left stick
                    print(f"Left stick axis {event.code}: {event.value}")
                elif event.code in [3, 4]:  # Right stick
                    print(f"Right stick axis {event.code}: {event.value}")
    except KeyboardInterrupt:
        print("\nTest stopped")
else:
    print("❌ No Xbox controller found")
    print("\nTroubleshooting:")
    print("1. Make sure controller is connected")
    print("2. Check: bluetoothctl info [MAC]")
    print("3. Try turning controller off and on again")