#!/usr/bin/env python3
"""
Test Xbox controller using evdev library
"""

import evdev
import time
from evdev import InputDevice, categorize, ecodes

print("Xbox Controller evdev Test")
print("="*40)

# List all input devices
devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
xbox_device = None

print("Available input devices:")
for device in devices:
    print(f"  {device.path}: {device.name}")
    if "xbox" in device.name.lower() or "360" in device.name.lower():
        xbox_device = device
        print(f"    -> Found Xbox controller!")

if not xbox_device:
    print("\nNo Xbox controller found in evdev devices")
    print("Controller may not be properly recognized by the kernel")
    exit(1)

print(f"\nUsing: {xbox_device.name}")
print("Reading input for 10 seconds...\n")

start_time = time.time()
input_count = 0

try:
    while time.time() - start_time < 10:
        # Non-blocking read
        events = xbox_device.read()
        for event in events:
            if event.type == ecodes.EV_KEY:
                # Button event
                button_state = "pressed" if event.value == 1 else "released" if event.value == 0 else "held"
                print(f"Button {event.code} {button_state}")
                input_count += 1

            elif event.type == ecodes.EV_ABS:
                # Analog stick or trigger
                print(f"Axis {event.code}: {event.value}")
                input_count += 1

except BlockingIOError:
    # No events available (non-blocking mode)
    pass
except Exception as e:
    print(f"Error reading input: {e}")

print(f"\nTest complete! {input_count} inputs received")

if input_count > 0:
    print("✓ Controller working with evdev!")