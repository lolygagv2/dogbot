#!/usr/bin/env python3
"""Debug camera control issues"""

import time
import struct
import os

# Test just reading the joystick values
js_device = '/dev/input/js0'

if not os.path.exists(js_device):
    print(f"No joystick at {js_device}")
    exit(1)

device = open(js_device, 'rb')
print("Reading raw joystick values...")
print("Move the RIGHT STICK to see values")
print("Press Ctrl+C to stop\n")

right_x = 0
right_y = 0

try:
    while True:
        event_data = device.read(8)
        if event_data:
            timestamp, value, event_type, number = struct.unpack('IhBB', event_data)

            # Only show axis events for right stick
            if event_type == 0x02:  # Axis event
                if number == 3:  # Right stick X
                    right_x = value / 32767.0
                    print(f"Right X: {right_x:6.3f} (raw: {value:6d})")
                elif number == 4:  # Right stick Y
                    right_y = value / 32767.0
                    print(f"Right Y: {right_y:6.3f} (raw: {value:6d})")

except KeyboardInterrupt:
    print("\nStopped")
finally:
    device.close()