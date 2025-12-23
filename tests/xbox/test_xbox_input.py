#!/usr/bin/env python3
"""Test Xbox controller input detection only"""

import struct

def test_xbox_input():
    """Test raw Xbox controller input"""
    device_path = '/dev/input/js0'

    try:
        print(f"Opening {device_path}...")
        device = open(device_path, 'rb')
        print("✅ Xbox controller device opened")

        print("Move Xbox controller sticks and press buttons...")
        print("Press Ctrl+C to stop")

        while True:
            # Read raw joystick events
            event_data = device.read(8)
            if len(event_data) != 8:
                continue

            timestamp, value, event_type, number = struct.unpack('IhBB', event_data)

            if event_type == 2:  # Axis events
                normalized = value / 32767.0
                print(f"AXIS {number}: {normalized:.3f} (raw: {value})")
            elif event_type == 1:  # Button events
                print(f"BUTTON {number}: {'PRESSED' if value else 'RELEASED'}")

    except KeyboardInterrupt:
        print("\nStopping...")
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        if 'device' in locals():
            device.close()

if __name__ == "__main__":
    test_xbox_input()
