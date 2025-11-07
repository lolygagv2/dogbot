#!/usr/bin/env python3
"""Emergency stop for DFPlayer Pro audio"""

import serial
import time

try:
    # Connect to DFPlayer
    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
    time.sleep(1)

    # Send stop commands
    commands = [
        'AT+STOP',      # Stop playback
        'AT+PAUSE',     # Pause playback
        'AT+VOL=0',     # Mute volume
        'AT+AMP=OFF',   # Turn off amplifier
    ]

    for cmd in commands:
        full_cmd = f"{cmd}\r\n"
        print(f"Sending: {cmd}")
        ser.write(full_cmd.encode())
        time.sleep(0.1)

        # Read any response
        if ser.in_waiting > 0:
            response = ser.read(100).decode('utf-8', errors='ignore')
            print(f"Response: {response.strip()}")

    ser.close()
    print("\n✅ DFPlayer stopped")

except Exception as e:
    print(f"Error: {e}")