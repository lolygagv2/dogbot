#!/usr/bin/env python3
"""Direct DFPlayer test - bypasses all services"""

import serial
import time

print("Direct DFPlayer Pro test")
print("-" * 40)

try:
    # Connect to DFPlayer
    ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)
    time.sleep(2)

    print("Connected to /dev/ttyAMA0")

    # Send status query
    print("\n1. Querying DFPlayer status...")
    ser.write(b"AT+QUERY=STATUS\r\n")
    time.sleep(0.5)
    if ser.in_waiting > 0:
        response = ser.read(100).decode('utf-8', errors='ignore')
        print(f"Response: {response.strip()}")

    # Set volume
    print("\n2. Setting volume to 20...")
    ser.write(b"AT+VOL=20\r\n")
    time.sleep(0.5)
    if ser.in_waiting > 0:
        response = ser.read(100).decode('utf-8', errors='ignore')
        print(f"Response: {response.strip()}")

    # Play track 8
    print("\n3. Playing track 8...")
    ser.write(b"AT+PLAYNUM=8\r\n")
    time.sleep(0.5)
    if ser.in_waiting > 0:
        response = ser.read(100).decode('utf-8', errors='ignore')
        print(f"Response: {response.strip()}")

    print("\n4. Waiting 3 seconds...")
    time.sleep(3)

    # Stop
    print("\n5. Stopping playback...")
    ser.write(b"AT+STOP\r\n")
    time.sleep(0.5)
    if ser.in_waiting > 0:
        response = ser.read(100).decode('utf-8', errors='ignore')
        print(f"Response: {response.strip()}")

    ser.close()
    print("\n✅ Test complete!")
    print("\nIf you heard audio, DFPlayer is working.")
    print("If not:")
    print("- Check power to DFPlayer")
    print("- Check speaker connections")
    print("- Check SD card is inserted")
    print("- Check audio files exist on SD card")

except Exception as e:
    print(f"Error: {e}")
    print("\nPossible issues:")
    print("- Serial port /dev/ttyAMA0 not available")
    print("- DFPlayer not connected")
    print("- Permission issues (try with sudo)")