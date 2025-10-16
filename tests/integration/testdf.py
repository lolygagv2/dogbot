#!/usr/bin/env python3
import serial
import time

ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=2)

# Clear buffers
ser.reset_input_buffer()
ser.reset_output_buffer()

print("Testing DFPlayer communication...")

# Some DFPlayers need an initialization sequence
init_commands = [
    b"AT\r\n",           # Basic AT
    b"AT+RESET\r\n",     # Reset command
    b"AT+VOL=10\r\n",    # Set volume
    b"AT+QUERY\r\n",     # Query status
]

for cmd in init_commands:
    print(f"\nSending: {cmd.decode().strip()}")
    ser.write(cmd)
    time.sleep(1)  # Longer delay after reset
    
    if ser.in_waiting:
        response = ser.read_all()
        print(f"Response: {response}")
    else:
        print("No response")

# Test if it can list files even without root files
ser.write(b"AT+QUERY\r\n")
time.sleep(0.5)
if ser.in_waiting:
    print(f"Query result: {ser.read_all()}")

ser.close()