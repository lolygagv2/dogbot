import serial
import time

try:
    ser = serial.Serial("/dev/serial0", 9600)
    print("✅ DFPlayer Pro serial initialized. Sending play command...")
    ser.write(b'\x7E\xFF\x06\x03\x00\x00\x01\xFE\xF7\xEF')  # Play track 1
    time.sleep(5)
    ser.close()
except Exception as e:
    print("❌ Error:", e)