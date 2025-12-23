#!/usr/bin/env python3
import os
import struct
import fcntl

print("Xbox Controller Test - Non-blocking")
print("="*40)

# Open js1 non-blocking
fd = os.open('/dev/input/js1', os.O_RDONLY | os.O_NONBLOCK)
print("Device opened!")
print("\n*** PRESS BUTTONS NOW! ***\n")

fmt = 'IhBB'
count = 0

import time
start = time.time()

while time.time() - start < 10:
    try:
        data = os.read(fd, 8)
        if len(data) == 8:
            t, v, ty, n = struct.unpack(fmt, data)
            if ty == 0x01:  # Button event
                count += 1
                print(f"BUTTON {n}: {'PRESSED' if v else 'RELEASED'}")
            elif ty == 0x02:  # Axis event
                if abs(v) > 10000:
                    print(f"AXIS {n}: {v}")
    except OSError:
        pass
    time.sleep(0.01)

os.close(fd)
print(f"\nDetected {count} button events in 10 seconds")
