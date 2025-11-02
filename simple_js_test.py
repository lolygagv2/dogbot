#!/usr/bin/env python3
import struct
import os
import time

print("Simple JS0 Test")

if not os.path.exists('/dev/input/js0'):
    print("No /dev/input/js0")
    exit(1)

js = open('/dev/input/js0', 'rb')
os.set_blocking(js.fileno(), False)
print("Opened js0, reading for 10 seconds...")
print("Press buttons or move sticks!")

fmt = 'IhBB'
size = struct.calcsize(fmt)
start = time.time()
count = 0

while time.time() - start < 10:
    try:
        data = js.read(size)
        if data:
            t, v, ty, n = struct.unpack(fmt, data)
            if not ty & 0x80:  # Skip init
                count += 1
                if ty & 0x01:
                    print(f"Button {n} {'pressed' if v else 'released'}")
                elif ty & 0x02:
                    norm = v / 32767.0
                    if abs(norm) > 0.15:
                        print(f"Axis {n}: {norm:.2f}")
    except:
        pass
    time.sleep(0.001)

js.close()
print(f"\nDone! {count} events detected")
