#!/usr/bin/env python3
import struct
import os

js = open('/dev/input/js1', 'rb')
print("Opened js1 - PRESS BUTTONS NOW!")

fmt = 'IhBB'
size = 8
count = 0

for i in range(100):  # Read 100 events max
    data = js.read(size)
    if data:
        t, v, ty, n = struct.unpack(fmt, data)
        if ty == 1:  # Button
            count += 1
            print(f"Button {n} {'pressed' if v else 'released'}")
            if count > 5:
                break

js.close()
print("Done!")
