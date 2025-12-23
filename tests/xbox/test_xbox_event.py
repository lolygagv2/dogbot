#!/usr/bin/env python3
"""Test Xbox using event device instead of js device"""
import struct
import os
import time

print("Xbox Event Device Test")
print("="*40)

event_dev = '/dev/input/event10'
if not os.path.exists(event_dev):
    print(f"No {event_dev}")
    exit(1)

# Event format for /dev/input/eventX
# struct input_event {
#     struct timeval time;
#     __u16 type;
#     __u16 code;
#     __s32 value;
# };
fmt = 'llHHI'  # long long unsigned_short unsigned_short unsigned_int
size = struct.calcsize(fmt)

event = open(event_dev, 'rb')
os.set_blocking(event.fileno(), False)
print(f"Opened {event_dev}")
print("Press buttons or move sticks for 10 seconds!\n")

start = time.time()
count = 0

# Event types
EV_KEY = 0x01
EV_ABS = 0x03

while time.time() - start < 10:
    try:
        data = event.read(size)
        if data:
            tv_sec, tv_usec, type_, code, value = struct.unpack(fmt, data)
            
            if type_ == EV_KEY:  # Button
                count += 1
                print(f"Button code={code} {'pressed' if value else 'released'}")
            elif type_ == EV_ABS:  # Axis
                count += 1
                if value != 0:  # Only show non-zero
                    print(f"Axis code={code} value={value}")
    except:
        pass
    time.sleep(0.001)

event.close()
print(f"\nDone! {count} events detected")
