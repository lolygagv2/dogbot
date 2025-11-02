#!/usr/bin/env python3
import struct, os, time

# Find Xbox event device
for i in range(20):
    path = f"/dev/input/event{i}"
    if os.path.exists(path):
        try:
            with open(f"/sys/class/input/event{i}/device/name", 'r') as f:
                if "Xbox Wireless Controller" in f.read():
                    print(f"Found Xbox at {path}")
                    event_path = path
                    break
        except:
            pass
else:
    print("No Xbox found!")
    exit(1)

# Open and read
event = open(event_path, 'rb')
print("Reading for 10 seconds... Press buttons!")

fmt = 'llHHI'
size = struct.calcsize(fmt)
start = time.time()
count = 0

while time.time() - start < 10:
    try:
        # Use select to avoid blocking
        import select
        r, _, _ = select.select([event], [], [], 0.01)
        if r:
            data = event.read(size)
            if data:
                _, _, type_, code, value = struct.unpack(fmt, data)
                if type_ == 1 and value:  # Button press
                    count += 1
                    print(f"Button {code} pressed!")
                elif type_ == 3 and abs(value) > 5000:  # Axis with threshold
                    print(f"Axis {code}: {value}")
    except:
        pass

event.close()
print(f"\nGot {count} button presses")
