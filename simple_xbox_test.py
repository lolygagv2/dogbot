import struct
import time

js = open('/dev/input/js0', 'rb')
print("Opened js0, reading for 5 seconds...")

start = time.time()
while time.time() - start < 5:
    try:
        data = js.read(8)
        if data:
            print(f"Got {len(data)} bytes")
    except:
        pass
    time.sleep(0.1)

js.close()
print("Done")
