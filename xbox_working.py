#!/usr/bin/env python3
"""Xbox controller reader that actually works"""
import struct
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.motion.motor import MotorService

print("Xbox Controller with Motors")
print("="*40)

# Initialize motor
motor = MotorService()
motor.initialize()

# Event device format
fmt = 'llHHI'
size = struct.calcsize(fmt)

# Open Xbox event device
event = open('/dev/input/event10', 'rb')
os.set_blocking(event.fileno(), False)
print("Controller connected!\n")
print("Controls: Left stick = move, B = stop\n")

# Event codes
BTN_A = 304
BTN_B = 305
ABS_X = 0
ABS_Y = 1

# State
axes = {0: 0, 1: 0}
last_update = 0

import time

try:
    while True:
        try:
            data = event.read(size)
            if data:
                tv_sec, tv_usec, type_, code, value = struct.unpack(fmt, data)
                
                if type_ == 1 and code == BTN_B and value == 1:  # B pressed
                    motor.manual_drive('stop', 0)
                    print("STOP")
                
                elif type_ == 3:  # Axis
                    if code in [ABS_X, ABS_Y]:
                        axes[code] = value / 32767.0
                        
                        # Update motors
                        if time.time() - last_update > 0.1:
                            y = -axes[1]  # Inverted
                            x = axes[0]
                            
                            if abs(y) > 0.2:
                                if y > 0:
                                    motor.manual_drive('forward', 50)
                                    print("Forward")
                                else:
                                    motor.manual_drive('backward', 50)
                                    print("Backward")
                            elif abs(x) > 0.2:
                                if x < 0:
                                    motor.manual_drive('left', 50)
                                    print("Left")
                                else:
                                    motor.manual_drive('right', 50) 
                                    print("Right")
                            else:
                                motor.manual_drive('stop', 0)
                            
                            last_update = time.time()
        except BlockingIOError:
            pass
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nStopping...")
    motor.manual_drive('stop', 0)
    event.close()
