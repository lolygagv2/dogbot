#!/usr/bin/env python3
"""Test Xbox controller with motor integration"""

import struct
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import motor service
from services.motion.motor import MotorService

print("Xbox Controller Motor Test")
print("="*40)

# Check joystick device
if not os.path.exists('/dev/input/js0'):
    print("ERROR: /dev/input/js0 not found!")
    print("Run ./fix_xbox_controller.sh first")
    exit(1)

# Initialize motor
motor = MotorService()
motor.initialize()
print("Motor service initialized")

# Open joystick
js = open('/dev/input/js0', 'rb')
os.set_blocking(js.fileno(), False)
print("Controller connected")

print("\nControls:")
print("  Left stick Y: Forward/Backward")
print("  Left stick X: Turn left/right")
print("  B button: Stop")
print("\nPress Ctrl+C to exit\n")

# Event format
fmt = 'IhBB'
event_size = struct.calcsize(fmt)

# State
axes = {}
last_motor_update = 0

read_count = 0
try:
    while True:
        try:
            event = js.read(event_size)
            if event:
                read_count += 1
                if read_count % 100 == 0:
                    print(f"Read {read_count} events...")
                time_ms, value, evt_type, number = struct.unpack(fmt, event)
                
                # Skip init
                if evt_type & 0x80:
                    continue
                
                # Button
                if evt_type & 0x01:
                    print(f"Button {number} {'pressed' if value else 'released'}")
                    if number == 1 and value:  # B button pressed
                        motor.manual_drive('stop', 0)
                        print("STOP")
                
                # Axis
                elif evt_type & 0x02:
                    normalized = value / 32767.0
                    axes[number] = normalized
                    if abs(normalized) > 0.15:  # Show significant movements
                        print(f"Axis {number}: {normalized:.2f}")
                    
                    # Update motors (throttled)
                    if time.time() - last_motor_update > 0.1:
                        left_y = -axes.get(1, 0)  # Inverted
                        left_x = axes.get(0, 0)
                        
                        # Deadzone
                        if abs(left_y) < 0.15:
                            left_y = 0
                        if abs(left_x) < 0.15:
                            left_x = 0
                        
                        speed = 50
                        
                        # Movement logic
                        if abs(left_y) > abs(left_x):
                            if left_y > 0.15:
                                motor.manual_drive('forward', speed)
                                print(f"Forward {speed}")
                            elif left_y < -0.15:
                                motor.manual_drive('backward', speed)
                                print(f"Backward {speed}")
                            else:
                                motor.manual_drive('stop', 0)
                        elif abs(left_x) > 0.15:
                            if left_x < -0.15:
                                motor.manual_drive('left', speed)
                                print(f"Left {speed}")
                            elif left_x > 0.15:
                                motor.manual_drive('right', speed)
                                print(f"Right {speed}")
                        else:
                            motor.manual_drive('stop', 0)
                        
                        last_motor_update = time.time()
        
        except BlockingIOError:
            time.sleep(0.01)
        except Exception as e:
            print(f"Error: {e}")
            break

except KeyboardInterrupt:
    print("\nStopping...")

motor.manual_drive('stop', 0)
js.close()
print("Test complete!")
