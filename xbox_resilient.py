#!/usr/bin/env python3
"""Xbox controller that handles disconnections"""
import struct
import os
import sys
import time
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from services.motion.motor import MotorService

MAC = "AC:8E:BD:4A:0F:97"

def find_xbox_event():
    """Find the Xbox event device"""
    for i in range(20):
        event_path = f"/dev/input/event{i}"
        if os.path.exists(event_path):
            try:
                name_path = f"/sys/class/input/event{i}/device/name"
                if os.path.exists(name_path):
                    with open(name_path, 'r') as f:
                        if "Xbox Wireless Controller" in f.read():
                            return event_path
            except:
                pass
    return None

def ensure_connected():
    """Ensure Xbox controller is connected"""
    result = subprocess.run(
        f"bluetoothctl info {MAC} | grep 'Connected: yes'",
        shell=True, capture_output=True
    )
    
    if result.returncode != 0:
        print("Controller disconnected, reconnecting...")
        subprocess.run(f"bluetoothctl connect {MAC}", shell=True, capture_output=True)
        time.sleep(2)
    
    event_path = find_xbox_event()
    if event_path:
        print(f"Controller at {event_path}")
        return event_path
    return None

print("Xbox Resilient Controller")
print("="*40)

# Initialize motor
motor = MotorService()
motor.initialize()
print("Motors ready")

# Event format
fmt = 'llHHI'
size = struct.calcsize(fmt)

# Event codes
BTN_B = 305
ABS_X = 0
ABS_Y = 1

# State
axes = {0: 0, 1: 0}
last_update = 0
event_file = None

print("\nStarting controller loop...")
print("Move left stick to drive, press B to stop")
print("Ctrl+C to exit\n")

try:
    while True:
        # Ensure connected
        if event_file is None:
            event_path = ensure_connected()
            if event_path:
                try:
                    event_file = open(event_path, 'rb')
                    os.set_blocking(event_file.fileno(), False)
                    print("Controller active!")
                except Exception as e:
                    print(f"Can't open device: {e}")
                    event_file = None
                    time.sleep(2)
                    continue
            else:
                print("Waiting for controller...")
                time.sleep(2)
                continue
        
        # Read events
        try:
            data = event_file.read(size)
            if data:
                tv_sec, tv_usec, type_, code, value = struct.unpack(fmt, data)
                
                if type_ == 1:  # Button
                    if code == BTN_B and value == 1:
                        motor.manual_drive('stop', 0)
                        print("STOP (B button)")
                
                elif type_ == 3:  # Axis
                    if code in [ABS_X, ABS_Y]:
                        axes[code] = value / 32767.0
                        
                        # Update motors (throttled)
                        now = time.time()
                        if now - last_update > 0.1:
                            y = -axes[1]  # Inverted
                            x = axes[0]
                            
                            # Simple threshold-based control
                            if abs(y) > 0.3:
                                if y > 0:
                                    motor.manual_drive('forward', 50)
                                    print(f"Forward (Y={y:.2f})")
                                else:
                                    motor.manual_drive('backward', 50)
                                    print(f"Backward (Y={y:.2f})")
                            elif abs(x) > 0.3:
                                if x < 0:
                                    motor.manual_drive('left', 50)
                                    print(f"Left (X={x:.2f})")
                                else:
                                    motor.manual_drive('right', 50)
                                    print(f"Right (X={x:.2f})")
                            
                            last_update = now
                            
        except OSError as e:
            if e.errno == 19:  # No such device
                print("Controller disconnected!")
                if event_file:
                    event_file.close()
                event_file = None
                time.sleep(1)
        except BlockingIOError:
            pass  # No data available
        
        time.sleep(0.01)

except KeyboardInterrupt:
    print("\nShutting down...")
    motor.manual_drive('stop', 0)
    if event_file:
        event_file.close()
    print("Done!")
