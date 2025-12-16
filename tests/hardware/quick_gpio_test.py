#!/usr/bin/env python3
"""
Quick GPIO state test - check if encoder pins change when motors run
"""
import subprocess
import time

def read_gpio(pin):
    try:
        result = subprocess.run(['gpioget', 'gpiochip0', str(pin)],
                               capture_output=True, text=True, timeout=0.1)
        return int(result.stdout.strip()) if result.returncode == 0 else -1
    except:
        return -1

print("=== Quick GPIO State Test ===")
print("Reading encoder GPIO states for 5 seconds")
print("If encoders work, you should see changing 0/1 values")
print()

# Test pins: LEFT(4,23) RIGHT(5,6)
pins = {'L_A': 4, 'L_B': 23, 'R_A': 5, 'R_B': 6}

print("Time | L_A L_B | R_A R_B")
print("-" * 25)

for i in range(50):  # 5 seconds at 10Hz
    states = {name: read_gpio(pin) for name, pin in pins.items()}

    print(f"{i/10:4.1f} | {states['L_A']:2d} {states['L_B']:2d} | {states['R_A']:2d} {states['R_B']:2d}")

    time.sleep(0.1)

print("\nIf all values stayed 0 or 1 (no changes), encoders may have issues")
print("If values changed randomly, encoders are working!")