#!/usr/bin/env python3
"""
Direct GPIO test with joystick input - bypasses motor controller entirely
This test MUST work if the motors can be controlled at all
"""
import time
import struct
from gpiozero import PWMOutputDevice, DigitalOutputDevice

# GPIO Pin mapping from hardware_specs.md
LEFT_IN1 = 17   # Left motor direction
LEFT_IN2 = 18   # Left motor direction
RIGHT_IN3 = 27  # Right motor direction
RIGHT_IN4 = 22  # Right motor direction
LEFT_ENA = 13   # Left motor PWM
RIGHT_ENB = 19  # Right motor PWM

# Config
PWM_MIN = 35    # Minimum PWM to overcome static friction
PWM_MAX = 70    # Maximum safe PWM
DEADZONE = 0.15

print("=" * 60)
print("DIRECT GPIO + JOYSTICK TEST")
print("=" * 60)
print(f"Left motor:  IN1={LEFT_IN1}, IN2={LEFT_IN2}, ENA={LEFT_ENA}")
print(f"Right motor: IN3={RIGHT_IN3}, IN4={RIGHT_IN4}, ENB={RIGHT_ENB}")
print(f"PWM range: {PWM_MIN}% - {PWM_MAX}%")
print()

# Initialize GPIO for motor direction
print("Initializing GPIO...")
left_in1 = DigitalOutputDevice(LEFT_IN1)
left_in2 = DigitalOutputDevice(LEFT_IN2)
right_in3 = DigitalOutputDevice(RIGHT_IN3)
right_in4 = DigitalOutputDevice(RIGHT_IN4)

# Initialize PWM for motor speed
left_en = PWMOutputDevice(LEFT_ENA, frequency=1000)
right_en = PWMOutputDevice(RIGHT_ENB, frequency=1000)

# Set direction: FORWARD (matching proper_pid_motor_controller.py)
print("Setting motor direction: FORWARD")
# Left motor: IN1=0, IN2=1 for forward
left_in1.off()
left_in2.on()
# Right motor: IN3=1, IN4=0 for forward
right_in3.on()
right_in4.off()
print("Direction pins set (L: IN1=0/IN2=1, R: IN3=1/IN4=0)")

# Test motors work at all
print("\nQuick motor test (50% PWM for 0.5 seconds)...")
left_en.value = 0.5
right_en.value = 0.5
time.sleep(0.5)
left_en.value = 0
right_en.value = 0
print("Did motors spin? If not, check wiring!")
time.sleep(1)

# Open joystick
print("\nOpening joystick...")
import os
try:
    js_fd = os.open('/dev/input/js0', os.O_RDONLY | os.O_NONBLOCK)
    js = os.fdopen(js_fd, 'rb', buffering=0)
    print("Joystick ready")
except Exception as e:
    print(f"ERROR: Cannot open joystick: {e}")
    left_en.close()
    right_en.close()
    exit(1)

print("\n" + "=" * 60)
print("Move LEFT STICK forward to control motors")
print("Motors should speed up gradually as you push further")
print("Press Ctrl+C to stop")
print("=" * 60 + "\n")

left_y = 0.0
last_print = 0

try:
    while True:
        # Read joystick
        try:
            data = js.read(8)
            if data:
                time_ms, value, event_type, axis = struct.unpack('IhBB', data)
                if event_type & 0x02 and axis == 1:  # Left Y axis
                    left_y = -value / 32767.0  # Invert for forward = positive
        except BlockingIOError:
            pass

        # Calculate PWM
        if left_y < DEADZONE:
            pwm_pct = 0
        else:
            # Map joystick (deadzone to 1.0) to PWM (PWM_MIN to PWM_MAX)
            normalized = (left_y - DEADZONE) / (1.0 - DEADZONE)
            pwm_pct = PWM_MIN + normalized * (PWM_MAX - PWM_MIN)

        # Apply PWM
        if pwm_pct > 0:
            pwm_val = pwm_pct / 100.0
            left_en.value = pwm_val
            right_en.value = pwm_val
        else:
            left_en.value = 0
            right_en.value = 0

        # Print every 200ms
        now = time.time()
        if now - last_print > 0.2:
            last_print = now
            status = "RUNNING" if pwm_pct > 0 else "STOPPED"
            print(f"Stick: {left_y:5.2f} | PWM: {pwm_pct:5.1f}% | {status}")

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\n\nStopping motors...")

finally:
    left_en.value = 0
    right_en.value = 0
    left_en.close()
    right_en.close()
    left_in1.close()
    left_in2.close()
    right_in3.close()
    right_in4.close()
    js.close()
    print("Done")
