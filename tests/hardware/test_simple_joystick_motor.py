#!/usr/bin/env python3
"""
SIMPLEST POSSIBLE joystick to motor test.
No abstractions, no motor controller, just joystick → GPIO.
"""
import os
import time
import struct
from gpiozero import OutputDevice, PWMOutputDevice

# Pins from pins.py
LEFT_IN1 = 17
LEFT_IN2 = 18
LEFT_ENA = 13
RIGHT_IN3 = 27
RIGHT_IN4 = 22
RIGHT_ENB = 19

print("=== SIMPLE JOYSTICK → MOTOR TEST ===")
print("No motor controller, direct GPIO only")
print()

# Init GPIO
left_in1 = OutputDevice(LEFT_IN1)
left_in2 = OutputDevice(LEFT_IN2)
left_ena = PWMOutputDevice(LEFT_ENA, frequency=1000)
right_in3 = OutputDevice(RIGHT_IN3)
right_in4 = OutputDevice(RIGHT_IN4)
right_enb = PWMOutputDevice(RIGHT_ENB, frequency=1000)

# Set forward direction
left_in1.off()
left_in2.on()
right_in3.on()
right_in4.off()

# Start stopped
left_ena.value = 0
right_enb.value = 0
print("GPIO ready, motors stopped")

# Open joystick
js_fd = os.open('/dev/input/js0', os.O_RDONLY | os.O_NONBLOCK)
js = os.fdopen(js_fd, 'rb', buffering=0)
print("Joystick ready")

print()
print("Move LEFT STICK forward slowly")
print("Watch PWM values - should be gradual")
print("Ctrl+C to stop")
print()

left_y = 0.0
DEADZONE = 0.15
LEFT_MULT = 0.5   # Left motor multiplier (it's overpowered)
RIGHT_MULT = 1.0
last_print = 0

try:
    while True:
        # Read joystick
        try:
            data = js.read(8)
            if data:
                _, value, event_type, axis = struct.unpack('IhBB', data)
                if event_type & 0x02 and axis == 1:
                    left_y = -value / 32767.0
        except BlockingIOError:
            pass

        # Calculate PWM (0-70% range)
        if abs(left_y) < DEADZONE:
            left_pwm = 0
            right_pwm = 0
        else:
            # Map (deadzone to 1.0) to (0% to 70%)
            magnitude = (abs(left_y) - DEADZONE) / (1.0 - DEADZONE)
            base_pwm = magnitude * 70  # 0-70%

            left_pwm = base_pwm * LEFT_MULT
            right_pwm = base_pwm * RIGHT_MULT

        # Apply to motors
        left_ena.value = left_pwm / 100.0
        right_enb.value = right_pwm / 100.0

        # Print
        now = time.time()
        if now - last_print > 0.1:
            last_print = now
            print(f"Stick={left_y:5.2f} → L_PWM={left_pwm:5.1f}% R_PWM={right_pwm:5.1f}%")

        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    left_ena.value = 0
    right_enb.value = 0
    left_in1.off()
    left_in2.off()
    right_in3.off()
    right_in4.off()
    left_in1.close()
    left_in2.close()
    left_ena.close()
    right_in3.close()
    right_in4.close()
    right_enb.close()
    print("Done")
