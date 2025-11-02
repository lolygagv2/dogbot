#!/usr/bin/env python3
"""
Minimal Xbox controller test with non-blocking input
"""

import pygame
import sys
import time

print("Xbox Controller Minimal Test")
print("="*40)

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller found!")
    sys.exit(1)

controller = pygame.joystick.Joystick(0)
controller.init()

print(f"Controller: {controller.get_name()}")
print(f"ID: {controller.get_instance_id()}")
print("\nReading for 10 seconds (non-blocking)...")
print("Press some buttons!\n")

start_time = time.time()
button_pressed = False

# Non-blocking event loop
while time.time() - start_time < 10:
    # Process events without blocking
    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:
            print(f"Button {event.button} pressed!")
            button_pressed = True
        elif event.type == pygame.JOYBUTTONUP:
            print(f"Button {event.button} released")
        elif event.type == pygame.JOYAXISMOTION:
            if abs(event.value) > 0.5:  # Only significant movements
                print(f"Axis {event.axis} moved to {event.value:.2f}")

    # Small delay to prevent CPU spinning
    time.sleep(0.01)

if button_pressed:
    print("\n✓ Controller input detected successfully!")
else:
    print("\nNo buttons pressed, but controller stayed connected!")

# Check if still connected
if controller.get_init():
    print("Controller still initialized ✓")

pygame.quit()
print("Test complete!")