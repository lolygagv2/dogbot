#!/usr/bin/env python3
"""
Quick Xbox controller test - 10 second non-blocking read
"""

import pygame
import time
import sys

print("Xbox Controller Quick Test (10 seconds)")
print("="*40)

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No controller detected!")
    sys.exit(1)

controller = pygame.joystick.Joystick(0)
controller.init()

print(f"Controller: {controller.get_name()}")
print(f"Axes: {controller.get_numaxes()}")
print(f"Buttons: {controller.get_numbuttons()}")
print("\nPress buttons or move sticks!\n")

start_time = time.time()
input_received = False

while time.time() - start_time < 10:
    for event in pygame.event.get():
        if event.type == pygame.JOYBUTTONDOWN:
            print(f"Button {event.button} pressed!")
            input_received = True
            if event.button == 0:
                print("  -> A button!")
            elif event.button == 1:
                print("  -> B button!")
            elif event.button == 2:
                print("  -> X button!")
            elif event.button == 3:
                print("  -> Y button!")

        elif event.type == pygame.JOYBUTTONUP:
            print(f"Button {event.button} released")

        elif event.type == pygame.JOYAXISMOTION:
            if abs(event.value) > 0.3:  # Deadzone
                axis_names = {0: "L-X", 1: "L-Y", 3: "R-X", 4: "R-Y"}
                name = axis_names.get(event.axis, f"Axis{event.axis}")
                print(f"{name}: {event.value:.2f}")
                input_received = True

    time.sleep(0.01)

if input_received:
    print("\n✓ Success! Controller input detected!")
else:
    print("\nNo input detected, but controller stayed connected")

pygame.quit()
print("Test complete!")