#!/usr/bin/env python3
"""
Test Xbox controller with pygame (now working with xpadneo)
"""

import pygame
import sys

print("="*50)
print("XBOX CONTROLLER TEST - PYGAME")
print("="*50)

pygame.init()
pygame.joystick.init()

count = pygame.joystick.get_count()
print(f"Controllers detected: {count}")

if count == 0:
    print("No controller found!")
    sys.exit(1)

controller = pygame.joystick.Joystick(0)
controller.init()

print(f"Controller name: {controller.get_name()}")
print(f"Axes: {controller.get_numaxes()}")
print(f"Buttons: {controller.get_numbuttons()}")
print(f"Hats: {controller.get_numhats()}")

print("\nControls:")
print("  A = Button 0")
print("  B = Button 1")
print("  X = Button 2")
print("  Y = Button 3")
print("  Left stick = Axes 0,1")
print("  Right stick = Axes 3,4")

print("\nPress buttons or move sticks (Ctrl+C to exit)...")

clock = pygame.time.Clock()

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                print(f"Button {event.button} pressed")
                if event.button == 0:
                    print("  -> A button!")
                elif event.button == 1:
                    print("  -> B button!")
                elif event.button == 2:
                    print("  -> X button!")
                elif event.button == 3:
                    print("  -> Y button!")

            elif event.type == pygame.JOYAXISMOTION:
                if abs(event.value) > 0.1:  # Deadzone
                    axis_names = {
                        0: "Left X",
                        1: "Left Y",
                        3: "Right X",
                        4: "Right Y",
                        2: "Left Trigger",
                        5: "Right Trigger"
                    }
                    name = axis_names.get(event.axis, f"Axis {event.axis}")
                    print(f"{name}: {event.value:.2f}")

        clock.tick(30)

except KeyboardInterrupt:
    print("\nTest complete!")

pygame.quit()