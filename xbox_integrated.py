#!/usr/bin/env python3
"""
Xbox Controller Integration - Simplified version that properly loads all services
"""

import os
import sys
import time

# Fix Python path for sudo execution
sys.path.insert(0, '/home/morgan/dogbot')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import the working Xbox controller
from xbox_controller_final import XboxControllerFinal

# Try to import and test each service
print("Testing service availability...")
print("-" * 40)

# Test servo
try:
    from core.hardware.servo_controller import ServoController
    servo = ServoController()
    print("✓ Servo controller available")
except Exception as e:
    print(f"✗ Servo: {e}")

# Test dispenser
try:
    from services.reward.dispenser import get_dispenser_service
    dispenser = get_dispenser_service()
    print("✓ Dispenser service available")
except Exception as e:
    print(f"✗ Dispenser: {e}")

# Test sound
try:
    from services.media.sfx import get_sfx_service
    sfx = get_sfx_service()
    print("✓ Sound effects available")
except Exception as e:
    print(f"✗ Sound: {e}")

print("-" * 40)

# Run the controller
if __name__ == "__main__":
    controller = XboxControllerFinal()

    # Override the initialization to ensure services are loaded
    if controller.initialize():
        try:
            controller.start()
        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            controller.stop()
    else:
        print("Failed to initialize controller")