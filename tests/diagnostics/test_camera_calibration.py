#!/usr/bin/env python3
"""
Camera Servo Calibration Test
Test camera movements to diagnose direction issues
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.motion.pan_tilt import get_pantilt_service

def test_camera_movements():
    """Test camera movements to determine correct mapping"""

    print("🎥 Camera Servo Calibration Test")
    print("=" * 50)

    # Get pan/tilt service
    pantilt = get_pantilt_service()

    # Initialize if needed
    if not pantilt.servo_initialized:
        print("Initializing pan/tilt service...")
        if not pantilt.initialize():
            print("❌ Failed to initialize pan/tilt service")
            return
        time.sleep(1)

    print("✅ Pan/tilt service ready")
    print()

    # Test sequence
    test_sequence = [
        ("Center position", 90, 90),
        ("Far left (should pan left)", 30, 90),
        ("Center", 90, 90),
        ("Far right (should pan right)", 150, 90),
        ("Center", 90, 90),
        ("Looking up (should tilt up)", 90, 60),
        ("Center", 90, 90),
        ("Looking down (should tilt down)", 90, 120),
        ("Center", 90, 90)
    ]

    print("🎯 Testing camera movements:")
    print("Watch the camera and note if movements match descriptions")
    print()

    for description, pan, tilt in test_sequence:
        print(f"Moving to: {description} (pan={pan}°, tilt={tilt}°)")
        pantilt.move_camera(pan=pan, tilt=tilt)
        time.sleep(2)  # Wait for movement and user observation

    print()
    print("📋 Diagnosis Questions:")
    print("1. Did 'Far left' actually move the camera LEFT?")
    print("2. Did 'Far right' actually move the camera RIGHT?")
    print("3. Did 'Looking up' actually tilt the camera UP?")
    print("4. Did 'Looking down' actually tilt the camera DOWN?")
    print()
    print("If any directions were wrong, we need to update the mapping.")

    # Current mapping in HTML
    print("Current HTML mapping:")
    print("  Up button → tilt=60°")
    print("  Down button → tilt=120°")
    print("  Left button → pan=50°")
    print("  Right button → pan=130°")

if __name__ == "__main__":
    test_camera_movements()