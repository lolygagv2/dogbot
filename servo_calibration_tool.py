#!/usr/bin/env python3
"""
Servo Calibration Tool for DogBot
Quick test tool to calibrate camera servos
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.hardware.servo_controller import ServoController
import time

def main():
    print("🔧 DogBot Servo Calibration Tool")
    print("="*50)

    try:
        servo = ServoController()

        if not servo.pca:
            print("❌ Servo controller not initialized!")
            return

        print("✅ Servo controller ready")
        print()

        while True:
            print("\n📋 Servo Calibration Options:")
            print("1. Test Pan Range (10° to 200°)")
            print("2. Test Pitch Range (30° to 120°)")
            print("3. Center Camera")
            print("4. Manual Pan Control")
            print("5. Manual Pitch Control")
            print("6. Scan Left/Right Test")
            print("7. Current Positions")
            print("0. Exit")

            choice = input("\nChoice (0-7): ").strip()

            if choice == '0':
                break
            elif choice == '1':
                test_pan_range(servo)
            elif choice == '2':
                test_pitch_range(servo)
            elif choice == '3':
                center_camera(servo)
            elif choice == '4':
                manual_pan_control(servo)
            elif choice == '5':
                manual_pitch_control(servo)
            elif choice == '6':
                scan_test(servo)
            elif choice == '7':
                show_current_positions(servo)
            else:
                print("❌ Invalid choice")

    except KeyboardInterrupt:
        print("\n⏹️ Calibration interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        print("🔧 Releasing servos...")
        if 'servo' in locals():
            servo.release_all_servos()

def test_pan_range(servo):
    """Test the full pan range"""
    print("\n🔄 Testing Pan Range...")
    positions = [10, 50, 100, 150, 200, 100]  # End at center

    for pos in positions:
        print(f"Pan to {pos}°...")
        servo.set_camera_pan(pos)
        time.sleep(1.5)

    print("✅ Pan range test complete")

def test_pitch_range(servo):
    """Test the full pitch range"""
    print("\n↕️ Testing Pitch Range...")
    positions = [30, 45, 55, 80, 120, 55]  # End at center

    for pos in positions:
        print(f"Pitch to {pos}°...")
        servo.set_camera_pitch(pos)
        time.sleep(1.5)

    print("✅ Pitch range test complete")

def center_camera(servo):
    """Center the camera"""
    print("\n📹 Centering camera...")
    servo.center_camera()
    print(f"✅ Camera centered (Pan: {servo.current_pan}°, Pitch: {servo.current_pitch}°)")

def manual_pan_control(servo):
    """Manual pan control"""
    print("\n🎮 Manual Pan Control (10-200°, 'q' to quit)")

    while True:
        try:
            value = input("Pan angle (10-200): ").strip()
            if value.lower() == 'q':
                break

            angle = float(value)
            if 10 <= angle <= 200:
                servo.set_camera_pan(angle)
                print(f"✅ Pan set to {angle}°")
            else:
                print("❌ Angle must be between 10-200°")

        except ValueError:
            print("❌ Invalid number")
        except KeyboardInterrupt:
            break

def manual_pitch_control(servo):
    """Manual pitch control"""
    print("\n🎮 Manual Pitch Control (30-120°, 'q' to quit)")

    while True:
        try:
            value = input("Pitch angle (30-120): ").strip()
            if value.lower() == 'q':
                break

            angle = float(value)
            if 30 <= angle <= 120:
                servo.set_camera_pitch(angle)
                print(f"✅ Pitch set to {angle}°")
            else:
                print("❌ Angle must be between 30-120°")

        except ValueError:
            print("❌ Invalid number")
        except KeyboardInterrupt:
            break

def scan_test(servo):
    """Test the scan function"""
    print("\n👀 Testing scan function...")
    servo.scan_left_right(cycles=2, delay=1.0)
    print("✅ Scan test complete")

def show_current_positions(servo):
    """Show current servo positions"""
    print(f"\n📍 Current Positions:")
    print(f"   Pan: {servo.current_pan}°")
    print(f"   Pitch: {servo.current_pitch}°")

if __name__ == "__main__":
    main()