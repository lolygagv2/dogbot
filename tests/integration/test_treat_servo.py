#!/usr/bin/env python3
"""
Test script for treat dispenser servo (carousel rotation)
Tests the winch servo on PCA9685 channel 2
"""

import time
import board
import busio
from adafruit_pca9685 import PCA9685

def pulse_to_duty(pulse_us):
    """Convert pulse width in microseconds to duty cycle"""
    return int((pulse_us / 20000.0) * 0xFFFF)

def test_treat_dispenser():
    """Test the treat dispenser carousel servo"""
    print("=" * 50)
    print("TREAT DISPENSER SERVO TEST")
    print("=" * 50)
    print("This will test the carousel rotation servo on channel 2")
    print()

    try:
        # Initialize I2C and PCA9685
        print("Initializing PCA9685...")
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50  # Standard servo frequency

        # Get channel 2 (winch/carousel)
        carousel_servo = pca.channels[2]

        print("✅ PCA9685 initialized successfully")
        print("📍 Using Channel 2 for carousel servo")
        print()

        # Test options
        while True:
            print("\n" + "=" * 50)
            print("TREAT DISPENSER TEST MENU")
            print("=" * 50)
            print("1. Single forward rotation (dispense 1 treat)")
            print("2. Single backward rotation")
            print("3. Burst sequence (2 treats)")
            print("4. Continuous forward (hold for 3 seconds)")
            print("5. Test different pulse values")
            print("6. Stop servo (release)")
            print("7. Exit")
            print()

            choice = input("Enter choice (1-7): ").strip()

            if choice == '1':
                print("\n🍖 Dispensing single treat (forward rotation)...")
                # Forward rotation for 0.08 seconds (from ServoController)
                carousel_servo.duty_cycle = pulse_to_duty(1700)
                time.sleep(0.08)
                carousel_servo.duty_cycle = 0  # Stop
                print("✅ Single treat dispensed")

            elif choice == '2':
                print("\n⬅️  Backward rotation...")
                carousel_servo.duty_cycle = pulse_to_duty(1300)
                time.sleep(0.08)
                carousel_servo.duty_cycle = 0  # Stop
                print("✅ Backward rotation complete")

            elif choice == '3':
                print("\n🍖🍖 Dispensing 2 treats (burst sequence)...")
                for i in range(2):
                    print(f"  Burst {i+1}/2")
                    carousel_servo.duty_cycle = pulse_to_duty(1700)
                    time.sleep(0.12)
                    carousel_servo.duty_cycle = 0
                    if i < 1:
                        time.sleep(0.3)  # Pause between bursts
                print("✅ 2 treats dispensed")

            elif choice == '4':
                print("\n🔄 Continuous forward rotation for 3 seconds...")
                print("   Watch the carousel rotation speed")
                carousel_servo.duty_cycle = pulse_to_duty(1700)
                time.sleep(3.0)
                carousel_servo.duty_cycle = 0
                print("✅ Continuous rotation complete")

            elif choice == '5':
                print("\n⚙️  Custom pulse test")
                print("   Typical values:")
                print("   - 1500: Stop/neutral")
                print("   - 1700: Forward")
                print("   - 1300: Backward")
                print("   - Range: 500-2500")

                try:
                    pulse = int(input("Enter pulse value (500-2500): "))
                    if 500 <= pulse <= 2500:
                        duration = float(input("Duration in seconds (0.1-5.0): "))
                        if 0.1 <= duration <= 5.0:
                            print(f"Testing pulse={pulse} for {duration}s...")
                            carousel_servo.duty_cycle = pulse_to_duty(pulse)
                            time.sleep(duration)
                            carousel_servo.duty_cycle = 0
                            print("✅ Custom test complete")
                        else:
                            print("❌ Duration out of range")
                    else:
                        print("❌ Pulse value out of range")
                except ValueError:
                    print("❌ Invalid input")

            elif choice == '6':
                print("\n🛑 Stopping servo...")
                carousel_servo.duty_cycle = 0
                print("✅ Servo released")

            elif choice == '7':
                print("\n👋 Exiting...")
                carousel_servo.duty_cycle = 0  # Make sure servo is stopped
                break

            else:
                print("❌ Invalid choice")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure:")
        print("- The PCA9685 is properly connected via I2C")
        print("- The servo is connected to channel 2")
        print("- Power is supplied to the servo")
        return False

    finally:
        # Always release servo on exit
        try:
            carousel_servo.duty_cycle = 0
            print("🛑 Servo released")
        except:
            pass

    return True

if __name__ == "__main__":
    print("Starting Treat Dispenser Servo Test...")
    print("Press Ctrl+C at any time to emergency stop")
    print()

    try:
        success = test_treat_dispenser()
        if success:
            print("\n✅ Test completed successfully!")
        else:
            print("\n⚠️  Test encountered issues")
    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        print("Servo stopped.")