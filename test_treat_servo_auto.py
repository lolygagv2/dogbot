#!/usr/bin/env python3
"""
Automated test for treat dispenser servo
Runs through basic servo operations without user input
"""

import time
import board
import busio
from adafruit_pca9685 import PCA9685

def pulse_to_duty(pulse_us):
    """Convert pulse width in microseconds to duty cycle"""
    return int((pulse_us / 20000.0) * 0xFFFF)

def test_treat_dispenser_auto():
    """Automated test of treat dispenser carousel servo"""
    print("=" * 50)
    print("AUTOMATED TREAT DISPENSER SERVO TEST")
    print("=" * 50)
    print("Testing carousel servo on PCA9685 channel 2")
    print()

    try:
        # Initialize I2C and PCA9685
        print("🔧 Initializing PCA9685...")
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50  # Standard servo frequency

        # Get channel 2 (winch/carousel)
        carousel_servo = pca.channels[2]

        print("✅ PCA9685 initialized successfully")
        print("📍 Using Channel 2 for carousel servo\n")

        # Run automated tests
        print("Starting automated test sequence...")
        print("-" * 40)

        # Test 1: Single forward rotation
        print("\n🧪 Test 1: Single forward rotation (0.08s)")
        print("   This should dispense 1 treat")
        carousel_servo.duty_cycle = pulse_to_duty(1700)
        time.sleep(0.08)
        carousel_servo.duty_cycle = 0  # Stop
        print("   ✅ Complete")
        time.sleep(1)

        # Test 2: Single backward rotation
        print("\n🧪 Test 2: Single backward rotation (0.08s)")
        print("   Testing reverse direction")
        carousel_servo.duty_cycle = pulse_to_duty(1300)
        time.sleep(0.08)
        carousel_servo.duty_cycle = 0  # Stop
        print("   ✅ Complete")
        time.sleep(1)

        # Test 3: Burst sequence
        print("\n🧪 Test 3: Burst sequence (2 treats)")
        print("   Two rotations with pause between")
        for i in range(2):
            print(f"   Burst {i+1}/2...")
            carousel_servo.duty_cycle = pulse_to_duty(1700)
            time.sleep(0.12)
            carousel_servo.duty_cycle = 0
            if i < 1:
                time.sleep(0.3)  # Pause between bursts
        print("   ✅ Complete")
        time.sleep(1)

        # Test 4: Different speeds
        print("\n🧪 Test 4: Testing different rotation speeds")
        speeds = [
            (1600, "Slow forward"),
            (1700, "Normal forward"),
            (1800, "Fast forward")
        ]

        for pulse, description in speeds:
            print(f"   Testing {description} (pulse={pulse})...")
            carousel_servo.duty_cycle = pulse_to_duty(pulse)
            time.sleep(0.5)
            carousel_servo.duty_cycle = 0
            time.sleep(0.5)
        print("   ✅ Complete")

        # Test 5: Continuous rotation
        print("\n🧪 Test 5: Continuous rotation (2 seconds)")
        print("   Watch carousel for smooth rotation")
        carousel_servo.duty_cycle = pulse_to_duty(1700)
        for i in range(4):
            print(f"   {i+1}/4...")
            time.sleep(0.5)
        carousel_servo.duty_cycle = 0
        print("   ✅ Complete")

        print("\n" + "=" * 50)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 50)

        return True

    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        print("\nTroubleshooting:")
        print("1. Check PCA9685 is connected via I2C (SDA/SCL)")
        print("2. Verify servo is connected to channel 2")
        print("3. Ensure servo has proper power supply")
        print("4. Run 'i2cdetect -y 1' to verify I2C connection")
        return False

    finally:
        # Always release servo on exit
        try:
            carousel_servo.duty_cycle = 0
            print("\n🛑 Servo released and stopped")
        except:
            pass

if __name__ == "__main__":
    print("Starting Automated Treat Dispenser Test")
    print("The carousel should rotate during this test")
    print("Press Ctrl+C to emergency stop\n")

    try:
        success = test_treat_dispenser_auto()

        if success:
            print("\n✅ Servo test passed!")
            print("The treat dispenser carousel servo is working correctly")
        else:
            print("\n⚠️ Servo test failed")
            print("Check the hardware connections and try again")

    except KeyboardInterrupt:
        print("\n\n🛑 Test interrupted by user")
        print("Servo stopped.")