#!/usr/bin/env python3
"""
Treat Loader - Interactive carousel loading assistant
Slowly rotates the carousel allowing even distribution of treats
"""

import time
import board
import busio
from adafruit_pca9685 import PCA9685

def pulse_to_duty(pulse_us):
    """Convert pulse width in microseconds to duty cycle"""
    return int((pulse_us / 20000.0) * 0xFFFF)

def treat_loader_mode():
    """Interactive treat loading mode with slow carousel rotation"""

    print("=" * 50)
    print("🍖 TREAT LOADER MODE")
    print("=" * 50)
    print("This mode slowly rotates the carousel for even treat distribution")
    print()

    try:
        # Initialize PCA9685
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50
        carousel_servo = pca.channels[2]

        print("✅ Carousel servo initialized")
        print()
        print("CONTROLS:")
        print("  SPACE or ENTER - Start/Stop rotation")
        print("  F - Faster rotation")
        print("  S - Slower rotation")
        print("  R - Reverse direction")
        print("  1-9 - Jump to compartment")
        print("  Q - Quit")
        print()

        # Settings
        current_speed = 1550  # Slow forward
        direction = 1  # 1=forward, -1=reverse
        is_rotating = False
        compartment_count = 8  # Assuming 8 compartments

        while True:
            print("\n" + "=" * 50)
            print("TREAT LOADER MENU")
            print("=" * 50)
            print(f"Current Speed: {'Forward' if direction > 0 else 'Reverse'} @ {abs(1500 - current_speed)} units")
            print(f"Status: {'🔄 ROTATING' if is_rotating else '⏸️  STOPPED'}")
            print()
            print("1. Start continuous slow rotation")
            print("2. Stop rotation")
            print("3. Single compartment advance")
            print("4. Rotate to specific compartment")
            print("5. Speed adjustment")
            print("6. Fill mode (rotate-pause-rotate)")
            print("7. Test dispense from current position")
            print("8. Exit")
            print()

            choice = input("Enter choice (1-8): ").strip()

            if choice == '1':
                print("\n🔄 Starting slow continuous rotation...")
                print("   Press ENTER to stop")
                carousel_servo.duty_cycle = pulse_to_duty(current_speed)
                input()  # Wait for ENTER
                carousel_servo.duty_cycle = 0
                print("⏸️  Rotation stopped")

            elif choice == '2':
                print("\n⏸️  Stopping rotation...")
                carousel_servo.duty_cycle = 0
                is_rotating = False

            elif choice == '3':
                print("\n⏭️  Advancing one compartment...")
                # Rotate for time needed for one compartment
                time_per_compartment = 0.15  # Adjust based on your carousel
                carousel_servo.duty_cycle = pulse_to_duty(1600)
                time.sleep(time_per_compartment)
                carousel_servo.duty_cycle = 0
                print("✅ Advanced one compartment")

            elif choice == '4':
                try:
                    target = int(input(f"Target compartment (1-{compartment_count}): "))
                    if 1 <= target <= compartment_count:
                        print(f"🎯 Rotating to compartment {target}...")
                        # Calculate rotation needed
                        time_per_compartment = 0.15
                        rotation_time = time_per_compartment * (target - 1)
                        carousel_servo.duty_cycle = pulse_to_duty(1600)
                        time.sleep(rotation_time)
                        carousel_servo.duty_cycle = 0
                        print(f"✅ At compartment {target}")
                    else:
                        print("❌ Invalid compartment number")
                except ValueError:
                    print("❌ Invalid input")

            elif choice == '5':
                print("\n⚙️  Speed Adjustment")
                print("Current speed setting:", current_speed)
                print("1. Slower (1520)")
                print("2. Slow (1550)")
                print("3. Medium (1600)")
                print("4. Fast (1700)")
                print("5. Custom")

                speed_choice = input("Select speed: ").strip()
                speed_map = {
                    '1': 1520,
                    '2': 1550,
                    '3': 1600,
                    '4': 1700
                }

                if speed_choice in speed_map:
                    current_speed = speed_map[speed_choice]
                    print(f"✅ Speed set to {current_speed}")
                elif speed_choice == '5':
                    try:
                        custom = int(input("Enter pulse value (1500-1800): "))
                        if 1500 <= custom <= 1800:
                            current_speed = custom
                            print(f"✅ Custom speed set to {custom}")
                        else:
                            print("❌ Value out of range")
                    except ValueError:
                        print("❌ Invalid input")

            elif choice == '6':
                print("\n🔄 Fill Mode - Rotate and pause for loading")
                print("   Carousel will rotate to each compartment and pause")
                print("   Press ENTER after loading each compartment")

                for i in range(1, compartment_count + 1):
                    print(f"\n📦 Compartment {i}/{compartment_count}")
                    print("   Load treats now...")
                    input("   Press ENTER when ready to continue")

                    if i < compartment_count:
                        print("   Advancing to next compartment...")
                        carousel_servo.duty_cycle = pulse_to_duty(1600)
                        time.sleep(0.15)
                        carousel_servo.duty_cycle = 0

                print("\n✅ All compartments loaded!")

            elif choice == '7':
                print("\n🧪 Test dispensing from current position...")
                carousel_servo.duty_cycle = pulse_to_duty(1700)
                time.sleep(0.12)
                carousel_servo.duty_cycle = 0
                print("✅ Test dispense complete")

            elif choice == '8':
                print("\n👋 Exiting treat loader...")
                carousel_servo.duty_cycle = 0
                break

            else:
                print("❌ Invalid choice")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False

    finally:
        try:
            carousel_servo.duty_cycle = 0
            print("🛑 Carousel stopped")
        except:
            pass

    return True

if __name__ == "__main__":
    print("🍖 TreatBot Carousel Loader")
    print("Load treats evenly into all compartments")
    print()

    try:
        success = treat_loader_mode()
        if success:
            print("\n✅ Treat loading complete!")
        else:
            print("\n⚠️  Loading encountered issues")
    except KeyboardInterrupt:
        print("\n\n🛑 Loading interrupted")