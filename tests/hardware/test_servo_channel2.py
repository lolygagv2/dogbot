#!/usr/bin/env python3
"""
Diagnostic test for treat dispenser servo (PCA9685 channel 2)
Tests whether the servo physically responds to commands.
Also tests channels 0 and 1 to compare.
"""

import sys
import time
sys.path.append('/home/morgan/dogbot')

from core.hardware.servo_controller import ServoController
from config.config_loader import get_config

def pulse_to_duty(pulse_us):
    return int((pulse_us / 20000.0) * 0xFFFF)

def main():
    print("=" * 50)
    print("SERVO CHANNEL 2 (TREAT DISPENSER) DIAGNOSTIC")
    print("=" * 50)

    # Load config
    config = get_config()
    print(f"\nConfig: slow_pulse={config.servo.slow_pulse}, dispense_duration={config.dispenser.dispense_duration}")

    # Init servo controller
    servo = ServoController()
    if not servo.is_initialized():
        print("\nFAILED: PCA9685 not initialized!")
        return

    print("\nPCA9685 initialized OK")
    print(f"Frequency: {servo.settings.SERVO_FREQUENCY}Hz")

    # --- Test 1: Channel 0 (pan) - quick wiggle ---
    print("\n--- TEST 1: Channel 0 (Pan) - small wiggle ---")
    input("Press ENTER to test pan servo (ch0)... ")
    servo.set_camera_pan(80)
    time.sleep(0.5)
    servo.set_camera_pan(100)
    time.sleep(0.5)
    servo.pca.channels[0].duty_cycle = 0
    print("Did the pan servo move? (If yes, PCA9685 board is working)")

    # --- Test 2: Channel 1 (tilt) - quick wiggle ---
    print("\n--- TEST 2: Channel 1 (Tilt) - small wiggle ---")
    input("Press ENTER to test tilt servo (ch1)... ")
    servo.set_camera_pitch(80)
    time.sleep(0.5)
    servo.set_camera_pitch(100)
    time.sleep(0.5)
    servo.pca.channels[1].duty_cycle = 0
    print("Did the tilt servo move?")

    # --- Test 3: Channel 2 (winch) - current config ---
    print("\n--- TEST 3: Channel 2 (Winch) - CURRENT CONFIG ---")
    print(f"  Pulse: {config.servo.slow_pulse}us, Duration: {config.dispenser.dispense_duration}s")
    input("Press ENTER to test with current config... ")
    servo.rotate_winch('slow', config.dispenser.dispense_duration)
    time.sleep(0.5)
    print("Did the treat servo move?")

    # --- Test 4: Channel 2 - longer duration ---
    print("\n--- TEST 4: Channel 2 (Winch) - LONGER DURATION (1 second) ---")
    print(f"  Pulse: {config.servo.slow_pulse}us, Duration: 1.0s")
    input("Press ENTER to test with 1 second duration... ")
    servo.rotate_winch('slow', 1.0)
    time.sleep(0.5)
    print("Did the treat servo move?")

    # --- Test 5: Channel 2 - different pulse values ---
    for pulse in [1300, 1500, 1600, 1700]:
        print(f"\n--- TEST 5: Channel 2 - RAW PULSE {pulse}us for 1s ---")
        input(f"Press ENTER to try pulse {pulse}us... ")
        servo.winch_channel.duty_cycle = pulse_to_duty(pulse)
        time.sleep(1.0)
        servo.winch_channel.duty_cycle = 0
        time.sleep(0.3)
        print(f"Any movement at pulse {pulse}?")

    # --- Test 6: Channel 2 - full power forward/backward ---
    print("\n--- TEST 6: Channel 2 - FORWARD (1700us) 2 seconds ---")
    input("Press ENTER for full forward... ")
    servo.rotate_winch('forward', 2.0)
    time.sleep(0.5)

    print("\n--- TEST 7: Channel 2 - BACKWARD (1300us) 2 seconds ---")
    input("Press ENTER for full backward... ")
    servo.rotate_winch('backward', 2.0)
    time.sleep(0.5)

    # Cleanup
    servo.release_all_servos()
    print("\n" + "=" * 50)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 50)
    print("""
RESULTS GUIDE:
- Ch0+Ch1 work, Ch2 doesn't = Channel 2 output driver is fried
- Nothing works = PCA9685 board is dead (but I2C showed it alive?)
- Ch2 moves with longer duration but not short = Increase dispense_duration
- Ch2 moves with different pulse = Update slow_pulse config value
- Ch2 works in all tests = Problem is upstream (API/dispenser service)
""")

if __name__ == "__main__":
    main()
