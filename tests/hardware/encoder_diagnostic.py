#!/usr/bin/env python3
"""
Encoder Diagnostic Script
Tests encoder hardware directly without PID interference
"""

import time
import subprocess
import logging
from gpiozero import OutputDevice, PWMOutputDevice

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class TreatBotPins:
    """Pin definitions"""
    # Motor control pins (CORRECTED TO MATCH WORKING TEST)
    MOTOR_ENA = 13    # Left motor PWM (was wrong: 18)
    MOTOR_IN1 = 17    # Left motor direction 1
    MOTOR_IN2 = 18    # Left motor direction 2 (was wrong: 27)
    MOTOR_ENB = 19    # Right motor PWM (was wrong: 13)
    MOTOR_IN3 = 27    # Right motor direction 1 (was wrong: 22)
    MOTOR_IN4 = 22    # Right motor direction 2 (was wrong: 19)

    # Encoder pins (CORRECTED FROM HARDWARE SPECS)
    ENCODER_L_A = 4   # Left encoder A1 (GPIO4 = Pin 7)
    ENCODER_L_B = 23  # Left encoder B1 (GPIO23 = Pin 16)
    ENCODER_R_A = 5   # Right encoder A2 (GPIO5 = Pin 29)
    ENCODER_R_B = 6   # Right encoder B2 (GPIO6 = Pin 31)

class EncoderDiagnostic:
    def __init__(self):
        self.pins = TreatBotPins()

        # Motor control using gpiozero
        self.left_in1 = OutputDevice(self.pins.MOTOR_IN1)
        self.left_in2 = OutputDevice(self.pins.MOTOR_IN2)
        self.left_ena = PWMOutputDevice(self.pins.MOTOR_ENA)
        self.right_in3 = OutputDevice(self.pins.MOTOR_IN3)
        self.right_in4 = OutputDevice(self.pins.MOTOR_IN4)
        self.right_enb = PWMOutputDevice(self.pins.MOTOR_ENB)

        print("=== ENCODER DIAGNOSTIC SCRIPT ===")
        print("This will test encoder hardware by:")
        print("1. Running LEFT motor at 50% PWM for 3 seconds")
        print("2. Counting encoder ticks during that time")
        print("3. Calculating RPM from tick count")
        print("4. Repeating for RIGHT motor")
        print("\nExpected: ~1000 ticks in 3 seconds if working")
        print("Problem indicators:")
        print("- <100 ticks: wiring/power problem")
        print("- 0 ticks: encoder not connected or wrong pins")
        print("\nPID CONTROL IS COMPLETELY DISABLED\n")

    def read_gpio_pin(self, pin):
        """Read GPIO pin value directly"""
        try:
            result = subprocess.run(['gpioget', 'gpiochip0', str(pin)],
                                  capture_output=True, text=True, timeout=0.1)
            if result.returncode == 0:
                return int(result.stdout.strip())
            return 0
        except:
            return 0

    def count_encoder_ticks(self, pin_a, pin_b, duration=3.0, motor_name=""):
        """Count encoder ticks for specified duration"""
        print(f"\n--- Testing {motor_name} Motor Encoder ---")
        print(f"Encoder pins: A={pin_a}, B={pin_b}")

        tick_count = 0
        last_a = self.read_gpio_pin(pin_a)
        last_b = self.read_gpio_pin(pin_b)

        start_time = time.time()
        end_time = start_time + duration

        print(f"Counting ticks for {duration} seconds...")
        changes_detected = 0

        while time.time() < end_time:
            current_a = self.read_gpio_pin(pin_a)
            current_b = self.read_gpio_pin(pin_b)

            # Count ANY state change (more reliable than just rising edges)
            if current_a != last_a or current_b != last_b:
                changes_detected += 1

                # Count transitions on A channel for tick counting
                if current_a != last_a:
                    tick_count += 1

            last_a = current_a
            last_b = current_b
            time.sleep(0.001)  # 1kHz polling (more reliable than 10kHz)

        elapsed = time.time() - start_time

        # Calculate RPM (assuming 660 ticks per revolution)
        ticks_per_rev = 660
        revolutions = tick_count / ticks_per_rev
        rpm = (revolutions / elapsed) * 60

        print(f"Results for {motor_name} motor:")
        print(f"  Duration: {elapsed:.2f} seconds")
        print(f"  State changes: {changes_detected}")
        print(f"  A-channel ticks: {tick_count}")
        print(f"  Revolutions: {revolutions:.2f}")
        print(f"  Calculated RPM: {rpm:.1f}")

        if changes_detected == 0:
            print(f"  ❌ ENCODER NOT WORKING - No state changes detected!")
            print(f"     Check wiring, power, or pin assignments")
        elif changes_detected < 10:
            print(f"  ⚠️  LOW ACTIVITY - Possible mechanical or wiring issue")
        elif tick_count > 0:
            print(f"  ✅ Encoder working - {changes_detected} total changes detected")
        else:
            print(f"  ⚠️  States changing but no A-channel transitions")

        return tick_count, rpm

    def stop_all_motors(self):
        """Emergency stop all motors"""
        try:
            self.left_in1.off()
            self.left_in2.off()
            self.left_ena.value = 0
            self.right_in3.off()
            self.right_in4.off()
            self.right_enb.value = 0
            print("All motors stopped")
        except Exception as e:
            print(f"Error stopping motors: {e}")

    def test_left_motor(self):
        """Test left motor and encoder"""
        print("\n🔴 STARTING LEFT MOTOR TEST")

        try:
            # Set left motor forward at 50% PWM (NO PID!)
            self.left_in1.off()    # IN1=low
            self.left_in2.on()     # IN2=high
            self.left_ena.value = 0.5  # 50% PWM

            # Count encoder ticks
            ticks, rpm = self.count_encoder_ticks(
                self.pins.ENCODER_L_A, self.pins.ENCODER_L_B, 3.0, "LEFT"
            )

        finally:
            # Stop left motor
            self.left_ena.value = 0
            self.left_in2.off()

        return ticks, rpm

    def test_right_motor(self):
        """Test right motor and encoder"""
        print("\n🔵 STARTING RIGHT MOTOR TEST")

        try:
            # Set right motor forward at 50% PWM (NO PID!)
            self.right_in3.on()     # IN3=high
            self.right_in4.off()    # IN4=low
            self.right_enb.value = 0.5  # 50% PWM

            # Count encoder ticks
            ticks, rpm = self.count_encoder_ticks(
                self.pins.ENCODER_R_A, self.pins.ENCODER_R_B, 3.0, "RIGHT"
            )

        finally:
            # Stop right motor
            self.right_enb.value = 0
            self.right_in3.off()

        return ticks, rpm

    def run_full_diagnostic(self):
        """Run complete encoder diagnostic"""
        try:
            # Test left motor
            left_ticks, left_rpm = self.test_left_motor()
            time.sleep(1)  # Brief pause

            # Test right motor
            right_ticks, right_rpm = self.test_right_motor()

            # Summary
            print("\n" + "="*50)
            print("DIAGNOSTIC SUMMARY")
            print("="*50)
            print(f"LEFT motor:  {left_ticks:4d} ticks, {left_rpm:5.1f} RPM")
            print(f"RIGHT motor: {right_ticks:4d} ticks, {right_rpm:5.1f} RPM")

            if left_ticks == 0 and right_ticks == 0:
                print("\n❌ BOTH ENCODERS FAILED")
                print("   Check encoder power (3.3V or 5V)")
                print("   Verify all encoder wiring")
                print("   Test with multimeter")

            elif left_ticks == 0:
                print("\n❌ LEFT ENCODER FAILED")
                print("   Check pins 23/24 wiring")

            elif right_ticks == 0:
                print("\n❌ RIGHT ENCODER FAILED")
                print("   Check pins 25/16 wiring")

            else:
                print("\n✅ Both encoders detected movement")
                if left_ticks < 100 or right_ticks < 100:
                    print("⚠️  But tick counts are low - check connections")

        except Exception as e:
            print(f"\nDiagnostic failed: {e}")
        finally:
            self.stop_all_motors()
            print("\nDiagnostic complete - all motors stopped")

if __name__ == "__main__":
    try:
        diagnostic = EncoderDiagnostic()
        print("Starting diagnostic in 2 seconds...")
        time.sleep(2)
        diagnostic.run_full_diagnostic()
    except KeyboardInterrupt:
        print("\nDiagnostic aborted by user")
        if 'diagnostic' in locals():
            diagnostic.stop_all_motors()
    except Exception as e:
        print(f"Fatal error: {e}")
        if 'diagnostic' in locals():
            diagnostic.stop_all_motors()