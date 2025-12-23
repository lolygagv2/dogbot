#!/usr/bin/env python3
"""
Direct test of lgpio encoder reading while running motor
Compare with motor controller polling system
"""

import lgpio
import time
import threading
import sys
from pathlib import Path
from gpiozero import OutputDevice, PWMOutputDevice

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Motor pins (left motor)
MOTOR_IN1 = 17
MOTOR_IN2 = 18
MOTOR_ENA = 13

# Encoder pins (left motor)
ENCODER_A = 4   # GPIO4
ENCODER_B = 23  # GPIO23

class DirectEncoderTest:
    def __init__(self):
        # Motor setup
        self.motor_in1 = OutputDevice(MOTOR_IN1)
        self.motor_in2 = OutputDevice(MOTOR_IN2)
        self.motor_ena = PWMOutputDevice(MOTOR_ENA)

        # Encoder setup
        self.gpio_handle = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_input(self.gpio_handle, ENCODER_A, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.gpio_handle, ENCODER_B, lgpio.SET_PULL_UP)

        # State tracking
        self.count = 0
        self.changes = 0
        self.lock = threading.Lock()
        self.last_a = lgpio.gpio_read(self.gpio_handle, ENCODER_A)
        self.last_b = lgpio.gpio_read(self.gpio_handle, ENCODER_B)

        print(f"✅ Direct encoder test initialized")
        print(f"Initial states: A={self.last_a}, B={self.last_b}")

    def start_motor(self, pwm_percent):
        """Start left motor forward"""
        print(f"🚗 Starting motor at {pwm_percent}% PWM")
        self.motor_in1.off()  # Forward direction
        self.motor_in2.on()
        self.motor_ena.value = pwm_percent / 100.0

    def stop_motor(self):
        """Stop motor"""
        print("🛑 Stopping motor")
        self.motor_ena.value = 0
        self.motor_in1.off()
        self.motor_in2.off()

    def poll_encoder_once(self):
        """Single encoder poll - same logic as motor controller"""
        current_a = lgpio.gpio_read(self.gpio_handle, ENCODER_A)
        current_b = lgpio.gpio_read(self.gpio_handle, ENCODER_B)

        with self.lock:
            # Detect changes (same as motor controller)
            if current_a != self.last_a or current_b != self.last_b:
                self.changes += 1

                # Quadrature decoding (same as motor controller)
                if self.last_a == 0 and current_a == 1:
                    # Rising edge on A
                    if current_b == 0:
                        self.count += 1  # Forward
                    else:
                        self.count -= 1  # Backward
                elif self.last_a == 1 and current_a == 0:
                    # Falling edge on A
                    if current_b == 1:
                        self.count += 1  # Forward
                    else:
                        self.count -= 1  # Backward

                # Debug first few changes
                if self.changes <= 10:
                    print(f"Change {self.changes}: A={self.last_a}→{current_a}, B={self.last_b}→{current_b}, Count={self.count}")

            # Update states
            self.last_a = current_a
            self.last_b = current_b

    def get_counts(self):
        with self.lock:
            return self.count, self.changes

    def cleanup(self):
        self.stop_motor()
        self.motor_ena.close()
        self.motor_in1.close()
        self.motor_in2.close()
        lgpio.gpiochip_close(self.gpio_handle)

def main():
    print("🔍 DIRECT LGPIO ENCODER TEST")
    print("===========================")
    print("Testing same logic as motor controller but with debug output")
    print()

    try:
        test = DirectEncoderTest()

        # Skip manual test, go straight to motor test

        # Test 2: Motor running at 80% PWM
        print("📋 Test 2: Motor ON at 80% PWM for 3 seconds")
        test.start_motor(80)

        start_time = time.time()
        while time.time() - start_time < 3:
            test.poll_encoder_once()
            time.sleep(0.0005)  # 2000Hz polling

        test.stop_motor()

        count, changes = test.get_counts()
        print(f"Motor running result: {count} counts, {changes} changes")

        if count > 10:
            print("✅ Direct lgpio reading works with motor running!")
        else:
            print("❌ Still getting low counts with direct lgpio")
            print("Possible issues:")
            print("1. Motor not spinning fast enough")
            print("2. Encoder wiring issue")
            print("3. Quadrature decoding logic bug")

        test.cleanup()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()