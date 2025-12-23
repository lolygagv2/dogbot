#!/usr/bin/env python3
"""
Left Motor Encoder Debug Test
Tests the exact pin configuration that worked in our direction test
but using the PID system's encoder decoding to isolate the issue
"""

import lgpio
import time
from gpiozero import OutputDevice, PWMOutputDevice

# Motor control pins
LEFT_IN1 = 17
LEFT_IN2 = 18
LEFT_ENA = 13

# Encoder pins
LEFT_ENCODER_A = 4
LEFT_ENCODER_B = 23

class LeftMotorEncoderDebugger:
    def __init__(self):
        # GPIO setup
        self.gpio_handle = lgpio.gpiochip_open(0)

        # Motor control (direct, not through PID)
        self.left_in1 = OutputDevice(LEFT_IN1)
        self.left_in2 = OutputDevice(LEFT_IN2)
        self.left_ena = PWMOutputDevice(LEFT_ENA)

        # Encoder setup (same as PID system)
        lgpio.gpio_claim_input(self.gpio_handle, LEFT_ENCODER_A, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.gpio_handle, LEFT_ENCODER_B, lgpio.SET_PULL_UP)

        # Encoder tracking (same logic as PID system)
        self.encoder_count = 0
        self.last_a = 0
        self.last_b = 0

        print("🔧 Left Motor Encoder Debugger Initialized")
        print(f"   Motor: IN1={LEFT_IN1}, IN2={LEFT_IN2}, ENA={LEFT_ENA}")
        print(f"   Encoder: A={LEFT_ENCODER_A}, B={LEFT_ENCODER_B}")

    def read_encoder_pins(self):
        """Read current encoder pin states"""
        a = lgpio.gpio_read(self.gpio_handle, LEFT_ENCODER_A)
        b = lgpio.gpio_read(self.gpio_handle, LEFT_ENCODER_B)
        return a, b

    def decode_quadrature_pid_style(self, current_a, current_b):
        """Use EXACT same quadrature decoding as PID system"""
        if current_a != self.last_a or current_b != self.last_b:
            # This is the EXACT logic from the PID system
            if self.last_a == 0 and current_a == 1:
                # Rising edge on A
                self.encoder_count += 1 if current_b == 0 else -1
            elif self.last_a == 1 and current_a == 0:
                # Falling edge on A
                self.encoder_count += 1 if current_b == 1 else -1

            self.last_a = current_a
            self.last_b = current_b

    def stop_motor(self):
        """Stop motor completely"""
        self.left_in1.off()
        self.left_in2.off()
        self.left_ena.value = 0
        print("🛑 Motor stopped")

    def test_known_good_configuration(self):
        """Test the exact configuration that gave us +814 counts"""
        print("\n🧪 TESTING KNOWN GOOD CONFIGURATION")
        print("Configuration: IN1=1, IN2=0 (gave +814 counts in direction test)")
        print("=" * 60)

        # Stop motor first
        self.stop_motor()
        time.sleep(0.5)

        # Reset encoder count and read initial state
        self.encoder_count = 0
        a, b = self.read_encoder_pins()
        self.last_a, self.last_b = a, b
        print(f"Initial encoder state: A={a}, B={b}")
        print(f"Initial encoder count: {self.encoder_count}")

        print("\n🚀 Starting motor with IN1=1, IN2=0 at 30% PWM...")

        # Apply the EXACT configuration from our successful test
        self.left_in1.on()    # IN1=1
        self.left_in2.off()   # IN2=0
        self.left_ena.value = 0.30  # 30% PWM

        print("Running for 3 seconds, monitoring encoder with PID-style decoding...")

        # Monitor encoder for 3 seconds using PID system logic
        start_time = time.time()
        last_print = start_time

        while time.time() - start_time < 3.0:
            # Read encoder pins
            current_a, current_b = self.read_encoder_pins()

            # Decode using PID system logic
            self.decode_quadrature_pid_style(current_a, current_b)

            # Print status every 0.5 seconds
            if time.time() - last_print >= 0.5:
                elapsed = time.time() - start_time
                print(f"   {elapsed:.1f}s: Encoder count = {self.encoder_count}, Current A={current_a}, B={current_b}")
                last_print = time.time()

            time.sleep(0.001)  # 1ms polling like PID system

        # Stop motor
        self.stop_motor()
        final_count = self.encoder_count

        print(f"\n📊 RESULTS:")
        print(f"   Configuration: IN1=1, IN2=0")
        print(f"   Duration: 3 seconds at 30% PWM")
        print(f"   Final encoder count: {final_count}")

        # Analysis
        print(f"\n📈 ANALYSIS:")
        if final_count > 50:
            print(f"   ✅ POSITIVE counts ({final_count}) - Encoder decoding is CORRECT")
            print(f"   The issue is elsewhere (likely motor direction in PID system)")
            return "encoder_correct"
        elif final_count < -50:
            print(f"   ❌ NEGATIVE counts ({final_count}) - Encoder decoding is INVERTED")
            print(f"   Same motor motion as direction test, but opposite encoder counts!")
            print(f"   Need to invert quadrature decoding for left motor in PID system")
            return "encoder_inverted"
        else:
            print(f"   ⚠️  LOW counts ({final_count}) - Encoder may not be working properly")
            return "encoder_weak"

def main():
    print("🔍 LEFT MOTOR ENCODER DEBUG TEST")
    print("=" * 50)
    print("This test uses the EXACT pin configuration that worked")
    print("in our direction test, but with PID system encoder decoding")
    print("to isolate whether the issue is encoder direction or motor direction")
    print("=" * 50)

    debugger = LeftMotorEncoderDebugger()

    try:
        result = debugger.test_known_good_configuration()

        print(f"\n🎯 CONCLUSION:")
        if result == "encoder_correct":
            print("   The encoder decoding is correct in the PID system")
            print("   The issue is likely motor direction mapping in PID _apply_pwm method")
            print("   Recommendation: Check motor direction logic in PID system")
        elif result == "encoder_inverted":
            print("   The encoder decoding is INVERTED in the PID system")
            print("   Same motor motion gives opposite encoder counts vs direction test")
            print("   Recommendation: Invert quadrature decoding for left motor only")
        elif result == "encoder_weak":
            print("   The encoder signal is weak - possible wiring/connection issue")
            print("   Recommendation: Check encoder connections")

    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        debugger.stop_motor()
        lgpio.gpiochip_close(debugger.gpio_handle)
        print("\n✅ Test cleanup complete")

if __name__ == "__main__":
    main()