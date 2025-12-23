#!/usr/bin/env python3
"""
MINIMAL encoder test using lgpio interrupts (not polling)
Tests LEFT motor encoder: GPIO4 (A) and GPIO23 (B)
Spin LEFT motor BY HAND to verify encoder counting
"""

import lgpio
import time
import threading

# LEFT motor encoder pins
ENCODER_A = 4   # GPIO4 (Pin 7)
ENCODER_B = 23  # GPIO23 (Pin 16)

class EncoderTest:
    def __init__(self):
        # Open GPIO chip
        self.h = lgpio.gpiochip_open(0)

        # Configure inputs with pull-up
        lgpio.gpio_claim_input(self.h, ENCODER_A, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.h, ENCODER_B, lgpio.SET_PULL_UP)

        # Encoder state
        self.ticks = 0
        self.lock = threading.Lock()

        # Set up edge detection callbacks for BOTH edges on BOTH channels
        lgpio.callback(self.h, ENCODER_A, lgpio.BOTH_EDGES, self._encoder_a_callback)
        lgpio.callback(self.h, ENCODER_B, lgpio.BOTH_EDGES, self._encoder_b_callback)

        print("✅ lgpio encoder test initialized")
        print(f"Monitoring GPIO{ENCODER_A} (A) and GPIO{ENCODER_B} (B)")
        print("Spin LEFT motor BY HAND...")

    def _encoder_a_callback(self, chip, gpio, level, tick):
        """Callback for encoder A channel edge"""
        with self.lock:
            # Read both channels for quadrature decoding
            a_state = lgpio.gpio_read(self.h, ENCODER_A)
            b_state = lgpio.gpio_read(self.h, ENCODER_B)

            # Simple tick counting (not direction-aware for this test)
            self.ticks += 1

            # Debug output for first few ticks
            if self.ticks <= 10:
                print(f"A edge: A={a_state} B={b_state} tick={self.ticks}")

    def _encoder_b_callback(self, chip, gpio, level, tick):
        """Callback for encoder B channel edge"""
        with self.lock:
            # Read both channels for quadrature decoding
            a_state = lgpio.gpio_read(self.h, ENCODER_A)
            b_state = lgpio.gpio_read(self.h, ENCODER_B)

            # Simple tick counting (not direction-aware for this test)
            self.ticks += 1

            # Debug output for first few ticks
            if self.ticks <= 10:
                print(f"B edge: A={a_state} B={b_state} tick={self.ticks}")

    def get_ticks(self):
        with self.lock:
            return self.ticks

    def reset_ticks(self):
        with self.lock:
            self.ticks = 0

    def cleanup(self):
        lgpio.gpiochip_close(self.h)

def main():
    try:
        # Initialize encoder test
        encoder = EncoderTest()

        print("\n🧪 Testing for 10 seconds...")
        print("SPIN LEFT MOTOR BY HAND NOW!")
        print("Expected: Ticks should increase when you rotate motor")
        print("-" * 50)

        # Monitor for 10 seconds
        for second in range(10):
            time.sleep(1)
            ticks = encoder.get_ticks()
            print(f"Second {second + 1}: {ticks} total ticks")

        final_ticks = encoder.get_ticks()
        print("-" * 50)
        print(f"🎯 Final result: {final_ticks} ticks in 10 seconds")

        if final_ticks > 0:
            print("✅ SUCCESS: Encoder working!")
        else:
            print("❌ FAIL: No encoder ticks detected")
            print("Check wiring: Green wire to GPIO4, Yellow wire to GPIO23")

        encoder.cleanup()

    except Exception as e:
        print(f"❌ ERROR: {e}")
        print("Make sure to install lgpio: pip install lgpio")

if __name__ == "__main__":
    main()