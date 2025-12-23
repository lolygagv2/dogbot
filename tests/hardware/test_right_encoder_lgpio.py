#!/usr/bin/env python3
"""
Test RIGHT motor encoder: GPIO5 (A) and GPIO6 (B)
Spin RIGHT motor BY HAND to verify encoder counting
"""

import lgpio
import time
import threading

# RIGHT motor encoder pins
ENCODER_A = 5   # GPIO5 (Pin 29)
ENCODER_B = 6   # GPIO6 (Pin 31)

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

        print("✅ RIGHT encoder lgpio test initialized")
        print(f"Monitoring GPIO{ENCODER_A} (A) and GPIO{ENCODER_B} (B)")
        print("Spin RIGHT motor BY HAND...")

    def _encoder_a_callback(self, chip, gpio, level, tick):
        """Callback for encoder A channel edge"""
        with self.lock:
            a_state = lgpio.gpio_read(self.h, ENCODER_A)
            b_state = lgpio.gpio_read(self.h, ENCODER_B)
            self.ticks += 1
            if self.ticks <= 10:
                print(f"RIGHT A edge: A={a_state} B={b_state} tick={self.ticks}")

    def _encoder_b_callback(self, chip, gpio, level, tick):
        """Callback for encoder B channel edge"""
        with self.lock:
            a_state = lgpio.gpio_read(self.h, ENCODER_A)
            b_state = lgpio.gpio_read(self.h, ENCODER_B)
            self.ticks += 1
            if self.ticks <= 10:
                print(f"RIGHT B edge: A={a_state} B={b_state} tick={self.ticks}")

    def get_ticks(self):
        with self.lock:
            return self.ticks

    def cleanup(self):
        lgpio.gpiochip_close(self.h)

def main():
    try:
        encoder = EncoderTest()

        print("\n🧪 Testing RIGHT motor for 10 seconds...")
        print("SPIN RIGHT MOTOR BY HAND NOW!")
        print("-" * 50)

        for second in range(10):
            time.sleep(1)
            ticks = encoder.get_ticks()
            print(f"Second {second + 1}: {ticks} total ticks")

        final_ticks = encoder.get_ticks()
        print("-" * 50)
        print(f"🎯 RIGHT encoder result: {final_ticks} ticks in 10 seconds")

        if final_ticks > 0:
            print("✅ RIGHT encoder working!")
        else:
            print("❌ RIGHT encoder also failed")

        encoder.cleanup()

    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    main()