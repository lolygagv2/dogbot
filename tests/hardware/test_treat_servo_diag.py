#!/usr/bin/env python3
"""
Treat servo diagnostic — non-interactive version
Watch the servos and report back what moved.

Run: python3 tests/hardware/test_treat_servo_diag.py
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def pulse_to_duty(pulse_us):
    return int((pulse_us / 20000.0) * 0xFFFF)


def main():
    print("=" * 60)
    print("WIM-Z Treat Servo Diagnostic (non-interactive)")
    print("=" * 60)

    # === STEP 1: I2C Scan ===
    print("\n--- STEP 1: I2C Bus Scan ---")
    try:
        import board
        import busio
        i2c = busio.I2C(board.SCL, board.SDA)
        while not i2c.try_lock():
            pass
        devices = i2c.scan()
        i2c.unlock()
        print(f"I2C devices: {[hex(d) for d in devices]}")
        if 0x40 not in devices:
            print("❌ PCA9685 NOT found at 0x40 — check wiring!")
            return
        print("✅ PCA9685 at 0x40")
    except Exception as e:
        print(f"❌ I2C scan failed: {e}")
        return

    # === STEP 2: PCA9685 Init ===
    print("\n--- STEP 2: PCA9685 Init ---")
    try:
        from core.hardware.i2c_bus import get_i2c_bus
        from adafruit_pca9685 import PCA9685
        i2c = get_i2c_bus()
        pca = PCA9685(i2c)
        pca.reset()
        time.sleep(0.1)
        pca.mode2 = 0x04
        pca.frequency = 50
        print("✅ PCA9685 initialized at 50Hz")
    except Exception as e:
        print(f"❌ PCA9685 init failed: {e}")
        return

    ch0 = pca.channels[0]  # pan
    ch1 = pca.channels[1]  # pitch
    ch2 = pca.channels[2]  # winch/carousel

    # === STEP 3: Test pan servo (channel 0) ===
    print("\n--- STEP 3: Pan servo (ch0) — should move left then right ---")
    ch0.duty_cycle = pulse_to_duty(1200)
    time.sleep(0.7)
    ch0.duty_cycle = pulse_to_duty(1800)
    time.sleep(0.7)
    ch0.duty_cycle = pulse_to_duty(1500)
    time.sleep(0.3)
    ch0.duty_cycle = 0
    print("  Done. Did pan servo move?")

    # === STEP 4: Test pitch servo (channel 1) ===
    print("\n--- STEP 4: Pitch servo (ch1) — should tilt up then down ---")
    ch1.duty_cycle = pulse_to_duty(1200)
    time.sleep(0.7)
    ch1.duty_cycle = pulse_to_duty(1800)
    time.sleep(0.7)
    ch1.duty_cycle = pulse_to_duty(1500)
    time.sleep(0.3)
    ch1.duty_cycle = 0
    print("  Done. Did pitch servo move?")

    # === STEP 5: Test winch (channel 2) at config pulse ===
    print("\n--- STEP 5: Winch (ch2) at 1544us for 0.17s (normal dispense) ---")
    ch2.duty_cycle = pulse_to_duty(1544)
    time.sleep(0.17)
    ch2.duty_cycle = 0
    time.sleep(0.5)
    print("  Done. Did carousel move?")

    # === STEP 6: Winch at various pulses, 1 second each ===
    print("\n--- STEP 6: Winch at various pulse widths, 1s each ---")
    tests = [
        (1544, "slow config"),
        (1600, "medium"),
        (1700, "fast forward"),
        (1800, "max forward"),
        (1300, "backward"),
        (1200, "max backward"),
    ]
    for pulse, label in tests:
        print(f"  {label} ({pulse}us) for 1 second...")
        ch2.duty_cycle = pulse_to_duty(pulse)
        time.sleep(1.0)
        ch2.duty_cycle = 0
        time.sleep(0.5)

    # === STEP 7: Winch LONG run at max power ===
    print("\n--- STEP 7: Winch at 1800us for 3 seconds (max power, overcome jam) ---")
    ch2.duty_cycle = pulse_to_duty(1800)
    time.sleep(3.0)
    ch2.duty_cycle = 0
    time.sleep(0.5)
    print("  Done.")

    # === STEP 8: Try channel 3 as alternate winch output ===
    print("\n--- STEP 8: Channel 3 (unused) at 1700us for 1s ---")
    print("  (If you swap the winch servo wire to ch3, this tests the servo separately)")
    ch3 = pca.channels[3]
    ch3.duty_cycle = pulse_to_duty(1700)
    time.sleep(1.0)
    ch3.duty_cycle = 0
    print("  Done.")

    # Release all
    for ch in [ch0, ch1, ch2, ch3]:
        ch.duty_cycle = 0

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE — Report what you observed:")
    print("=" * 60)
    print("""
  A) Pan & pitch moved, winch did NOT     → Winch servo dead or ch2 damaged
  B) Pan & pitch moved, winch only at high → Servo weak / physical jam / needs higher pulse
  C) Nothing moved at all                  → Servo power rail dead (check V+ on PCA9685)
  D) Everything moved fine                 → Intermittent issue, may need to test under load
  E) Winch moved in step 6/7 but not 5    → dispense_duration too short, increase to 0.25+
""")


if __name__ == "__main__":
    main()
