#!/usr/bin/env python3
"""
Stepper motor test script for NEMA 17 + TMC2209 treat dispenser
Calibrates steps-per-slot and speed under load.

GPIO Wiring (Board Pin → BCM):
  STEP:     Pin 32 → GPIO 12
  DIR:      Pin 36 → GPIO 16
  EN:       Pin 18 → GPIO 24  (LOW = enabled, HIGH = disabled)
  VCC_IO:   Pi 3.3V → TMC2209 Pin 7
  UART TX:  Pin 8  → GPIO 14  (1K resistor) ──┐
  UART RX:  Pin 10 → GPIO 15  (direct)    ────┘── TMC2209 Pin 4 (PDN_UART)

Motor: NEMA 17, 200 steps/rev, 1.8°/step
TMC2209: MS1=HIGH → 32x microstepping hardware default
  but mstep_reg_select=1 overrides to 8x via UART
Carousel: 12 slots per level, 4 levels, 44 treats total
"""

import lgpio
import serial
import struct
import time
import sys

# === GPIO Pin Assignments (BCM) ===
STEP_PIN = 12   # Board Pin 32
DIR_PIN = 16    # Board Pin 36
EN_PIN = 24     # Board Pin 18

# === UART ===
UART_PORT = '/dev/ttyAMA0'
UART_BAUD = 115200

# === Motor Constants ===
MICROSTEPS = 8
FULL_STEPS_PER_REV = 200
MICROSTEPS_PER_REV = FULL_STEPS_PER_REV * MICROSTEPS  # 1600

# Carousel geometry
SLOTS_PER_LEVEL = 12
STEPS_PER_SLOT = 137  # Calibrated value

# Direction
CW = 1
CCW = 0

# Default step delay
DEFAULT_DELAY = 0.010  # 10ms half-period = 20ms/step


# =============================================================================
# TMC2209 UART
# =============================================================================

def _tmc_crc(data):
    crc = 0
    for byte in data:
        for _ in range(8):
            if (crc >> 7) ^ (byte & 0x01):
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
            byte >>= 1
    return crc


def tmc_write(ser, reg, value):
    datagram = bytes([0x05, 0x00, reg | 0x80]) + struct.pack('>I', value)
    datagram += bytes([_tmc_crc(datagram)])
    ser.write(datagram)
    time.sleep(0.005)
    ser.reset_input_buffer()


def tmc_read(ser, reg):
    ser.reset_input_buffer()
    datagram = bytes([0x05, 0x00, reg])
    datagram += bytes([_tmc_crc(datagram)])
    ser.write(datagram)
    time.sleep(0.01)
    response = ser.read(12)
    if len(response) >= 12:
        return struct.unpack('>I', response[7:11])[0]
    return None


def configure_tmc(ser):
    """Configure TMC2209 for carousel operation"""
    # GCONF: I_scale_analog=1, mstep_reg_select=1 (UART overrides MS pins)
    tmc_write(ser, 0x00, 0x00000081)
    # IRUN=31 (max), IHOLD=5, IHOLDDELAY=6
    tmc_write(ser, 0x10, (6 << 16) | (31 << 8) | 5)
    # CHOPCONF: 8x microstep (MRES=5), vsense=0 (high current range)
    tmc_write(ser, 0x6C, (5 << 24) | 0x00000053)
    # TPOWERDOWN
    tmc_write(ser, 0x11, 20)
    print("TMC2209 configured: IRUN=31, 8x microstep, vsense=0")


# =============================================================================
# GPIO CONTROL
# =============================================================================

def init_gpio():
    chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(chip, STEP_PIN)
    lgpio.gpio_claim_output(chip, DIR_PIN)
    lgpio.gpio_claim_output(chip, EN_PIN)
    lgpio.gpio_write(chip, EN_PIN, 1)  # Disabled
    lgpio.gpio_write(chip, STEP_PIN, 0)
    lgpio.gpio_write(chip, DIR_PIN, CW)
    return chip


def enable_motor(chip):
    lgpio.gpio_write(chip, EN_PIN, 0)
    time.sleep(0.05)


def disable_motor(chip):
    lgpio.gpio_write(chip, EN_PIN, 1)


def step_motor(chip, steps, direction=CW, delay=DEFAULT_DELAY):
    lgpio.gpio_write(chip, DIR_PIN, direction)
    time.sleep(0.001)
    for i in range(steps):
        lgpio.gpio_write(chip, STEP_PIN, 1)
        time.sleep(delay)
        lgpio.gpio_write(chip, STEP_PIN, 0)
        time.sleep(delay)


def cleanup(chip, ser=None):
    try:
        lgpio.gpio_write(chip, EN_PIN, 1)
        lgpio.gpio_write(chip, STEP_PIN, 0)
        lgpio.gpiochip_close(chip)
    except Exception:
        pass
    if ser:
        try:
            ser.close()
        except Exception:
            pass


# =============================================================================
# TESTS
# =============================================================================

def test_speed_calibration(chip, ser):
    """Test single-slot dispense at different speeds"""
    print("\n" + "=" * 60)
    print("SPEED CALIBRATION (load treats first!)")
    print("=" * 60)
    print(f"Each test does {STEPS_PER_SLOT} steps at different speeds.")
    print()

    delays = [
        (0.010, "SLOW    — 20ms/step, ~2.7s per slot"),
        (0.006, "MEDIUM  — 12ms/step, ~1.6s per slot"),
        (0.004, "FAST    —  8ms/step, ~1.1s per slot"),
        (0.002, "FASTER  —  4ms/step, ~0.5s per slot"),
        (0.001, "FASTEST —  2ms/step, ~0.3s per slot"),
    ]

    enable_motor(chip)
    time.sleep(0.3)

    for idx, (delay, label) in enumerate(delays):
        input(f"Test {idx+1}/{len(delays)}: {label}\n  Press ENTER...")
        step_motor(chip, STEPS_PER_SLOT, CW, delay)
        print(f"  Done. Clean rotation? Treat dropped? Clicking/stall?")
        print()
        time.sleep(0.5)

    disable_motor(chip)


def test_step_calibration(chip, ser):
    """Fine-tune steps per slot"""
    print("\n" + "=" * 60)
    print("STEP COUNT CALIBRATION")
    print("=" * 60)
    print(f"Current: {STEPS_PER_SLOT} steps/slot")
    print("Options: [ENTER]=dispense 1 slot, [number]=change steps, [d]=done")
    print()

    steps = STEPS_PER_SLOT
    enable_motor(chip)
    time.sleep(0.3)

    while True:
        cmd = input(f"Steps={steps} > ").strip().lower()
        if cmd == 'd':
            break
        elif cmd == '':
            step_motor(chip, steps, CW)
            print(f"  Moved {steps} steps")
        elif cmd.startswith('-'):
            try:
                rev = int(cmd)
                step_motor(chip, abs(rev), CCW)
                print(f"  Reversed {abs(rev)} steps")
            except ValueError:
                print("  Invalid")
        else:
            try:
                steps = int(cmd)
                print(f"  Steps per slot set to {steps}")
            except ValueError:
                print("  Invalid")

    disable_motor(chip)
    print(f"\nFinal calibration: {steps} steps/slot")
    return steps


def test_dispense_multiple(chip, ser, count=5):
    """Dispense multiple treats in sequence"""
    print(f"\n{'=' * 60}")
    print(f"DISPENSING {count} TREATS")
    print("=" * 60)

    enable_motor(chip)
    time.sleep(0.3)

    for i in range(count):
        print(f"  Treat {i+1}/{count}...", end=" ", flush=True)
        step_motor(chip, STEPS_PER_SLOT, CW)
        print("dispensed")
        time.sleep(0.5)

    disable_motor(chip)
    print(f"Done — {count} treats dispensed")


def test_refill_mode(chip, ser):
    """Step through slots slowly for refilling"""
    print("\n" + "=" * 60)
    print("REFILL MODE")
    print("=" * 60)
    total = 56  # 4 levels x 12 slots + extra
    print(f"Will advance {total} slots. Fill each as it comes around.")
    print("Press Ctrl+C to stop early.")
    input("Press ENTER to start...")

    enable_motor(chip)
    time.sleep(0.3)

    try:
        for slot in range(total):
            print(f"  Slot {slot+1}/{total} — fill now...", end=" ", flush=True)
            time.sleep(2.0)
            step_motor(chip, STEPS_PER_SLOT, CW)
            print("advanced")
    except KeyboardInterrupt:
        print(f"\n  Stopped at slot {slot+1}")

    disable_motor(chip)


def test_anti_jam(chip, ser):
    """Test anti-jam reverse-and-retry"""
    print("\n" + "=" * 60)
    print("ANTI-JAM TEST")
    print("=" * 60)
    print("Simulates jam recovery: reverse 50 steps, forward full slot")
    input("Press ENTER...")

    enable_motor(chip)
    time.sleep(0.3)

    for cycle in range(3):
        print(f"  Cycle {cycle+1}/3: reverse...", end=" ", flush=True)
        step_motor(chip, 50, CCW, delay=0.006)
        time.sleep(0.2)
        print("forward...", end=" ", flush=True)
        step_motor(chip, STEPS_PER_SLOT + 50, CW, delay=0.008)
        print("done")
        time.sleep(0.5)

    disable_motor(chip)
    print("Anti-jam test complete")


def test_full_revolution(chip, ser):
    """12 slots to verify full revolution"""
    print(f"\n{'=' * 60}")
    print(f"FULL REVOLUTION: 12 x {STEPS_PER_SLOT} = {12 * STEPS_PER_SLOT} steps")
    print("=" * 60)
    input("Mark start position, press ENTER...")

    enable_motor(chip)
    time.sleep(0.3)

    for slot in range(12):
        step_motor(chip, STEPS_PER_SLOT, CW)
        print(f"  Slot {slot+1}/12")
        time.sleep(0.8)

    disable_motor(chip)
    print("Back to start? Check alignment.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global STEPS_PER_SLOT

    print("=" * 60)
    print("NEMA 17 + TMC2209 TREAT DISPENSER TEST")
    print("=" * 60)
    print(f"  Steps/slot: {STEPS_PER_SLOT}")
    print(f"  IRUN: 31 (max), 8x microstepping, vsense=0")
    print()

    chip = None
    ser = None

    try:
        chip = init_gpio()
        ser = serial.Serial(UART_PORT, UART_BAUD, timeout=0.2)
        configure_tmc(ser)

        # CLI shortcuts
        if len(sys.argv) > 1:
            cmd = sys.argv[1]
            enable_motor(chip)
            time.sleep(0.3)

            if cmd == "dispense":
                count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
                for i in range(count):
                    print(f"Dispensing {i+1}/{count}...")
                    step_motor(chip, STEPS_PER_SLOT, CW)
                    time.sleep(0.5)

            elif cmd == "reverse":
                steps = int(sys.argv[2]) if len(sys.argv) > 2 else STEPS_PER_SLOT
                print(f"Reversing {steps} steps...")
                step_motor(chip, steps, CCW)

            elif cmd == "step":
                steps = int(sys.argv[2]) if len(sys.argv) > 2 else 200
                delay = float(sys.argv[3]) / 1000 if len(sys.argv) > 3 else DEFAULT_DELAY
                print(f"Stepping {steps} (delay={delay*1000:.1f}ms)...")
                step_motor(chip, steps, CW, delay)

            elif cmd == "refill":
                disable_motor(chip)
                test_refill_mode(chip, ser)

            elif cmd == "unjam":
                disable_motor(chip)
                test_anti_jam(chip, ser)

            disable_motor(chip)
            cleanup(chip, ser)
            return

        # Interactive menu
        print("Tests:")
        print("  1. Speed calibration (test different speeds)")
        print("  2. Step count calibration (fine-tune steps/slot)")
        print("  3. Full revolution (12 slots)")
        print("  4. Dispense multiple treats")
        print("  5. Refill mode")
        print("  6. Anti-jam test")
        print("  q. Quit")

        while True:
            choice = input("\nSelect (1-6, q): ").strip().lower()

            if choice == 'q':
                break
            elif choice == '1':
                test_speed_calibration(chip, ser)
            elif choice == '2':
                result = test_step_calibration(chip, ser)
                if result > 0:
                    STEPS_PER_SLOT = result
            elif choice == '3':
                test_full_revolution(chip, ser)
            elif choice == '4':
                count = input("How many treats? [5]: ").strip()
                count = int(count) if count else 5
                test_dispense_multiple(chip, ser, count)
            elif choice == '5':
                test_refill_mode(chip, ser)
            elif choice == '6':
                test_anti_jam(chip, ser)
            else:
                print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\nAborted")
    finally:
        if chip is not None:
            cleanup(chip, ser)

    print(f"\n--- CALIBRATION ---")
    print(f"Steps per slot: {STEPS_PER_SLOT}")
    print(f"CW direction: DIR={CW}")


if __name__ == '__main__':
    main()
