#!/usr/bin/env python3
"""
Battery ADC diagnostic — reads ALL FOUR ADS1115 channels.

Purpose: the battery monitor only reads A0. If A0 reads ~0V while a real
battery is plugged into the JST, this checks whether the voltage-divider
tap is actually wired to a different channel (A1/A2/A3) on this PCB.

A 4S LiPo at 16.8V through a ~54:1 divider should present ~0.31V to
whichever channel the tap feeds. Look for a channel reading 0.25-0.35V.

Safe to run while treatbot.service is active — single-shot I2C reads.
"""
import time

from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
import board

CALIBRATION_FACTOR = 54.28  # treatbot4.yaml battery.calibration_factor


def main():
    i2c = board.I2C()
    ads = ADS1115(i2c, address=0x48)
    channels = {n: AnalogIn(ads, n) for n in range(4)}

    print("=== ADS1115 4-channel scan (0x48) ===")
    print(f"Calibration factor: {CALIBRATION_FACTOR}  (expect ~0.31V on divider channel at 16.8V)\n")
    print(f"{'CH':<5}{'raw counts':<14}{'volts @ pin':<16}{'x factor':<14}{'note'}")
    print("-" * 60)

    for _ in range(3):  # average a few samples
        for n, ch in channels.items():
            v = ch.voltage
            counts = ch.value
            scaled = v * CALIBRATION_FACTOR
            if 0.25 <= v <= 0.40:
                note = "<-- looks like the divider tap"
            elif abs(v) < 0.05:
                note = "floating / grounded"
            else:
                note = ""
            print(f"A{n:<4}{counts:<14}{v:<16.4f}{scaled:<14.2f}{note}")
        print("-" * 60)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
