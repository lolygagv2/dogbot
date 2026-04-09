#!/usr/bin/env python3
"""
TMC2209 diagnostic — reads back registers, tests current/microstepping combos.
Identifies why motor buzzes instead of stepping.
"""

import lgpio
import serial
import struct
import time

STEP_PIN = 12
DIR_PIN = 16
EN_PIN = 24
UART_PORT = '/dev/ttyAMA0'

def _crc(data):
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
    datagram += bytes([_crc(datagram)])
    ser.write(datagram)
    time.sleep(0.005)
    ser.reset_input_buffer()

def tmc_read(ser, reg):
    ser.reset_input_buffer()
    datagram = bytes([0x05, 0x00, reg])
    datagram += bytes([_crc(datagram)])
    ser.write(datagram)
    time.sleep(0.01)
    response = ser.read(12)
    if len(response) >= 12:
        return struct.unpack('>I', response[7:11])[0]
    return None

def step_motor(chip, steps, direction, delay=0.01):
    lgpio.gpio_write(chip, DIR_PIN, direction)
    time.sleep(0.005)
    for i in range(steps):
        lgpio.gpio_write(chip, STEP_PIN, 1)
        time.sleep(delay)
        lgpio.gpio_write(chip, STEP_PIN, 0)
        time.sleep(delay)

chip = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(chip, STEP_PIN)
lgpio.gpio_claim_output(chip, DIR_PIN)
lgpio.gpio_claim_output(chip, EN_PIN)
lgpio.gpio_write(chip, EN_PIN, 1)
lgpio.gpio_write(chip, STEP_PIN, 0)

ser = serial.Serial(UART_PORT, 115200, timeout=0.2)

try:
    print("=" * 60)
    print("TMC2209 DIAGNOSTIC")
    print("=" * 60)

    # 1. Read chip info
    ioin = tmc_read(ser, 0x06)
    if ioin is None:
        print("ERROR: TMC2209 not responding on UART!")
        exit(1)
    version = (ioin >> 24) & 0xFF
    print(f"TMC2209 version: 0x{version:02X}")
    print(f"IOIN raw: 0x{ioin:08X}")
    print(f"  STEP pin state: {(ioin >> 7) & 1}")
    print(f"  DIR pin state:  {(ioin >> 3) & 1}")
    print(f"  MS1 pin:        {(ioin >> 0) & 1}")
    print(f"  MS2 pin:        {(ioin >> 1) & 1}")
    print()

    # 2. Read current config before we change anything
    print("--- CURRENT REGISTER STATE (before config) ---")
    gconf = tmc_read(ser, 0x00)
    print(f"GCONF:     0x{gconf:08X}" if gconf else "GCONF:     read failed")
    chopconf = tmc_read(ser, 0x6C)
    print(f"CHOPCONF:  0x{chopconf:08X}" if chopconf else "CHOPCONF:  read failed")
    if chopconf:
        mres = (chopconf >> 24) & 0xF
        mres_map = {0: 256, 1: 128, 2: 64, 3: 32, 4: 16, 5: 8, 6: 4, 7: 2, 8: 1}
        print(f"  MRES={mres} → {mres_map.get(mres, '?')}x microstepping")
        print(f"  vsense={(chopconf >> 17) & 1}")
    drv_status = tmc_read(ser, 0x6F)
    print(f"DRV_STATUS: 0x{drv_status:08X}" if drv_status else "DRV_STATUS: read failed")
    if drv_status:
        print(f"  stst (standstill):  {(drv_status >> 31) & 1}")
        print(f"  ola (open load A):  {(drv_status >> 29) & 1}")
        print(f"  olb (open load B):  {(drv_status >> 30) & 1}")
        print(f"  s2ga (short GND A): {(drv_status >> 27) & 1}")
        print(f"  s2gb (short GND B): {(drv_status >> 28) & 1}")
        print(f"  s2vsa (short VS A): {(drv_status >> 25) & 1}")
        print(f"  s2vsb (short VS B): {(drv_status >> 26) & 1}")
        print(f"  ot (overtemp):      {(drv_status >> 25) & 1}")
        print(f"  otpw (temp warn):   {(drv_status >> 26) & 1}")
        print(f"  SG_RESULT:          {drv_status & 0x3FF}")
    print()

    # 3. Configure TMC2209
    print("--- CONFIGURING TMC2209 ---")
    # GCONF: I_scale_analog=1, mstep_reg_select=1
    tmc_write(ser, 0x00, 0x00000081)
    # IRUN=31, IHOLD=5, IHOLDDELAY=6
    tmc_write(ser, 0x10, (6 << 16) | (31 << 8) | 5)
    # CHOPCONF: 8x microstep (MRES=5), vsense=0
    tmc_write(ser, 0x6C, (5 << 24) | 0x00000053)
    # TPOWERDOWN
    tmc_write(ser, 0x11, 20)
    print("Config written.")
    print()

    # 4. Read back to verify
    print("--- REGISTER STATE (after config) ---")
    gconf = tmc_read(ser, 0x00)
    print(f"GCONF:     0x{gconf:08X}" if gconf else "GCONF:     read failed")
    if gconf:
        print(f"  I_scale_analog: {gconf & 1}")
        print(f"  internal_Rsense: {(gconf >> 1) & 1}")
        print(f"  en_spreadcycle: {(gconf >> 2) & 1}")
        print(f"  shaft (direction): {(gconf >> 4) & 1}")
        print(f"  mstep_reg_select: {(gconf >> 7) & 1}")
    chopconf = tmc_read(ser, 0x6C)
    print(f"CHOPCONF:  0x{chopconf:08X}" if chopconf else "CHOPCONF:  read failed")
    if chopconf:
        mres = (chopconf >> 24) & 0xF
        print(f"  MRES={mres} → {mres_map.get(mres, '?')}x microstepping")
    drv_status = tmc_read(ser, 0x6F)
    print(f"DRV_STATUS: 0x{drv_status:08X}" if drv_status else "DRV_STATUS: read failed")
    if drv_status:
        print(f"  stst (standstill):  {(drv_status >> 31) & 1}")
        print(f"  ola (open load A):  {(drv_status >> 29) & 1}")
        print(f"  olb (open load B):  {(drv_status >> 30) & 1}")
        print(f"  s2ga (short GND A): {(drv_status >> 27) & 1}")
        print(f"  s2gb (short GND B): {(drv_status >> 28) & 1}")
    print()

    # 5. Test motor with different configs
    print("--- MOTOR TESTS ---")
    print("Motor will be enabled. Watch for movement vs buzz.\n")

    # Test A: Full step mode (strongest torque)
    input("Test A: FULL STEP mode (1x), max current, 200 steps SLOW. Press Enter...")
    tmc_write(ser, 0x6C, (8 << 24) | 0x00000053)  # MRES=8 → 1x fullstep
    lgpio.gpio_write(chip, EN_PIN, 0)
    time.sleep(0.1)
    step_motor(chip, 200, 1, delay=0.02)
    time.sleep(0.3)
    lgpio.gpio_write(chip, EN_PIN, 1)
    print("  Result? (movement/buzz/nothing)")
    print()

    # Test B: Try SHAFT bit (internal direction invert in TMC2209)
    input("Test B: FULL STEP + SHAFT bit inverted. Press Enter...")
    tmc_write(ser, 0x00, 0x00000091)  # GCONF with shaft=1
    lgpio.gpio_write(chip, EN_PIN, 0)
    time.sleep(0.1)
    step_motor(chip, 200, 1, delay=0.02)
    time.sleep(0.3)
    lgpio.gpio_write(chip, EN_PIN, 1)
    tmc_write(ser, 0x00, 0x00000081)  # Reset shaft
    print("  Result?")
    print()

    # Test C: SpreadCycle mode instead of StealthChop
    input("Test C: FULL STEP + SpreadCycle (not StealthChop). Press Enter...")
    tmc_write(ser, 0x00, 0x00000085)  # en_spreadcycle=1
    lgpio.gpio_write(chip, EN_PIN, 0)
    time.sleep(0.1)
    step_motor(chip, 200, 1, delay=0.02)
    time.sleep(0.3)
    lgpio.gpio_write(chip, EN_PIN, 1)
    print("  Result?")
    print()

    # Test D: Ignore UART config entirely — let hardware defaults work
    input("Test D: Reset GCONF to 0 (hardware defaults, no UART override). Press Enter...")
    tmc_write(ser, 0x00, 0x00000001)  # Only I_scale_analog=1, mstep_reg_select=0
    lgpio.gpio_write(chip, EN_PIN, 0)
    time.sleep(0.1)
    step_motor(chip, 200, 1, delay=0.02)
    time.sleep(0.3)
    lgpio.gpio_write(chip, EN_PIN, 1)
    print("  Result?")
    print()

    # Read final DRV_STATUS for errors
    drv_status = tmc_read(ser, 0x6F)
    print(f"Final DRV_STATUS: 0x{drv_status:08X}" if drv_status else "DRV_STATUS: read failed")
    if drv_status:
        print(f"  open load A:  {(drv_status >> 29) & 1}")
        print(f"  open load B:  {(drv_status >> 30) & 1}")
        print(f"  short GND A:  {(drv_status >> 27) & 1}")
        print(f"  short GND B:  {(drv_status >> 28) & 1}")

    print("\nDiagnostic complete.")

except KeyboardInterrupt:
    print("\nAborted")
finally:
    lgpio.gpio_write(chip, EN_PIN, 1)
    lgpio.gpiochip_close(chip)
    ser.close()
