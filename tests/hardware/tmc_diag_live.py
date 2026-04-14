#!/usr/bin/env python3
"""
TMC2209 LIVE diagnostic — reads registers WHILE stepping.
Pinpoints why the motor won't rotate even though MSCNT advances.

Run on the BROKEN unit: sudo python3 tests/hardware/tmc_diag_live.py
"""
import lgpio, serial, struct, time, sys

STEP_PIN = 12
DIR_PIN = 16
EN_PIN = 24
UART_PORT = '/dev/ttyAMA0'
BAUD = 115200

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

def tmc_read(ser, reg):
    ser.reset_input_buffer()
    datagram = bytes([0x05, 0x00, reg])
    datagram += bytes([_crc(datagram)])
    ser.write(datagram)
    time.sleep(0.012)
    resp = ser.read(12)
    if len(resp) >= 12:
        return struct.unpack('>I', resp[7:11])[0]
    return None

def tmc_write(ser, reg, value):
    datagram = bytes([0x05, 0x00, reg | 0x80]) + struct.pack('>I', value)
    datagram += bytes([_crc(datagram)])
    ser.write(datagram)
    time.sleep(0.005)
    ser.reset_input_buffer()

def decode_drv(val):
    return {
        'sg_result': val & 0x3FF,
        's2vsa': (val >> 12) & 1,
        's2vsb': (val >> 13) & 1,
        'stealth': (val >> 14) & 1,
        'cs_actual': (val >> 16) & 0x1F,
        'ot': (val >> 25) & 1,
        'otpw': (val >> 26) & 1,
        's2ga': (val >> 27) & 1,
        's2gb': (val >> 28) & 1,
        'ola': (val >> 29) & 1,
        'olb': (val >> 30) & 1,
        'stst': (val >> 31) & 1,
    }

def decode_ioin(val):
    return {
        'enn': (val >> 0) & 1,
        'ms1': (val >> 2) & 1,
        'ms2': (val >> 3) & 1,
        'diag': (val >> 4) & 1,
        'pdn_uart': (val >> 6) & 1,
        'step': (val >> 7) & 1,
        'spread': (val >> 8) & 1,
        'dir': (val >> 9) & 1,
        'version': (val >> 24) & 0xFF,
    }

print("=" * 60)
print("TMC2209 LIVE STEPPING DIAGNOSTIC")
print("=" * 60)

chip = lgpio.gpiochip_open(0)
lgpio.gpio_claim_output(chip, STEP_PIN)
lgpio.gpio_claim_output(chip, DIR_PIN)
lgpio.gpio_claim_output(chip, EN_PIN)
lgpio.gpio_write(chip, EN_PIN, 1)  # disabled
lgpio.gpio_write(chip, STEP_PIN, 0)
lgpio.gpio_write(chip, DIR_PIN, 1)

ser = serial.Serial(UART_PORT, BAUD, timeout=0.2)
time.sleep(0.05)
ser.reset_input_buffer()

# ─── PHASE 1: Pre-config baseline ───
print("\n--- PHASE 1: Before any config (power-on defaults) ---")
ioin = tmc_read(ser, 0x06)
drv = tmc_read(ser, 0x6F)
gconf = tmc_read(ser, 0x00)
chopconf = tmc_read(ser, 0x6C)
ifcnt_before = tmc_read(ser, 0x02)

if ioin is not None:
    d = decode_ioin(ioin)
    print(f"  IOIN    = 0x{ioin:08X}  VERSION=0x{d['version']:02X}  ENN={d['enn']}")
if gconf is not None:
    print(f"  GCONF   = 0x{gconf:08X}  I_scale_analog={(gconf>>0)&1}  mstep_reg_select={(gconf>>7)&1}")
if chopconf is not None:
    toff = chopconf & 0xF
    vsense = (chopconf >> 17) & 1
    mres = (chopconf >> 24) & 0xF
    print(f"  CHOPCONF= 0x{chopconf:08X}  TOFF={toff}  VSENSE={vsense}  MRES={mres}")
if drv is not None:
    d = decode_drv(drv)
    print(f"  DRV_STS = 0x{drv:08X}  CS_ACTUAL={d['cs_actual']}  STST={d['stst']}  "
          f"OT={d['ot']}  s2g={d['s2ga']}/{d['s2gb']}  ol={d['ola']}/{d['olb']}")
print(f"  IFCNT   = {ifcnt_before}")

# ─── PHASE 2: Configure TMC2209 ───
print("\n--- PHASE 2: Configuring TMC2209 ---")

# First try with I_scale_analog=0 (internal reference, ignores VREF pin)
# This is the KEY difference to test — if VREF isn't connected, I_scale_analog=1
# means current = 0 regardless of IRUN setting
gconf_val = 0x00000080  # I_scale_analog=0, mstep_reg_select=1
tmc_write(ser, 0x00, gconf_val)
tmc_write(ser, 0x10, (6 << 16) | (31 << 8) | 5)  # IRUN=31, IHOLD=5
tmc_write(ser, 0x6C, (5 << 24) | 0x00000053)      # 8x microstep, TOFF=3
tmc_write(ser, 0x11, 20)                           # TPOWERDOWN

ifcnt_after = tmc_read(ser, 0x02)
gconf_readback = tmc_read(ser, 0x00)
chopconf_readback = tmc_read(ser, 0x6C)

print(f"  IFCNT: {ifcnt_before} -> {ifcnt_after} (delta={((ifcnt_after or 0)-(ifcnt_before or 0)) & 0xFF})")
print(f"  GCONF   = 0x{gconf_readback:08X}  I_scale_analog={(gconf_readback>>0)&1}")
print(f"  CHOPCONF= 0x{chopconf_readback:08X}  TOFF={chopconf_readback & 0xF}")

# ─── PHASE 3: Enable motor, read status ───
print("\n--- PHASE 3: Motor ENABLED (EN=LOW) ---")
lgpio.gpio_write(chip, EN_PIN, 0)
time.sleep(0.1)

ioin = tmc_read(ser, 0x06)
drv = tmc_read(ser, 0x6F)
if ioin is not None:
    d = decode_ioin(ioin)
    enn_ok = "OK (driver enabled)" if d['enn'] == 0 else "*** STILL HIGH — EN pin not reaching chip! ***"
    print(f"  IOIN    = 0x{ioin:08X}  ENN={d['enn']}  {enn_ok}")
if drv is not None:
    d = decode_drv(drv)
    cs_ok = "OK" if d['cs_actual'] > 0 else "*** ZERO — no current flowing! ***"
    print(f"  DRV_STS = 0x{drv:08X}  CS_ACTUAL={d['cs_actual']}  {cs_ok}")
    print(f"    stealth={d['stealth']}  STST={d['stst']}  OT={d['ot']}  OTPW={d['otpw']}")
    print(f"    s2ga={d['s2ga']}  s2gb={d['s2gb']}  s2vsa={d['s2vsa']}  s2vsb={d['s2vsb']}")
    print(f"    olA={d['ola']}  olB={d['olb']}")

# ─── PHASE 4: Step and read MSCNT + DRV_STATUS mid-motion ───
print("\n--- PHASE 4: Stepping 50 steps, reading registers mid-step ---")
lgpio.gpio_write(chip, DIR_PIN, 1)
time.sleep(0.001)

mscnt_before = tmc_read(ser, 0x6A)
print(f"  MSCNT before: {mscnt_before}")

# Step 25, pause, read, step 25 more
for i in range(25):
    lgpio.gpio_write(chip, STEP_PIN, 1)
    time.sleep(0.010)
    lgpio.gpio_write(chip, STEP_PIN, 0)
    time.sleep(0.010)

# Read mid-motion
mscnt_mid = tmc_read(ser, 0x6A)
drv_mid = tmc_read(ser, 0x6F)
ioin_mid = tmc_read(ser, 0x06)

for i in range(25):
    lgpio.gpio_write(chip, STEP_PIN, 1)
    time.sleep(0.010)
    lgpio.gpio_write(chip, STEP_PIN, 0)
    time.sleep(0.010)

mscnt_after = tmc_read(ser, 0x6A)
drv_after = tmc_read(ser, 0x6F)

print(f"  MSCNT: {mscnt_before} -> {mscnt_mid} -> {mscnt_after}")
if drv_mid is not None:
    d = decode_drv(drv_mid)
    print(f"  DRV_STS (mid-step)  = 0x{drv_mid:08X}")
    print(f"    CS_ACTUAL={d['cs_actual']}  stealth={d['stealth']}  STST={d['stst']}")
    print(f"    OT={d['ot']}  OTPW={d['otpw']}")
    print(f"    s2ga={d['s2ga']}  s2gb={d['s2gb']}  s2vsa={d['s2vsa']}  s2vsb={d['s2vsb']}")
    print(f"    olA={d['ola']}  olB={d['olb']}  SG_RESULT={d['sg_result']}")
if drv_after is not None:
    d = decode_drv(drv_after)
    print(f"  DRV_STS (after 50)  = 0x{drv_after:08X}")
    print(f"    CS_ACTUAL={d['cs_actual']}  stealth={d['stealth']}  STST={d['stst']}")

# ─── PHASE 5: Try I_scale_analog=1 (VREF pin) for comparison ───
print("\n--- PHASE 5: Switching to I_scale_analog=1 (VREF pin mode) ---")
tmc_write(ser, 0x00, 0x00000081)  # I_scale_analog=1
time.sleep(0.1)

drv_vref = tmc_read(ser, 0x6F)
if drv_vref is not None:
    d = decode_drv(drv_vref)
    print(f"  DRV_STS = 0x{drv_vref:08X}  CS_ACTUAL={d['cs_actual']}")
    if d['cs_actual'] == 0:
        print(f"  *** CS_ACTUAL dropped to 0 with I_scale_analog=1 ***")
        print(f"  *** THIS MEANS VREF PIN HAS NO VOLTAGE ***")
        print(f"  *** FIX: Use I_scale_analog=0 in GCONF (0x80 instead of 0x81) ***")
    else:
        print(f"  VREF is providing voltage, I_scale_analog mode works")

# Step again with I_scale_analog=1
print("  Stepping 50 with I_scale_analog=1...")
for i in range(50):
    lgpio.gpio_write(chip, STEP_PIN, 1)
    time.sleep(0.010)
    lgpio.gpio_write(chip, STEP_PIN, 0)
    time.sleep(0.010)

drv_vref2 = tmc_read(ser, 0x6F)
if drv_vref2 is not None:
    d = decode_drv(drv_vref2)
    print(f"  DRV_STS after step = 0x{drv_vref2:08X}  CS_ACTUAL={d['cs_actual']}")

# ─── Cleanup ───
print("\n--- Disabling motor ---")
lgpio.gpio_write(chip, EN_PIN, 1)
lgpio.gpio_write(chip, STEP_PIN, 0)
lgpio.gpiochip_close(chip)
ser.close()

# ─── Verdict ───
print("\n" + "=" * 60)
print("VERDICT")
print("=" * 60)

issues = []
if ioin is not None and (decode_ioin(ioin)['enn'] != 0):
    issues.append("EN pin not reaching TMC2209 — check GPIO24 wiring to EN input")
if drv_mid is not None:
    d = decode_drv(drv_mid)
    if d['cs_actual'] == 0:
        issues.append("CS_ACTUAL=0 during stepping — no motor current, check VREF or use I_scale_analog=0")
    if d['ot']:
        issues.append("Overtemp shutdown — driver is thermally shutting down")
    if d['s2ga'] or d['s2gb']:
        issues.append("Short to GND detected — check motor wiring for shorts")
    if d['s2vsa'] or d['s2vsb']:
        issues.append("Short to VS detected — check motor wiring for shorts")
    if d['stst']:
        issues.append("STST=1 during stepping — driver thinks motor is standstill (no chopper activity)")
if chopconf_readback is not None and (chopconf_readback & 0xF) == 0:
    issues.append("TOFF=0 — driver output stage is OFF, chopper disabled")

if issues:
    for i, issue in enumerate(issues):
        print(f"  [{i+1}] {issue}")
else:
    print("  No obvious issues found in registers — problem may be mechanical")
    print("  or the motor coil pairs are swapped (swap middle 2 wires)")

print()
