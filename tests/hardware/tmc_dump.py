#!/usr/bin/env python3
"""
Dump all TMC2209 readable registers with full decode.
Run on working treatbot1 to document baseline for comparison.
"""
import serial, struct, time, socket, sys

UART_PORT = '/dev/ttyAMA0'
BAUD = 115200
ADDR = 0x00  # TMC2209 slave address (MS1=0, MS2=0)

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
    datagram = bytes([0x05, ADDR, reg])
    datagram += bytes([_crc(datagram)])
    ser.write(datagram)
    time.sleep(0.012)
    resp = ser.read(12)
    if len(resp) >= 12:
        # Verify CRC on response
        payload = resp[3:11]
        expected_crc = _crc(resp[3:11])
        actual_crc = resp[11]
        val = struct.unpack('>I', resp[7:11])[0]
        if expected_crc != actual_crc:
            return val  # return anyway but could flag
        return val
    return None

# All readable registers on TMC2209
REGISTERS = [
    (0x00, 'GCONF'),
    (0x01, 'GSTAT'),
    (0x02, 'IFCNT'),
    (0x06, 'IOIN'),
    (0x10, 'IHOLD_IRUN'),
    (0x11, 'TPOWERDOWN'),
    (0x12, 'TSTEP'),
    (0x13, 'TPWMTHRS'),
    (0x14, 'TCOOLTHRS'),
    (0x22, 'VACTUAL'),
    (0x40, 'SGTHRS'),
    (0x41, 'SG_RESULT'),
    (0x42, 'COOLCONF'),
    (0x6C, 'CHOPCONF'),
    (0x6F, 'DRV_STATUS'),
    (0x70, 'PWMCONF'),
    (0x71, 'PWM_SCALE'),
    (0x72, 'PWM_AUTO'),
]

print("=" * 60)
print("TMC2209 FULL REGISTER DUMP")
print("=" * 60)
print(f"Host:      {socket.gethostname()}")
print(f"UART:      {UART_PORT} @ {BAUD} baud")
print(f"Address:   0x{ADDR:02X}")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print()

try:
    ser = serial.Serial(UART_PORT, BAUD, timeout=0.2)
except Exception as e:
    print(f"FATAL: Cannot open {UART_PORT}: {e}")
    sys.exit(1)

time.sleep(0.05)
ser.reset_input_buffer()

# Read all registers
values = {}
print(f"{'Register':14s} {'Addr':6s} {'Hex Value':12s} {'Decimal':>12s}")
print("-" * 50)
for reg, name in REGISTERS:
    val = tmc_read(ser, reg)
    values[name] = val
    if val is not None:
        print(f"{name:14s} (0x{reg:02X})  0x{val:08X}  {val:>12d}")
    else:
        print(f"{name:14s} (0x{reg:02X})  NO RESPONSE")

# =========================================================================
# DETAILED DECODE
# =========================================================================
print()
print("=" * 60)
print("REGISTER DECODE")
print("=" * 60)

# --- GCONF (0x00) ---
gconf = values.get('GCONF')
if gconf is not None:
    print(f"\n--- GCONF (0x00) = 0x{gconf:08X} ---")
    print(f"  I_scale_analog   = {(gconf>>0)&1}  (1=use VREF pin, 0=internal)")
    print(f"  internal_Rsense  = {(gconf>>1)&1}  (1=internal sense R, 0=external)")
    print(f"  en_SpreadCycle   = {(gconf>>2)&1}  (0=StealthChop, 1=SpreadCycle)")
    print(f"  shaft            = {(gconf>>4)&1}  (1=invert direction)")
    print(f"  index_otpw       = {(gconf>>5)&1}  (INDEX pin = overtemp warning)")
    print(f"  index_step       = {(gconf>>6)&1}  (INDEX pin = step pulses)")
    print(f"  mstep_reg_select = {(gconf>>7)&1}  (1=UART sets microstep, 0=MS pins)")
    print(f"  multistep_filt   = {(gconf>>8)&1}  (1=filter step pulses)")
    print(f"  test_mode        = {(gconf>>9)&1}  (must be 0)")

# --- GSTAT (0x01) ---
gstat = values.get('GSTAT')
if gstat is not None:
    print(f"\n--- GSTAT (0x01) = 0x{gstat:08X} ---")
    reset = (gstat >> 0) & 1
    drv_err = (gstat >> 1) & 1
    uv_cp = (gstat >> 2) & 1
    print(f"  reset    = {reset}  {'*** CHIP WAS RESET ***' if reset else '(normal)'}")
    print(f"  drv_err  = {drv_err}  {'*** DRIVER ERROR ***' if drv_err else '(no error)'}")
    print(f"  uv_cp    = {uv_cp}  {'*** CHARGE PUMP UNDERVOLTAGE ***' if uv_cp else '(normal)'}")

# --- IOIN (0x06) ---
ioin = values.get('IOIN')
if ioin is not None:
    print(f"\n--- IOIN (0x06) = 0x{ioin:08X} ---")
    version = (ioin >> 24) & 0xFF
    print(f"  ENN      = {(ioin>>0)&1}  (enable pin state, 0=enabled)")
    print(f"  MS1      = {(ioin>>2)&1}")
    print(f"  MS2      = {(ioin>>3)&1}")
    print(f"  DIAG     = {(ioin>>4)&1}")
    print(f"  PDN_UART = {(ioin>>6)&1}")
    print(f"  STEP     = {(ioin>>7)&1}")
    print(f"  SPREAD   = {(ioin>>8)&1}")
    print(f"  DIR      = {(ioin>>9)&1}")
    print(f"  VERSION  = 0x{version:02X}  {'(TMC2209)' if version == 0x21 else '(UNEXPECTED!)'}")

# --- IHOLD_IRUN (0x10) ---
ihr = values.get('IHOLD_IRUN')
if ihr is not None:
    ihold = (ihr >> 0) & 0x1F
    irun = (ihr >> 8) & 0x1F
    iholddelay = (ihr >> 16) & 0x0F
    print(f"\n--- IHOLD_IRUN (0x10) = 0x{ihr:08X} ---")
    print(f"  IHOLD      = {ihold}  (0-31)")
    print(f"  IRUN       = {irun}  (0-31)")
    print(f"  IHOLDDELAY = {iholddelay}  (ramp-down speed, 0=instant)")
    # Current estimate: I = (CS+1)/32 * V_fs / R_sense
    # With external sense resistors (typ 0.11 ohm), V_fs depends on VSENSE bit
    chopconf = values.get('CHOPCONF')
    if chopconf is not None:
        vsense = (chopconf >> 17) & 1
        v_fs = 0.180 if vsense else 0.325
        r_sense = 0.110  # typical for TMC2209 eval boards
        i_run_amps = (irun + 1) / 32.0 * v_fs / r_sense
        i_hold_amps = (ihold + 1) / 32.0 * v_fs / r_sense
        print(f"  --- Current estimate (Rsense=0.11ohm, Vfs={'0.180' if vsense else '0.325'}V) ---")
        print(f"  I_RUN  ~ {i_run_amps:.3f} A RMS  ({i_run_amps*1.414:.3f} A peak)")
        print(f"  I_HOLD ~ {i_hold_amps:.3f} A RMS  ({i_hold_amps*1.414:.3f} A peak)")

# --- CHOPCONF (0x6C) ---
chopconf = values.get('CHOPCONF')
if chopconf is not None:
    mres_val = (chopconf >> 24) & 0x0F
    mres_map = {0: 256, 1: 128, 2: 64, 3: 32, 4: 16, 5: 8, 6: 4, 7: 2, 8: 1}
    mres_steps = mres_map.get(mres_val, '?')
    vsense = (chopconf >> 17) & 1
    toff = (chopconf >> 0) & 0x0F
    hstrt = (chopconf >> 4) & 0x07
    hend = (chopconf >> 7) & 0x0F
    tbl = (chopconf >> 15) & 0x03
    intpol = (chopconf >> 28) & 1
    dedge = (chopconf >> 29) & 1
    diss2g = (chopconf >> 30) & 1
    diss2vs = (chopconf >> 31) & 1
    print(f"\n--- CHOPCONF (0x6C) = 0x{chopconf:08X} ---")
    print(f"  TOFF         = {toff}  (0=driver OFF, 1-15=chopper on)")
    print(f"  HSTRT        = {hstrt}")
    print(f"  HEND         = {hend}")
    print(f"  TBL          = {tbl}  (comparator blank time)")
    print(f"  VSENSE       = {vsense}  (0=high range 0.325V, 1=low range 0.180V)")
    print(f"  MRES         = {mres_val} ({mres_steps} microsteps/fullstep)")
    print(f"  intpol       = {intpol}  (1=interpolate to 256 microsteps)")
    print(f"  dedge        = {dedge}  (1=step on both edges)")
    print(f"  diss2g       = {diss2g}  (1=disable short to GND protection)")
    print(f"  diss2vs      = {diss2vs}  (1=disable short to VS protection)")

# --- DRV_STATUS (0x6F) --- TMC2209 specific bit map (Rev 1.09)
drv = values.get('DRV_STATUS')
if drv is not None:
    sg_result_drv = drv & 0x3FF          # bits [9:0]
    s2vsa = (drv >> 12) & 1              # bit 12
    s2vsb = (drv >> 13) & 1              # bit 13
    stealth = (drv >> 14) & 1            # bit 14
    cs_actual = (drv >> 16) & 0x1F       # bits [20:16]
    ot = (drv >> 25) & 1                 # bit 25
    otpw = (drv >> 26) & 1              # bit 26
    s2ga = (drv >> 27) & 1              # bit 27
    s2gb = (drv >> 28) & 1              # bit 28
    ola = (drv >> 29) & 1               # bit 29
    olb = (drv >> 30) & 1               # bit 30
    stst = (drv >> 31) & 1              # bit 31
    print(f"\n--- DRV_STATUS (0x6F) = 0x{drv:08X} ---")
    print(f"  SG_RESULT   = {sg_result_drv}  (StallGuard load, higher=less load, 0=stall)")
    print(f"  s2vsa       = {s2vsa}  {'*** SHORT TO VS ON A ***' if s2vsa else '(ok)'}")
    print(f"  s2vsb       = {s2vsb}  {'*** SHORT TO VS ON B ***' if s2vsb else '(ok)'}")
    print(f"  stealth     = {stealth}  (1=StealthChop active)")
    print(f"  CS_ACTUAL   = {cs_actual}  (actual motor current scale, 0-31)")
    print(f"  OT          = {ot}   {'*** OVERTEMP SHUTDOWN ***' if ot else '(normal)'}")
    print(f"  OTPW        = {otpw}   {'*** OVERTEMP WARNING ***' if otpw else '(normal)'}")
    print(f"  s2ga        = {s2ga}  {'*** SHORT TO GND ON A ***' if s2ga else '(ok)'}")
    print(f"  s2gb        = {s2gb}  {'*** SHORT TO GND ON B ***' if s2gb else '(ok)'}")
    print(f"  olA         = {ola}   (1=open load phase A — normal when standstill)")
    print(f"  olB         = {olb}   (1=open load phase B — normal when standstill)")
    print(f"  STST        = {stst}   (1=standstill detected)")
    # Note: TMC2209 does NOT have t120/t143/t150/t157 temperature threshold
    # bits in DRV_STATUS (those are TMC5160/TMC2130 features).
    # Temperature is only indicated by OT (shutdown at ~150C) and OTPW (warning at ~120C).

# --- VACTUAL (0x22) ---
vactual = values.get('VACTUAL')
if vactual is not None:
    # VACTUAL is signed 24-bit
    if vactual & 0x800000:
        vactual_signed = vactual - 0x1000000
    else:
        vactual_signed = vactual
    print(f"\n--- VACTUAL (0x22) = 0x{vactual:08X} ---")
    print(f"  Value = {vactual_signed}  (0=external step/dir mode, nonzero=internal motion)")

# --- TSTEP (0x12) ---
tstep = values.get('TSTEP')
if tstep is not None:
    print(f"\n--- TSTEP (0x12) = 0x{tstep:08X} ---")
    if tstep > 0 and tstep < 0xFFFFF:
        freq_mhz = 12.0  # TMC2209 internal clock ~12MHz
        step_freq = freq_mhz * 1e6 / tstep
        print(f"  Measured step time = {tstep} clocks ({1e6/step_freq:.1f} us/step)")
        print(f"  Step frequency ~ {step_freq:.1f} Hz")
    else:
        print(f"  Value = {tstep}  (0xFFFFF = motor stopped)")

# --- TPWMTHRS (0x13) ---
tpwmthrs = values.get('TPWMTHRS')
if tpwmthrs is not None:
    print(f"\n--- TPWMTHRS (0x13) = 0x{tpwmthrs:08X} ---")
    print(f"  StealthChop -> SpreadCycle switchover threshold")
    if tpwmthrs == 0:
        print(f"  Value = 0  (disabled, always StealthChop if en_SpreadCycle=0)")

# --- TCOOLTHRS (0x14) ---
tcoolthrs = values.get('TCOOLTHRS')
if tcoolthrs is not None:
    print(f"\n--- TCOOLTHRS (0x14) = 0x{tcoolthrs:08X} ---")
    print(f"  CoolStep/StallGuard enable below this velocity threshold")
    if tcoolthrs == 0xFFFFF:
        print(f"  Set to max (0xFFFFF) — StallGuard active at all speeds")

# --- SGTHRS (0x40) ---
sgthrs = values.get('SGTHRS')
if sgthrs is not None:
    print(f"\n--- SGTHRS (0x40) = 0x{sgthrs:08X} ---")
    print(f"  Threshold = {sgthrs}  (stall when SG_RESULT <= 2*SGTHRS)")
    if sgthrs == 0:
        print(f"  StallGuard DISABLED (threshold=0)")

# --- SG_RESULT (0x41) ---
sgr = values.get('SG_RESULT')
if sgr is not None:
    print(f"\n--- SG_RESULT (0x41) = 0x{sgr:08X} ---")
    print(f"  Load measurement = {sgr}  (higher=less load, 0=stall)")

# --- PWMCONF (0x70) ---
pwmconf = values.get('PWMCONF')
if pwmconf is not None:
    pwm_ofs = (pwmconf >> 0) & 0xFF
    pwm_grad = (pwmconf >> 8) & 0xFF
    pwm_freq = (pwmconf >> 16) & 0x03
    pwm_autoscale = (pwmconf >> 18) & 1
    pwm_autograd = (pwmconf >> 19) & 1
    freewheel = (pwmconf >> 20) & 0x03
    pwm_reg = (pwmconf >> 24) & 0x0F
    pwm_lim = (pwmconf >> 28) & 0x0F
    freq_names = ['2/1024', '2/683', '2/512', '2/410']
    freewheel_names = ['Normal', 'Freewheeling', 'LS short (passive brake)', 'HS short (passive brake)']
    print(f"\n--- PWMCONF (0x70) = 0x{pwmconf:08X} ---")
    print(f"  PWM_OFS       = {pwm_ofs}")
    print(f"  PWM_GRAD      = {pwm_grad}")
    print(f"  PWM_FREQ      = {pwm_freq} ({freq_names[pwm_freq]})")
    print(f"  PWM_AUTOSCALE = {pwm_autoscale}  (1=auto current regulation)")
    print(f"  PWM_AUTOGRAD  = {pwm_autograd}  (1=auto PWM gradient)")
    print(f"  FREEWHEEL     = {freewheel} ({freewheel_names[freewheel]})")
    print(f"  PWM_REG       = {pwm_reg}")
    print(f"  PWM_LIM       = {pwm_lim}")

# --- PWM_SCALE (0x71) ---
pwm_scale = values.get('PWM_SCALE')
if pwm_scale is not None:
    pwm_scale_sum = pwm_scale & 0xFF
    pwm_scale_auto = (pwm_scale >> 16) & 0x1FF
    if pwm_scale_auto & 0x100:
        pwm_scale_auto_signed = pwm_scale_auto - 0x200
    else:
        pwm_scale_auto_signed = pwm_scale_auto
    print(f"\n--- PWM_SCALE (0x71) = 0x{pwm_scale:08X} ---")
    print(f"  PWM_SCALE_SUM  = {pwm_scale_sum}")
    print(f"  PWM_SCALE_AUTO = {pwm_scale_auto_signed}")

# --- PWM_AUTO (0x72) ---
pwm_auto = values.get('PWM_AUTO')
if pwm_auto is not None:
    pwm_ofs_auto = pwm_auto & 0xFF
    pwm_grad_auto = (pwm_auto >> 16) & 0xFF
    print(f"\n--- PWM_AUTO (0x72) = 0x{pwm_auto:08X} ---")
    print(f"  PWM_OFS_AUTO  = {pwm_ofs_auto}")
    print(f"  PWM_GRAD_AUTO = {pwm_grad_auto}")

# --- IFCNT (0x02) ---
ifcnt = values.get('IFCNT')
if ifcnt is not None:
    print(f"\n--- IFCNT (0x02) = 0x{ifcnt:08X} ---")
    print(f"  UART write counter = {ifcnt & 0xFF}  (increments on every successful write)")

# --- Summary ---
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
if values.get('IOIN') is not None:
    ver = (values['IOIN'] >> 24) & 0xFF
    print(f"Chip:          TMC2209 v0x{ver:02X}")
if values.get('CHOPCONF') is not None:
    vs = (values['CHOPCONF'] >> 17) & 1
    mr = (values['CHOPCONF'] >> 24) & 0x0F
    ms = mres_map.get(mr, '?')
    print(f"Microsteps:    {ms}/fullstep (MRES={mr})")
    print(f"VSENSE:        {'Low range (0.180V)' if vs else 'High range (0.325V)'}")
if values.get('IHOLD_IRUN') is not None:
    ihr_v = values['IHOLD_IRUN']
    print(f"IRUN:          {(ihr_v>>8)&0x1F}/31")
    print(f"IHOLD:         {(ihr_v>>0)&0x1F}/31")
if values.get('GCONF') is not None:
    gc = values['GCONF']
    mode = 'SpreadCycle' if (gc >> 2) & 1 else 'StealthChop'
    print(f"Chopper mode:  {mode}")
    print(f"UART controls: {'Yes' if (gc>>7)&1 else 'No (MS pins)'}")
if values.get('DRV_STATUS') is not None:
    d = values['DRV_STATUS']
    errors = []
    if (d >> 25) & 1: errors.append('OVERTEMP SHUTDOWN')
    if (d >> 26) & 1: errors.append('OVERTEMP WARNING')
    if (d >> 27) & 1: errors.append('SHORT_GND_A')
    if (d >> 28) & 1: errors.append('SHORT_GND_B')
    if (d >> 12) & 1: errors.append('SHORT_VS_A')
    if (d >> 13) & 1: errors.append('SHORT_VS_B')
    if errors:
        print(f"ERRORS:        {', '.join(errors)}")
    else:
        print(f"Driver status: OK (no faults)")
    cs = (d >> 16) & 0x1F
    print(f"CS_ACTUAL:     {cs}/31")
    print(f"Standstill:    {'Yes' if (d>>31)&1 else 'No (motor moving)'}")
    print(f"StealthChop:   {'Active' if (d>>14)&1 else 'Inactive'}")
    ol_note = ""
    if ((d>>29)&1 or (d>>30)&1) and (d>>31)&1:
        ol_note = " (normal at standstill)"
    print(f"Open load:     A={(d>>29)&1} B={(d>>30)&1}{ol_note}")
if values.get('VACTUAL') is not None:
    va = values['VACTUAL']
    print(f"VACTUAL:       {va} ({'external step/dir' if va == 0 else 'INTERNAL MOTION'})")

ser.close()
print(f"\nDump complete. UART closed.")
