#!/usr/bin/env python3
"""
TMC2209 known-good register dump.

Read-only. Captures GCONF, IOIN, CHOPCONF, DRV_STATUS at slave 0x00
to establish a reference state for cross-fleet comparison.
"""

import serial
import struct
import time

UART_PORT = '/dev/ttyAMA0'
BAUD = 115200
SLAVE = 0x00

REGS = [
    ('GCONF',      0x00),
    ('IOIN',       0x06),
    ('CHOPCONF',   0x6C),
    ('DRV_STATUS', 0x6F),
]


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


def read_reg(ser, reg):
    ser.reset_input_buffer()
    datagram = bytes([0x05, SLAVE, reg])
    datagram += bytes([_crc(datagram)])
    ser.write(datagram)
    time.sleep(0.02)
    response = ser.read(32)
    # Strip the 4-byte TX echo
    if len(response) >= 12 and response[:4] == datagram:
        reply = response[4:12]
    elif len(response) >= 8:
        reply = response[-8:]
    else:
        return None, response
    value = struct.unpack('>I', reply[3:7])[0]
    return value, response


def decode_gconf(v):
    return {
        'I_scale_analog':   v & 1,
        'internal_Rsense':  (v >> 1) & 1,
        'en_spreadcycle':   (v >> 2) & 1,
        'shaft':            (v >> 4) & 1,
        'index_otpw':       (v >> 5) & 1,
        'index_step':       (v >> 6) & 1,
        'pdn_disable':      (v >> 6) & 1,
        'mstep_reg_select': (v >> 7) & 1,
        'multistep_filt':   (v >> 8) & 1,
    }


def decode_ioin(v):
    return {
        'version':  (v >> 24) & 0xFF,
        'STEP':     (v >> 7) & 1,
        'DIR':      (v >> 3) & 1,
        'MS1':      (v >> 0) & 1,
        'MS2':      (v >> 1) & 1,
        'PDN_UART': (v >> 6) & 1,
        'SEL_A':    (v >> 8) & 1,
        'DIAG':     (v >> 2) & 1,
        'ENN':      (v >> 4) & 1,
    }


def decode_chopconf(v):
    mres_map = {0: 256, 1: 128, 2: 64, 3: 32, 4: 16, 5: 8, 6: 4, 7: 2, 8: 1}
    mres = (v >> 24) & 0xF
    return {
        'TOFF':     v & 0xF,
        'HSTRT':    (v >> 4) & 0x7,
        'HEND':     (v >> 7) & 0xF,
        'TBL':      (v >> 15) & 0x3,
        'vsense':   (v >> 17) & 1,
        'MRES':     mres,
        'microsteps': mres_map.get(mres, '?'),
        'intpol':   (v >> 28) & 1,
        'dedge':    (v >> 29) & 1,
        'diss2g':   (v >> 30) & 1,
        'diss2vs':  (v >> 31) & 1,
    }


def decode_drv_status(v):
    return {
        'standstill':     (v >> 31) & 1,
        'open_load_A':    (v >> 29) & 1,
        'open_load_B':    (v >> 30) & 1,
        'short_GND_A':    (v >> 27) & 1,
        'short_GND_B':    (v >> 28) & 1,
        'short_VS_A':     (v >> 25) & 1,
        'short_VS_B':     (v >> 26) & 1,
        'overtemp':       (v >> 25) & 1,
        'overtemp_warn':  (v >> 26) & 1,
        'stealthchop':    (v >> 30) & 1,
        'cs_actual':      (v >> 16) & 0x1F,
        'sg_result':      v & 0x3FF,
    }


def main():
    ser = serial.Serial(UART_PORT, BAUD, timeout=0.1)
    print("=" * 60)
    print(f"TMC2209 reference register dump (slave=0x{SLAVE:02X})")
    print("=" * 60)
    for name, reg in REGS:
        value, raw = read_reg(ser, reg)
        if value is None:
            print(f"\n{name} (0x{reg:02X}): READ FAILED  raw={raw.hex(' ')}")
            continue
        print(f"\n{name} (0x{reg:02X}) = 0x{value:08X}")
        if name == 'GCONF':
            decoded = decode_gconf(value)
        elif name == 'IOIN':
            decoded = decode_ioin(value)
        elif name == 'CHOPCONF':
            decoded = decode_chopconf(value)
        elif name == 'DRV_STATUS':
            decoded = decode_drv_status(value)
        for k, val in decoded.items():
            if isinstance(val, int) and k == 'version':
                print(f"  {k:18s} = 0x{val:02X}")
            else:
                print(f"  {k:18s} = {val}")
    ser.close()


if __name__ == '__main__':
    main()
