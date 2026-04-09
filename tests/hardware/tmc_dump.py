#!/usr/bin/env python3
"""Dump all TMC2209 readable registers. Run on working treatbot1 to compare."""
import serial, struct, time

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
    time.sleep(0.01)
    resp = ser.read(12)
    if len(resp) >= 12:
        return struct.unpack('>I', resp[7:11])[0]
    return None

ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.2)

regs = [
    (0x00, 'GCONF'), (0x01, 'GSTAT'), (0x02, 'IFCNT'),
    (0x06, 'IOIN'), (0x6C, 'CHOPCONF'), (0x6F, 'DRV_STATUS'),
    (0x40, 'SGTHRS'), (0x41, 'SG_RESULT'), (0x42, 'COOLCONF'),
]

print("=== TMC2209 REGISTER DUMP ===")
import socket
print(f"Host: {socket.gethostname()}")
for reg, name in regs:
    val = tmc_read(ser, reg)
    if val is not None:
        print(f"{name:12s} (0x{reg:02X}): 0x{val:08X}")
    else:
        print(f"{name:12s} (0x{reg:02X}): NO RESPONSE")

# Decode key fields
ioin = tmc_read(ser, 0x06)
if ioin:
    print(f"\nMS1={(ioin>>0)&1} MS2={(ioin>>1)&1} DIAG={(ioin>>2)&1} PDN_UART={(ioin>>6)&1} STEP={(ioin>>7)&1} DIR={(ioin>>3)&1}")
    print(f"Version=0x{(ioin>>24)&0xFF:02X}")

drv = tmc_read(ser, 0x6F)
if drv:
    print(f"CS_ACTUAL={(drv>>16)&0x1F} stst={(drv>>31)&1} olA={(drv>>29)&1} olB={(drv>>30)&1} s2gA={(drv>>27)&1} s2gB={(drv>>28)&1}")

gconf = tmc_read(ser, 0x00)
if gconf:
    print(f"I_scale_analog={(gconf>>0)&1} internal_Rsense={(gconf>>1)&1} en_spreadcycle={(gconf>>2)&1} shaft={(gconf>>4)&1} mstep_reg_select={(gconf>>7)&1}")

ser.close()
