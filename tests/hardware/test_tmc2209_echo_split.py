#!/usr/bin/env python3
"""
TMC2209 UART probe with explicit echo/reply split.

In single-wire UART, Pi RX sees its own TX as an echo. A real chip reply
follows the echo. This script reads enough bytes to capture both and
prints them separated so we can distinguish:
  - 4 bytes back, all matching TX  -> only echo, chip silent
  - 12 bytes back (4 TX + 8 reply) -> chip responded
  - 0 bytes back                   -> wire physically broken (no TX/RX path)

Designed to be run under different physical conditions (RX connected vs
disconnected) to validate whether the probe is reading real hardware or
some internal loopback.
"""

import serial
import struct
import sys
import time

UART_PORT = '/dev/ttyAMA0'
BAUD = 115200


def crc(data):
    c = 0
    for b in data:
        for _ in range(8):
            if (c >> 7) ^ (b & 1):
                c = ((c << 1) ^ 0x07) & 0xFF
            else:
                c = (c << 1) & 0xFF
            b >>= 1
    return c


def probe(ser, addr, reg=0x00):
    """Send a read request to (addr, reg). Return (tx_bytes, rx_bytes)."""
    ser.reset_input_buffer()
    tx = bytes([0x05, addr, reg])
    tx += bytes([crc(tx)])
    ser.write(tx)
    time.sleep(0.03)              # wait long enough for any chip reply
    rx = ser.read(16)             # read more than needed; trailing zeros tell us nothing came after
    return tx, rx


def classify(tx, rx):
    """Return ('echo_only' | 'chip_replied' | 'silent' | 'odd', echo, reply)."""
    if len(rx) == 0:
        return 'silent', b'', b''
    if len(rx) < len(tx):
        return 'odd', rx, b''
    echo = rx[:len(tx)]
    reply = rx[len(tx):]
    if echo != tx:
        return 'odd', echo, reply
    if len(reply) == 0:
        return 'echo_only', echo, reply
    return 'chip_replied', echo, reply


def main():
    ser = serial.Serial(UART_PORT, BAUD, timeout=0.3)
    print(f"=== TMC2209 echo/reply probe on {UART_PORT} ===")
    for addr in (0, 1, 2, 3):
        tx, rx = probe(ser, addr, reg=0x00)  # GCONF
        verdict, echo, reply = classify(tx, rx)
        print(f"\naddr={addr}  TX={tx.hex()}")
        print(f"         RX[{len(rx)}b]={rx.hex()}")
        print(f"         echo={echo.hex()}  reply={reply.hex()}  -> {verdict}")
        if verdict == 'chip_replied' and len(reply) >= 8:
            # TMC reply format: sync(0x05) addr(0xFF) reg data[4] crc
            data = reply[3:7]
            val = struct.unpack('>I', data)[0]
            print(f"         REGISTER VALUE = 0x{val:08X}")
            # Also pull IOIN (0x06) so we can read chip version
            tx2, rx2 = probe(ser, addr, reg=0x06)
            v2, e2, r2 = classify(tx2, rx2)
            if v2 == 'chip_replied' and len(r2) >= 8:
                ioin = struct.unpack('>I', r2[3:7])[0]
                ver = ioin >> 24
                print(f"         IOIN=0x{ioin:08X}  CHIP VERSION = 0x{ver:02X}  "
                      f"{'(MATCH TMC2209)' if ver == 0x21 else '(UNEXPECTED)'}")
    ser.close()


if __name__ == '__main__':
    main()
