# TMC2209 UART Setup Guide (Per-Unit, Not in Git)

Each new WIM-Z unit needs this OS-level configuration before the stepper
dispenser will work. These settings live outside the repo and must be
applied manually once per Raspberry Pi.

## Wiring Reference

```
TMC2209 Pin 4 (PDN_UART) ──┬── 1K resistor ── GPIO14 TX (Pin 8)
                            └── direct ─────── GPIO15 RX (Pin 10)
TMC2209 STEP  ── GPIO12 (Pin 32)
TMC2209 DIR   ── GPIO16 (Pin 36)
TMC2209 EN    ── GPIO24 (Pin 18)
```

Single-wire half-duplex: TX and RX both go to the same TMC2209 PDN_UART
pin. The 1K resistor on TX prevents bus contention.

## Step 1: Enable UART in boot config

Edit `/boot/firmware/config.txt` and ensure these lines exist (under `[all]`):

```
dtparam=uart0=on
enable_uart=1
```

## Step 2: Disable serial console on UART

Check `/boot/firmware/cmdline.txt`. It must NOT contain `console=serial0,115200`
or `console=ttyAMA0,115200`. The kernel serial console would conflict with
TMC2209 communication.

**Correct** (serial console on tty1 only):
```
console=tty1 root=PARTUUID=... rootfstype=ext4 fsck.repair=yes rootwait
```

**Wrong** (serial console on UART — will break TMC2209):
```
console=serial0,115200 console=tty1 root=PARTUUID=... rootfstype=ext4 ...
```

If `console=serial0,115200` is present, remove it. Also run:
```bash
sudo raspi-config
# -> Interface Options -> Serial Port
# -> Login shell over serial: NO
# -> Serial port hardware enabled: YES
```

## Step 3: Add user to dialout group

```bash
sudo usermod -aG dialout morgan
```

The systemd service also needs `SupplementaryGroups=dialout` (already in
the treatbot.service unit file).

## Step 4: Reboot

```bash
sudo reboot
```

## Step 5: Verify

```bash
# Device exists with correct permissions
ls -la /dev/ttyAMA0
# Expected: crw-rw---- 1 root dialout 204, 64 ... /dev/ttyAMA0

# User has access
groups morgan | grep dialout

# No serial console on UART
cat /boot/firmware/cmdline.txt | grep -o "console=[^ ]*"
# Expected: only "console=tty1"

# Quick TMC2209 communication test
python3 -c "
import serial, struct
ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.2)
ser.reset_input_buffer()
# Read IOIN register (0x06) — returns chip version
data = bytes([0x05, 0x00, 0x06])
crc = 0
for b in data:
    for _ in range(8):
        if (crc >> 7) ^ (b & 0x01):
            crc = ((crc << 1) ^ 0x07) & 0xFF
        else:
            crc = (crc << 1) & 0xFF
        b >>= 1
ser.write(data + bytes([crc]))
import time; time.sleep(0.01)
resp = ser.read(12)
if len(resp) >= 12:
    version = (struct.unpack('>I', resp[7:11])[0] >> 24) & 0xFF
    print(f'TMC2209 detected: version=0x{version:02X}')
else:
    print(f'No response (got {len(resp)} bytes) — check wiring and config')
ser.close()
"
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `/dev/ttyAMA0` missing | UART not enabled | Add `dtparam=uart0=on` + `enable_uart=1` to config.txt, reboot |
| Permission denied on `/dev/ttyAMA0` | User not in dialout | `sudo usermod -aG dialout morgan`, logout/login |
| TMC2209 no response | Serial console on UART | Remove `console=serial0,115200` from cmdline.txt |
| TMC2209 no response | Wiring issue | Check 1K resistor on TX, verify PDN_UART connection |
| Garbled response | Wrong baud rate | Must be 115200 |
| Motor doesn't move but UART works | EN pin not low | Check GPIO24 wiring to TMC2209 EN |

## Reference: treatbot1 working config

`/boot/firmware/config.txt` relevant lines:
```
dtparam=spi=on
dtparam=audio=on
dtparam=pciex1=on
dtparam=uart0=on
enable_uart=1
dtparam=i2c_arm=on
dtoverlay=imx500
dtoverlay=disable-wifi
dtoverlay=cooling_fan
```

`/boot/firmware/cmdline.txt`:
```
console=tty1 root=PARTUUID=8049b1e8-02 rootfstype=ext4 fsck.repair=yes rootwait cfg80211.ieee80211_regdom=US
```
