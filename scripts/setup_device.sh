#!/usr/bin/env bash
#
# WIM-Z per-device setup — applies system-level config that `git pull` cannot.
#
# Run once on each robot after pulling the repo:
#     sudo bash scripts/setup_device.sh
#
# Idempotent: safe to re-run. Reboot afterwards to activate the WiFi driver
# change (or it takes effect on the next boot regardless).
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[setup] WIM-Z device setup from $REPO"

if [ "$(id -u)" -ne 0 ]; then
    echo "[setup] ERROR: must run as root (sudo bash scripts/setup_device.sh)" >&2
    exit 1
fi

# Owning user for /etc/wimz (the treatbot service runs as this user).
WIMZ_USER="${SUDO_USER:-morgan}"

# --- 1. Persistent volume state directory --------------------------------
echo "[setup] creating /etc/wimz (owned by $WIMZ_USER)"
mkdir -p /etc/wimz
chown "$WIMZ_USER":"$WIMZ_USER" /etc/wimz
chmod 755 /etc/wimz

# --- 2. Boot-time volume restore service ---------------------------------
echo "[setup] installing wimz-audio.service"
cp "$REPO/scripts/wimz-audio.service" /etc/systemd/system/wimz-audio.service
systemctl daemon-reload
systemctl enable wimz-audio.service
systemctl start wimz-audio.service

# --- 3. WiFi stability: disable rtw88 deep power-save --------------------
echo "[setup] installing /etc/modprobe.d/rtw88.conf"
cp "$REPO/scripts/rtw88.conf" /etc/modprobe.d/rtw88.conf

# --- 4. NetworkManager: disable WiFi power-save on all saved connections -
echo "[setup] disabling WiFi power-save on saved connections"
while IFS=: read -r conn ctype; do
    if [ "$ctype" = "802-11-wireless" ]; then
        nmcli connection modify "$conn" 802-11-wireless.powersave 2 \
            && echo "[setup]   powersave disabled: $conn" \
            || echo "[setup]   WARN: could not modify $conn"
    fi
done < <(nmcli -t -f NAME,TYPE connection show)

# --- 5. Free GPIO7 (SPI0 CE1, pin 26) for the treat through-beam sensor ---
# Safe on all units: NeoPixel uses CE0/spidev0.0 which spi0-1cs preserves.
# Units without the sensor just gain a free GPIO. Takes effect after the
# power cycle below. Sensor wiring/enable: docs/NEW_ROBOT_SETUP.md §beam.
BOOTCFG=/boot/firmware/config.txt
if grep -q "^dtoverlay=spi0-1cs" "$BOOTCFG"; then
    echo "[setup] spi0-1cs overlay already present"
else
    echo "[setup] adding dtoverlay=spi0-1cs to $BOOTCFG (frees GPIO7 for treat beam)"
    printf '\n# Free GPIO7 (SPI0 CE1, pin 26) for treat through-beam sensor.\n# Keeps CE0/spidev0.0 alive for the WS2812 NeoPixel driver.\ndtoverlay=spi0-1cs\n' >> "$BOOTCFG"
fi

# --- Verify ---------------------------------------------------------------
echo
echo "[setup] verification:"
[ -d /etc/wimz ] \
    && echo "  OK   /etc/wimz present (owner: $(stat -c '%U' /etc/wimz))" \
    || echo "  FAIL /etc/wimz missing"
systemctl is-enabled --quiet wimz-audio.service \
    && echo "  OK   wimz-audio.service enabled" \
    || echo "  FAIL wimz-audio.service not enabled"
[ -f /etc/modprobe.d/rtw88.conf ] \
    && echo "  OK   /etc/modprobe.d/rtw88.conf present" \
    || echo "  FAIL rtw88.conf missing"
grep -q "^dtoverlay=spi0-1cs" "$BOOTCFG" \
    && echo "  OK   spi0-1cs overlay in config.txt (GPIO7 free after power cycle)" \
    || echo "  FAIL spi0-1cs overlay missing"

echo
echo "[setup] Done. Reboot to activate the WiFi driver change + load new code:"
echo "[setup]   sudo reboot"
