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

echo
echo "[setup] Done. Reboot to activate the WiFi driver change:  sudo reboot"
