#!/usr/bin/env bash
# Install WIM-Z auto-recovery (hardware watchdog + liveness healthcheck) on
# this robot. Idempotent — safe to re-run after a git pull. Run with sudo.
#
#   sudo /home/morgan/dogbot/scripts/install-watchdog.sh
#
# Deploy to another unit: git pull there, then run this same command.
set -euo pipefail

REPO="/home/morgan/dogbot"
SD="$REPO/scripts/systemd"

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run with sudo (needs to write /etc/systemd and re-exec systemd)." >&2
    exit 1
fi

echo "[1/6] hardware watchdog drop-in -> /etc/systemd/system.conf.d/"
install -D -m644 "$SD/10-wimz-watchdog.conf" /etc/systemd/system.conf.d/10-wimz-watchdog.conf

echo "[2/6] treatbot stop-timeout drop-in -> /etc/systemd/system/treatbot.service.d/"
install -D -m644 "$SD/10-treatbot-timeout.conf" /etc/systemd/system/treatbot.service.d/10-timeout.conf

echo "[3/6] healthcheck service + timer -> /etc/systemd/system/"
install -m644 "$SD/wimz-healthcheck.service" /etc/systemd/system/wimz-healthcheck.service
install -m644 "$SD/wimz-healthcheck.timer"   /etc/systemd/system/wimz-healthcheck.timer
chmod +x "$REPO/scripts/wimz-healthcheck.sh"

echo "[4/6] nm-online timeout cap -> /etc/systemd/system/NetworkManager-wait-online.service.d/"
# Bounds the network-online.target wait that treatbot.service now orders on,
# so a unit with no WiFi still boots promptly. See the drop-in for rationale.
install -D -m644 "$SD/20-nm-online-timeout.conf" \
    /etc/systemd/system/NetworkManager-wait-online.service.d/20-nm-online-timeout.conf

echo "[5/6] reload units + enable timer"
systemctl daemon-reload
systemctl enable --now wimz-healthcheck.timer

echo "[6/6] apply hardware watchdog live (systemd re-exec)"
systemctl daemon-reexec

echo
echo "Done. Verify:"
echo "  systemctl show -p RuntimeWatchdogUSec        # expect 15000000"
echo "  systemctl list-timers wimz-healthcheck.timer # expect a next-run time"
echo "  systemctl cat treatbot.service | grep TimeoutStopSec"
