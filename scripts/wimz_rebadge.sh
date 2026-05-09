#!/bin/bash
# wimz_rebadge.sh — post-clone identity reset for treatbot3/4/5
#
# Run as root immediately after first boot of a freshly cloned SD card. Resets
# OS-level identity (machine-id, SSH host keys, hostname, /etc/hosts), updates
# WIMZ relay credentials in .env, and prints the new DEVICE_SECRET so it can be
# registered on the Lightsail relay.
#
# Usage:  sudo bash /home/morgan/dogbot/scripts/wimz_rebadge.sh <3|4|5>
#
# Idempotent: safe to re-run if any step fails partway. Does not auto-reboot.
# After it completes:
#   1. Capture the printed DEVICE_SECRET and register on Lightsail relay
#   2. sudo reboot
#   3. On the laptop side: ssh-keygen -R <old-hostname>.local

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "ERROR: must run as root (use sudo)" >&2
  exit 1
fi

N="${1:-}"
if [[ -z "$N" ]]; then
  echo "usage: sudo bash $0 <robot_number 3|4|5>" >&2
  exit 1
fi
if [[ ! "$N" =~ ^[3-5]$ ]]; then
  echo "ERROR: robot number must be 3, 4, or 5 (got: $N)" >&2
  exit 1
fi

NN=$(printf "%02d" "$N")
HOST="treatbot${N}"
DEVID="wimz_robot_${NN}"
ENV_FILE="/home/morgan/dogbot/.env"

echo "=================================================="
echo "WIM-Z Rebadge: -> hostname=$HOST  DEVICE_ID=$DEVID"
echo "=================================================="

echo
echo ">>> [1/4] Resetting machine-id"
rm -f /etc/machine-id /var/lib/dbus/machine-id
systemd-machine-id-setup
dbus-uuidgen --ensure
echo "    new machine-id: $(cat /etc/machine-id)"

echo
echo ">>> [2/4] Regenerating SSH host keys"
rm -f /etc/ssh/ssh_host_*
DEBIAN_FRONTEND=noninteractive dpkg-reconfigure openssh-server
systemctl restart ssh
echo "    new ed25519 fingerprint:"
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub | sed 's/^/        /'

echo
echo ">>> [3/4] Setting hostname to $HOST"
hostnamectl set-hostname "$HOST"
# Replace any existing treatbotN or wimz-NNN token in /etc/hosts with the new hostname
sed -i -E "s/\b(treatbot[0-9]+|wimz-[0-9]+)\b/$HOST/g" /etc/hosts
echo "    hostname:     $(hostname)"
echo "    /etc/hosts:   $(grep -E '127\.0\.1\.1' /etc/hosts || echo '(no 127.0.1.1 line)')"

echo
echo ">>> [4/4] Updating $ENV_FILE with new DEVICE_ID + fresh DEVICE_SECRET"
NEW_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
cat > "$ENV_FILE" <<EOF
DEVICE_ID=$DEVID
DEVICE_SECRET=$NEW_SECRET
EOF
chown morgan:morgan "$ENV_FILE"
chmod 600 "$ENV_FILE"

echo
echo "=================================================="
echo "DONE. Register on the Lightsail relay:"
echo
echo "    DEVICE_ID     = $DEVID"
echo "    DEVICE_SECRET = $NEW_SECRET"
echo
echo "Then: sudo reboot"
echo "=================================================="
