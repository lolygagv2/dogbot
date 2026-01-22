#!/bin/bash
#
# install_wifi_provision.sh - Install WiFi provisioning systemd service
#
# Run with: sudo ./scripts/install_wifi_provision.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=================================="
echo "WIM-Z WiFi Provisioning Installer"
echo "=================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (sudo)"
    exit 1
fi

# Create logs directory if needed
mkdir -p "$PROJECT_DIR/logs"
chown morgan:morgan "$PROJECT_DIR/logs"

# Create the systemd service file
echo "Creating wifi-provision.service..."
cat > /etc/systemd/system/wifi-provision.service << 'EOF'
[Unit]
Description=WIM-Z WiFi Provisioning
Documentation=https://github.com/wimzai/dogbot
Before=treatbot.service
After=NetworkManager.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
User=root
WorkingDirectory=/home/morgan/dogbot
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/morgan/dogbot/env_new/bin/python /home/morgan/dogbot/wifi_provision_main.py
TimeoutStartSec=180
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Create polkit rules for nmcli access (if running as non-root user)
echo "Creating polkit rules for NetworkManager..."
cat > /etc/polkit-1/rules.d/50-wifi-provisioning.rules << 'EOF'
// Allow wifi-provisioning service to manage NetworkManager
polkit.addRule(function(action, subject) {
    if (action.id.indexOf("org.freedesktop.NetworkManager") == 0 &&
        subject.user == "root") {
        return polkit.Result.YES;
    }
});
EOF

# Modify treatbot.service to depend on wifi-provision
TREATBOT_SERVICE="/etc/systemd/system/treatbot.service"
if [ -f "$TREATBOT_SERVICE" ]; then
    echo "Updating treatbot.service dependencies..."
    # Check if already has the dependency
    if ! grep -q "After=.*wifi-provision.service" "$TREATBOT_SERVICE"; then
        # Add wifi-provision.service to After= line
        sed -i '/^\[Unit\]/,/^\[/ s/^After=\(.*\)/After=\1 wifi-provision.service/' "$TREATBOT_SERVICE"
    fi
    if ! grep -q "Requires=wifi-provision.service" "$TREATBOT_SERVICE"; then
        # Add Requires line after After
        sed -i '/^After=/a Requires=wifi-provision.service' "$TREATBOT_SERVICE"
    fi
fi

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

# Enable the service
echo "Enabling wifi-provision.service..."
systemctl enable wifi-provision.service

echo ""
echo "=================================="
echo "Installation complete!"
echo "=================================="
echo ""
echo "The WiFi provisioning service is now installed."
echo ""
echo "To test manually:"
echo "  sudo systemctl start wifi-provision.service"
echo "  sudo journalctl -u wifi-provision.service -f"
echo ""
echo "The service will run automatically before treatbot.service on boot."
echo ""
echo "To verify installation:"
echo "  systemctl status wifi-provision.service"
echo ""
