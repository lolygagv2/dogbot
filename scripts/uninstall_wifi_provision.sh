#!/bin/bash
#
# uninstall_wifi_provision.sh - Remove WiFi provisioning systemd service
#
# Run with: sudo ./scripts/uninstall_wifi_provision.sh
#

set -e

echo "=================================="
echo "WIM-Z WiFi Provisioning Uninstaller"
echo "=================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Error: This script must be run as root (sudo)"
    exit 1
fi

# Stop the service if running
echo "Stopping wifi-provision.service..."
systemctl stop wifi-provision.service 2>/dev/null || true

# Disable the service
echo "Disabling wifi-provision.service..."
systemctl disable wifi-provision.service 2>/dev/null || true

# Remove the service file
echo "Removing service file..."
rm -f /etc/systemd/system/wifi-provision.service

# Remove polkit rules
echo "Removing polkit rules..."
rm -f /etc/polkit-1/rules.d/50-wifi-provisioning.rules

# Remove dependency from treatbot.service
TREATBOT_SERVICE="/etc/systemd/system/treatbot.service"
if [ -f "$TREATBOT_SERVICE" ]; then
    echo "Removing dependency from treatbot.service..."
    sed -i 's/ wifi-provision.service//g' "$TREATBOT_SERVICE"
    sed -i '/^Requires=wifi-provision.service/d' "$TREATBOT_SERVICE"
fi

# Reload systemd
echo "Reloading systemd daemon..."
systemctl daemon-reload

echo ""
echo "=================================="
echo "Uninstallation complete!"
echo "=================================="
echo ""
echo "The WiFi provisioning service has been removed."
echo "Note: The Python files in services/network/ were not removed."
echo ""
