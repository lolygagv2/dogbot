#!/bin/bash
# Keep Xbox controller connected

MAC="AC:8E:BD:4A:0F:97"

echo "Xbox Controller Keeper"
echo "====================="
echo "This will keep trying to maintain connection"
echo ""

# Ensure ERTM is disabled
echo 1 | sudo tee /sys/module/bluetooth/parameters/disable_ertm > /dev/null
echo "ERTM disabled for stability"

while true; do
    # Check connection
    if ! bluetoothctl info $MAC | grep -q "Connected: yes"; then
        echo "[$(date '+%H:%M:%S')] Controller disconnected, reconnecting..."

        # Try to connect
        bluetoothctl connect $MAC 2>&1 | grep -E "successful|Failed"

        # Wait a bit
        sleep 2

        # Check if connected
        if bluetoothctl info $MAC | grep -q "Connected: yes"; then
            echo "[$(date '+%H:%M:%S')] ✓ Reconnected successfully"

            # Check for js0
            if [ -e "/dev/input/js0" ]; then
                echo "  Joystick device: /dev/input/js0"
            fi
        fi
    fi

    # Check every 5 seconds
    sleep 5
done