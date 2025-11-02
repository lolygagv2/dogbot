#!/bin/bash

echo "Fixing USB Audio power management..."

# Find and fix the USB Audio device
for device in /sys/bus/usb/devices/*; do
    if [ -f "$device/idVendor" ] && [ -f "$device/idProduct" ]; then
        vendor=$(cat "$device/idVendor" 2>/dev/null)
        product=$(cat "$device/idProduct" 2>/dev/null)

        if [ "$vendor" = "0d8c" ] && [ "$product" = "0014" ]; then
            echo "Found USB Audio at: $device"

            # Set power control to 'on'
            if [ -f "$device/power/control" ]; then
                echo on | sudo tee "$device/power/control"
                echo "  Set power/control to: $(cat $device/power/control)"
            fi

            # Disable autosuspend
            if [ -f "$device/power/autosuspend" ]; then
                echo -1 | sudo tee "$device/power/autosuspend"
                echo "  Set autosuspend to: $(cat $device/power/autosuspend)"
            fi

            echo "✅ USB Audio fixed!"
        fi
    fi
done

echo ""
echo "Testing with simple recording..."
timeout 2 arecord -D hw:2,0 -f S16_LE -r 44100 -d 1 /tmp/test_mic.wav 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Recording works!"
    echo "File size: $(stat -c%s /tmp/test_mic.wav) bytes"
else
    echo "❌ Recording failed"
fi