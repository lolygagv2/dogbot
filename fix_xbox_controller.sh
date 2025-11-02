#!/bin/bash
# Fix Xbox controller connection

echo "Fixing Xbox Controller connection..."

# Remove any existing connection
bluetoothctl disconnect AC:8E:BD:4A:0F:97 2>/dev/null
sleep 1

# Power cycle Bluetooth
echo "Restarting Bluetooth..."
sudo rfkill block bluetooth
sleep 1
sudo rfkill unblock bluetooth
sleep 2

# Ensure controller is trusted
bluetoothctl trust AC:8E:BD:4A:0F:97

echo "Now turn on your Xbox controller (press Xbox button)"
echo "Waiting for controller to be ready..."
sleep 5

# Connect with retry
for i in {1..3}; do
    echo "Connection attempt $i..."
    bluetoothctl connect AC:8E:BD:4A:0F:97

    # Check if connected
    sleep 2
    if bluetoothctl info AC:8E:BD:4A:0F:97 | grep -q "Connected: yes"; then
        echo "Controller connected!"

        # Wait for device to stabilize
        sleep 2

        # Check what input device was created
        echo "Looking for controller input device..."
        dmesg | grep -i "xbox.*input" | tail -1

        # Find the event device and create js device
        for event in /dev/input/event*; do
            name=$(cat /sys/class/input/$(basename $event)/device/name 2>/dev/null)
            if [[ "$name" == *"Xbox"* ]]; then
                echo "Found Xbox controller at $event"

                # Install jstest if needed
                if ! command -v jstest &> /dev/null; then
                    echo "Installing joystick tools..."
                    sudo apt-get install -y joystick
                fi

                # Force reload joydev to create js device
                echo "Creating joystick device..."
                sudo modprobe -r joydev 2>/dev/null
                sudo modprobe joydev
                sleep 1

                # Check if js device was created
                if ls /dev/input/js* 2>/dev/null; then
                    echo "Joystick device created!"
                    ls -la /dev/input/js*
                    echo ""
                    echo "Testing with jstest (press buttons/move sticks):"
                    timeout 10 jstest --normal /dev/input/js0
                else
                    echo "No js device created, testing with evtest instead:"
                    sudo timeout 10 evtest $event
                fi
                exit 0
            fi
        done

        echo "Controller connected but device not found in /dev/input/"
        exit 1
    fi

    sleep 2
done

echo "Failed to connect after 3 attempts"
exit 1