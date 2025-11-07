#!/bin/bash
# Simple Xbox Controller Restart Script

echo "🎮 Restarting Xbox Controller..."

# Kill any existing processes
echo "Stopping existing Xbox controller processes..."
sudo pkill -f xbox_hybrid_controller.py
sleep 2

# Check if joystick device exists
if [ ! -e /dev/input/js0 ]; then
    echo "❌ Error: No Xbox controller found at /dev/input/js0"
    echo "Make sure controller is connected and run:"
    echo "  sudo ./fix_xbox_controller.sh"
    exit 1
fi

# Test servo controller first
echo "Testing servo controller..."
cd /home/morgan/dogbot
if ! /home/morgan/dogbot/env_new/bin/python -c "
from core.hardware.servo_controller import ServoController
servo = ServoController()
print('✅ Servo controller OK')
"; then
    echo "❌ Error: Servo controller test failed"
    exit 1
fi

# Start Xbox controller
echo "Starting Xbox controller..."
/home/morgan/dogbot/env_new/bin/python /home/morgan/dogbot/xbox_hybrid_controller.py &

# Wait and check if it started
sleep 3
if pgrep -f xbox_hybrid_controller.py > /dev/null; then
    echo "✅ Xbox controller started successfully"
    echo "🎮 Camera controls ready on right stick!"
else
    echo "❌ Error: Xbox controller failed to start"
    exit 1
fi