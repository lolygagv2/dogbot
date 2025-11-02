#!/bin/bash
# WIM-Z Bluetooth Controller Setup Script

echo "========================================="
echo "  WIM-Z BLUETOOTH CONTROLLER SETUP"
echo "========================================="
echo ""
echo "Make sure your controller is in pairing mode:"
echo "  • PS4/PS5: Hold PS + Share buttons"
echo "  • Xbox: Hold pairing button"
echo "  • Generic: Check manual for pairing mode"
echo ""
echo "Press Enter when ready..."
read

echo "Scanning for Bluetooth devices..."
bluetoothctl scan on &
SCAN_PID=$!
sleep 10
kill $SCAN_PID 2>/dev/null

echo ""
echo "Available devices:"
bluetoothctl devices

echo ""
echo "Enter the MAC address of your controller (e.g., XX:XX:XX:XX:XX:XX):"
read MAC_ADDRESS

echo "Pairing with $MAC_ADDRESS..."
bluetoothctl pair $MAC_ADDRESS
bluetoothctl trust $MAC_ADDRESS
bluetoothctl connect $MAC_ADDRESS

echo ""
echo "Testing controller connection..."
python3 -c "
import pygame
pygame.init()
pygame.joystick.init()
count = pygame.joystick.get_count()
if count > 0:
    j = pygame.joystick.Joystick(0)
    j.init()
    print(f'✅ Controller connected: {j.get_name()}')
    print(f'   Axes: {j.get_numaxes()}')
    print(f'   Buttons: {j.get_numbuttons()}')
else:
    print('❌ No controller detected')
"

echo ""
echo "Setup complete! To use:"
echo "1. Turn on WIM-Z"
echo "2. Run: sudo python3 /home/morgan/dogbot/main_treatbot.py"
echo "3. The controller will be active in MANUAL mode"
echo ""
echo "Controls:"
echo "  Left Stick: Drive (forward/back/turn)"
echo "  Right Stick: Camera pan/tilt"
echo "  A: Dispense treat"
echo "  B: Play sound"
echo "  X: Toggle AI detection"
echo "  Y: Emergency stop"
echo "  Bumpers: Adjust speed"