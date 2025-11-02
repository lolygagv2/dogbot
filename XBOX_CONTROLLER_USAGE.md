# Xbox Controller Manual Control Mode

## Setup
1. Ensure Xbox controller is paired via Bluetooth (solid light, not pulsing)
2. Start API server: `python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000`
3. Start Xbox controller: `./run_xbox_controller.sh start`

## Controller Script Commands
- `./run_xbox_controller.sh start` - Start controller
- `./run_xbox_controller.sh stop` - Stop controller
- `./run_xbox_controller.sh status` - Check if running
- `./run_xbox_controller.sh logs` - Watch live logs
- `./run_xbox_controller.sh restart` - Restart controller

## Xbox Controller Mapping

### Movement (Left Stick)
- **Left Stick Y-axis**: Forward/Backward movement
- **Left Stick X-axis**: Left/Right turning
- **Right Trigger (RT)**: Variable speed control (30-100% throttle)

### Camera Control (Right Stick)
- **Right Stick X-axis**: Pan camera left/right
- **Right Stick Y-axis**: Tilt camera up/down

### Actions
- **Left Bumper (LB)**: Dispense treat
- **Right Bumper (RB)**: Take photo (saved to /captures)
- **Y Button**: Play selected sound effect
- **A Button**: Emergency stop
- **B Button**: Stop motors

### Audio Control (D-Pad)
- **D-Pad Left**: Previous sound effect
- **D-Pad Right**: Next sound effect
- **D-Pad Down**: Play current sound
- **D-Pad Up**: Audio off

## Available Sound Effects
1. success
2. bark
3. whistle
4. celebrate
5. startup
6. shutdown
7. alert
8. reward

## Troubleshooting

### Controller Not Detected
```bash
# Check if joystick device exists
ls -la /dev/input/js*

# Reconnect controller
sudo ./fix_xbox_controller.sh
```

### API Not Responding
```bash
# Check if API is running
curl http://localhost:8000/health

# Restart API server
pkill -f uvicorn
python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000
```

### Controller Disconnecting
- Ensure controller has solid light (not pulsing)
- Check Bluetooth connection: `bluetoothctl info [MAC]`
- May need firmware update if authentication incomplete

## Files
- `/home/morgan/dogbot/xbox_api_controller.py` - Main controller script
- `/home/morgan/dogbot/run_xbox_controller.sh` - Service management script
- `/home/morgan/dogbot/xbox_controller.log` - Controller logs
- `/home/morgan/dogbot/api/server.py` - REST API server