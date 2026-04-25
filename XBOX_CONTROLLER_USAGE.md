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

*Updated: April 2026 (xbox_hybrid_controller.py)*

### Movement (Left Stick)
- **Left Stick Y-axis**: Forward/Backward movement
- **Left Stick X-axis**: Left/Right turning
- **Right Trigger (RT)**: Play "Good" audio (dog-aware path if dog detected)

### Camera Control (Right Stick)
- **Right Stick X-axis**: Pan camera left/right
- **Right Stick Y-axis**: Tilt camera up/down

### Actions
- **A Button**: Emergency Stop
- **B Button**: Cycle trick commands (Sit → Speak → Stay → Quiet → LieDown → Spin)
- **X Button**: Blue LED effect
- **Y Button**: Play treat sound

### Audio & Treats
- **Left Bumper (LB)**: Dispense treat (hold 5s for refill mode)
- **Right Bumper (RB)**: Play "No" audio (dog-aware path if dog detected)
- **Right Trigger (RT)**: Play "Good" audio (dog-aware path if dog detected)
- **Left Trigger (LT)**: Cycle NeoPixel LED modes

### D-Pad
- **D-Pad Left**: Cycle songs
- **D-Pad Right**: Quiet/Come commands

### System
- **SELECT**: Cycle modes (MANUAL → IDLE → COACH → SILENT_GUARDIAN)
- **START**: Shutdown robot

## Dog-Aware Audio
When a dog is detected (e.g., "Cooper"), RT/RB look for dog-specific audio:
- `/VOICEMP3/talks/Cooper/good.mp3` (if exists)
- Falls back to `/VOICEMP3/talks/default/good.mp3`

Audio paths are resolved via `_get_active_dog()` which queries the `/status` API.

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