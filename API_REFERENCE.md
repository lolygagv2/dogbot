# TreatBot API Reference

**Base URL:** `http://localhost:8000` (when running locally)
**Content-Type:** `application/json`

## Quick Start

1. **Start the API server:**
   ```bash
   cd /home/morgan/dogbot
   python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000
   ```

2. **Check if it's working:**
   ```bash
   curl http://localhost:8000/health
   ```

## Core Endpoints

### 🏠 **System Status**

#### `GET /health`
Check if the API is running
```bash
curl http://localhost:8000/health
```
**Response:**
```json
{
  "status": "healthy",
  "mode": "detection",
  "uptime": 1234.5,
  "components": {
    "database": true,
    "ai": true,
    "hardware": true
  }
}
```

#### `GET /system/status`
Get comprehensive system status
```bash
curl http://localhost:8000/system/status
```

#### `GET /telemetry`
Get real-time system metrics
```bash
curl http://localhost:8000/telemetry
```

---

### 🎮 **Mode Control**

#### `GET /mode`
Get current system mode
```bash
curl http://localhost:8000/mode
```

#### `POST /mode/set`
Change system mode
```bash
curl -X POST http://localhost:8000/mode/set \
  -H "Content-Type: application/json" \
  -d '{"mode": "detection"}'
```
**Available modes:** `idle`, `detection`, `vigilant`, `photography`, `manual`

#### `POST /mode/clear_override`
Clear mode override
```bash
curl -X POST http://localhost:8000/mode/clear_override
```

---

### 🍖 **Treat Control**

#### `POST /treat/dispense`
Dispense treats manually
```bash
curl -X POST http://localhost:8000/treat/dispense \
  -H "Content-Type: application/json" \
  -d '{"dog_id": "test_dog", "reason": "manual", "count": 1}'
```

#### `GET /treat/status`
Get treat dispenser status
```bash
curl http://localhost:8000/treat/status
```

#### `POST /treat/force_reward`
Force a reward for testing (bypasses cooldowns)
```bash
curl -X POST "http://localhost:8000/treat/force_reward?dog_id=test_dog"
```

---

### 🎯 **Mission Control**

#### `GET /missions/status`
Get current mission status
```bash
curl http://localhost:8000/missions/status
```

#### `POST /missions/start`
Start a training mission
```bash
curl -X POST http://localhost:8000/missions/start \
  -H "Content-Type: application/json" \
  -d '{"mission_name": "train_sit_daily", "parameters": {}}'
```

#### `POST /missions/stop`
Stop current mission
```bash
curl -X POST http://localhost:8000/missions/stop
```

---

### 🎬 **Sequence Control**

#### `POST /sequence/execute`
Execute a celebration sequence
```bash
curl -X POST http://localhost:8000/sequence/execute \
  -H "Content-Type: application/json" \
  -d '{"sequence_name": "celebrate", "context": {"dog_id": "test"}}'
```

#### `GET /sequence/status`
Get sequence engine status
```bash
curl http://localhost:8000/sequence/status
```

#### `POST /sequence/stop_all`
Stop all running sequences
```bash
curl -X POST http://localhost:8000/sequence/stop_all
```

---

### 🎮 **Manual Control (RC Car Mode)**

#### `POST /manual/drive`
Drive the robot manually
```bash
curl -X POST http://localhost:8000/manual/drive \
  -H "Content-Type: application/json" \
  -d '{"left_speed": 50, "right_speed": 50, "duration": 2.0}'
```

#### `POST /manual/keyboard`
Send keyboard commands
```bash
curl -X POST http://localhost:8000/manual/keyboard \
  -H "Content-Type: application/json" \
  -d '{"key": "w", "pressed": true}'
```
**Keys:** `w` (forward), `s` (backward), `a` (left), `d` (right), `space` (stop)

#### `POST /manual/emergency_stop`
Emergency stop
```bash
curl -X POST http://localhost:8000/manual/emergency_stop
```

#### `GET /manual/status`
Get manual control status
```bash
curl http://localhost:8000/manual/status
```

---

### 🐕 **Dog Management**

#### `GET /dogs`
Get all registered dogs
```bash
curl http://localhost:8000/dogs
```

#### `GET /dogs/{dog_id}`
Get specific dog info
```bash
curl http://localhost:8000/dogs/test_dog
```

#### `GET /dogs/{dog_id}/rewards`
Get dog's reward history
```bash
curl "http://localhost:8000/dogs/test_dog/rewards?days=7"
```

---

### 📊 **Events & Logging**

#### `GET /events/recent`
Get recent system events
```bash
curl "http://localhost:8000/events/recent?limit=10&event_type=vision"
```

#### `GET /events/stats`
Get event statistics
```bash
curl http://localhost:8000/events/stats
```

---

### 🎵 **DFPlayer Audio Control**

#### `GET /audio/status`
Get DFPlayer and audio relay status
```bash
curl http://localhost:8000/audio/status
```

#### `GET /audio/files`
Get list of available audio files
```bash
curl http://localhost:8000/audio/files
```

#### `POST /audio/play/file`
Play audio file by path
```bash
curl -X POST http://localhost:8000/audio/play/file \
  -H "Content-Type: application/json" \
  -d '{"filepath": "/talks/0008.mp3"}'
```

#### `POST /audio/play/number`
Play audio file by number
```bash
curl -X POST http://localhost:8000/audio/play/number \
  -H "Content-Type: application/json" \
  -d '{"number": 1}'
```

#### `POST /audio/play/sound`
Play audio by sound name (from AudioFiles)
```bash
curl -X POST http://localhost:8000/audio/play/sound \
  -H "Content-Type: application/json" \
  -d '{"sound_name": "good_dog"}'
```

#### `POST /audio/volume`
Set DFPlayer volume (0-30)
```bash
curl -X POST http://localhost:8000/audio/volume \
  -H "Content-Type: application/json" \
  -d '{"volume": 20}'
```

#### `POST /audio/pause`
Pause/resume audio playback
```bash
curl -X POST http://localhost:8000/audio/pause
```

#### `POST /audio/next`
Play next track
```bash
curl -X POST http://localhost:8000/audio/next
```

#### `POST /audio/previous`
Play previous track
```bash
curl -X POST http://localhost:8000/audio/previous
```

#### `POST /audio/relay/pi`
Switch audio relay to Pi USB audio
```bash
curl -X POST http://localhost:8000/audio/relay/pi
```

#### `POST /audio/relay/dfplayer`
Switch audio relay to DFPlayer
```bash
curl -X POST http://localhost:8000/audio/relay/dfplayer
```

#### `GET /audio/relay/status`
Get audio relay status
```bash
curl http://localhost:8000/audio/relay/status
```

#### `POST /audio/test`
Test audio system (relay switching)
```bash
curl -X POST http://localhost:8000/audio/test
```

---

### 💡 **LED Control**

#### `GET /leds/status`
Get LED system status
```bash
curl http://localhost:8000/leds/status
```

#### `GET /leds/colors`
Get list of available LED colors
```bash
curl http://localhost:8000/leds/colors
```

#### `GET /leds/modes`
Get available LED modes
```bash
curl http://localhost:8000/leds/modes
```

#### `POST /leds/color`
Set LEDs to solid color
```bash
curl -X POST http://localhost:8000/leds/color \
  -H "Content-Type: application/json" \
  -d '{"color": "blue"}'
```

#### `POST /leds/custom_color`
Set LEDs to custom RGB color
```bash
curl -X POST http://localhost:8000/leds/custom_color \
  -H "Content-Type: application/json" \
  -d '{"red": 255, "green": 100, "blue": 50}'
```

#### `POST /leds/brightness`
Set LED brightness (0.1 to 1.0)
```bash
curl -X POST http://localhost:8000/leds/brightness \
  -H "Content-Type: application/json" \
  -d '{"brightness": 0.5}'
```

#### `POST /leds/mode`
Set LED mode with animations
```bash
curl -X POST http://localhost:8000/leds/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "searching"}'
```
**Available modes:** `off`, `idle`, `searching`, `dog_detected`, `treat_launching`, `error`, `charging`

#### `POST /leds/animation`
Start custom LED animation
```bash
curl -X POST http://localhost:8000/leds/animation \
  -H "Content-Type: application/json" \
  -d '{"animation": "pulse_color", "color": "green", "delay": 0.05, "steps": 20}'
```
**Available animations:** `spinning_dot`, `pulse_color`, `rainbow_cycle`

#### `POST /leds/stop`
Stop all LED animations
```bash
curl -X POST http://localhost:8000/leds/stop
```

#### `POST /leds/blue/on`
Turn blue LED on
```bash
curl -X POST http://localhost:8000/leds/blue/on
```

#### `POST /leds/blue/off`
Turn blue LED off
```bash
curl -X POST http://localhost:8000/leds/blue/off
```

#### `POST /leds/off`
Turn all LEDs off
```bash
curl -X POST http://localhost:8000/leds/off
```

---

### 🚨 **Emergency Control**

#### `POST /emergency/stop`
Trigger emergency stop
```bash
curl -X POST http://localhost:8000/emergency/stop
```

#### `POST /emergency/clear`
Clear emergency state
```bash
curl -X POST http://localhost:8000/emergency/clear
```

---

## Common Use Cases

### 1. **Start Training Session**
```bash
# 1. Check system health
curl http://localhost:8000/health

# 2. Set detection mode
curl -X POST http://localhost:8000/mode/set \
  -d '{"mode": "detection"}'

# 3. Start training mission
curl -X POST http://localhost:8000/missions/start \
  -d '{"mission_name": "train_sit_daily"}'

# 4. Monitor status
curl http://localhost:8000/missions/status
```

### 2. **Manual Testing**
```bash
# 1. Force a reward
curl -X POST "http://localhost:8000/treat/force_reward?dog_id=test"

# 2. Play celebration
curl -X POST http://localhost:8000/sequence/execute \
  -d '{"sequence_name": "celebrate"}'

# 3. Check what happened
curl "http://localhost:8000/events/recent?limit=5"
```

### 3. **Remote Control**
```bash
# 1. Set manual mode
curl -X POST http://localhost:8000/manual/mode/manual

# 2. Drive forward
curl -X POST http://localhost:8000/manual/keyboard \
  -d '{"key": "w", "pressed": true}'

# 3. Stop
curl -X POST http://localhost:8000/manual/keyboard \
  -d '{"key": "space", "pressed": true}'
```

### 4. **DFPlayer Audio Control**
```bash
# 1. Check available sounds
curl http://localhost:8000/audio/files | jq .

# 2. Play a predefined sound
curl -X POST http://localhost:8000/audio/play/sound \
  -d '{"sound_name": "good_dog"}'

# 3. Set volume
curl -X POST http://localhost:8000/audio/volume \
  -d '{"volume": 25}'

# 4. Test relay switching
curl -X POST http://localhost:8000/audio/test
```

### 5. **LED Light Show Control**
```bash
# 1. Check available colors and modes
curl http://localhost:8000/leds/colors | jq .
curl http://localhost:8000/leds/modes | jq .

# 2. Set solid color
curl -X POST http://localhost:8000/leds/color \
  -d '{"color": "purple"}'

# 3. Custom RGB color
curl -X POST http://localhost:8000/leds/custom_color \
  -d '{"red": 255, "green": 165, "blue": 0}'

# 4. Start spinning animation
curl -X POST http://localhost:8000/leds/animation \
  -d '{"animation": "spinning_dot", "color": "cyan", "delay": 0.1}'

# 5. Set brightness
curl -X POST http://localhost:8000/leds/brightness \
  -d '{"brightness": 0.8}'

# 6. Dog detected mode
curl -X POST http://localhost:8000/leds/mode \
  -d '{"mode": "dog_detected"}'
```

### 6. **Monitor System**
```bash
# Real-time monitoring
watch -n 2 'curl -s http://localhost:8000/telemetry | jq .'

# Check for errors
curl "http://localhost:8000/events/recent?event_type=error"

# System overview
curl http://localhost:8000/system/status | jq .
```

---

## Error Handling

All endpoints return standard HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **404**: Not Found
- **500**: Internal Server Error

**Error Response Format:**
```json
{
  "detail": "Error description",
  "error_type": "ValidationError",
  "timestamp": "2025-10-22T20:59:00Z"
}
```

---

## Request/Response Models

### **ModeRequest**
```json
{
  "mode": "detection",
  "duration": 30.0  // optional timeout
}
```

### **TreatRequest**
```json
{
  "dog_id": "test_dog",    // optional
  "reason": "manual",      // manual, reward, test
  "count": 1               // number of treats
}
```

### **ManualDriveRequest**
```json
{
  "left_speed": 50,    // -100 to 100
  "right_speed": 50,   // -100 to 100
  "duration": 2.0      // seconds
}
```

### **KeyboardControlRequest**
```json
{
  "key": "w",          // w, a, s, d, space
  "pressed": true      // true = press, false = release
}
```

### **DFPlayerPlayRequest**
```json
{
  "filepath": "/talks/0008.mp3"    // Path to audio file
}
```

### **DFPlayerVolumeRequest**
```json
{
  "volume": 20         // Volume level 0-30
}
```

### **DFPlayerNumberRequest**
```json
{
  "number": 1          // File number to play
}
```

### **DFPlayerSoundRequest**
```json
{
  "sound_name": "good_dog"    // Predefined sound name
}
```

### **LEDColorRequest**
```json
{
  "color": "blue"             // Color name from available colors
}
```

### **LEDCustomColorRequest**
```json
{
  "red": 255,                 // RGB red value (0-255)
  "green": 165,               // RGB green value (0-255)
  "blue": 0                   // RGB blue value (0-255)
}
```

### **LEDBrightnessRequest**
```json
{
  "brightness": 0.5           // Brightness level (0.1-1.0)
}
```

### **LEDModeRequest**
```json
{
  "mode": "searching"         // LED mode from available modes
}
```

### **LEDAnimationRequest**
```json
{
  "animation": "pulse_color", // Animation type
  "color": "green",           // Color for animation (optional)
  "delay": 0.05,              // Animation delay (optional)
  "steps": 20                 // Steps for pulse animation (optional)
}
```

---

## Interactive API Documentation

When the server is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These provide interactive documentation where you can test endpoints directly in your browser.

---

## Integration Examples

### **Python Client**
```python
import requests

# Start training
response = requests.post('http://localhost:8000/missions/start',
                        json={'mission_name': 'train_sit_daily'})
print(response.json())

# Monitor status
status = requests.get('http://localhost:8000/missions/status')
print(f"Mission: {status.json()}")
```

### **JavaScript/Web**
```javascript
// Dispense treat
fetch('http://localhost:8000/treat/dispense', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({dog_id: 'buddy', count: 1})
})
.then(response => response.json())
.then(data => console.log(data));
```

### **Shell Script Monitoring**
```bash
#!/bin/bash
# monitor.sh - Simple system monitor

while true; do
  echo "=== TreatBot Status ==="
  curl -s http://localhost:8000/health | jq .
  curl -s http://localhost:8000/missions/status | jq .
  echo ""
  sleep 5
done
```

---

**Need help?** Check the logs at `/var/log/treatbot/` or use `curl http://localhost:8000/system/status` for diagnostics.