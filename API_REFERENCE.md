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

### 4. **Monitor System**
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