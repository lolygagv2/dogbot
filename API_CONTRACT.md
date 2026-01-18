# WIM-Z API Contract v1.0

> **Single source of truth** for Robot, Mobile App, and Cloud Relay Server.
> When changing any API, update this file FIRST, then update each project to match.

Last Updated: 2026-01-18

---

## Architecture Overview

```
┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
│   Mobile App    │◄───────►│  Cloud Relay    │◄───────►│    WIM-Z Robot  │
│   (Flutter)     │   WS    │  (FastAPI)      │   WS    │  (Raspberry Pi) │
└─────────────────┘         └─────────────────┘         └─────────────────┘
        │                          │                           │
        │      Local Network Mode (Phase 1)                    │
        └──────────────────────────────────────────────────────┘
                              Direct WS/REST
```

**Connection Modes:**
- **Local Mode**: App connects directly to Robot on same WiFi
- **Cloud Mode**: Both App and Robot connect to Relay Server

---

## REST API Endpoints

Base URL: `http://{host}:8000` (local) or `https://api.wimz.io` (cloud)

### Health & Status

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/health` | Server alive check | - | `{"status": "ok"}` |
| GET | `/telemetry` | Full system status | - | `Telemetry` object |

### Motor Control

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| POST | `/motor/speed` | Set motor speeds | `{"left": float, "right": float}` | `{"success": true}` |
| POST | `/motor/stop` | Stop all motors | - | `{"success": true}` |
| POST | `/motor/emergency` | Emergency stop | - | `{"success": true}` |

**Motor speed values:** -1.0 (full reverse) to 1.0 (full forward)

### Camera & Servos

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/camera/stream` | MJPEG video stream | - | `multipart/x-mixed-replace` |
| GET | `/camera/snapshot` | Single JPEG frame | - | `image/jpeg` |
| POST | `/servo/pan` | Set pan angle | `{"angle": float}` | `{"success": true}` |
| POST | `/servo/tilt` | Set tilt angle | `{"angle": float}` | `{"success": true}` |
| POST | `/servo/center` | Center camera | - | `{"success": true}` |

**Servo angle limits:**
- Pan: -90° to +90°
- Tilt: -45° to +45°

### Treat Dispenser

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| POST | `/treat/dispense` | Dispense one treat | - | `{"success": true, "remaining": int}` |
| POST | `/treat/carousel/rotate` | Rotate carousel | - | `{"success": true}` |

### LED Control

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| POST | `/led/pattern` | Set LED pattern | `{"pattern": string}` | `{"success": true}` |
| POST | `/led/color` | Set RGB color | `{"r": int, "g": int, "b": int}` | `{"success": true}` |
| POST | `/led/off` | Turn off LEDs | - | `{"success": true}` |

**Available patterns:** `breathing`, `rainbow`, `celebration`, `searching`, `alert`, `idle`

### Audio

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| POST | `/audio/play` | Play audio file | `{"file": string}` | `{"success": true}` |
| POST | `/audio/stop` | Stop playback | - | `{"success": true}` |
| POST | `/audio/volume` | Set volume | `{"level": int}` | `{"success": true}` |
| GET | `/audio/files` | List audio files | - | `{"files": string[]}` |

**Volume range:** 0-100

### Mode Control

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/mode/get` | Get current mode | - | `{"mode": string}` |
| POST | `/mode/set` | Set mode | `{"mode": string}` | `{"success": true}` |

**Available modes:** `idle`, `guardian`, `training`, `manual`, `docking`

### Missions

| Method | Endpoint | Description | Request | Response |
|--------|----------|-------------|---------|----------|
| GET | `/missions` | List all missions | - | `Mission[]` |
| GET | `/missions/{id}` | Get mission details | - | `Mission` |
| POST | `/missions/{id}/start` | Start mission | - | `{"success": true}` |
| POST | `/missions/{id}/stop` | Stop mission | - | `{"success": true}` |
| GET | `/missions/active` | Get active mission | - | `Mission` or `null` |

---

## WebSocket API

Endpoint: `ws://{host}:8000/ws` (local) or `wss://api.wimz.io/ws` (cloud)

### Connection

```json
// On connect, client sends auth (cloud mode only)
{"type": "auth", "token": "jwt_token_here", "device_id": "optional_for_robot"}

// Server responds
{"type": "auth_result", "success": true}
```

### Events: Robot → App

#### Detection Event
```json
{
  "event": "detection",
  "data": {
    "detected": true,
    "behavior": "sitting",
    "confidence": 0.92,
    "bbox": [0.2, 0.3, 0.4, 0.5]
  },
  "timestamp": "2025-01-18T12:00:00Z"
}
```
**bbox format:** `[x, y, width, height]` normalized 0-1

#### Status Event
```json
{
  "event": "status",
  "data": {
    "battery": 78.5,
    "temperature": 42.0,
    "mode": "training",
    "is_charging": false,
    "treats_remaining": 12
  },
  "timestamp": "2025-01-18T12:00:00Z"
}
```

#### Treat Event
```json
{
  "event": "treat",
  "data": {
    "dispensed": true,
    "remaining": 11
  },
  "timestamp": "2025-01-18T12:00:00Z"
}
```

#### Mission Event
```json
{
  "event": "mission",
  "data": {
    "id": "sit_training",
    "status": "running",
    "progress": 0.6,
    "rewards_given": 3,
    "success_count": 5,
    "fail_count": 2
  },
  "timestamp": "2025-01-18T12:00:00Z"
}
```

#### Error Event
```json
{
  "event": "error",
  "data": {
    "code": "LOW_BATTERY",
    "message": "Battery below 20%",
    "severity": "warning"
  },
  "timestamp": "2025-01-18T12:00:00Z"
}
```
**Error codes:** `LOW_BATTERY`, `OVERHEAT`, `MOTOR_FAULT`, `CAMERA_FAULT`, `NETWORK_ERROR`

### Commands: App → Robot

#### Motor Command (send at 20Hz for smooth control)
```json
{
  "command": "motor",
  "left": 0.5,
  "right": 0.5
}
```

#### Servo Command
```json
{
  "command": "servo",
  "pan": 15.0,
  "tilt": -10.0
}
```

#### Treat Command
```json
{
  "command": "treat"
}
```

#### LED Command
```json
{
  "command": "led",
  "pattern": "celebration"
}
```

#### Audio Command
```json
{
  "command": "audio",
  "file": "good_dog.mp3"
}
```

#### Mode Command
```json
{
  "command": "mode",
  "mode": "training"
}
```

### Ping/Pong (Keepalive)
```json
// Client sends every 30 seconds
{"type": "ping"}

// Server responds
{"type": "pong"}
```

---

## Data Models

### Telemetry
```json
{
  "battery": 78.5,
  "temperature": 42.0,
  "mode": "idle",
  "dog_detected": false,
  "current_behavior": null,
  "confidence": null,
  "is_charging": false,
  "treats_remaining": 15,
  "last_treat_time": "2025-01-18T11:30:00Z",
  "active_mission_id": null,
  "wifi_strength": -45,
  "uptime_seconds": 3600
}
```

### Mission
```json
{
  "id": "sit_training",
  "name": "Sit Training",
  "description": "Reward dog for sitting on command",
  "target_behavior": "sit",
  "required_duration": 3.0,
  "cooldown_seconds": 15,
  "daily_limit": 10,
  "is_active": false,
  "rewards_given": 0,
  "progress": 0.0,
  "success_count": 0,
  "fail_count": 0,
  "created_at": "2025-01-01T00:00:00Z",
  "last_run": null
}
```

### Detection
```json
{
  "detected": true,
  "behavior": "sitting",
  "confidence": 0.92,
  "bbox": [0.2, 0.3, 0.4, 0.5],
  "timestamp": "2025-01-18T12:00:00Z"
}
```

### Device (for cloud relay)
```json
{
  "device_id": "wimz_abc123",
  "name": "Living Room WIM-Z",
  "owner_id": "user_xyz",
  "is_online": true,
  "last_seen": "2025-01-18T12:00:00Z",
  "firmware_version": "1.0.0",
  "local_ip": "192.168.1.50"
}
```

### User (for cloud relay)
```json
{
  "user_id": "user_xyz",
  "email": "user@example.com",
  "devices": ["wimz_abc123"],
  "created_at": "2025-01-01T00:00:00Z"
}
```

---

## Cloud Relay Server Specifics

### Device Registration (Robot → Relay)
```
POST /api/device/register
Authorization: HMAC-SHA256(device_secret)

{
  "device_id": "wimz_abc123",
  "firmware_version": "1.0.0"
}

Response: {"success": true, "websocket_url": "wss://api.wimz.io/ws/device"}
```

### User Authentication (App → Relay)
```
POST /api/auth/login
{
  "email": "user@example.com",
  "password": "..."
}

Response: {"token": "jwt_token", "expires_in": 86400}
```

### Device Pairing (App → Relay)
```
POST /api/device/pair
Authorization: Bearer {jwt_token}
{
  "pairing_code": "ABC123"
}

Response: {"success": true, "device_id": "wimz_abc123"}
```

### WebSocket Routing (Cloud Mode)

When robot and app both connect to relay:

```
App connects: wss://api.wimz.io/ws/app?token={jwt}
Robot connects: wss://api.wimz.io/ws/device?device_id={id}&sig={hmac}

Relay routes:
- App command → finds robot by user's device list → forwards to robot
- Robot event → finds connected app sessions for owner → forwards to apps
```

---

## Error Responses

All REST endpoints return errors in this format:
```json
{
  "success": false,
  "error": {
    "code": "INVALID_PARAMETER",
    "message": "Motor speed must be between -1.0 and 1.0"
  }
}
```

**HTTP Status Codes:**
- `200` - Success
- `400` - Bad request / invalid parameters
- `401` - Unauthorized (cloud mode)
- `404` - Not found
- `500` - Server error

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-18 | Initial contract |

---

## Implementation Checklist

### Robot (Raspberry Pi)
- [x] REST endpoints implemented
- [x] Contract-compliant REST endpoints added (2026-01-18)
  - [x] `/motor/speed`, `/motor/stop`, `/motor/emergency`
  - [x] `/camera/stream`, `/camera/snapshot`
  - [x] `/servo/pan`, `/servo/tilt`, `/servo/center`
  - [x] `/treat/dispense`, `/treat/carousel/rotate`
  - [x] `/led/pattern`, `/led/color`, `/led/off`
  - [x] `/audio/play`, `/audio/stop`, `/audio/volume`, `/audio/files`
  - [x] `/mode/get`, `/mode/set`
  - [x] `/missions`, `/missions/{id}`, `/missions/{id}/start`, `/missions/{id}/stop`, `/missions/active`
  - [x] `/telemetry/contract` (contract-format telemetry)
- [x] WebSocket server implemented
  - [x] Ping/pong handling
  - [x] Auth handling (local mode accepts all)
  - [x] Contract command format support
  - [x] Contract event format broadcasting
  - [x] Status event broadcasting (every 5s)
- [ ] Cloud relay client (Phase 3)

### Mobile App (Flutter)
- [x] REST client implemented
- [x] WebSocket client implemented
- [ ] Cloud mode connection (Phase 3)

### Cloud Relay (FastAPI)
- [ ] Device WebSocket handler
- [ ] App WebSocket handler
- [ ] Message routing
- [ ] JWT authentication
- [ ] Device registration
- [ ] User management
