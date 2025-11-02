# WIM-Z/TreatBot API Documentation for iPhone App

## Overview
The WIM-Z robot provides both REST API and WebSocket endpoints for control from the custom iPhone app (Phase 5.3 of product roadmap).

## Base URL
```
http://<robot-ip>:8000
```

## Authentication
Currently no authentication (will add in Phase 5.4)

## Motor Control Endpoints

### Direct Motor Control
```http
POST /motor/control
Content-Type: application/json

{
  "left_speed": 50,    // -100 to 100
  "right_speed": 50,   // -100 to 100
  "duration": 2.0      // Optional, seconds
}

Response:
{
  "success": true,
  "left_speed": 50,
  "right_speed": 50,
  "duration": 2.0
}
```

### Virtual Joystick Control
```http
POST /motor/joystick
Content-Type: application/json

{
  "x": 0.5,   // -1.0 to 1.0 (left/right)
  "y": 0.8    // -1.0 to 1.0 (forward/back)
}

Response:
{
  "success": true,
  "joystick": {"x": 0.5, "y": 0.8},
  "motors": {"left": 130, "right": 30}
}
```

### Emergency Stop
```http
POST /motor/stop
Content-Type: application/json

{
  "reason": "user_initiated"  // Optional
}

Response:
{
  "success": true,
  "message": "Motors stopped: user_initiated"
}
```

## Camera Control Endpoints

### Pan/Tilt Control
```http
POST /camera/pantilt
Content-Type: application/json

{
  "pan": 90,    // 0-180 degrees (optional)
  "tilt": 45,   // 0-180 degrees (optional)
  "speed": 5    // Movement speed (optional)
}

Response:
{
  "success": true,
  "pan": 90,
  "tilt": 45,
  "current_position": {"pan": 90, "tilt": 45}
}
```

### Get Camera Position
```http
GET /camera/position

Response:
{
  "success": true,
  "position": {"pan": 90, "tilt": 90}
}
```

### Center Camera
```http
POST /camera/center

Response:
{
  "success": true,
  "message": "Camera centered",
  "position": {"pan": 90, "tilt": 90}
}
```

## Treat Dispensing

### Dispense Treat
```http
POST /treat/dispense
Content-Type: application/json

{
  "dog_id": "buddy",
  "reason": "good_behavior",
  "count": 1
}

Response:
{
  "success": true,
  "message": "Treat dispensed",
  "dog_id": "buddy",
  "count": 1
}
```

## WebSocket Control (Real-time)

### Connection
```javascript
ws://[robot-ip]:8000/ws/control
```

### Commands

#### Motor Control
```json
// Send
{
  "command": "motor",
  "left": 50,
  "right": 50
}

// Receive
{
  "type": "motor_ack",
  "left": 50,
  "right": 50
}
```

#### Joystick Control
```json
// Send
{
  "command": "joystick",
  "x": 0.5,
  "y": 0.8
}

// Receive
{
  "type": "joystick_ack",
  "motors": {"left": 130, "right": 30}
}
```

#### Camera Control
```json
// Send
{
  "command": "camera",
  "pan": 90,
  "tilt": 45
}

// Receive
{
  "type": "camera_ack",
  "position": {"pan": 90, "tilt": 45}
}
```

#### Emergency Stop
```json
// Send
{
  "command": "stop"
}

// Receive
{
  "type": "stop_ack",
  "message": "Motors stopped"
}
```

#### Treat Dispense
```json
// Send
{
  "command": "treat"
}

// Receive
{
  "type": "treat_ack",
  "message": "Treat dispensed"
}
```

#### Keep-Alive
```json
// Send
{
  "command": "ping",
  "timestamp": 1234567890
}

// Receive
{
  "type": "pong",
  "timestamp": 1234567890
}
```

## System Status

### Get Full System Status
```http
GET /system/status

Response:
{
  "state": {
    "mode": "IDLE",
    "emergency": false,
    "hardware_status": {...}
  },
  "services": {
    "detector": {...},
    "pantilt": {...},
    "dispenser": {...}
  }
}
```

### Get Mode
```http
GET /mode

Response:
{
  "mode": "IDLE",
  "mode_info": {...}
}
```

### Set Mode
```http
POST /mode
Content-Type: application/json

{
  "mode": "MANUAL",
  "duration": 300  // Optional, seconds
}
```

## iOS App Implementation Notes

### Swift WebSocket Example
```swift
import Foundation

class RobotController {
    var webSocket: URLSessionWebSocketTask?

    func connect() {
        let url = URL(string: "ws://192.168.1.100:8000/ws/control")!
        webSocket = URLSession.shared.webSocketTask(with: url)
        webSocket?.resume()
        receiveMessage()
    }

    func sendJoystick(x: Float, y: Float) {
        let command = [
            "command": "joystick",
            "x": x,
            "y": y
        ]

        if let data = try? JSONSerialization.data(withJSONObject: command) {
            let message = URLSessionWebSocketTask.Message.data(data)
            webSocket?.send(message) { error in
                if let error = error {
                    print("WebSocket send error: \(error)")
                }
            }
        }
    }

    func receiveMessage() {
        webSocket?.receive { [weak self] result in
            switch result {
            case .success(let message):
                // Handle received message
                self?.receiveMessage() // Continue listening
            case .failure(let error):
                print("WebSocket receive error: \(error)")
            }
        }
    }
}
```

### SwiftUI Joystick Control View
```swift
struct JoystickView: View {
    @State private var joystickPosition = CGSize.zero
    let controller = RobotController()

    var body: some View {
        GeometryReader { geometry in
            Circle()
                .fill(Color.gray.opacity(0.3))
                .frame(width: 200, height: 200)
                .overlay(
                    Circle()
                        .fill(Color.blue)
                        .frame(width: 50, height: 50)
                        .offset(joystickPosition)
                )
                .gesture(
                    DragGesture()
                        .onChanged { value in
                            let x = Float(value.translation.width / 100)
                            let y = Float(-value.translation.height / 100)

                            // Clamp to unit circle
                            let magnitude = sqrt(x*x + y*y)
                            if magnitude <= 1.0 {
                                joystickPosition = value.translation
                                controller.sendJoystick(x: x, y: y)
                            }
                        }
                        .onEnded { _ in
                            joystickPosition = .zero
                            controller.sendJoystick(x: 0, y: 0)
                        }
                )
        }
    }
}
```

## Testing with curl

### Test Motor Control
```bash
curl -X POST http://192.168.1.100:8000/motor/control \
  -H "Content-Type: application/json" \
  -d '{"left_speed": 50, "right_speed": 50, "duration": 2}'
```

### Test Emergency Stop
```bash
curl -X POST http://192.168.1.100:8000/motor/stop
```

### Test WebSocket with wscat
```bash
npm install -g wscat
wscat -c ws://192.168.1.100:8000/ws/control
> {"command": "joystick", "x": 0.5, "y": 0.5}
```

## Error Handling

All endpoints return standard HTTP status codes:
- 200: Success
- 400: Bad Request (invalid parameters)
- 500: Internal Server Error

Error response format:
```json
{
  "detail": "Error message here"
}
```

## Rate Limiting

- REST API: No hard limit, but recommended < 10 requests/second
- WebSocket: Can handle 30-60 Hz control updates

## Future Enhancements (Phase 5.4)

1. JWT Authentication
2. SSL/TLS encryption
3. User profiles and permissions
4. Video streaming endpoint
5. Cloud connectivity
6. Push notifications for events