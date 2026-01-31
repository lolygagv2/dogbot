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

---

## Missions API

Training missions are structured training sessions with specific goals (e.g., "sit 5 times", "stay quiet for 2 minutes").

### Get Available Missions
```http
GET /missions/available

Response:
{
  "missions": [
    {
      "name": "sit_training",
      "description": "Basic sit training",
      "enabled": true
    },
    ...
  ]
}
```

### Get Mission Status
```http
GET /missions/status

Response:
{
  "active": true,
  "mission_name": "sit_training",
  "state": "running",
  "progress": {
    "current_stage": 2,
    "total_stages": 5,
    "successes": 3,
    "failures": 1
  },
  "start_time": "2026-01-30T10:00:00",
  "elapsed_seconds": 120
}
```

### Start Mission
```http
POST /missions/start
Content-Type: application/json

{
  "mission_name": "sit_training",
  "parameters": {
    "dog_id": "elsa"
  }
}

Response:
{
  "success": true,
  "message": "Mission 'sit_training' started successfully",
  "mission_name": "sit_training"
}
```

### Stop Mission
```http
POST /missions/stop

Response:
{
  "success": true,
  "message": "Mission stopped"
}
```

### Pause Mission
```http
POST /missions/pause

Response:
{
  "success": true,
  "message": "Mission paused"
}
```

### Resume Mission
```http
POST /missions/resume

Response:
{
  "success": true,
  "message": "Mission resumed"
}
```

---

## Mission Scheduler API

Auto-start missions based on time schedules (e.g., "morning quiet training at 8am").

### Get Scheduler Status
```http
GET /missions/schedule

Response:
{
  "success": true,
  "scheduler": {
    "enabled": false,
    "running": false,
    "check_interval": 60,
    "last_started": {},
    "scheduled_missions": [...]
  }
}
```

### Enable Auto-Scheduling
```http
POST /missions/schedule/enable

Response:
{
  "success": true,
  "message": "Scheduler enabled"
}
```

### Disable Auto-Scheduling
```http
POST /missions/schedule/disable

Response:
{
  "success": true,
  "message": "Scheduler disabled"
}
```

### List Scheduled Missions
```http
GET /missions/schedule/list

Response:
{
  "success": true,
  "missions": [
    {
      "name": "morning_quiet_2hr",
      "description": "Morning quiet time",
      "enabled": true,
      "schedule_type": "daily",
      "start_time": "08:00",
      "end_time": "10:00",
      "days_of_week": ["mon", "tue", "wed", "thu", "fri"],
      "next_run": "2026-01-31T08:00:00",
      "last_run": null
    }
  ]
}
```

### Force Start Scheduled Mission
```http
POST /missions/schedule/force/{mission_name}

Response:
{
  "success": true,
  "message": "Mission morning_quiet_2hr started"
}
```

---

## Training Programs API

Programs are collections of multiple missions that run sequentially (e.g., "Puppy Basics" = sit + down + quiet).

### Get Available Programs
```http
GET /programs/available

Response:
{
  "success": true,
  "programs": [
    {
      "name": "puppy_basics",
      "display_name": "Puppy Basics",
      "description": "Foundation training for new puppies",
      "missions": ["sit_training", "down_sustained", "quiet_progressive"],
      "created_by": "preset",
      "daily_treat_limit": 15
    },
    {
      "name": "quiet_dog",
      "display_name": "Quiet Dog",
      "description": "Bark reduction focus",
      "missions": ["quiet_progressive", "bark_prevention"],
      "created_by": "preset",
      "daily_treat_limit": 12
    }
  ],
  "count": 5
}
```

### Get Program Details
```http
GET /programs/{name}

Example: GET /programs/puppy_basics

Response:
{
  "success": true,
  "program": {
    "name": "puppy_basics",
    "display_name": "Puppy Basics",
    "description": "Foundation training for new puppies - teaches sit, down, and quiet behavior",
    "missions": ["sit_training", "down_sustained", "quiet_progressive"],
    "created_by": "preset",
    "repeat": false,
    "daily_treat_limit": 15,
    "rest_between_missions_sec": 60
  }
}
```

### Create Custom Program
```http
POST /programs/create
Content-Type: application/json

{
  "name": "my_routine",
  "display_name": "My Training Routine",
  "description": "Custom daily routine for Elsa",
  "missions": ["sit_training", "down_sustained"],
  "daily_treat_limit": 12,
  "rest_between_missions_sec": 60
}

Response:
{
  "success": true,
  "message": "Program 'my_routine' created",
  "program": {...}
}
```

### Delete Custom Program
```http
DELETE /programs/{name}

Example: DELETE /programs/my_routine

Response:
{
  "success": true,
  "message": "Program 'my_routine' deleted"
}

Note: Preset programs (puppy_basics, quiet_dog, etc.) cannot be deleted.
```

### Start Program
```http
POST /programs/start
Content-Type: application/json

{
  "program_name": "puppy_basics",
  "dog_id": "elsa"
}

Response:
{
  "success": true,
  "message": "Program 'puppy_basics' started",
  "status": {
    "state": "running",
    "program_name": "puppy_basics",
    "current_mission": "sit_training",
    "current_mission_index": 0,
    "total_missions": 3,
    "treats_dispensed": 0
  }
}
```

### Stop Program
```http
POST /programs/stop

Response:
{
  "success": true,
  "message": "Program stopped"
}
```

### Pause Program
```http
POST /programs/pause

Response:
{
  "success": true,
  "message": "Program paused"
}
```

### Resume Program
```http
POST /programs/resume

Response:
{
  "success": true,
  "message": "Program resumed"
}
```

### Get Program Status
```http
GET /programs/status

Response:
{
  "state": "running",
  "program_name": "puppy_basics",
  "display_name": "Puppy Basics",
  "current_mission": "down_sustained",
  "current_mission_index": 1,
  "total_missions": 3,
  "missions_completed": ["sit_training"],
  "missions_failed": [],
  "treats_dispensed": 4,
  "daily_treat_limit": 15,
  "elapsed_seconds": 180
}
```

### Reload Programs from Disk
```http
POST /programs/reload

Response:
{
  "success": true,
  "message": "Reloaded 5 programs"
}
```

---

## Reports & Analytics API

Weekly summaries, behavior trends, and per-dog progress tracking.

### Get Current Week Summary
```http
GET /reports/weekly

Response:
{
  "success": true,
  "report": {
    "report_type": "weekly_summary",
    "generated_at": "2026-01-30T12:00:00",
    "week_start": "2026-01-27T00:00:00",
    "week_end": "2026-02-02T23:59:59",
    "week_number": 5,
    "year": 2026,
    "bark_stats": {
      "total": 45,
      "avg_loudness": 72.5,
      "by_emotion": {"alert": 20, "attention": 15, "anxious": 10},
      "by_dog": {"elsa": 30, "bezik": 15}
    },
    "reward_stats": {
      "total_treats": 28,
      "by_behavior": {"sit": 12, "down": 8, "quiet": 8},
      "by_dog": {"elsa": 18, "bezik": 10}
    },
    "silent_guardian": {
      "interventions": 5,
      "success_rate": 0.8
    },
    "coaching": {
      "sessions": 12,
      "tricks_practiced": {"sit": 5, "down": 4, "speak": 3},
      "success_rate": 0.75
    },
    "highlights": [
      "Elsa improved sit success rate by 15%",
      "Bark frequency down 20% from last week"
    ]
  }
}
```

### Get Weekly Report for Specific Date
```http
GET /reports/weekly/{date}

Example: GET /reports/weekly/2026-01-15

Response:
{
  "success": true,
  "report": {...}  // Same format as /reports/weekly
}
```

### Get Behavior Trends (Multi-Week)
```http
GET /reports/trends?weeks=8

Response:
{
  "success": true,
  "trends": {
    "weeks_analyzed": 8,
    "bark_trend": [
      {"week": 1, "count": 80},
      {"week": 2, "count": 65},
      {"week": 3, "count": 55},
      ...
    ],
    "reward_trend": [...],
    "coaching_success_trend": [...],
    "summary": {
      "bark_change_percent": -31.25,
      "treat_efficiency_change": 12.5,
      "best_performing_dog": "elsa"
    }
  }
}
```

### Get Individual Dog Progress
```http
GET /reports/dog/{dog_id}?weeks=8

Example: GET /reports/dog/elsa?weeks=8

Response:
{
  "success": true,
  "progress": {
    "dog_id": "elsa",
    "dog_name": "Elsa",
    "weeks_analyzed": 8,
    "tricks": {
      "sit": {"attempts": 45, "successes": 38, "rate": 0.84},
      "down": {"attempts": 30, "successes": 22, "rate": 0.73},
      "speak": {"attempts": 20, "successes": 15, "rate": 0.75}
    },
    "bark_stats": {
      "total": 180,
      "weekly_average": 22.5,
      "trend": "decreasing"
    },
    "treats_earned": 85,
    "coaching_sessions": 28,
    "improvement_areas": ["down", "crosses"],
    "strengths": ["sit", "speak"]
  }
}
```

### Compare All Dogs
```http
GET /reports/compare

Response:
{
  "success": true,
  "comparison": {
    "dogs": [
      {
        "dog_id": "elsa",
        "name": "Elsa",
        "total_treats": 85,
        "coaching_sessions": 28,
        "bark_count": 180,
        "best_trick": "sit",
        "needs_work": "crosses"
      },
      {
        "dog_id": "bezik",
        "name": "Bezik",
        "total_treats": 62,
        "coaching_sessions": 22,
        "bark_count": 95,
        "best_trick": "down",
        "needs_work": "speak"
      }
    ],
    "leader_board": {
      "most_treats": "elsa",
      "quietest": "bezik",
      "best_learner": "elsa"
    }
  }
}
```

### Export Report to File
```http
POST /reports/export?format=markdown

Formats: "markdown" or "csv"

Response:
{
  "success": true,
  "filepath": "/home/morgan/dogbot/reports/weekly_report_2026_w05_20260130_120000.md",
  "format": "markdown"
}
```

### Get HTML Weekly Report (Browser-Viewable)
```http
GET /reports/html/weekly

Response: HTML page with formatted weekly report
```

### Get HTML Dog Report (Browser-Viewable)
```http
GET /reports/html/dog/{dog_id}

Example: GET /reports/html/dog/elsa

Response: HTML page with formatted dog progress report
```

---

## iOS App Implementation Notes - New Features

### Swift Program Control Example
```swift
class ProgramController {
    let baseURL = "http://192.168.1.100:8000"

    func getPrograms() async throws -> [Program] {
        let url = URL(string: "\(baseURL)/programs/available")!
        let (data, _) = try await URLSession.shared.data(from: url)
        let response = try JSONDecoder().decode(ProgramsResponse.self, from: data)
        return response.programs
    }

    func startProgram(_ name: String, dogId: String?) async throws {
        let url = URL(string: "\(baseURL)/programs/start")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = ["program_name": name, "dog_id": dogId]
        request.httpBody = try JSONEncoder().encode(body)

        let (_, response) = try await URLSession.shared.data(for: request)
        // Handle response
    }

    func getProgramStatus() async throws -> ProgramStatus {
        let url = URL(string: "\(baseURL)/programs/status")!
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(ProgramStatus.self, from: data)
    }
}
```

### Swift Reports Example
```swift
class ReportsController {
    let baseURL = "http://192.168.1.100:8000"

    func getWeeklyReport() async throws -> WeeklyReport {
        let url = URL(string: "\(baseURL)/reports/weekly")!
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(WeeklyReportResponse.self, from: data).report
    }

    func getDogProgress(_ dogId: String, weeks: Int = 8) async throws -> DogProgress {
        let url = URL(string: "\(baseURL)/reports/dog/\(dogId)?weeks=\(weeks)")!
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(DogProgressResponse.self, from: data).progress
    }

    func getTrends(weeks: Int = 8) async throws -> Trends {
        let url = URL(string: "\(baseURL)/reports/trends?weeks=\(weeks)")!
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(TrendsResponse.self, from: data).trends
    }
}
```

---

## Complete Endpoint Reference

### Motor Control
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/motor/control` | Direct motor control |
| POST | `/motor/joystick` | Virtual joystick control |
| POST | `/motor/stop` | Emergency stop |

### Camera Control
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/camera/pantilt` | Set pan/tilt position |
| GET | `/camera/position` | Get current position |
| POST | `/camera/center` | Center camera |

### Treats
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/treat/dispense` | Dispense treat |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/system/status` | Full system status |
| GET | `/mode` | Get current mode |
| POST | `/mode` | Set mode |

### Missions
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/missions/available` | List all missions |
| GET | `/missions/status` | Current mission status |
| POST | `/missions/start` | Start a mission |
| POST | `/missions/stop` | Stop current mission |
| POST | `/missions/pause` | Pause current mission |
| POST | `/missions/resume` | Resume paused mission |

### Mission Scheduler
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/missions/schedule` | Scheduler status |
| POST | `/missions/schedule/enable` | Enable auto-scheduling |
| POST | `/missions/schedule/disable` | Disable auto-scheduling |
| GET | `/missions/schedule/list` | List scheduled missions |
| POST | `/missions/schedule/force/{name}` | Force start mission |

### Training Programs
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/programs/available` | List all programs |
| GET | `/programs/{name}` | Get program details |
| POST | `/programs/create` | Create custom program |
| DELETE | `/programs/{name}` | Delete custom program |
| POST | `/programs/start` | Start a program |
| POST | `/programs/stop` | Stop current program |
| POST | `/programs/pause` | Pause current program |
| POST | `/programs/resume` | Resume paused program |
| GET | `/programs/status` | Get program status |
| POST | `/programs/reload` | Reload from disk |

### Reports & Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/reports/weekly` | Current week summary |
| GET | `/reports/weekly/{date}` | Specific week summary |
| GET | `/reports/trends` | Multi-week trends |
| GET | `/reports/dog/{dog_id}` | Per-dog progress |
| GET | `/reports/compare` | Compare all dogs |
| POST | `/reports/export` | Export to file |
| GET | `/reports/html/weekly` | HTML weekly report |
| GET | `/reports/html/dog/{dog_id}` | HTML dog report |

### WebSocket
| Endpoint | Description |
|----------|-------------|
| `/ws/control` | Real-time control |

---

## Future Enhancements (Phase 5.4)

1. JWT Authentication
2. SSL/TLS encryption
3. User profiles and permissions
4. Video streaming endpoint
5. Cloud connectivity
6. Push notifications for events