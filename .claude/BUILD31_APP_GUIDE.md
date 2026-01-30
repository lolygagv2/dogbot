# WIM-Z Build 31 - Complete App Integration Guide

**Date:** January 30, 2026
**For:** iOS App Developer
**Robot API Base URL:** `http://<robot-ip>:8000`
**WebSocket Relay:** `wss://api.wimzai.com/ws/app`

---

## Table of Contents

1. [What Changed in Build 31](#1-what-changed-in-build-31)
2. [Connection Architecture](#2-connection-architecture)
3. [Missions API](#3-missions-api)
4. [Programs API](#4-programs-api)
5. [Mode Control](#5-mode-control)
6. [Reports & Analytics](#6-reports--analytics)
7. [Real-time WebSocket Events](#7-real-time-websocket-events)
8. [Video Overlay](#8-video-overlay)
9. [TURN Server Issue](#9-turn-server-issue-action-required)
10. [Swift Implementation](#10-swift-implementation)
11. [Checklist](#11-checklist)

---

## 1. What Changed in Build 31

### Mission Engine Rewrite
The robot now uses **coach-style flow** for missions:

```
WAITING_FOR_DOG → GREETING → COMMAND → WATCHING → SUCCESS/FAILURE
```

**Before:** Mission silently polled for poses, no audio, no feedback.
**After:** Mission greets dog by name, plays trick command audio, shows progress, retries on failure.

### New WebSocket Events
The robot now broadcasts `mission_progress` events in real-time so the app can show live status.

### Mode Locking
When a mission is active, the mode is **locked** and cannot be changed. The app should disable the mode selector.

### Bark Detection
Microphone only enabled in specific modes now:
- SILENT_GUARDIAN: Always on
- COACH: Always on (for speak trick)
- MISSION: Only for bark/quiet missions
- IDLE/MANUAL: Off

### Video Overlay
Status text now appears on the WebRTC video stream (large text at top showing current state).

---

## 2. Connection Architecture

### Direct Connection (Same Network)
```
iPhone App  ←──REST/WebSocket──→  Robot (192.168.x.x:8000)
```

### Cloud Relay (Remote Access)
```
iPhone App  ←──WebSocket──→  Relay Server  ←──WebSocket──→  Robot
                           (api.wimzai.com)
```

| Scenario | Connection | Why |
|----------|------------|-----|
| Same WiFi | Direct REST | Fastest, lowest latency |
| Remote/Cellular | Cloud Relay | NAT traversal |
| Video streaming | WebRTC via Relay | ICE negotiation |
| Real-time events | WebSocket | Push notifications |

---

## 3. Missions API

Missions are single training goals (e.g., "sit 5 times", "hold down for 30 seconds").

### 3.1 List Available Missions

```http
GET /missions/available
```

**Response:**
```json
{
  "missions": [
    {
      "name": "sit_training",
      "description": "Practice sit command 5 times",
      "enabled": true,
      "max_rewards": 5,
      "duration_minutes": 30,
      "stages": 5
    },
    {
      "name": "down_sustained",
      "description": "Hold down position for 30 seconds",
      "enabled": true,
      "max_rewards": 3,
      "duration_minutes": 15,
      "stages": 3
    },
    {
      "name": "quiet_training",
      "description": "Bark prevention training",
      "enabled": true,
      "max_rewards": 5,
      "duration_minutes": 20,
      "stages": 5
    }
  ]
}
```

---

### 3.2 Start a Mission

```http
POST /missions/start
Content-Type: application/json

{
  "mission_name": "sit_training",
  "parameters": {
    "dog_id": "elsa"
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Mission 'sit_training' started successfully",
  "mission_name": "sit_training",
  "mission_id": 42
}
```

**What Happens on Robot:**
1. Mode changes to MISSION (locked)
2. Robot waits for dog to appear
3. When dog detected: greeting → command → watch → reward cycle
4. WebSocket `mission_progress` events sent at each state change

**App Should:**
- Disable mode selector (mode is locked)
- Show "Mission Active" indicator
- Listen for `mission_progress` events
- Display current stage and trick

---

### 3.3 Get Mission Status

```http
GET /missions/status
```

**Response (mission active):**
```json
{
  "active": true,
  "mission_id": 42,
  "mission_name": "sit_training",
  "dog_id": "aruco_315",
  "dog_name": "Elsa",
  "state": "watching",
  "trick_requested": "sit",
  "current_stage": 2,
  "total_stages": 5,
  "stage_info": {
    "name": "Sit Practice 2",
    "timeout": 60.0,
    "elapsed": 15.3,
    "success_event": "VisionEvent.Pose.Sit"
  },
  "rewards_given": 1,
  "max_rewards": 5,
  "duration": 180.5,
  "max_duration": 1800
}
```

**Response (no mission):**
```json
{
  "active": false
}
```

---

### 3.4 Stop/Pause/Resume Mission

```http
POST /missions/stop
POST /missions/pause
POST /missions/resume
```

**Response:**
```json
{
  "success": true,
  "message": "Mission stopped"
}
```

---

## 4. Programs API

Programs are sequences of multiple missions with rest periods between (e.g., "Puppy Basics" = sit + down + quiet).

### 4.1 List Available Programs

```http
GET /programs/available
```

**Response:**
```json
{
  "success": true,
  "programs": [
    {
      "name": "puppy_basics",
      "display_name": "Puppy Basics",
      "description": "Foundation training: sit, down, quiet",
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
    },
    {
      "name": "trick_master",
      "display_name": "Trick Master",
      "description": "Advanced tricks training",
      "missions": ["spin_training", "speak_training"],
      "created_by": "preset",
      "daily_treat_limit": 20
    }
  ],
  "count": 5
}
```

---

### 4.2 Start a Program

```http
POST /programs/start
Content-Type: application/json

{
  "program_name": "puppy_basics",
  "dog_id": "elsa"
}
```

**Response:**
```json
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

---

### 4.3 Get Program Status

```http
GET /programs/status
```

**Response:**
```json
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
  "elapsed_seconds": 420
}
```

**Suggested App Display:**
```
Puppy Basics
━━━━━━━━━━━━━━━━━━━━━━
Mission 2 of 3: Down Sustained
✓ Sit Training (complete)
● Down Sustained (in progress)
○ Quiet Progressive (pending)

Treats: 4/15 | Time: 7:00
```

---

### 4.4 Stop/Pause/Resume Program

```http
POST /programs/stop
POST /programs/pause
POST /programs/resume
```

---

### 4.5 Create Custom Program

```http
POST /programs/create
Content-Type: application/json

{
  "name": "morning_routine",
  "display_name": "Morning Routine",
  "description": "Quick morning training session",
  "missions": ["sit_training", "quiet_progressive"],
  "daily_treat_limit": 8,
  "rest_between_missions_sec": 30
}
```

---

### 4.6 Delete Custom Program

```http
DELETE /programs/{name}
```

Note: Preset programs cannot be deleted.

---

## 5. Mode Control

### 5.1 Get Current Mode

```http
GET /mode
```

**Response:**
```json
{
  "mode": "mission",
  "mode_info": {
    "locked": true,
    "lock_reason": "Mission active: sit_training",
    "since": "2026-01-30T10:15:30"
  }
}
```

**Modes:**
| Mode | Description | App Behavior |
|------|-------------|--------------|
| `idle` | Standby, no AI | Default state |
| `coach` | Opportunistic trick training | Show coaching status |
| `mission` | Structured mission | Show mission progress |
| `silent_guardian` | Bark monitoring | Show bark stats |
| `manual` | Xbox/joystick control | Show manual controls |

---

### 5.2 Set Mode

```http
POST /mode
Content-Type: application/json

{
  "mode": "coach"
}
```

**Response (success):**
```json
{
  "success": true,
  "mode": "coach",
  "previous_mode": "idle"
}
```

**Response (locked):**
```json
{
  "success": false,
  "error": "Mode locked: Mission active: sit_training"
}
```

**Important:** When `locked: true`, disable the mode selector in the app.

---

## 6. Reports & Analytics

### 6.1 Weekly Summary

```http
GET /reports/weekly
```

**Response:**
```json
{
  "success": true,
  "report": {
    "week_start": "2026-01-27T00:00:00",
    "week_end": "2026-02-02T23:59:59",
    "week_number": 5,
    "year": 2026,

    "bark_stats": {
      "total": 45,
      "avg_loudness": 72.5,
      "by_emotion": {"alert": 20, "attention": 15, "anxious": 10},
      "by_dog": {"Elsa": 30, "Bezik": 15}
    },

    "reward_stats": {
      "total_treats": 28,
      "by_behavior": {"sit": 12, "down": 8, "quiet": 8},
      "by_dog": {"Elsa": 18, "Bezik": 10}
    },

    "coaching": {
      "sessions": 12,
      "success_rate": 0.75,
      "tricks_practiced": {"sit": 5, "down": 4, "speak": 3}
    },

    "highlights": [
      "Elsa improved sit success rate by 15%",
      "Bark frequency down 20% from last week"
    ]
  }
}
```

---

### 6.2 Individual Dog Progress

```http
GET /reports/dog/elsa?weeks=8
```

**Response:**
```json
{
  "success": true,
  "progress": {
    "dog_id": "aruco_315",
    "dog_name": "Elsa",
    "weeks_analyzed": 8,

    "tricks": {
      "sit": {"attempts": 45, "successes": 38, "rate": 0.84, "trend": "improving"},
      "down": {"attempts": 30, "successes": 22, "rate": 0.73, "trend": "stable"},
      "speak": {"attempts": 20, "successes": 15, "rate": 0.75, "trend": "new"}
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

---

### 6.3 Trends Over Time

```http
GET /reports/trends?weeks=8
```

**Response:**
```json
{
  "success": true,
  "trends": {
    "weeks_analyzed": 8,
    "bark_trend": [
      {"week": 1, "count": 80},
      {"week": 2, "count": 65},
      {"week": 3, "count": 55},
      {"week": 4, "count": 50},
      {"week": 5, "count": 48},
      {"week": 6, "count": 45},
      {"week": 7, "count": 42},
      {"week": 8, "count": 40}
    ],
    "summary": {
      "bark_change_percent": -50.0,
      "best_performing_dog": "Elsa"
    }
  }
}
```

---

### 6.4 Compare All Dogs

```http
GET /reports/compare
```

**Response:**
```json
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

---

## 7. Real-time WebSocket Events

### Connection

**Via Relay:**
```
wss://api.wimzai.com/ws/app?device_id=<robot_id>&token=<auth_token>
```

**Direct to Robot:**
```
ws://<robot-ip>:8000/ws/control
```

### Event Format

```json
{
  "type": "event",
  "event": "<event_name>",
  "data": { ... },
  "timestamp": "2026-01-30T10:15:30.123Z"
}
```

---

### 7.1 mission_progress (NEW - Most Important)

Sent whenever mission state changes. **This is how the app shows live training progress.**

```json
{
  "type": "event",
  "event": "mission_progress",
  "data": {
    "status": "watching",
    "trick": "sit",
    "stage": 2,
    "total_stages": 5,
    "dog_name": "Elsa",
    "rewards": 1,
    "progress": 2.5,
    "target_sec": 5.0
  }
}
```

**Status Values and Suggested UI:**

| Status | Meaning | App UI |
|--------|---------|--------|
| `waiting_for_dog` | Waiting for dog to appear | Pulsing "Waiting for dog..." |
| `greeting` | Playing dog's name | "Greeting Elsa..." |
| `command` | Playing trick command | "Commanding: SIT" |
| `watching` | Watching for behavior | **Progress bar** (progress/target_sec) |
| `success` | Trick completed | Green checkmark + celebration |
| `failed` | Stage failed | Brief red indicator |
| `retry` | Retrying after failure | "Trying again..." |
| `completed` | Mission finished | Show summary screen |

**Progress Bar (watching state):**
- `progress` = current hold time in seconds
- `target_sec` = required hold time
- Show: `progress / target_sec` as percentage

---

### 7.2 mode_changed

```json
{
  "type": "event",
  "event": "mode_changed",
  "data": {
    "mode": "mission",
    "previous_mode": "idle",
    "locked": true,
    "lock_reason": "Mission active: sit_training"
  }
}
```

**App Action:** When `locked: true`, disable mode selector.

---

### 7.3 audio_state (NEW)

Sent when audio playback state changes. **Use this to sync music player button state.**

```json
{
  "type": "event",
  "event": "audio_state",
  "data": {
    "state": "playing",
    "track": "default/Wimz_theme.mp3",
    "playing": true,
    "playlist_index": 3,
    "playlist_length": 14
  }
}
```

**State Values:**

| State | Meaning | App UI |
|-------|---------|--------|
| `playing` | Music is playing | Show pause/stop button |
| `stopped` | Music stopped | Show play button |
| `paused` | Music paused | Show resume button |

**App Action:**
- Update music player button icons based on `playing` boolean
- Update track name display from `track`
- Update playlist position from `playlist_index`

---

### 7.4 bark_detected

```json
{
  "type": "event",
  "event": "bark_detected",
  "data": {
    "emotion": "alert",
    "confidence": 0.85,
    "loudness_db": 75.2,
    "dog_id": "aruco_315",
    "dog_name": "Elsa"
  }
}
```

---

### 7.5 dog_detected

```json
{
  "type": "event",
  "event": "dog_detected",
  "data": {
    "dog_id": "aruco_315",
    "dog_name": "Elsa",
    "bbox": [100, 150, 400, 500],
    "confidence": 0.92
  }
}
```

---

### 7.6 treat_dispensed

```json
{
  "type": "event",
  "event": "treat_dispensed",
  "data": {
    "dog_id": "aruco_315",
    "dog_name": "Elsa",
    "behavior": "sit",
    "daily_count": 5,
    "daily_limit": 15
  }
}
```

---

## 8. Video Overlay

The robot now displays status text on the WebRTC video stream:
- Large text at top center
- Shows: Mode + State (e.g., "MISSION [2/5]: Watching for SIT")
- Color-coded (yellow=waiting, cyan=watching, green=success)

### Disable Overlay (Optional)

If the app shows its own native UI, you can disable the robot overlay:

```http
POST /video/overlay/disable
```

```http
POST /video/overlay/enable
```

---

## 9. TURN Server Issue (ACTION REQUIRED)

### Problem

The robot logs show WebRTC TURN authentication failures:
```
aioice.stun.TransactionFailed: STUN transaction failed (401 - )
```

### Cause

TURN server credentials are expiring or invalid. Video relay fails when direct P2P isn't possible.

### Required Fix (Relay Server Side)

1. Check TURN credential generation - ensure sufficient TTL
2. Implement credential refresh endpoint
3. Robot should request new credentials before expiration

### Suggested Credential Refresh API

```http
GET /api/turn/credentials
Authorization: Bearer <device_token>

Response:
{
  "urls": ["turn:turn.wimzai.com:3478"],
  "username": "device_abc123_1706648400",
  "credential": "generated_password",
  "ttl": 86400
}
```

---

## 10. Swift Implementation

### 10.1 API Client

```swift
import Foundation

class WIMZAPIClient {
    static let shared = WIMZAPIClient()

    var baseURL: URL?
    private let session = URLSession.shared

    // MARK: - Missions

    func getAvailableMissions() async throws -> [Mission] {
        let url = baseURL!.appendingPathComponent("/missions/available")
        let (data, _) = try await session.data(from: url)
        let response = try JSONDecoder().decode(MissionsResponse.self, from: data)
        return response.missions
    }

    func startMission(_ name: String, dogId: String? = nil) async throws -> Bool {
        let url = baseURL!.appendingPathComponent("/missions/start")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var body: [String: Any] = ["mission_name": name]
        if let dogId = dogId {
            body["parameters"] = ["dog_id": dogId]
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, _) = try await session.data(for: request)
        let response = try JSONDecoder().decode(GenericResponse.self, from: data)
        return response.success
    }

    func getMissionStatus() async throws -> MissionStatus {
        let url = baseURL!.appendingPathComponent("/missions/status")
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode(MissionStatus.self, from: data)
    }

    func stopMission() async throws {
        let url = baseURL!.appendingPathComponent("/missions/stop")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        _ = try await session.data(for: request)
    }

    // MARK: - Programs

    func getAvailablePrograms() async throws -> [Program] {
        let url = baseURL!.appendingPathComponent("/programs/available")
        let (data, _) = try await session.data(from: url)
        let response = try JSONDecoder().decode(ProgramsResponse.self, from: data)
        return response.programs
    }

    func startProgram(_ name: String, dogId: String? = nil) async throws -> Bool {
        let url = baseURL!.appendingPathComponent("/programs/start")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var body: [String: Any] = ["program_name": name]
        if let dogId = dogId { body["dog_id"] = dogId }
        request.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, _) = try await session.data(for: request)
        let response = try JSONDecoder().decode(GenericResponse.self, from: data)
        return response.success
    }

    func getProgramStatus() async throws -> ProgramStatus {
        let url = baseURL!.appendingPathComponent("/programs/status")
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode(ProgramStatus.self, from: data)
    }

    // MARK: - Reports

    func getWeeklyReport() async throws -> WeeklyReport {
        let url = baseURL!.appendingPathComponent("/reports/weekly")
        let (data, _) = try await session.data(from: url)
        let response = try JSONDecoder().decode(WeeklyReportResponse.self, from: data)
        return response.report
    }

    func getDogProgress(_ dogId: String, weeks: Int = 8) async throws -> DogProgress {
        var components = URLComponents(url: baseURL!.appendingPathComponent("/reports/dog/\(dogId)"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "weeks", value: "\(weeks)")]

        let (data, _) = try await session.data(from: components.url!)
        let response = try JSONDecoder().decode(DogProgressResponse.self, from: data)
        return response.progress
    }

    func getTrends(weeks: Int = 8) async throws -> Trends {
        var components = URLComponents(url: baseURL!.appendingPathComponent("/reports/trends"), resolvingAgainstBaseURL: false)!
        components.queryItems = [URLQueryItem(name: "weeks", value: "\(weeks)")]

        let (data, _) = try await session.data(from: components.url!)
        let response = try JSONDecoder().decode(TrendsResponse.self, from: data)
        return response.trends
    }

    // MARK: - Mode

    func getMode() async throws -> ModeInfo {
        let url = baseURL!.appendingPathComponent("/mode")
        let (data, _) = try await session.data(from: url)
        return try JSONDecoder().decode(ModeInfo.self, from: data)
    }

    func setMode(_ mode: String) async throws -> Bool {
        let url = baseURL!.appendingPathComponent("/mode")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONSerialization.data(withJSONObject: ["mode": mode])

        let (data, _) = try await session.data(for: request)
        let response = try JSONDecoder().decode(GenericResponse.self, from: data)
        return response.success
    }
}
```

---

### 10.2 WebSocket Manager

```swift
import Foundation

class WIMZWebSocket: NSObject, URLSessionWebSocketDelegate {
    static let shared = WIMZWebSocket()

    private var webSocket: URLSessionWebSocketTask?
    private var session: URLSession!

    // Event callbacks
    var onMissionProgress: ((MissionProgress) -> Void)?
    var onModeChanged: ((ModeChange) -> Void)?
    var onBarkDetected: ((BarkEvent) -> Void)?
    var onTreatDispensed: ((TreatEvent) -> Void)?
    var onDogDetected: ((DogEvent) -> Void)?
    var onDisconnect: (() -> Void)?

    override init() {
        super.init()
        session = URLSession(configuration: .default, delegate: self, delegateQueue: nil)
    }

    func connect(to url: URL) {
        disconnect()
        webSocket = session.webSocketTask(with: url)
        webSocket?.resume()
        receiveMessage()
    }

    func disconnect() {
        webSocket?.cancel(with: .goingAway, reason: nil)
        webSocket = nil
    }

    private func receiveMessage() {
        webSocket?.receive { [weak self] result in
            switch result {
            case .success(let message):
                self?.handleMessage(message)
                self?.receiveMessage()

            case .failure(let error):
                print("WebSocket error: \(error)")
                DispatchQueue.main.async {
                    self?.onDisconnect?()
                }
                // Auto-reconnect after 5 seconds
                DispatchQueue.main.asyncAfter(deadline: .now() + 5) { [weak self] in
                    if let url = self?.webSocket?.originalRequest?.url {
                        self?.connect(to: url)
                    }
                }
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        guard case .string(let text) = message,
              let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let eventName = json["event"] as? String,
              let eventData = json["data"] as? [String: Any] else {
            return
        }

        DispatchQueue.main.async { [weak self] in
            switch eventName {
            case "mission_progress":
                if let progress = self?.decode(MissionProgress.self, from: eventData) {
                    self?.onMissionProgress?(progress)
                }

            case "mode_changed":
                if let change = self?.decode(ModeChange.self, from: eventData) {
                    self?.onModeChanged?(change)
                }

            case "bark_detected":
                if let bark = self?.decode(BarkEvent.self, from: eventData) {
                    self?.onBarkDetected?(bark)
                }

            case "treat_dispensed":
                if let treat = self?.decode(TreatEvent.self, from: eventData) {
                    self?.onTreatDispensed?(treat)
                }

            case "dog_detected":
                if let dog = self?.decode(DogEvent.self, from: eventData) {
                    self?.onDogDetected?(dog)
                }

            default:
                break
            }
        }
    }

    private func decode<T: Decodable>(_ type: T.Type, from dict: [String: Any]) -> T? {
        guard let data = try? JSONSerialization.data(withJSONObject: dict) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    // Send commands
    func sendCommand(_ command: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: command),
              let string = String(data: data, encoding: .utf8) else { return }
        webSocket?.send(.string(string)) { _ in }
    }
}
```

---

### 10.3 Mission Progress View (SwiftUI)

```swift
import SwiftUI

struct MissionProgressView: View {
    @StateObject var viewModel = MissionProgressViewModel()

    var body: some View {
        VStack(spacing: 24) {
            // Mission name
            Text(viewModel.missionName)
                .font(.title)
                .bold()

            // Stage dots
            HStack(spacing: 8) {
                ForEach(1...max(viewModel.totalStages, 1), id: \.self) { stage in
                    Circle()
                        .fill(stageColor(stage))
                        .frame(width: 16, height: 16)
                }
            }
            Text("Stage \(viewModel.currentStage) of \(viewModel.totalStages)")
                .font(.caption)
                .foregroundColor(.secondary)

            Spacer()

            // Main status display
            statusView
                .frame(maxWidth: .infinity)
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(16)

            Spacer()

            // Treat counter
            HStack {
                Image(systemName: "leaf.fill")
                    .foregroundColor(.green)
                Text("\(viewModel.treatsGiven) / \(viewModel.maxTreats) treats")
            }
            .font(.headline)

            // Stop button
            Button(action: { Task { await viewModel.stopMission() } }) {
                Text("Stop Mission")
                    .frame(maxWidth: .infinity)
                    .padding()
            }
            .buttonStyle(.borderedProminent)
            .tint(.red)
        }
        .padding()
        .onAppear { viewModel.startListening() }
        .onDisappear { viewModel.stopListening() }
    }

    @ViewBuilder
    var statusView: some View {
        switch viewModel.status {
        case "waiting_for_dog":
            VStack(spacing: 16) {
                ProgressView()
                    .scaleEffect(1.5)
                Text("Waiting for dog...")
                    .font(.title2)
            }
            .foregroundColor(.yellow)

        case "greeting", "command":
            VStack(spacing: 16) {
                Image(systemName: "speaker.wave.3.fill")
                    .font(.system(size: 50))
                Text("Commanding: \(viewModel.trick.uppercased())")
                    .font(.title2)
            }
            .foregroundColor(.orange)

        case "watching":
            VStack(spacing: 16) {
                Text("Watching for \(viewModel.trick.uppercased())")
                    .font(.title2)

                // Progress bar
                GeometryReader { geo in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.gray.opacity(0.3))
                        RoundedRectangle(cornerRadius: 8)
                            .fill(Color.cyan)
                            .frame(width: geo.size.width * min(viewModel.progress / viewModel.targetSec, 1.0))
                    }
                }
                .frame(height: 24)

                Text("\(String(format: "%.1f", viewModel.progress))s / \(String(format: "%.0f", viewModel.targetSec))s")
                    .font(.headline)

                if let dogName = viewModel.dogName {
                    Text("Dog: \(dogName)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                }
            }

        case "success":
            VStack(spacing: 16) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 70))
                Text("SUCCESS!")
                    .font(.largeTitle)
                    .bold()
            }
            .foregroundColor(.green)

        case "failed", "retry":
            VStack(spacing: 16) {
                Image(systemName: "arrow.clockwise.circle")
                    .font(.system(size: 50))
                Text("Trying again...")
                    .font(.title2)
            }
            .foregroundColor(.orange)

        default:
            Text(viewModel.status)
                .font(.title2)
        }
    }

    func stageColor(_ stage: Int) -> Color {
        if stage < viewModel.currentStage { return .green }
        if stage == viewModel.currentStage { return .blue }
        return .gray.opacity(0.3)
    }
}

@MainActor
class MissionProgressViewModel: ObservableObject {
    @Published var missionName = "Mission"
    @Published var status = "waiting_for_dog"
    @Published var trick = ""
    @Published var currentStage = 1
    @Published var totalStages = 1
    @Published var dogName: String?
    @Published var treatsGiven = 0
    @Published var maxTreats = 5
    @Published var progress: Double = 0
    @Published var targetSec: Double = 10

    func startListening() {
        WIMZWebSocket.shared.onMissionProgress = { [weak self] event in
            self?.status = event.status
            self?.trick = event.trick ?? ""
            self?.currentStage = event.stage
            self?.totalStages = event.totalStages
            self?.dogName = event.dogName
            self?.treatsGiven = event.rewards
            self?.progress = event.progress ?? 0
            self?.targetSec = event.targetSec ?? 10
        }
    }

    func stopListening() {
        WIMZWebSocket.shared.onMissionProgress = nil
    }

    func stopMission() async {
        try? await WIMZAPIClient.shared.stopMission()
    }
}
```

---

### 10.4 Data Models

```swift
// MARK: - Generic

struct GenericResponse: Codable {
    let success: Bool
    let message: String?
}

// MARK: - Missions

struct Mission: Codable, Identifiable {
    var id: String { name }
    let name: String
    let description: String
    let enabled: Bool
    let maxRewards: Int
    let durationMinutes: Int
    let stages: Int

    enum CodingKeys: String, CodingKey {
        case name, description, enabled, stages
        case maxRewards = "max_rewards"
        case durationMinutes = "duration_minutes"
    }
}

struct MissionsResponse: Codable {
    let missions: [Mission]
}

struct MissionStatus: Codable {
    let active: Bool
    let missionId: Int?
    let missionName: String?
    let dogId: String?
    let dogName: String?
    let state: String?
    let trickRequested: String?
    let currentStage: Int?
    let totalStages: Int?
    let rewardsGiven: Int?
    let maxRewards: Int?
    let duration: Double?
    let maxDuration: Double?

    enum CodingKeys: String, CodingKey {
        case active, state, duration
        case missionId = "mission_id"
        case missionName = "mission_name"
        case dogId = "dog_id"
        case dogName = "dog_name"
        case trickRequested = "trick_requested"
        case currentStage = "current_stage"
        case totalStages = "total_stages"
        case rewardsGiven = "rewards_given"
        case maxRewards = "max_rewards"
        case maxDuration = "max_duration"
    }
}

struct MissionProgress: Codable {
    let status: String
    let trick: String?
    let stage: Int
    let totalStages: Int
    let dogName: String?
    let rewards: Int
    let progress: Double?
    let targetSec: Double?

    enum CodingKeys: String, CodingKey {
        case status, trick, stage, rewards, progress
        case totalStages = "total_stages"
        case dogName = "dog_name"
        case targetSec = "target_sec"
    }
}

// MARK: - Programs

struct Program: Codable, Identifiable {
    var id: String { name }
    let name: String
    let displayName: String
    let description: String
    let missions: [String]
    let createdBy: String
    let dailyTreatLimit: Int

    enum CodingKeys: String, CodingKey {
        case name, description, missions
        case displayName = "display_name"
        case createdBy = "created_by"
        case dailyTreatLimit = "daily_treat_limit"
    }
}

struct ProgramsResponse: Codable {
    let success: Bool
    let programs: [Program]
    let count: Int
}

struct ProgramStatus: Codable {
    let state: String
    let programName: String
    let displayName: String
    let currentMission: String
    let currentMissionIndex: Int
    let totalMissions: Int
    let missionsCompleted: [String]
    let missionsFailed: [String]
    let treatsDispensed: Int
    let dailyTreatLimit: Int
    let elapsedSeconds: Double

    enum CodingKeys: String, CodingKey {
        case state
        case programName = "program_name"
        case displayName = "display_name"
        case currentMission = "current_mission"
        case currentMissionIndex = "current_mission_index"
        case totalMissions = "total_missions"
        case missionsCompleted = "missions_completed"
        case missionsFailed = "missions_failed"
        case treatsDispensed = "treats_dispensed"
        case dailyTreatLimit = "daily_treat_limit"
        case elapsedSeconds = "elapsed_seconds"
    }
}

// MARK: - Mode

struct ModeInfo: Codable {
    let mode: String
    let modeInfo: ModeDetails?

    enum CodingKeys: String, CodingKey {
        case mode
        case modeInfo = "mode_info"
    }
}

struct ModeDetails: Codable {
    let locked: Bool
    let lockReason: String?
    let since: String?

    enum CodingKeys: String, CodingKey {
        case locked
        case lockReason = "lock_reason"
        case since
    }
}

struct ModeChange: Codable {
    let mode: String
    let previousMode: String
    let locked: Bool
    let lockReason: String?

    enum CodingKeys: String, CodingKey {
        case mode, locked
        case previousMode = "previous_mode"
        case lockReason = "lock_reason"
    }
}

// MARK: - Reports

struct WeeklyReport: Codable {
    let weekStart: String
    let weekEnd: String
    let weekNumber: Int
    let year: Int
    let barkStats: BarkStats
    let rewardStats: RewardStats
    let coaching: CoachingStats
    let highlights: [String]

    enum CodingKeys: String, CodingKey {
        case year, coaching, highlights
        case weekStart = "week_start"
        case weekEnd = "week_end"
        case weekNumber = "week_number"
        case barkStats = "bark_stats"
        case rewardStats = "reward_stats"
    }
}

struct WeeklyReportResponse: Codable {
    let success: Bool
    let report: WeeklyReport
}

struct BarkStats: Codable {
    let total: Int
    let avgLoudness: Double
    let byEmotion: [String: Int]
    let byDog: [String: Int]

    enum CodingKeys: String, CodingKey {
        case total
        case avgLoudness = "avg_loudness"
        case byEmotion = "by_emotion"
        case byDog = "by_dog"
    }
}

struct RewardStats: Codable {
    let totalTreats: Int
    let byBehavior: [String: Int]
    let byDog: [String: Int]

    enum CodingKeys: String, CodingKey {
        case totalTreats = "total_treats"
        case byBehavior = "by_behavior"
        case byDog = "by_dog"
    }
}

struct CoachingStats: Codable {
    let sessions: Int
    let successRate: Double
    let tricksPracticed: [String: Int]

    enum CodingKeys: String, CodingKey {
        case sessions
        case successRate = "success_rate"
        case tricksPracticed = "tricks_practiced"
    }
}

struct DogProgress: Codable {
    let dogId: String
    let dogName: String
    let weeksAnalyzed: Int
    let tricks: [String: TrickStats]
    let barkStats: DogBarkStats
    let treatsEarned: Int
    let coachingSessions: Int
    let improvementAreas: [String]
    let strengths: [String]

    enum CodingKeys: String, CodingKey {
        case tricks, strengths
        case dogId = "dog_id"
        case dogName = "dog_name"
        case weeksAnalyzed = "weeks_analyzed"
        case barkStats = "bark_stats"
        case treatsEarned = "treats_earned"
        case coachingSessions = "coaching_sessions"
        case improvementAreas = "improvement_areas"
    }
}

struct DogProgressResponse: Codable {
    let success: Bool
    let progress: DogProgress
}

struct TrickStats: Codable {
    let attempts: Int
    let successes: Int
    let rate: Double
    let trend: String
}

struct DogBarkStats: Codable {
    let total: Int
    let weeklyAverage: Double
    let trend: String

    enum CodingKeys: String, CodingKey {
        case total, trend
        case weeklyAverage = "weekly_average"
    }
}

struct Trends: Codable {
    let weeksAnalyzed: Int
    let barkTrend: [WeekData]
    let summary: TrendSummary

    enum CodingKeys: String, CodingKey {
        case weeksAnalyzed = "weeks_analyzed"
        case barkTrend = "bark_trend"
        case summary
    }
}

struct TrendsResponse: Codable {
    let success: Bool
    let trends: Trends
}

struct WeekData: Codable {
    let week: Int
    let count: Int
}

struct TrendSummary: Codable {
    let barkChangePercent: Double
    let bestPerformingDog: String

    enum CodingKeys: String, CodingKey {
        case barkChangePercent = "bark_change_percent"
        case bestPerformingDog = "best_performing_dog"
    }
}

// MARK: - Events

struct BarkEvent: Codable {
    let emotion: String
    let confidence: Double
    let loudnessDb: Double
    let dogId: String?
    let dogName: String?

    enum CodingKeys: String, CodingKey {
        case emotion, confidence
        case loudnessDb = "loudness_db"
        case dogId = "dog_id"
        case dogName = "dog_name"
    }
}

struct TreatEvent: Codable {
    let dogId: String?
    let dogName: String?
    let behavior: String
    let dailyCount: Int
    let dailyLimit: Int

    enum CodingKeys: String, CodingKey {
        case behavior
        case dogId = "dog_id"
        case dogName = "dog_name"
        case dailyCount = "daily_count"
        case dailyLimit = "daily_limit"
    }
}

struct DogEvent: Codable {
    let dogId: String
    let dogName: String?
    let bbox: [Int]?
    let confidence: Double

    enum CodingKeys: String, CodingKey {
        case bbox, confidence
        case dogId = "dog_id"
        case dogName = "dog_name"
    }
}
```

---

## 11. Checklist

### Must Have
- [ ] Handle `mission_progress` WebSocket events
- [ ] Show progress bar during `watching` state
- [ ] Disable mode selector when `locked: true`
- [ ] Display stage indicator (1/5, 2/5, etc.)

### Should Have
- [ ] Animate success/failure states
- [ ] Show dog name when identified
- [ ] Implement Programs UI (multi-mission)
- [ ] Add Reports dashboard

### Nice to Have
- [ ] Option to disable video overlay (use native UI)
- [ ] Custom program creation
- [ ] Trend charts for analytics

### Relay Server
- [ ] Fix TURN credential refresh
- [ ] Ensure credential TTL is sufficient

---

## Quick Reference

### Core Endpoints

| Action | Method | Endpoint |
|--------|--------|----------|
| List missions | GET | `/missions/available` |
| Start mission | POST | `/missions/start` |
| Mission status | GET | `/missions/status` |
| Stop mission | POST | `/missions/stop` |
| List programs | GET | `/programs/available` |
| Start program | POST | `/programs/start` |
| Program status | GET | `/programs/status` |
| Get mode | GET | `/mode` |
| Set mode | POST | `/mode` |
| Weekly report | GET | `/reports/weekly` |
| Dog progress | GET | `/reports/dog/{id}` |
| Trends | GET | `/reports/trends` |

### WebSocket Events

| Event | When |
|-------|------|
| `mission_progress` | Mission state changes |
| `mode_changed` | Mode changes |
| `bark_detected` | Dog barks |
| `treat_dispensed` | Treat given |
| `dog_detected` | Dog appears |

---

*Document version: Build 31 - January 30, 2026*
