# WIM-Z (Watchful Intelligent Mobile Zen) Product Roadmap
*Last Updated: February 2, 2026*

## Mission Statement
Build the world's first autonomous AI-powered pet training robot - the WIM-Z (Watchful Intelligent Mobile Zen) - that combines mobility, edge AI inference, and behavioral learning to create a premium pet care experience.

---

## Current Status: Build 40 Complete - Validation Phase

### Build Phase: **COMPLETE**
All core hardware and software systems are operational. Build 40 implemented critical fixes for app integration (mission events, AI display, tracking). Awaiting live testing validation.

### Recent Build History
| Build | Date | Focus | Status |
|-------|------|-------|--------|
| 40 | Feb 2 | Mission fields, AI display, coach events | ✅ Code complete |
| 38 | Feb 1 | Video overlay, bounding boxes, nudge tracking | ✅ Reviewed |
| 36 | Jan 31 | Mission aliases, frame freshness, faster detection | ✅ Reviewed |
| 35 | Jan 31 | Schedule API (dog_id, type fields) | ✅ Reviewed |
| 34 | Jan 31 | Mission pipeline, dog ID, servo safety | ✅ Reviewed |

---

## ✅ Completed Systems (Reviewed)

### Hardware (100% Complete)
- [x] Devastator chassis with DFRobot DC motors + encoders (6V 210RPM 10Kg.cm)
- [x] Raspberry Pi 5 + Hailo-8 HAT (26 TOPS)
- [x] IMX500 camera with pan/tilt servos
- [x] Treat dispenser carousel with servo
- [x] 50W amplifier + 2x 4ohm 5W speakers
- [x] NeoPixels (165 LEDs) + Blue LED tube lighting
- [x] 4S2P 21700 battery pack with BMS
- [x] Power distribution (4x buck converters)
- [x] Ugreen USB Audio Adapter (mic + speaker)
- [x] Conference microphone for bark detection
- [x] Xbox Wireless Controller (Bluetooth)
- [x] Roomba-style charging pads (16.8V/5A max)

### Core Infrastructure (100% Complete)
- [x] Event bus (`core/bus.py`) - Thread-safe pub/sub messaging
- [x] State manager (`core/state.py`) - System mode tracking
- [x] Safety monitor (`core/safety.py`) - Battery/temp/CPU monitoring
- [x] Data store (`core/store.py`) - SQLite persistence
- [x] Behavior interpreter (`core/behavior_interpreter.py`) - Pose detection wrapper
- [x] Dog tracker (`core/dog_tracker.py`) - Presence-based detection + ArUco ID

### Service Layer (100% Complete)
- [x] Perception: `detector.py` (YOLOv8), `bark_detector.py` (TFLite + bandpass filter)
- [x] Motion: `motor.py` (PID control), `pan_tilt.py` (servo tracking with nudge mode)
- [x] Reward: `dispenser.py` (treat carousel)
- [x] Media: `led.py` (165 NeoPixels), `usb_audio.py` (pygame playback)
- [x] Control: `xbox_controller.py`, `bluetooth_esc.py`
- [x] Power: `battery_monitor.py` (charging detection)
- [x] Cloud: `relay_client.py` (WebSocket to relay server)
- [x] Streaming: `webrtc.py`, `video_track.py` (with overlay)

### Orchestration Layer (100% Complete)
- [x] Mode FSM (`mode_fsm.py`) - IDLE, COACH, SILENT_GUARDIAN, MANUAL, MISSION
- [x] Coaching Engine (`coaching_engine.py`) - Trick training + coach_progress events
- [x] Silent Guardian (`silent_guardian.py`) - Bark detection + quiet training
- [x] Reward Logic (`reward_logic.py`) - Rules-based reward decisions
- [x] Sequence Engine (`sequence_engine.py`) - Celebration sequences
- [x] Mission Engine (`mission_engine.py`) - Formal mission execution with proper events
- [x] Program Engine (`program_engine.py`) - Training programs

### API Layer (100% Complete)
- [x] REST API (`api/server.py`) - All endpoints including GET /missions
- [x] WebSocket (`api/ws.py`) - Commands including download_song
- [x] Schedule API - CRUD with dog_id, schedule_id, type fields

---

## ❓ Unknown Status (Need User Input)

### App/Relay Integration
- [x] ❓ Is relay forwarding mission_progress events correctly?
- [x] ❓ Is app displaying video overlay with AI confidence?
- [ ] ❓ Is servo tracking checkbox working in app?
- [x] ❓ Is MP3 upload/download flow working?

### Coach Mode Live Testing
- [x] ❓ Bark filter rejecting claps/voice?
- [x] ❓ Pose thresholds accurate (sitting ≠ down)?
- [x] ❓ Full coaching session working end-to-end?

### Silent Guardian Live Testing
- [x] ❓ Bark → intervention → reward flow working?
- [x] ❓ Escalation and cooldown working?

### Hardware Status
- [ ] ❓ Servo calibration still accurate?
- [x] ❓ Treat dispenser reliable?
- [x] ❓ Audio playback consistent?

---

## ✅ Build 40 Code Changes (Reviewed)

### Mission Events (P0-R1)
- [x] Changed `mission_name` → `mission_id` in all events
- [x] Changed `stage` → `stage_number` in all events
- [x] Added `action` field to all mission_progress events

### AI Display (P0-R2)
- [x] Added `update_dog_behavior()` call in detector.py:778
- [x] Bridges behavior data to dog_tracker for video overlay

### Servo Tracking (P0-R3)
- [x] Auto-enable tracking when entering COACH mode
- [x] Debug logging for tracking state changes

### MP3 Download (P1-R4)
- [x] Constructs full URL from relay's relative path
- [x] Saves to dog-specific folder (VOICEMP3/songs/{dog_id}/)

### Coach Events (P1-R5)
- [x] coach_progress events: greeting, command, watching
- [x] coach_reward event on success

### Missions Endpoint (P2-R6)
- [x] GET /missions returns mission catalog

---

## 🔄 Needs Testing/Rework

### Weekly Summary (`core/weekly_summary.py`)
- [ ] ❓ Has this ever been tested with real data?
- [ ] Mission stats added in Build 38

### Mission Scheduler (`core/mission_scheduler.py`)
- [x] ❓ Has auto-scheduling been tested?
- [x] Type logic (once/daily/weekly) added in Build 35

---

## Xbox Controller (Complete)
| Button | Function |
|--------|----------|
| Left Stick | Drive (forward/back/turn) |
| Right Stick | Camera pan/tilt |
| A | Emergency stop |
| B | Stop motors |
| X | Toggle blue LED |
| Y | Play treat sound |
| LB | Dispense treat |
| RB | Take photo (4K in MANUAL, 640x640 otherwise) |
| LT | Cycle LED modes |
| RT | Play "good dog" audio |
| Start | Record audio (press again to save) |
| Select | Cycle modes |
| Guide | Cycle tricks (Coach mode only) |
| D-pad | Audio control (Left/Right: cycle, Down: play, Up: stop) |

---

## Future Enhancements

### Analytics System
- [ ] Daily summary endpoints
- [ ] Bark frequency trends
- [ ] Treat usage statistics
- [ ] 1-5 bone rating system

### Session Management
- [ ] 8-hour session tracking
- [ ] Session reset at midnight
- [x] Max 11 treats enforcement

### Photography Mode
- [ ] Burst mode with quality scoring
- [ ] Auto-select best photos

### Push Notifications (BUILD 41)
- [x] AWS SNS notification service created
- [x] API endpoints: `/notifications/*`
- [ ] AWS credentials configuration
- [ ] Event integrations (mission complete, bark alerts, low battery)

---

## Dropped Features

### IR Navigation/Docking
**Status:** DROPPED - Hardware caused Pi startup failures

### Direct LAN WebSocket Server
**Status:** Deferred - Not needed since all clients connect via relay

**Note:** WebSocket IS used in production:
- Robot ↔ AWS Lightsail Relay (WebSocket connection)
- App ↔ AWS Lightsail Relay (WebSocket connection)
- WebRTC signaling via TURN server through CloudFlare

What was deferred: A direct WebSocket server on the robot for local LAN clients (bypassing relay)

---

## Technical Specifications

### AI Pipeline
- **Detection:** YOLOv8s on Hailo-8 (26 TOPS)
- **Pose Estimation:** Custom dog pose model
- **Bark Detection:** TFLite classifier with 400-4000Hz bandpass filter
- **Dog ID:** ArUco markers (DICT_4X4_1000)

### Performance Targets
- Detection: 2s + 55% presence
- Pose thresholds: 0.75 for lie/cross
- Servo nudge: 2°/sec max, 500ms stability delay

### Audio Organization (Updated Build 40)
- `VOICEMP3/talks/default/` - Default voice commands
- `VOICEMP3/talks/dog_{id}/` - Per-dog custom voices
- `VOICEMP3/songs/default/` - Default songs
- `VOICEMP3/songs/dog_{id}/` - Per-dog uploaded songs
- `VOICEMP3/wimz/` - System sounds

---

*Updated: February 2, 2026 - Build 40 review*
