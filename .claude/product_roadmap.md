# WIM-Z (Watchful Intelligent Mobile Zen) Product Roadmap
*Last Updated: June 2026 · Build 106*

## Mission Statement
Build the world's first autonomous AI-powered pet training robot - the WIM-Z (Watchful Intelligent Mobile Zen) - that combines mobility, edge AI inference, and behavioral learning to create a premium pet care experience.

---

## 📊 Investor Status Summary (June 2026)

**Where we are:** Functional, multi-unit prototype fleet. **Five robots (treatbot1–5) are built and operational**, all running one shared codebase with per-unit calibration. The platform is past proof-of-concept and into reliability hardening + manufacturing prep.

**What works today (demonstrable):**
- **Autonomous behavior modes** — Coach (active trick training), Silent Guardian (anti-nuisance-barking), Manual, and Mission modes, driven by on-robot edge AI.
- **Edge AI** — dog detection + pose estimation on Hailo-8 (26 TOPS) at 30+ FPS; bark detection via a TFLite classifier with bandpass filtering.
- **Reward loop** — vision/audio → decision → treat dispense → reward logged to cloud history, closed end-to-end.
- **Remote + local control** — live WebRTC video, manual driving, and treat dispensing from a mobile app over an AWS relay, *and* a no-internet local-AP mode (phone ↔ robot direct).
- **Cloud activity history** — barks, guardian events, treats, and battery telemetry forwarded to the relay for the app's history feed.

**Known reliability work in flight (see Known Issues / Roadmap below):**
- Rare silent hard-freezes under investigation (power-button single-point-of-failure + power-delivery; no software crash trace).
- Mission Scheduler auto-start and Weekly Summary accuracy still need validation.
- Data layer to be refactored into ML/analytics-friendly schema (design pending).

**Two hardware generations** (full detail in `hardware_specs.md`): Gen-1 (tb1–2, L298N + encoder motors + IMX500, PID-capable) and Gen-2 (tb3–5, Cytron MDD10A + 9V brushed + IMX708 Wide; tb4 = NoIR night-vision).

---

## Current Status: Build 106 — Fleet Reliability Hardening

### Build Phase: **CORE COMPLETE — HARDENING**
All core hardware and software systems are operational across the 5-unit fleet. Focus since April 2026 has shifted from feature-build to per-unit calibration, fleet reliability (WiFi, timing, power), and cloud-history correctness.

### Recent Build History (selected)
| Period | Focus | Status |
|-------|-------|--------|
| Jun 2026 | Cloud activity history (guardian/treat/bark events), in-app BT controller pairing, WebRTC ICE grace-period | ✅ Shipped |
| May 2026 | Silent Guardian overhaul (BPM fast-escalation, offline catch-up), local-AP demo mode, per-dog/house voice, night mode, adaptive-bitrate WebRTC, fleet-wide monotonic-timing fix | ✅ Shipped |
| May 2026 | Fleet bring-up: per-unit battery/gimbal/dispenser calibration (tb2–tb5), Cytron drivetrain, IMX708 cameras, WiFi onboard-only | ✅ Shipped |
| Feb 2026 | Build 34–40: mission events, AI overlay, coach events, schedule API | ✅ Shipped |

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

## ✅ Verified Working (April 2026)

### App/Relay Integration
- [x] Relay forwarding mission_progress events (not all missions tested yet)
- [x] App displaying video overlay with AI confidence
- [x] Servo tracking checkbox working in app
- [x] MP3 upload/download flow working

### Coach Mode Live Testing
- [x] Bark filter rejecting claps/voice (mostly)
- [x] Pose thresholds accurate (sitting ≠ down)
- [x] Full coaching session working end-to-end

### Silent Guardian Live Testing
- [x] Bark → intervention → reward flow working
- [x] Escalation and cooldown working

### Hardware Status
- [x] Servo calibration accurate (needs tweaking per unit during manufacturing)
- [x] Treat dispenser reliable
- [x] Audio playback consistent

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

## 🚀 Shipped Since Build 40 (April → June 2026)

### Fleet Bring-Up & Calibration
- [x] **5-unit fleet operational** (treatbot1–5), one codebase + per-unit `robot_profiles/*.yaml`
- [x] Gen-2 drivetrain: **Cytron MDD10A** + 9V brushed motors (tb3–5)
- [x] **IMX708 Wide** cameras on Gen-2; **NoIR** night-vision variant on tb4
- [x] **Per-unit ADS1115 battery calibration** (factors 3.6–54×) — fixes false SoC readings
- [x] Per-unit gimbal center/limits + dispenser steps-per-slot tuning across the fleet
- [x] TMC2209 UART brought up on Gen-2 dispensers (tb4/tb5 wiring fixes)

### Reliability & Safety
- [x] **Fleet-wide monotonic-timing fix** — no RTC battery → clock jumps at boot; all safety/timeout checks moved to `time.monotonic()` (motor dead-man watchdogs, charge gate)
- [x] Motor **dead-man's watchdog** + stop-motors on WebRTC teardown
- [x] **WiFi onboard-only** — removed flaky USB dongles, MAC-locked NM profiles, closed dual-AP-manager conflict
- [x] Blue-LED GPIO25 self-heal (survives boot race)

### Connectivity
- [x] **Local-AP demo mode** — 5 GHz `WIMZ-*` AP, phone ↔ robot direct (WebRTC data-channel + MJPEG fallback), no internet
- [x] **Adaptive-bitrate/resolution WebRTC** + memory monitoring
- [x] **In-app Bluetooth game-controller pairing** over relay and local-AP
- [x] Relay commands: `mood_led`, `audio_volume`, `motor`, volume telemetry

### Behavior & Training
- [x] **Silent Guardian overhaul** — BPM-based fast-escalation, offline catch-up, sustained-bark segmentation, anti-farming treat cooldown
- [x] **Per-dog voice + house voice** — owner's first dog acts as house voice before shipped defaults; command audio routed per-dog
- [x] **Night mode** — kills blue tube light; pairs with tb4 NoIR camera
- [x] Coach robustness: 2-consecutive-frame commit for `spin`, bark-duration rejection, per-dog `force_trick`

### Cloud Activity History
- [x] Robot forwards **guardian / bark / treat_dispensed / battery** events to relay → app history feed
- [x] Silent Guardian session-summary column-shift + zero-fill bug fixed (correct kwargs + 3 new counters)

---

## 🩺 Known Issues / Risk Register (June 2026)

| Issue | Impact | Status |
|-------|--------|--------|
| **Rare silent hard-freeze** (no kernel/software trace; corrupt journal) | Requires physical power-pull to recover | Under investigation — power-delivery/brownout + power-button SPOF suspected |
| **Power-button single-point-of-failure** | GPIO21 relay routes all power-off authority through a software watcher; if the system hangs, the button can't power it off | Redesign needed (button must keep a hardware-direct OFF path) |
| **Mission Scheduler** auto-start / time windows | Scheduled missions may not fire | Implemented, **not yet validated** |
| **Weekly Summary** accuracy | Investor/owner-facing stats may be wrong | Tested, not 100% accurate |
| **SG post-cap behavior** | After the 11-treat session cap, SG keeps intervening but never rewards → possible behavior extinction | Design decision pending |
| **Data schema** not ML/analytics-friendly | Limits continual-learning + reporting | Refactor design doc pending (do not backfill old rows) |

---

## 🛰️ Fleet Status (All 5 Built & Operational)

| Unit | Role | Drivetrain | Camera | Notes |
|------|------|-----------|--------|-------|
| treatbot1 | Primary demo | L298N + 6V encoder (PID) | IMX500 | Best-calibrated; matched motors |
| treatbot2 | Backup | L298N + 6V encoder (PID) | IMX500 | Motor compensation; rebuilt battery divider |
| treatbot3 | Production #3 | Cytron + 9V brushed | IMX708 Wide | High-ratio battery divider |
| treatbot4 | Production #4 | Cytron + 9V brushed | IMX708 Wide **NoIR** | Night-vision; cam on CSI0 (cam1 port damaged) |
| treatbot5 | Production #5 | Cytron + 9V brushed | IMX708 Wide | UART/motor wiring resolved May 2026 |

---

## 🔄 Needs Testing/Rework

### Weekly Summary (`core/weekly_summary.py`)
- [x] Tested with real data (not 100% accurate)
- [x] Mission stats added in Build 38 (works mostly)
- [ ] Verify report accuracy before it becomes an owner/investor-facing metric

### Mission Scheduler (`core/mission_scheduler.py`)
- [ ] Auto-scheduling not yet tested
- [ ] Type logic (once/daily/weekly) not yet tested

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
| Right Stick Push | Center camera |
| Left Stick Push | Anti-jam action for treat dispenser |

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

### Direct LAN Connection
**Status:** IMPLEMENTED (April 2026)

Phone can connect directly to robot WiFi hotspot (WIMZ-*) without internet:
- Local API on `192.168.4.1:8000`
- WebSocket at `/ws`
- WebRTC at `/ws/webrtc/{session_id}`

**Production connections:**
- Robot ↔ AWS Lightsail Relay (WebSocket) — for remote access
- App ↔ AWS Lightsail Relay (WebSocket) — for remote access
- App ↔ Robot direct (local WiFi) — no internet required

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

*Updated: June 2026 — Build 106. Fleet status, April→June shipped features, known-issues risk register, investor status summary.*
