# WIM-Z (Watchful Intelligent Mobile Zen) Product Roadmap
*Last Updated: January 11, 2026*

## Mission Statement
Build the world's first autonomous AI-powered pet training robot - the WIM-Z (Watchful Intelligent Mobile Zen) - that combines mobility, edge AI inference, and behavioral learning to create a premium pet care experience.

---

## Current Status: Live Testing & Bug Fixes

### Build Phase: **COMPLETE**
All core hardware and software systems are operational. Currently in live testing phase with real dogs, focusing on stability and accuracy improvements.

---

## Completed Systems

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
- [x] Motion: `motor.py` (PID control), `pan_tilt.py` (servo tracking)
- [x] Reward: `dispenser.py` (treat carousel)
- [x] Media: `led.py` (165 NeoPixels), `usb_audio.py` (pygame playback)
- [x] Control: `xbox_controller.py`, `bluetooth_esc.py`
- [x] Power: `battery_monitor.py` (charging detection)

### Orchestration Layer (100% Complete)
- [x] Mode FSM (`mode_fsm.py`) - IDLE, COACH, SILENT_GUARDIAN, MANUAL, MISSION
- [x] Coaching Engine (`coaching_engine.py`) - Trick training with retry logic
- [x] Silent Guardian (`silent_guardian.py`) - Bark detection + quiet training
- [x] Reward Logic (`reward_logic.py`) - Rules-based reward decisions
- [x] Sequence Engine (`sequence_engine.py`) - Celebration sequences
- [x] Mission Engine (`mission_engine.py`) - Formal mission execution

### API & Dashboard (100% Complete)
- [x] REST API (`api/server.py`) - All endpoints functional
- [x] Web Dashboard - Working frontend for monitoring/control

### Xbox Controller (100% Complete)
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

## In Progress: Live Testing

### Coach Mode Testing
- [x] Dog detection triggers coaching session (presence-based, 3s + 66%)
- [x] ArUco identification (optional - just provides name)
- [x] Trick rotation: sit, down, crosses, spin, speak
- [x] Retry on first failure (2 attempts per session)
- [x] Threading race condition fix (timestamp validation)
- [ ] **TESTING:** Verify bark filter rejects claps/voice
- [ ] **TESTING:** Verify pose thresholds (0.75 for lie/cross)
- [ ] **TESTING:** Full coaching session end-to-end

### Silent Guardian Testing
- [x] Bark detection with 400-4000Hz bandpass filter
- [x] Escalating quiet protocol (10 commands over 90s)
- [x] Give-up timeout with 2-min cooldown
- [ ] **TESTING:** Full bark → quiet → reward flow
- [ ] **TESTING:** Non-bark sound rejection

---

## Needs Rework

### Weekly Summary & Mission Scheduler
**Status:** Implemented but needs rework

Files:
- `core/weekly_summary.py` - Report generation
- `core/mission_scheduler.py` - Auto-start missions

Issues to address:
- [ ] Verify report data accuracy
- [ ] Test mission scheduler time windows
- [ ] Integration with current Coach/SG modes

---

## Next Priority: Bug Fixes

Focus on stabilizing Coach and Silent Guardian modes:

1. **Bark Detection Accuracy**
   - Verify bandpass filter effectiveness
   - Test with various non-bark sounds (claps, voice, HVAC)
   - Tune confidence threshold if needed

2. **Pose Detection Reliability**
   - Verify 0.75 threshold for lie/cross poses
   - Test with sitting dogs (should NOT trigger lie/cross)
   - Monitor false positive rate

3. **Coaching Session Flow**
   - Verify timestamp validation prevents race conditions
   - Test retry logic (2 attempts per session)
   - Verify audio plays correctly for each trick

4. **System Stability**
   - Monitor CPU/temperature during extended operation
   - Verify motor safety auto-stop
   - Test Xbox controller reconnection

---

## Future Enhancements (Post-Stabilization)

### Analytics System
- [ ] Daily summary endpoints
- [ ] Bark frequency trends
- [ ] Treat usage statistics
- [ ] 1-5 bone rating system

### Session Management
- [ ] 8-hour session tracking
- [ ] Session reset at midnight
- [ ] Max 11 treats per session enforcement

### Photography Mode
- [ ] High-res capture on demand
- [ ] Burst mode with quality scoring
- [ ] Auto-select best photos

### Social Features
- [ ] Auto photo capture of good moments
- [ ] LLM captioning (GPT-4 Vision)
- [ ] SMS/app notifications

---

## Dropped Features

### IR Navigation/Docking
**Status:** DROPPED

Originally planned Roomba-style IR beacon navigation with:
- 3x rear IR sensors (Left, Center, Right)
- Dead reckoning with encoder feedback
- Automatic return-to-base

**Reason:** Hardware caused Pi startup failures. Charging pads work manually.

### WebSocket Real-time Server
**Status:** Deferred (REST API sufficient for current needs)

---

## Technical Specifications

### AI Pipeline
- **Detection:** YOLOv8s on Hailo-8 (26 TOPS)
- **Pose Estimation:** Custom dog pose model
- **Bark Detection:** TFLite classifier with scipy bandpass filter
- **Dog ID:** ArUco markers (DICT_4X4_1000, 4cm markers)
  - Elsa: ID 315 (Green)
  - Bezik: ID 832 (Magenta)

### Performance Targets
- Pose detection accuracy: >90%
- False positive rate: <5%
- Battery life: >4 hours continuous
- Detection confidence: 0.5 (dog), 0.75 (lie/cross poses)

### Operating Temperatures
- Normal range: 65-75C under AI load
- Warning threshold: 76C
- Critical threshold: 85C

---

## File Reference

### Entry Points
- `main_treatbot.py` - Main WIM-Z system
- `xbox_hybrid_controller.py` - Xbox controller subprocess

### Mode Handlers
- `modes/silent_guardian.py` - Bark quiet training
- `orchestrators/coaching_engine.py` - Trick coaching

### Configuration
- `config/robot_config.yaml` - Robot settings (PROTECTED)
- `configs/trick_rules.yaml` - Trick definitions + thresholds
- `configs/rules/silent_guardian_rules.yaml` - SG config

### Audio Files
- `VOICEMP3/talks/` - Voice commands (sit, down, quiet, etc.)
- `VOICEMP3/songs/` - Music files
- `VOICEMP3/wimz/` - System sounds (mode announcements, alerts)

---

*This roadmap reflects actual implementation status as of January 11, 2026*
