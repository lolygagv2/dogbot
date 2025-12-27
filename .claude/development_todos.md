# WIM-Z Development TODO List
*Last Updated: December 25, 2025*

## 🏆 CURRENT STATUS: Silent Guardian Live Testing

### ✅ COMPLETED - Mode System Redesign (Dec 25, 2025)
- [x] Renamed DETECTION → COACH, VIGILANT → SILENT_GUARDIAN
- [x] Updated SystemMode enum with backward compatibility
- [x] Created `modes/silent_guardian.py` - Full FSM implementation
- [x] Created `orchestrators/coaching_engine.py` - Trick training
- [x] Created `configs/rules/silent_guardian_rules.yaml`
- [x] Updated boot sequence to SILENT_GUARDIAN as default
- [x] Database tables for SG sessions and interventions
- [x] Fixed event bus subscription (string vs class)
- [x] Fixed `log_sg_intervention()` parameter bug
- [x] Implemented escalating quiet protocol (10 commands over 90s)
- [x] Implemented give-up timeout with 2-min cooldown

### ✅ COMPLETED - Core Infrastructure
- [x] Event bus (`/core/bus.py`) - Thread-safe pub/sub
- [x] State manager (`/core/state.py`) - System mode tracking
- [x] Safety monitor (`/core/safety.py`) - Battery/temp monitoring
- [x] Data store (`/core/store.py`) - SQLite persistence with SG tables
- [x] Bark frequency tracker (`/core/bark_frequency_tracker.py`)

### ✅ COMPLETED - Service Layer
- [x] Bark detector (`/services/perception/bark_detector.py`) - TFLite classifier
- [x] USB Audio (`/services/media/usb_audio.py`) - Pygame-based playback
- [x] Pan/tilt service (`/services/motion/pan_tilt.py`)
- [x] Motor service with PID control
- [x] Dispenser service (`/services/reward/dispenser.py`)
- [x] LED service (`/services/media/led.py`) - 165 NeoPixels
- [x] Xbox controller (`xbox_hybrid_controller.py`) - Full button mapping

### ✅ COMPLETED - Orchestration Layer
- [x] Mode FSM (`/orchestrators/mode_fsm.py`) - Updated transitions
- [x] Sequence engine (`/orchestrators/sequence_engine.py`)
- [x] Reward logic (`/orchestrators/reward_logic.py`)
- [x] Rules engine (`/missions/rules_engine.py`) - YAML-driven
- [x] Coaching engine (`/orchestrators/coaching_engine.py`)

### ✅ COMPLETED - Hardware
- [x] Xbox controller with all buttons mapped
- [x] Audio recording feature (Start button)
- [x] NeoPixel LED system (165 LEDs, fire/chase/gradient patterns)
- [x] USB audio (conference mic + speakers)
- [x] DFRobot motors with encoders
- [x] Pan/tilt camera servos
- [x] Treat dispenser carousel

---

## 🔄 IN PROGRESS - Live Testing

### Silent Guardian Mode Testing
- [x] Bark detection working (classifying audio every 3s)
- [x] Threshold set to 0.43 confidence for testing
- [x] Intervention triggers on bark threshold exceeded
- [ ] **TEST:** Full intervention → quiet → reward flow
- [ ] **TEST:** Escalating quiet commands work correctly
- [ ] **TEST:** 90-second timeout and 2-min cooldown
- [ ] **TEST:** Treat dispensing at end of successful sequence
- [ ] **TEST:** Database logging of interventions

### Coach Mode Testing
- [ ] **TEST:** Dog detection triggers coaching session
- [ ] **TEST:** ArUco identification works
- [ ] **TEST:** Trick rotation (sit, down, stay, spin)
- [ ] **TEST:** Success/failure audio plays correctly
- [ ] **TEST:** Treat dispensed on successful trick

---

## 🎯 PRIORITY 1: Complete Live Testing (Today)

### 1. Silent Guardian End-to-End Test
**Goal:** Verify complete bark → intervention → quiet → treat flow

**Test Steps:**
1. Have dogs bark (2+ barks above 0.43 confidence in 60s)
2. Verify "quiet" audio plays
3. Wait 10 seconds of quiet
4. Verify reward sequence: "come" → "treat" → "sit" → "quiet"
5. Verify treat dispenses
6. Check database logs

**Monitor with:**
```bash
journalctl -u treatbot -f | grep -i "bark\|intervention\|quiet\|reward\|treat"
```

### 2. Escalation Test
**Goal:** Verify escalating quiet commands when dog ignores

**Test Steps:**
1. Trigger intervention
2. Keep making noise/barking
3. Verify quiet commands increase in frequency
4. Verify firmness increases (quiet → quiet+quiet → quiet+no)
5. Test 90-second timeout
6. Verify 2-minute cooldown

### 3. Coach Mode Quick Test
**Goal:** Verify trick detection and reward

**Test Steps:**
1. Switch to COACH mode via API
2. Present dog with ArUco marker
3. Verify greeting plays
4. Issue trick command
5. Test success/failure paths

---

## 🎯 PRIORITY 2: Polish & Analytics (This Week)

### Analytics Endpoints
- [ ] `GET /analytics/daily` - Daily summary
- [ ] `GET /analytics/bark-stats` - Bark frequency trends
- [ ] `GET /analytics/treat-usage` - Treat dispensing stats
- [ ] Bone score rating system (1-5 bones)

### Session Management
- [ ] Proper 8-hour session tracking
- [ ] Session reset at midnight or manual
- [ ] Max 11 treats per session enforced

### Audio File Verification
- [ ] Verify all referenced MP3s exist:
  - [x] quiet.mp3
  - [x] no.mp3
  - [x] sit.mp3
  - [x] dogs_come.mp3
  - [x] treat.mp3
  - [x] good_dog.mp3
  - [x] elsa.mp3
  - [x] bezik.mp3
  - [ ] calming music file exists

---

## 🎯 PRIORITY 3: Future Enhancements

### Mission Scheduler
- [ ] Time-based mission triggers
- [ ] `configs/mission_schedule.yaml`
- [ ] Weekday vs weekend schedules

### Photography Mode
- [ ] High-res capture on demand
- [ ] Burst mode with quality scoring
- [ ] Auto-select best photos

### Return-to-Base
- [ ] IR beacon navigation
- [ ] Encoder-based dead reckoning
- [ ] Obstacle avoidance

### Web Dashboard
- [ ] Real-time camera feed
- [ ] Mission control interface
- [ ] Training history graphs

---

## 📊 COMPLETION GATES STATUS

| Gate | Status | Notes |
|------|--------|-------|
| Event Bus Working | ✅ | Services publish/subscribe |
| AI Detection Active | ✅ | Dog detection in COACH mode |
| Bark Detection Active | ✅ | TFLite classifier running |
| Behavior Recognition | ✅ | Sit/down/stand poses |
| Reward Logic | ✅ | Rules engine evaluates |
| Sequence Execution | ✅ | Lights + sound + treat |
| Database Logging | ✅ | SQLite with SG tables |
| Cooldown Enforcement | ✅ | Time between rewards |
| Session Limits | ✅ | Max 11 treats/session |
| API Monitoring | ✅ | REST endpoints working |
| Silent Guardian Loop | 🔄 | Testing now |
| Coach Mode Loop | ⏳ | Ready to test |

**Legend:** ✅ Verified | 🔄 Testing | ⏳ Ready to test

---

## 📁 Key Files Reference

### Mode System
- `core/state.py` - SystemMode enum
- `orchestrators/mode_fsm.py` - Mode transitions
- `modes/silent_guardian.py` - SG mode handler
- `orchestrators/coaching_engine.py` - Coach mode

### Bark Detection
- `services/perception/bark_detector.py` - TFLite classifier
- `core/bark_frequency_tracker.py` - Threshold tracking
- `configs/rules/silent_guardian_rules.yaml` - SG config

### Audio
- `services/media/usb_audio.py` - Playback
- `VOICEMP3/talks/` - Voice commands
- `VOICEMP3/songs/` - Music files

### Database
- `core/store.py` - SQLite operations
- `data/treatbot.db` - Database file

---

*This TODO list reflects actual implementation status as of December 25, 2025*
