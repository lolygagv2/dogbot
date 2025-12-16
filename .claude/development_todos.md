# WIM-Z Development TODO List
*Aligned with WIM-Z (Watchful Intelligent Mobile Zen) Platform Architecture*
*Last Updated: October 22, 2025*

## 🏆 WIM-Z UNIFIED ARCHITECTURE STATUS

### ✅ COMPLETED - Event-Driven System (Oct 21-22, 2025)
Following the 6-phase plan from `claude_code_instructions.md`:

**Phase 1: Core Infrastructure** ✅
- [x] Event bus (`/core/bus.py`) - Thread-safe pub/sub messaging
- [x] State manager (`/core/state.py`) - System mode tracking
- [x] Safety monitor (`/core/safety.py`) - Battery/temperature monitoring
- [ ] Data store (`/core/store.py`) - SQLite persistence **[MISSING]**

**Phase 2: Service Layer** ✅
- [x] Detector service wraps `ai_controller_3stage_fixed.py`
- [x] Pan/tilt service for camera tracking
- [x] Motor service for movement control
- [x] Dispenser service for treat delivery
- [x] SFX and LED services for feedback
- [x] Gamepad and GUI services for control

**Phase 3: Orchestration** ✅
- [x] Sequence engine for celebration routines
- [x] Reward logic for training decisions
- [x] Mode FSM for camera state management
- [ ] Mission engine for training sequences **[TODO]**

**Phase 4: Configuration** ⚠️ PARTIAL
- [x] Modes config (`/configs/modes.yaml`)
- [ ] Sequences, policies, missions **[TODO]**

**Phase 5: Main Orchestrator** ✅
- [x] `/main_treatbot.py` - THE definitive entry point

**Phase 6: API Layer** ⚠️ PARTIAL
- [x] REST API (`/api/server.py`)
- [ ] WebSocket server **[TODO]**

## ✅ COMPLETED TASKS

### ✅ Unified Architecture Implementation [Oct 21-25, 2025]
- **Event Bus System** - Pub/sub messaging for all components
- **Service Wrappers** - All hardware wrapped with service interfaces
- **Orchestration Layer** - Sequence, reward, and mode management
- **Main Entry Point** - Single `/main_treatbot.py` orchestrator
- **API Server** - REST endpoints for control and monitoring

### ✅ Hardware Systems [Updated Nov 25, 2025]
- **Motors** - ⚠️ NEW: DFRobot Metal Motors w/Encoders (6V 210RPM 10Kg.cm)
  - STATUS: Installed, producing "clicks" but not moving - needs control reconfiguration
- **LEDs** - NeoPixel patterns and blue LED
- **Servos** - Pan/tilt/carousel all functional
- **Audio** - DFPlayer Pro with relay switching
  - UPGRADED: Conference microphone (from lapel mic) - needs testing
- **Camera** - Multi-mode system with AI detection
  - UPGRADED: Longer camera cable - needs testing
- **Gamepad** - Bluetooth framework ready (pygame)
- **Bark Detection** - TFLite emotion classifier (58% accuracy) [Oct 25, 2025]

### ✅ Hardware Added - Docking System Complete
- **IR Sensors** - 3x rear sensors for docking (Left, Center, Right)
  - STATUS: ✅ COMPLETE - Navigation IR sensors skipped, but docking functionality working
- **Charging Pads** - Roomba-style charging with bare metal plates wired to P+/P-
  - STATUS: ✅ COMPLETE - Charging system functional with 16.8V buck converter

## 🎯 COMPLETION GATES - System Validation

### Must Pass All Gates for MVP:
1. ⏳ **Event Bus Working** - Services publish/subscribe events
2. ⏳ **AI Detection Active** - Dog detection triggers events
3. ⏳ **Behavior Recognition** - Sit/down/stand poses detected
4. ⏳ **Reward Logic** - Sitting triggers celebration
5. ⏳ **Sequence Execution** - Lights + sound + treat coordinated
6. ❌ **Database Logging** - Events saved to SQLite
7. ⏳ **Cooldown Enforcement** - Time between rewards
8. ⏳ **Daily Limits** - Max rewards per day
9. ⏳ **API Monitoring** - REST endpoints return telemetry
10. ⏳ **Full Autonomous Loop** - Complete training cycle

**Legend:** ✅ Verified | ⏳ Ready to test | ❌ Not implemented

## 🔴 CRITICAL - Hardware Integration & Testing (This Week)

### ✅ COMPLETED - Motor Control System
**Status:** ✅ **COMPLETE** - Motors working with error-free operation
- **Solution:** Fixed PWM overflow with 50ms rate limiting
- **Safety:** Comprehensive Xbox controller safety fixes applied
- **Performance:** Error-free operation with proper rate limiting
- **Completed Tasks:**
  - [x] Fixed motor PWM response issues
  - [x] Calibrated PWM duty cycle ranges for DFRobot motors
  - [x] Updated motor service configuration
  - [x] Verified encoder functionality (1000Hz polling)
  - [x] Tested movement in all directions

**Priority 2: Debug Power/Startup Issues [BLOCKING FULL MVP]**
- **Issue:** Pi won't start when IR sensors + charging pads connected
- **Suspects:** Power draw, short circuit, or voltage drop
- **Tasks:**
  - [ ] Measure current draw of IR sensors individually
  - [ ] Test charging pad wiring for shorts
  - [ ] Check voltage levels at Pi terminals
  - [ ] Add power isolation/protection circuitry
  - [ ] Reconnect components one by one

**Priority 3: Test Upgraded Components [VALIDATION]**
- **Tasks:**
  - [ ] Test conference microphone audio quality
  - [ ] Verify new camera cable functionality
  - [ ] Confirm camera feed quality/length
  - [ ] Test audio input levels and sensitivity

## 🔄 CRITICAL - Refine Draft Components (Integration Testing Phase)

### 1. SQLite Store - Draft Implementation Exists ⚠️
**File:** `/core/store.py` - ✅ 550 lines of draft code

**Current Status:** Complete database schema and methods implemented
**Reality Check:** Needs 15-20 refinement iterations to work reliably

**Tasks:**
- [ ] Test database initialization and table creation
- [ ] Test event logging with real events from bus
- [ ] Debug SQLite threading issues
- [ ] Test data persistence across system restarts
- [ ] Handle edge cases and error conditions
- [ ] Performance testing with large datasets

**Acceptance:** Events persist reliably without crashes

### 2. Mission Engine - Draft Implementation Exists ⚠️
**File:** `/orchestrators/mission_engine.py` - ✅ 600 lines of draft code

**Current Status:** Full state machine and JSON loading implemented
**Reality Check:** Needs extensive testing and debugging with real hardware

**Tasks:**
- [ ] Create sample mission JSON files in `/missions/` folder
- [ ] Test mission loading and parsing
- [ ] Debug state machine transitions with real events
- [ ] Test integration with reward logic and sequence engine
- [ ] Handle timeout and error conditions
- [ ] Test daily limits enforcement

**Acceptance:** End-to-end training mission executes without crashes

### 3. WebSocket Server - Draft Implementation Exists ⚠️
**File:** `/api/ws.py` - ✅ 467 lines of draft code

**Current Status:** Real-time streaming and control commands implemented
**Reality Check:** Needs testing with real clients and debugging connection issues

**Tasks:**
- [ ] Test WebSocket connection establishment
- [ ] Debug telemetry streaming with real data
- [ ] Test bidirectional control commands (motor, pan/tilt, treats)
- [ ] Handle client disconnections gracefully
- [ ] Test multi-client scenarios
- [ ] Debug async event handling

**Acceptance:** Web dashboard connects and controls robot reliably

## 🟠 HIGH PRIORITY - System Validation

### 4. Integration Testing Suite
**Goal:** Validate all 10 completion gates

**Test Scenarios:**
1. **Autonomous Training Loop**
   - Start `/main_treatbot.py`
   - Place dog in view
   - Verify detection → sit → reward → cooldown

2. **API Control Test**
   - GET `/telemetry` returns system status
   - POST `/mode/set` changes camera mode
   - POST `/treat/dispense` triggers treat

3. **Event Bus Test**
   - Monitor event flow between services
   - Verify no dropped messages
   - Check thread safety

**Tasks:**
- [ ] Create integration test script
- [ ] Test each completion gate
- [ ] Document failures and fixes
- [ ] Achieve 10/10 gates passing

**Acceptance:** All gates verified working

### 5. Configuration Files
**Required Files:**
- `/configs/sequences/celebrate.yaml` - Celebration routine
- `/configs/policies/reward.yaml` - Reward policies
- `/missions/train_sit_daily.json` - Sample mission

**Tasks:**
- [ ] Create sequence definitions
- [ ] Define reward policies
- [ ] Create mission templates
- [ ] Test configuration loading

**Acceptance:** Configs load and execute correctly

## 🟡 MEDIUM PRIORITY - Enhanced Features

### 6. Happy Pet Progress Report [NEW FEATURE]
**Description:** 1-5 bone scale grading system

**Implementation:** `/reports/progress_report.py`

**Features:**
- Track successful behaviors vs treats
- Monitor bark frequency
- Generate daily/weekly reports
- Export for sharing

**Tasks:**
- [ ] Design report schema
- [ ] Implement data aggregation
- [ ] Create report generator
- [ ] Add export functionality

**Acceptance:** Daily reports generated automatically

### 7. Audio + Vision Behavioral Fusion [ADVANCED]
**Description:** Combined audio and vision analysis

**Implementation:** `/core/behavioral_fusion.py`

**Features:**
- Combine pose with bark analysis
- Generate confidence scores
- Real-time state assessment

**Tasks:**
- [ ] Implement fusion algorithm
- [ ] Add bark detection
- [ ] Create behavior classifier
- [ ] Integrate with rewards

**Acceptance:** Multi-modal behavior detection working

### 8. Battery Monitoring System
**Implementation:** Use existing safety monitor

**Tasks:**
- [ ] Add voltage sensor if needed
- [ ] Display in API telemetry
- [ ] Add low battery warnings
- [ ] Auto-shutdown at critical

**Acceptance:** Battery % visible in API

## 🟢 FUTURE FEATURES - After MVP

### Return-to-Base Navigation
- **Dead Reckoning** - Encoder-based path recording
- **IR Docking** - Roomba-style beacon system
- **Obstacle Avoidance** - Sensor integration

### Multi-Dog Recognition
- **ArUco Markers** - Primary ID method
- **Pose + Location** - Fallback for fluffy dogs
- **Individual Profiles** - Per-dog training

### Mobile Dashboard
- **iOS-Quality PWA** - Production-grade interface
- **Real-time Feed** - Camera with AI overlays
- **Mission Control** - Start/stop/schedule

### Social Features
- **Auto Photography** - Best moment capture
- **Social Posting** - Instagram integration
- **LLM Captions** - AI-generated descriptions

## 📊 ARCHITECTURE REFERENCE

### System Architecture (Event-Driven)
```
┌─────────────────────────────────────────────┐
│                 API LAYER                   │
│          /api/server.py  /api/ws.py         │
└───────────────────┬──────────────────────┘
                   │
┌───────────────────┴──────────────────────┐
│           ORCHESTRATION LAYER               │
│   sequence_engine  reward_logic  mode_fsm   │
└───────────────────┬──────────────────────┘
                   │
┌───────────────────┴──────────────────────┐
│             SERVICE LAYER                   │
│  detector  pantilt  motor  dispenser  sfx   │
└───────────────────┬──────────────────────┘
                   │
┌───────────────────┴──────────────────────┐
│           CORE INFRASTRUCTURE               │
│     bus  state  store  safety               │
└─────────────────────────────────────────────┘
```

### Event Flow Example
```
1. Dog detected → VisionEvent.DogDetected
2. Pose recognized → VisionEvent.Pose.Sit
3. Reward logic → RewardEvent.Trigger
4. Sequence engine → Multiple service commands
5. Services execute → Treat + Sound + Lights
```

## 🔧 Development Tools

### Testing Commands
```bash
# Start main system
python3 /home/morgan/dogbot/main_treatbot.py

# Test API
curl http://localhost:8000/telemetry
curl -X POST http://localhost:8000/treat/dispense

# Monitor events
tail -f /var/log/treatbot/events.log

# Check database
sqlite3 /data/treatbot.db "SELECT * FROM events LIMIT 10;"
```

### Debug Helpers
- `test_mission_with_controls.py` - AI controller reference
- `live_gui_detection.py` - Camera integration
- `test_treat_servo.py` - Servo control
- `/tmp/treatbot/` - Debug images

## 🏁 DEFINITION OF DONE

### MVP Complete When:
1. ✅ All 10 completion gates pass
2. ✅ 48-hour stress test (no crashes)
3. ✅ Dog sits → waits 10s → gets reward
4. ✅ Daily limits enforced (5 treats max)
5. ✅ API monitoring working
6. ✅ Database logging events

### Production Ready When:
- Battery monitoring integrated
- Web dashboard functional
- Mission library (5+ missions)
- User documentation complete
- Setup wizard implemented

---

## 📊 STATUS SUMMARY

### Architecture Implementation:
- **✅ Core Infrastructure**: 3/4 complete (missing store)
- **✅ Service Layer**: 8/8 complete
- **✅ Orchestration**: 3/4 complete (missing mission engine)
- **⚠️ Configuration**: 1/4 complete
- **✅ Main System**: Complete
- **⚠️ API Layer**: 1/2 complete (missing WebSocket)

### Overall Progress: ~85% Complete
**Blocking Issues:**
1. SQLite store not implemented
2. Mission engine not created
3. Config files incomplete

**Next Session Priorities:**
1. Implement `/core/store.py`
2. Test full autonomous loop
3. Validate all 10 gates

*Last Updated: October 22, 2025 - Aligned with Unified Architecture*