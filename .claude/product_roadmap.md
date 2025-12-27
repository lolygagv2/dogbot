# WIM-Z (Watchful Intelligent Mobile Zen) Product Roadmap
*Last Updated: October 2025*

## 🎯 Mission Statement
Build the world's first autonomous AI-powered pet training robot - the WIM-Z (Watchful Intelligent Mobile Zen) - that combines mobility, edge AI inference, and behavioral learning to create a premium pet care experience.

## 🏆 Strategic Positioning

### Must-Have Features to Lead Field
- **Robust AI pose + audio fusion** with intelligent reward logic
- **Return-to-dock charging** with fully autonomous missions
- **Multi-dog ID and behavior history** with individual profiles
- **Happy Pet Progress Report** - 1-5 bone scale grading system tracking:
  - Successful behaviors performed
  - Treats dispensed vs earned
  - Bark frequency analysis
- **Dual-mode camera system:**
  - Photography Mode: High-quality photos with AI presets
  - AI Detection Mode: Real-time behavior boxing and recognition
- **Behavioral Analysis Module** - Combined audio + vision synopsis

### High-Value Differentiators
- **True autonomous training sequences** - Not just scheduled dispensing
  - Example: "Sit 5 times today" with 3/5 random reward rate
  - Teaches 100% obedience with variable reinforcement
- **AI-curated social content** - Automatic best moments capture
- **Offline LLM capability** - Local command processing without internet
- **Open API architecture** - Third-party trainer/IoT integration

### The Killer Combo (Our Unique Moat)
1. ✅ True AI-powered autonomous training (not scheduled treats)
2. ✅ Individual dog recognition without collars (pose + ArUco)
3. ✅ Real-time pose detection with adaptive learning
4. ✅ Social media auto-posting (viral marketing built-in)
5. ✅ Multi-dog household support (competitors fail here)
6. ✅ Mobility + Edge AI + Training in ONE device

## 📊 Competitive Analysis

| Competitor | What They Do | What They're Missing | Our Advantage |
|------------|--------------|---------------------|---------------|
| **Furbo/Petcube** | Camera + treat toss | No mobility, no training AI | Mobile + AI training |
| **Anki Vector** | Cute robot companion | No dog training, discontinued | Pet-focused AI |
| **Treat&Train** | Stationary trainer | No mobility, no AI, manual only | Autonomous + mobile |
| **Robot vacuums** | Mobility + sensors | No pet interaction purpose | Purpose-built for pets |

**Our Positioning:** "WIM-Z - The first autonomous AI companion that moves, learns, and trains multiple pets independently"

## 🎯 Market Gap Analysis

Most AI pet robots (Sony Aibo, Tombot, Joy for All) simulate companionship for humans. WIM-Z's hybrid vision-audio reinforcement loop creates the first true **AI companion for animals**, not just humans.

- **Precision AI Focus:** Vision + sound fusion via Hailo-8 + IMX500
- **Premium Build Quality:** Dyson-level trust with dog-centric design
- **Safety Certified:** Professional-grade power electronics

## 📊 Current Status: Unified Architecture Implementation

### ✅ COMPLETED - Hardware Foundation (Original Phase 1)
- [x] Devastator chassis with 2x DC motor with Encoders
- [x] Raspberry Pi 5 + Hailo-8 (26 TOPS HAT)
- [x] IMX500 camera with pan/tilt servos (2x)
- [x] Treat dispenser carousel with servo
- [x] YOLOv8s dog detection/pose inference working
- [x] 50W amplifier + 2x speakers
- [x] NeoPixels + Blue LED tube lighting
- [x] 4S2P 21700 battery pack with BMS
- [x] Power distribution (4x buck converters)
- [x] Basic motor/servo control tested

### ✅ COMPLETED - Unified Architecture (Oct 21-22, 2025)

#### Phase 1: Core Infrastructure ⚠️ DRAFT
- [x] Event bus (`/core/bus.py`) - Pub/sub messaging
- [x] State manager (`/core/state.py`) - System mode tracking
- [x] Safety monitor (`/core/safety.py`) - Battery/temp monitoring
- [⚠️] Data store (`/core/store.py`) - SQLite persistence [DRAFT - 550 lines, needs testing]

#### Phase 2: Service Layer ✅
- [x] Perception service (`/services/perception/detector.py`)
- [x] Pan/tilt service (`/services/motion/pan_tilt.py`)
- [x] Motor service (`/services/motion/motor.py`)
- [x] Dispenser service (`/services/reward/dispenser.py`)
- [x] SFX service (`/services/media/sfx.py`)
- [x] LED service (`/services/media/led.py`)
- [x] Gamepad service (`/services/input/gamepad.py`)
- [x] GUI service (`/services/ui/gui.py`)

#### Phase 3: Orchestration Layer ⚠️ DRAFT
- [x] Sequence engine (`/orchestrators/sequence_engine.py`)
- [x] Reward logic (`/orchestrators/reward_logic.py`)
- [x] Mode FSM (`/orchestrators/mode_fsm.py`)
- [⚠️] Mission engine (`/orchestrators/mission_engine.py`) [DRAFT - 600 lines, needs testing]

#### Phase 4: Configuration ⚠️ PARTIAL
- [x] Modes config (`/configs/modes.yaml`)
- [ ] Sequences config (`/configs/sequences/*.yaml`)
- [ ] Policies config (`/configs/policies/*.yaml`)
- [ ] Mission definitions (`/missions/*.json`)

#### Phase 5: Main Orchestrator ✅
- [x] Main entry point (`/main_treatbot.py`)
- [x] Startup/shutdown sequences
- [x] Service coordination

#### Phase 6: API Layer ⚠️ DRAFT
- [x] REST API (`/api/server.py`)
- [⚠️] WebSocket server (`/api/ws.py`) [DRAFT - 467 lines, needs testing]

### 🔄 HARDWARE UPDATES [November 25, 2025]

**✅ COMPLETED UPGRADES:**
- **Motors:** Upgraded to DFRobot Metal DC Motors w/Encoders (6V 210RPM 10Kg.cm)
  - **Higher torque:** 10 kg·cm (vs 4.5 kg·cm previous)
  - **Higher speed:** 210 RPM (vs 133 RPM previous)
  - **Built-in encoders:** Quadrature feedback for precise control
  - **✅ STATUS:** Working with error-free operation, safety fixes complete

**Audio System:**
- ✅ **Ugreen USB Audio Adapter** - Unified microphone input and speaker output
- ✅ **Conference Microphone** - 2.5" disc, omnidirectional pickup for bark detection
- ✅ **Upgraded Speakers** - 4Ω 5W speakers for improved audio output
- ✅ **VOICEMP3 folder** - Organized local audio files (/talks/ and /songs/)
- ✅ **Complete Testing** - Recording/playback verified working
- ✅ **Simplified architecture** - No DFPlayer, relays, or external amplifier needed

**Charging Pads:** Roomba-style bare metal charging plates
- **Connection:** Direct to P+/P- 
- ✅ **Status:** Hardware working without errors, but power is throttled to 16.8V and maximum of 5A power.

**Camera System:**
- ✅ **Longer camera cable** installed
- ✅ **STATUS:** Tested and functioning adequately

**🔧 OFFLINE - Postponed for future development:**
- **IR Sensors:** 3x rear sensors installed (Left, Center, Right)
  - **Issue:** Caused Pi startup failure when connected
  - **Status:** Hardware present but disconnected

**Sensor Additions (Still Planned):**
- [ ] Bumper sensors (collision detection)
- [ ] Ultrasonic sensors (obstacle avoidance, optional)
- [ ] Cliff sensors (edge detection)

---

## 🎯 System Completion Gates

### Critical Gates for MVP (Must Pass All):
1. **Event Bus Working** ⏳ - Services can publish/subscribe events
2. **AI Detection Active** ⏳ - Dog detection triggers VisionEvent.DogDetected
3. **Behavior Recognition** ⏳ - Detects sit/down/stand poses reliably
4. **Reward Logic** ⏳ - Sitting behavior triggers celebration sequence
5. **Sequence Execution** ⏳ - Coordinated lights + sound + treat
6. **Database Logging** ❌ - Events saved to SQLite store
7. **Cooldown Enforcement** ⏳ - Time between rewards enforced
8. **Daily Limits** ⏳ - Max rewards per day working
9. **API Monitoring** ⏳ - REST endpoints return telemetry
10. **Full Autonomous Loop** ⏳ - Complete training cycle works

**Status Legend:** ✅ Verified | ⏳ Implemented, needs testing | ❌ Not implemented

## 🚀 Remaining Development Tasks

### **IMMEDIATE: Complete Core System (This Week)**

#### Priority 1: Missing Components [CRITICAL]
- [ ] **SQLite Store** (`/core/store.py`) - Event persistence
- [ ] **Mission Engine** (`/orchestrators/mission_engine.py`) - Training sequences
- [ ] **WebSocket Server** (`/api/ws.py`) - Real-time updates
- [ ] **Config Files** - Sequences, policies, missions
- [ ] **Integration Tests** - Verify all 10 completion gates

#### Priority 2: System Validation [HIGH]
- [ ] **Test Autonomous Training Loop**
  - [ ] Dog detection → pose recognition → reward
  - [ ] Cooldown and daily limit enforcement
  - [ ] Mission state persistence

- [ ] **Hardware Integration Testing**
  - [ ] Motors respond to events
  - [ ] Servos track detected dogs
  - [ ] Treat dispenser accurate
  - [ ] LED patterns synchronized

- [ ] **API Validation**
  - [ ] All REST endpoints functional
  - [ ] WebSocket streaming works
  - [ ] Remote control via API
  - [ ] Telemetry reporting accurate

### **NEXT PHASE: Enhanced Features (After MVP)**
#### Advanced AI Features
- [ ] **Multi-dog Recognition** - ArUco markers or pose-based ID
- [ ] **Bark Detection** - Audio analysis for "quiet" training
- [ ] **Behavioral Patterns** - Learning curves and analytics

#### User Interface
- [ ] **Mobile Dashboard** - iOS-quality PWA
- [ ] **Remote Control** - Bluetooth gamepad support
- [ ] **Social Features** - Auto photo capture and sharing

### **FUTURE: Production Features**

#### Navigation & Autonomy - [Optional]
- [ ] **Return-to-Base** - IR beacon docking system
- [ ] **Obstacle Avoidance** - Sensor integration
- [ ] **Path Recording** - Dead reckoning with encoders

#### Navigation Enhancements [Optional]
- [ ] **Waypoint Mapping System**
  - [ ] Visual landmark recognition
  - [ ] GPS integration for outdoor mode
  - [ ] Patrol route definition
  - [ ] Auto-exploration mapping

- [ ] **Human Detection Mode** [Security Feature]
  - [ ] YOLOv8 person detection model
  - [ ] Stranger vs family member recognition
  - [ ] Alert notifications
  - [ ] Security patrol mode

### **Additional WIM-Z Features**

#### 3.1 Reward Logic System [Priority: HIGH]
**Goal:** Rules-based training system with configurable parameters

```yaml
# Example: Sit Training Mission
mission_type: "sit_training"
rules:
  - condition: "dog_pose == 'sit' AND duration >= 3.0"
    action: "dispense_treat"
    cooldown: 15  # seconds before next treat
    daily_limit: 5
  
  - condition: "consecutive_success >= 3"
    action: ["dispense_treat", "play_audio", "led_celebration"]
    
schedule:
  frequency: "5x per day"
  active_hours: [8, 20]  # 8 AM - 8 PM
  dog_detection_timeout: 10  # minutes before mission abort
```

**Implementation:**
- [ ] YAML-based rule engine
- [ ] Condition parser (pose, duration, count)
- [ ] Action executor (treat, audio, lights, movement)
- [ ] Mission logger (success/fail tracking)
- [ ] Daily schedule manager

#### 3.2 Event Logging & Pattern Recognition [Priority: MEDIUM]
- [ ] **Event Database**
  - [ ] SQLite for local storage
  - [ ] Tables: sessions, detections, rewards, errors
  - [ ] Export to CSV/JSON
  
- [ ] **Pattern Analysis**
  - [ ] Success rate by time of day
  - [ ] Dog learning curves
  - [ ] Optimal training times
  - [ ] Behavior trends over weeks

#### 3.3 Advanced Features [Priority: LOW]
- [ ] **ArUco Marker Dog ID** (Optional)
  - [ ] Individual dog profiles
  - [ ] Per-dog training progress
  - [ ] Collar-mounted marker detection
  - [ ] Fallback: pose + location for ID

### **PHASE 4: Navigation & Autonomy**  [Optional]

#### 4.1 Obstacle Avoidance [Priority: HIGH]  [Optional]
**Sensor Strategy:**
- **IR Sensors:** Primary (short-range, 0-30cm)
- **Cliff Sensors:** Edge detection (prevent falls)
- **Bumper Sensors:** Last-resort collision detection
- **Ultrasonic:** Optional (medium-range, 30-200cm)

**Implementation:**
- [ ] Sensor fusion algorithm
- [ ] Collision avoidance behavior
- [ ] Emergency stop on bumper hit
- [ ] Safe zone boundaries

#### 4.2 Return-to-Base Navigation [Priority: MEDIUM]  [Optional]
**Approach:** Dead Reckoning + IR Beacon

**Dead Reckoning:**
- [ ] Odometry from motor encoders
- [ ] Path recording (position + timestamp)
- [ ] Reverse path calculation
- [ ] Cumulative error compensation

**IR Beacon Docking (Roomba-style approach):**
- [ ] IR transmitter on charging dock (360° pulses, 38 kHz modulation)
- [ ] 3-4 IR receivers on robot perimeter
- [ ] Two rear-facing receivers angled ±15° for alignment
- [ ] Signal strength differential for centering while reversing
- [ ] Triangulation algorithm (3+ meter range)
- [ ] Final approach alignment with contact bumpers
- [ ] Virtual wall capability (keep-out zones)
- [ ] Combine with wheel encoder timing for precision

**Failure Recovery:**
- [ ] Progress tracking (0-100%)
- [ ] Stuck detection (no movement for 30s)
- [ ] SMS/app notification: "Help needed at 73%"
- [ ] Manual remote control override

### **PHASE 5: User Interface (Now)**

#### 5.1 Web Dashboard [Priority: HIGH]
**Tech Stack:** Flask/FastAPI + React/Vue
- [ ] **Real-time Monitoring**
  - [ ] Live camera feed
  - [ ] Battery status
  - [ ] Current mission status
  - [ ] System health indicators

- [ ] **Mission Control**
  - [ ] Start/stop missions
  - [ ] Schedule editor
  - [ ] Training history graphs
  - [ ] Dog behavior analytics

- [ ] **Settings**
  - [ ] Treat dispenser calibration
  - [ ] Audio volume controls
  - [ ] Detection sensitivity sliders
  - [ ] Safe zone boundaries

#### 5.2 Remote Control [Priority: MEDIUM] ✅ XBOX COMPLETE
**Connectivity:** WiFi API + Bluetooth Xbox Controller

**✅ COMPLETED - Xbox Wireless Controller (Dec 2025)**
- [x] **Manual Control Mode**
  - [x] Left stick for direction (forward/back/turn)
  - [x] Right stick for camera pan/tilt
  - [x] Emergency stop (A/B buttons)
  - [x] Manual treat dispense (LB)

- [x] **Xbox Controller Features**
  - [x] Auto-detect on Bluetooth pairing
  - [x] Spawned as subprocess by treatbot.service
  - [x] Priority over autonomous mode (AUTO → MANUAL transition)
  - [x] 2-second watchdog timeout + stale command detection

- [x] **Button Mapping**
  | Button | Function |
  |--------|----------|
  | Left Stick | Direction (forward/back/turn) |
  | Right Stick | Camera pan/tilt |
  | RT | Play "good dog" audio |
  | LT | Cycle LED modes |
  | A | Emergency stop |
  | B | Stop motors |
  | X | Toggle blue LED |
  | Y | Play treat sound |
  | LB | Dispense treat |
  | RB | Take photo |
  | Start (☰) | Record audio (2 sec) |
  | D-pad Left | Cycle songs |
  | D-pad Right | Cycle talks |
  | D-pad Down | Play queued audio |
  | D-pad Up | Stop audio |

- [x] **Audio Recording (Start Button)**
  - [x] Press Start → Beep + Fire LED + 2 sec recording
  - [x] Automatic playback of recording
  - [x] Press Start again within 10s → Save to VOICEMP3/talks/
  - [x] Dynamic audio folder rescan after save

**Implementation Files:**
- `xbox_hybrid_controller.py` - Main controller logic
- `core/hardware/proper_pid_motor_controller.py` - PID motor control
- `core/motor_command_bus.py` - Motor command routing

#### 5.3 Mobile App [Priority: HIGH - NEXT]
**Platform:** React Native or Flutter
- [ ] Responsive web app as MVP
- [ ] Native app if needed (iOS/Android)
- [ ] Push notifications
- [ ] Photo gallery sync
- [ ] Remote control via web interface

### **PHASE 6: Social & AI Integration (Mar 2026)**

#### 6.1 Photography System [Priority: MEDIUM]
**Goal:** Auto-capture and select best photos for social media

**Implementation:**
- [ ] Burst mode (10 photos in 2 seconds)
- [ ] Quality scoring algorithm:
  - Dog in center frame (YOLOv8 bbox)
  - Focus score (Laplacian variance)
  - Good lighting (histogram analysis)
  - No motion blur
- [ ] Top 3 photo selection
- [ ] Optional: LLM captioning (GPT-4 Vision)

**Packages:**
- `pillow` - Image processing
- `opencv-python` - Quality metrics
- `requests` - API calls

#### 6.2 Social Media Auto-Posting [Priority: LOW]
- [ ] Instagram API integration
- [ ] SMS reporting (Twilio)
- [ ] WeChat/KakaoTalk support (region-specific)
- [ ] Daily digest: "Your dog trained 5x today! 🐕"

#### 6.3 LLM Integration [Priority: LOW]
- [ ] Voice command parsing
- [ ] Natural language mission creation
  - Example: "Train Benny to be quiet" → JSON mission
- [ ] Training tips via ChatGPT
- [ ] Behavior analysis summaries
- [ ] Offline LLM mode with preset library
- [ ] Text summaries via SMS/Telegram

---

## 🔧 Technical Architecture

### API Server Design
```
api/
├── server.py           # FastAPI main (monolithic - all routes inline)
├── ws.py               # WebSocket server (real-time updates)
└── static/             # Static assets for web interface
```

**Current Endpoints in server.py:**
- `/telemetry` - System health and status
- `/mode/set`, `/mode/get` - Mode FSM control
- `/motor/*` - Motor control (speed, stop, emergency)
- `/servo/*` - Pan/tilt/carousel control
- `/led/*` - LED patterns and modes
- `/audio/*` - Playback, recording, file management
- `/camera/*` - Snapshot, stream control
- `/treat/dispense` - Treat dispenser trigger

### Mission Module System
**Unified API for all scripts:**
```python
from missions import MissionController

mission = MissionController("sit_training")
mission.start()
mission.wait_for_condition("sit", duration=3.0)
mission.reward(treat=True, audio="good_dog.mp3", lights="celebration")
mission.log_event("success")
mission.end()
```

---

## 📦 Deliverables Checklist

### Hardware Finalization
- [x] Conference mic connected and tested
- [x] IR sensors installed
- [ ] Final enclosure assembly
- [x] Charging dock with IR beacon

### Software MVP
- [ ] API server running on boot
- [ ] Camera tracking functional
- [ ] 3+ training missions working
- [ ] Web dashboard accessible
- [ ] Return-to-base tested
- [ ] Battery monitoring live

### Production Ready
- [ ] All sensors calibrated
- [ ] Mission library (10+ sequences)
- [ ] User manual written
- [ ] Setup wizard implemented
- [ ] OTA update system
- [ ] 48-hour stress test passed

---

## 🎯 Success Metrics

**Technical KPIs:**
- Pose detection accuracy: >90%
- False positive rate: <5%
- Battery life: >4 hours continuous
- Docking success rate: >95%
- Mission completion rate: >85%

**Business KPIs:**
- User satisfaction: 4.5+ stars
- Viral coefficient: >1.2 (social sharing)
- Customer support tickets: <10/month
- Hardware failure rate: <2%

---

## 🚨 Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| ArUco fails on fluffy dogs | Medium | Use pose + location for ID |
| WiFi range limits control | High | Add Bluetooth fallback |
| Docking accuracy issues | Medium | Multi-sensor approach (IR + encoders) |
| AI inference too slow | Critical | Already using Hailo-8 (26 TOPS) |
| Battery degrades quickly | Medium | Smart BMS + cycle monitoring |

---

## 📅 Timeline

- **Phase 2 (Software):**  1 week
- **Phase 3 (Behavioral):** 1 week
- **Phase 4 (Navigation):** 1 week
- **Phase 5 (UI):** 1 week
- **Phase 6 (Social/AI):** 1 week
- **Beta Testing:** 1 week
- **Production Launch:** December 1 2025 week

**Total Development:** ~2 months from now

---

## 🔮 Future Enhancements (Post-Launch)

- Multi-robot coordination (2+ TreatBots)
- Outdoor GPS navigation
- Veterinary behavior alerts
- Subscription training content
- Third-party integrations (Alexa, Google Home)
- Custom trick designer (drag-and-drop)

---

*This roadmap is a living document. Update as priorities shift.*