# TreatBot Development TODO List
*Priority-Sorted Action Items for Claude Code*

## ✅ COMPLETED TASKS

### ✅ Camera Mode System [COMPLETED - Oct 19, 2025]
- **Status**: Complete
- **Priority**: Was Critical (1)
- **Dependencies**: None
- **Description**: Multi-mode camera system with automatic switching
- **Implementation**: `core/camera_mode_controller.py`
- **Acceptance Criteria**: ✅ All 4 modes implemented and tested
  - ✅ Photography Mode (4K, manual controls, no AI)
  - ✅ AI Detection Mode (640x640 @ 30 FPS)
  - ✅ Vigilant Mode (HD with 3x2 tiling)
  - ✅ Auto-switching based on vehicle motion

### ✅ Pose Detection Status [COMPLETED]
- **Status**: Complete
- **Priority**: Was Critical (1)
- **Description**: Pose detection verified working reliably
- **Acceptance Criteria**: ✅ Pose detection running at required FPS with good accuracy

### ✅ Audio Relay Hardware Integration [COMPLETED]
- **Status**: Complete
- **Priority**: Was Critical (1)
- **Description**: Two analog relays (Left/Right channels) switching Pi audio and DFPlayer based on GPIO
- **Acceptance Criteria**: ✅ Hardware relay system implemented and working

## 🔴 CRITICAL - Do First (Week 1-2)

### 1. Create Unified Mission API
**Context:** "Seems like we're almost there?" - need to verify and consolidate

**Goal:** All training scripts use the same API

**File:** `missions/__init__.py`
```python
class MissionController:
    """
    Unified API for all training missions
    """
    def __init__(self, mission_name: str, config: dict = None):
        self.mission_name = mission_name
        self.config = config or self.load_config(mission_name)
        self.logger = Logger(mission_name)
        self.hardware = HardwareInterface()
        
    def start(self):
        """Initialize mission and log start time"""
        
    def wait_for_condition(self, pose: str, duration: float = 0):
        """Wait until pose detected for X seconds"""
        
    def reward(self, treat: bool = True, audio: str = None, lights: str = None):
        """Trigger reward actions"""
        
    def log_event(self, event_type: str, data: dict = None):
        """Log to database"""
        
    def end(self):
        """Cleanup and log completion"""
```

**Tasks:**
- [ ] Create `missions/` module
- [ ] Implement `MissionController` base class
- [ ] Migrate existing mission scripts to use API
- [ ] Create mission config schema (YAML)
- [ ] Document API in `docs/mission_api.md`

**Acceptance:** 3+ existing missions refactored to use unified API

---

### 📋 NEW - Happy Pet Progress Report System
- **Status**: Not Started
- **Priority**: 1 (Critical MVP)
- **Dependencies**: Event Logging, Mission API
- **Description**: 1-5 bone scale grading system for dog behavior analysis
- **Implementation**: `reports/progress_report.py`
- **Acceptance Criteria**:
  - [ ] Track successful behaviors vs treats dispensed
  - [ ] Monitor bark frequency and patterns
  - [ ] Generate daily/weekly bone scale reports
  - [ ] Export reports for user sharing

### 📋 NEW - Audio + Vision Behavioral Analysis
- **Status**: Not Started
- **Priority**: 1 (Critical differentiator)
- **Dependencies**: Audio Intelligence, Pose Detection
- **Description**: Combined audio and pattern recognition for behavior synopsis
- **Implementation**: `core/behavioral_fusion.py`
- **Acceptance Criteria**:
  - [ ] Combine pose detection with bark analysis
  - [ ] Generate behavior confidence scores
  - [ ] Real-time behavioral state assessment
  - [ ] Integration with reward logic

---

## 🟠 HIGH PRIORITY (Week 3-4)

### 2. Build Reward Logic Rules Engine
**Reference:** User's example YAML config

**File:** `missions/rules_engine.py`
```python
class RulesEngine:
    def __init__(self, rules_file: str):
        self.rules = self.load_rules(rules_file)
        self.state = {}  # Track counters, cooldowns, etc.
        
    def evaluate_condition(self, condition: dict, current_data: dict) -> bool:
        """
        condition: {"pose": "sit", "duration": 3.0}
        current_data: {"pose": "sit", "duration": 3.2, "confidence": 0.92}
        """
        
    def execute_action(self, action: dict):
        """
        action: {"type": "reward", "treat": true, "audio": "good_dog.mp3"}
        """
        
    def check_cooldown(self, rule_id: str) -> bool:
        """Prevent spam-rewarding"""
        
    def check_daily_limit(self, rule_id: str) -> bool:
        """Max 5 treats per day, etc."""
```

**Tasks:**
- [ ] Implement YAML rule parser
- [ ] Create condition evaluator (AND/OR logic)
- [ ] Add cooldown tracking (per-rule state)
- [ ] Implement daily limit counters
- [ ] Create example rule files:
  - `missions/rules/sit_training.yaml`
  - `missions/rules/quiet_training.yaml`
  - `missions/rules/stay_training.yaml`

**Acceptance:** Rule engine successfully runs 5+ day simulation without errors

---

### 3. Event Logging & Pattern Recognition
**File:** `utils/event_logger.py`

**Database Schema:**
```sql
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    mission_type TEXT,
    dog_id INTEGER,
    success BOOLEAN
);

CREATE TABLE detections (
    id INTEGER PRIMARY KEY,
    session_id INTEGER,
    timestamp TIMESTAMP,
    pose TEXT,
    confidence FLOAT,
    duration FLOAT,
    bbox TEXT  -- JSON string
);

CREATE TABLE rewards (
    id INTEGER PRIMARY KEY,
    session_id INTEGER,
    timestamp TIMESTAMP,
    treat_dispensed BOOLEAN,
    audio_played TEXT,
    lights_activated TEXT
);
```

**Tasks:**
- [ ] Create SQLite database schema
- [ ] Implement `EventLogger` class
- [ ] Add logging calls to mission scripts
- [ ] Create analysis notebook: `analysis/training_patterns.ipynb`
- [ ] Generate weekly reports (success rates, trends)

**Acceptance:** 7 days of logged data with working analysis notebook

---

## 🟡 MEDIUM PRIORITY (Week 5-6)

### 4. Battery Monitoring System
**File:** `hardware/power_management.py`

**Formula:**
```python
def calculate_battery_percentage(voltage: float) -> float:
    """
    4S LiPo: 16.8V max (4.2V per cell), 12.0V min (3.0V per cell)
    """
    V_MAX = 16.8
    V_MIN = 12.0
    V_CRITICAL = 13.2  # 3.3V per cell - shutdown threshold
    
    if voltage >= V_MAX:
        return 100.0
    elif voltage <= V_MIN:
        return 0.0
    else:
        return ((voltage - V_MIN) / (V_MAX - V_MIN)) * 100
```

**Tasks:**
- [ ] Add INA219 voltage sensor (if not present)
- [ ] Implement voltage reading in `power_management.py`
- [ ] Add battery % to API `/api/system/battery`
- [ ] Display in web dashboard
- [ ] Implement low battery warning (20%)
- [ ] Auto-shutdown at critical (10%)

**Acceptance:** Live battery monitoring visible in dashboard

---

### 8. "Quiet" Training Mission
**Requirements:**
- Sit pose detected: ✅
- No barking sound (>X dB): ✅ for 10 seconds
- Reward: 1 treat + "good dog" audio + lights

**File:** `missions/quiet_training.py`

**Tasks:**
- [ ] Add lapel mic to audio input
- [ ] Implement bark detection in `audio/bark_detection.py`
- [ ] Set decibel threshold (configurable)
- [ ] Create "quiet" mission with dual conditions
- [ ] Test with actual barking dog

**Acceptance:** Successfully rewards dog after 10 seconds of sit + silence

---

### 9. Return-to-Base Navigation - Phase 1 (Dead Reckoning)
**File:** `navigation/return_to_base.py`

**Approach:**
```python
class PathRecorder:
    def __init__(self):
        self.path = []  # List of (x, y, heading, timestamp)
        
    def record_position(self, delta_left: float, delta_right: float):
        """
        Called every 100ms with wheel encoder deltas
        Calculate position using differential drive kinematics
        """
        
class PathNavigator:
    def return_home(self, path: list):
        """
        Reverse the recorded path
        Handle cumulative error with course correction
        """
```

**Tasks:**
- [ ] Implement odometry from motor encoders
- [ ] Add `PathRecorder` to mission controller
- [ ] Create reverse path algorithm
- [ ] Test: Drive 5m → return to start (within 30cm)
- [ ] Add progress tracking (0-100%)
- [ ] Implement stuck detection (no movement for 30s)

**Acceptance:** Returns to base within 50cm of start point after 5m journey

---

### 📋 NEW - Autonomous Training Sequences
- **Status**: Not Started
- **Priority**: 2 (Enhanced feature)
- **Dependencies**: Mission API, Rules Engine
- **Description**: Variable reinforcement training (e.g., 3/5 reward rate for "sit 5 times")
- **Implementation**: `missions/autonomous_training.py`
- **Acceptance Criteria**:
  - [ ] Configurable training schedules (daily limits)
  - [ ] Variable reward ratios (teach 100% obedience with <100% rewards)
  - [ ] Progress tracking per training session
  - [ ] Integration with Happy Pet Progress Report

### 📋 NEW - Individual Dog Recognition System
- **Status**: Not Started
- **Priority**: 2 (High-value differentiator)
- **Dependencies**: ArUco system, Pose detection
- **Description**: Multi-dog household support with individual profiles
- **Implementation**: `ai/dog_identification.py`
- **Acceptance Criteria**:
  - [ ] ArUco marker detection (primary method)
  - [ ] Pose + location fallback for fluffy dogs
  - [ ] Individual dog profiles and training progress
  - [ ] Multi-dog session management

### 📋 REVISED - Production-Grade Mobile Dashboard
- **Status**: Not Started
- **Priority**: 2 (High-value user interface)
- **Dependencies**: API Server, Mission System
- **Description**: High-end Apple iOS compatible app-level dashboard
- **Implementation**: Progressive Web App (PWA) or native React Native
- **Acceptance Criteria**:
  - [ ] iOS Safari compatible interface
  - [ ] Real-time camera feed with AI overlays
  - [ ] Mission control (start/stop/schedule)
  - [ ] Battery and system health monitoring
  - [ ] Dog behavior analytics dashboard
  - [ ] Push notifications for events
  - [ ] Offline capability for core functions
  - [ ] App Store quality UI/UX design

---

## 🟢 LOW PRIORITY (Week 7+)

### 8. IR Sensor Docking System
**Reference:** Roomba approach

**Hardware:**
- IR transmitter on dock (360° beacon)
- 3-4 IR receivers on robot perimeter
- Modulated signal (38kHz carrier)

**File:** `navigation/ir_docking.py`

**Tasks:**
- [ ] Research Roomba IR protocol (if not documented)
- [ ] Add IR receivers to robot (3x 120° apart)
- [ ] Build IR beacon transmitter for dock
- [ ] Implement triangulation algorithm
- [ ] Final approach alignment (within 5cm)
- [ ] Test 20+ docking attempts

**Acceptance:** >90% docking success rate from 3m distance

---

### 13. Obstacle Avoidance Sensor Integration
**Sensor Priority:**
1. **Bumper sensors** (last resort, collision)
2. **IR sensors** (0-30cm, primary)
3. **Cliff sensors** (edge detection)
4. **Ultrasonic** (30-200cm, optional)

**File:** `navigation/obstacle_avoidance.py`

**Tasks:**
- [ ] Wire all sensors to GPIO
- [ ] Implement sensor polling (100Hz)
- [ ] Create collision avoidance behavior:
  - Detected obstacle → Stop → Back up → Turn 45° → Continue
- [ ] Add safe zone boundaries (virtual walls)
- [ ] Test in cluttered room

**Acceptance:** Navigates 10m obstacle course without human intervention

---

### 14. Bluetooth Controller Support
**Requirements:**
- Cheap gamepad (Xbox/PS4 compatible)
- Auto-detect on pairing
- Priority over web interface when connected

**File:** `control/bluetooth_controller.py`

**Tasks:**
- [ ] Install `pygame` for joystick input
- [ ] Detect controller connection
- [ ] Map buttons: WASD → motors, L/R → camera
- [ ] Add mode toggle: `BT_PRIMARY` config
- [ ] Test with 3+ different controllers

**Acceptance:** Can drive robot with gamepad, no lag

---

### 15. Social Media Photography System
**Pipeline:**
1. Detect dog in frame
2. Capture 10 burst photos
3. Score each photo (focus, lighting, centering)
4. Select top 3
5. Optional: LLM caption generation
6. Auto-post to Instagram

**File:** `social/photo_system.py`

**Tasks:**
- [ ] Implement burst capture mode
- [ ] Create quality scoring algorithm:
  ```python
  def score_photo(image, bbox):
      focus_score = calculate_focus(image)  # Laplacian variance
      center_score = calculate_centering(bbox, image.shape)
      lighting_score = calculate_lighting(image)  # Histogram
      return (focus_score * 0.4 + center_score * 0.3 + lighting_score * 0.3)
  ```
- [ ] Instagram API integration (Meta Graph API)
- [ ] Test with 50+ photos

**Acceptance:** Successfully auto-posts 1 photo per day

---

### 📋 NEW - Offline LLM Integration
- **Status**: Not Started
- **Priority**: 3 (Advanced feature)
- **Dependencies**: API Server, Mission System
- **Description**: Local LLM for command processing without internet
- **Implementation**: `ai/offline_llm.py`
- **Acceptance Criteria**:
  - [ ] Research suitable offline LLM (lightweight)
  - [ ] Natural language to JSON mission conversion
  - [ ] "Train Benny to be quiet" → structured mission
  - [ ] Preset command library for common tasks
  - [ ] Fallback to cloud LLM when available

### 📋 NEW - Open API for Third-Party Extensions
- **Status**: Not Started
- **Priority**: 3 (Business differentiator)
- **Dependencies**: API Server, Authentication
- **Description**: REST + WebSocket API for external integrations
- **Implementation**: `api/external_api.py`
- **Acceptance Criteria**:
  - [ ] Authenticated API endpoints
  - [ ] Real-time telemetry streaming
  - [ ] Mission control API
  - [ ] Dog behavior data export
  - [ ] Third-party trainer integration capability

### 📋 NEW - AI-Curated Social Content
- **Status**: Not Started
- **Priority**: 3 (Marketing feature)
- **Dependencies**: Photography System, AI Controller
- **Description**: Automatic "best moments" capture and curation
- **Implementation**: `social/content_curator.py`
- **Acceptance Criteria**:
  - [ ] Real-time moment detection (play, tricks, etc.)
  - [ ] Quality scoring (focus, lighting, centering)
  - [ ] Top 3 daily photo selection
  - [ ] Instagram/social media integration
  - [ ] User approval workflow

---

## 🔧 Infrastructure Tasks

### Code Quality
- [ ] Add type hints to all functions
- [ ] Write docstrings (Google style)
- [ ] Set up `black` for formatting
- [ ] Add `pytest` for unit tests
- [ ] Configure `pre-commit` hooks

### Documentation
- [ ] Update README with quick start
- [ ] Create `docs/API_REFERENCE.md`
- [ ] Add mission creation tutorial
- [ ] Write troubleshooting guide
- [ ] Record video demo

### Deployment
- [ ] Systemd service for auto-start
- [ ] OTA update mechanism (rsync or git pull)
- [ ] Backup/restore user configs
- [ ] Factory reset script

---

## 📋 Testing Checklist (Before Production)

- [ ] 48-hour stress test (no crashes)
- [ ] All sensors calibrated
- [ ] 10+ successful docking attempts
- [ ] 20+ training missions completed
- [ ] Battery lasts 4+ hours
- [ ] Web dashboard tested on 3+ devices
- [ ] User manual complete
- [ ] Setup wizard functional

---

## 🎯 Next Steps for Claude Code

**Immediate Actions (This Session):**
1. Verify pose detection status (`python3 tests/test_camera.py`)
2. Check if camera tracking exists (`ls hardware/servo_control.py`)
3. Search for existing mission code (`grep -r "mission" .`)
4. List current TODO status: `git status`

---

## 📊 Summary of Updates (Phase 4 - Oct 19, 2025)

### Task Status Count:
- **✅ Completed**: 3 items (Camera Mode System, Pose Detection, Audio Relay)
- **🔴 Critical**: 1 item (Mission API)
- **🟠 High Priority**: 5 items (includes 2 new strategic features)
- **🟡 Medium Priority**: 8 items (includes 2 new multi-dog features + revised dashboard)
- **🟢 Low Priority**: 15 items (includes 3 new advanced features)

### New Items Added (7 total):
1. **Happy Pet Progress Report System** (Priority 1)
2. **Audio + Vision Behavioral Analysis** (Priority 1)
3. **Autonomous Training Sequences** (Priority 2)
4. **Individual Dog Recognition System** (Priority 2)
5. **Offline LLM Integration** (Priority 3)
6. **Open API for Third-Party Extensions** (Priority 3)
7. **AI-Curated Social Content** (Priority 3)

### Items Marked Complete (3 total):
- **Camera Mode System** - Multi-mode system with auto-switching
- **Pose Detection Status** - Working reliably at required performance
- **Audio Relay Hardware** - Two-channel analog relay system implemented

### Items Removed/Obsolete (4 total):
- **Verify Pose Detection Status** - ✅ Completed and moved
- **Camera Servo Tracking** - ❌ Obsolete (superseded by camera modes)
- **ArUco Research Task** - ❌ Obsolete (research phase done)
- **Audio Relay Implementation** - ✅ Completed and moved

### Items Revised (1 total):
- **Web Dashboard MVP** → **Production-Grade Mobile Dashboard** - Upgraded scope for iOS app-level quality

*This TODO list now aligns with the comprehensive strategic roadmap and technical requirements.*