# WIM-Z Project Directory Structure
*Last Updated: 2025-10-27 - Session with ArUco, Bark Detection, and Bluetooth Control*

## ⚠️ IMPORTANT NOTES
- **Duplicate vision folders** need consolidation:
  - `/vision/` - Root level (should be archived)
  - `/core/vision/` - CORRECT location for vision modules
  - `/tests/vision/` - CORRECT location for vision tests
- **New additions this session:** ArUco, bark detection, Bluetooth control, modes folder

## 📁 Active Project Structure

```
/home/morgan/dogbot/   # WIM-Z (Watchful Intelligent Mobile Zen) Robot Platform

   📂 .claude/                    # Claude AI session management
      CLAUDE.md                   # Development rules (DO NOT DELETE)
      DEVELOPMENT_PROTOCOL.md     # Development workflow rules
      WIM-Z_Project_Directory_Structure.md  # THIS FILE
      resume_chat.md              # Session history
      product_roadmap.md          # WIM-Z project phases
      development_todos.md        # Priority tasks
      hardware_specs.md           # Hardware configuration
      commands/                   # Session commands
          session_start.md        # Session initialization
          session_end.md          # Session cleanup

   📂 core/                        # Core system components (ACTIVE)
      ai_controller_3stage_fixed.py   ✅ CURRENT - 3-stage AI pipeline
      bus.py                       Event bus system
      state.py                     System state manager
      store.py                     SQLite database (IMPLEMENTED)
      safety.py                    Safety monitor
      camera_mode_controller.py    Camera mode management
      camera_positioning_system.py  Camera positioning
      vision/                      # Vision modules (CORRECT LOCATION)
          camera_manager.py        Unified camera interface
          detection_plugins/
              aruco_detector.py    ArUco marker detector

   📂 services/                    # Service layer (ACTIVE)
      perception/
         detector.py              AI detection service wrapper
         bark_detector.py         🆕 Bark detection service
      motion/
         motor.py                 Motor control service
         pan_tilt.py              Pan/tilt servo service
      reward/
         dispenser.py             Treat dispenser service
      media/
         led.py                   LED control service
         sfx.py                   Sound effects service
      control/
          bluetooth_esc.py        🆕 Bluetooth ESC gamepad control
          gamepad.py              Gamepad input service (placeholder)
          gui.py                  GUI monitoring service

   📂 orchestrators/               # High-level coordination (ACTIVE)
      sequence_engine.py           Celebration sequences
      reward_logic.py              Reward decision engine
      mode_fsm.py                  Mode state machine
      mission_engine.py            ✅ Training missions (IMPLEMENTED)

   📂 api/                         # REST API (ACTIVE)
      server.py                    FastAPI server
      ws.py                        WebSocket server (TODO)

   📂 configs/                     # Configuration files (ACTIVE)
      config.json                  Main AI config
      robot_config.yaml            🔒 DO NOT MODIFY
      modes.yaml                   Camera modes config
      sequences/
          celebrate.yaml           Celebration sequence
          startup.yaml             Startup sequence
          shutdown.yaml            Shutdown sequence

   📂 modes/                       # 🆕 Autonomous operation modes
      treat_on_sit.py              Automatic treat-on-sit training

   📂 ai/                          # AI models and classifiers
      models/
          dogdetector_14.hef       Detection model
          dogpose_14.hef           Pose model
          dog_bark_classifier.tflite  🆕 Bark emotion model
          emotion_mapping.json     🆕 Emotion labels
          behavior_14.ts           ❌ MISSING - Temporal behavior model
          config.json              Model config
      bark_classifier.py           🆕 Bark emotion classifier

   📂 audio/                       # Audio processing
      bark_buffer.py               🆕 Circular audio buffer

   📂 hardware/                    # Hardware control (LEGACY - being phased out)
      led_controller.py            ➡️ Moving to services/media/led.py
      servo_controller.py          ➡️ Moving to services/
      audio_controller.py          ➡️ Moving to services/

   📂 tests/                       # All test files (ACTIVE)
      integration/
         test_10_gates_validation.py  System validation
      hardware/
         leds_v3.py               LED testing
      vision/
          test_camera_*.py         Camera tests
      test_behavior_fusion.py      🆕 Visual+audio fusion test
      test_bark_quiet_training.py  🆕 Bark training test
      test_bark_classifier.py      🆕 Bark emotion classifier test

   📂 vision/                      # ⚠️ DUPLICATE - Should be archived
      [Various old vision files]

   📂 data/                        # Runtime data (ACTIVE)
      treatbot.db                  SQLite database file

   📂 Archive/                     # Obsolete files (DO NOT USE)
      ai/                          Old AI implementations
      vision/                      Old vision code
      core/                        Old core files

   📂 docs/                        # Documentation (DO NOT DELETE)
      IR_DOCKING_SYSTEM.md         IR beacon docking guide
      *.md                         Other reference docs

   📄 Entry Points
       main_treatbot.py             Main WIM-Z autonomous system
       test_3stage_fixed.py         Working AI test
       live_gui_detection.py        Real-time detection GUI
       live_gui_with_simple_modes.py  GUI with modes
       live_gui_with_aruco.py      🆕 GUI with ArUco markers
       detect_aruco_id.py           ArUco detection utility

```

## 📋 File Status Legend
- ✅ **ACTIVE** - Currently in use and working
- 🆕 **NEW** - Added in current session
- ⏳ **TODO** - Needs implementation
- ➡️ **MIGRATING** - Being moved/refactored
- ❌ **MISSING** - Required but not found
- 🔒 **PROTECTED** - Do not modify without permission
- ⚠️ **ISSUE** - Needs attention/cleanup

## 🔍 Key Files by Function

### **Core AI Pipeline**
- `core/ai_controller_3stage_fixed.py` - Main AI processing
- `ai/models/dogdetector_14.hef` - Detection model
- `ai/models/dogpose_14.hef` - Pose estimation
- `ai/bark_classifier.py` - 🆕 Bark emotion detection

### **Event-Driven Architecture**
- `core/bus.py` - Event pub/sub system
- `core/state.py` - Global state management
- `orchestrators/mode_fsm.py` - Mode transitions

### **Control Systems**
- `services/control/bluetooth_esc.py` - 🆕 Bluetooth gamepad
- `modes/treat_on_sit.py` - 🆕 Autonomous training
- `api/server.py` - REST API control

### **Dog Identification**
- `detect_aruco_id.py` - ArUco marker detection
- `live_gui_with_aruco.py` - 🆕 GUI with dog ID overlay

### **Audio Processing**
- `ai/bark_classifier.py` - 🆕 Bark emotion classifier
- `audio/bark_buffer.py` - 🆕 Real-time audio buffer
- `services/perception/bark_detector.py` - 🆕 Bark service

## 🚨 Cleanup Needed

1. **Consolidate vision folders:**
   - Move useful files from `/vision/` to `/core/vision/`
   - Archive `/vision/` folder

2. **Complete hardware migration:**
   - Finish moving `/hardware/` to `/services/`

3. **Remove duplicate test files:**
   - Organize all tests under `/tests/` subdirectories

## 📝 How Claude Finds Files

When answering questions about functionality:

1. **For "is X working?"** → Check test files in `/tests/`
2. **For "how does X work?"** → Check implementation in `/core/` or `/services/`
3. **For "unified architecture"** → Check `/orchestrators/` and `main_treatbot.py`
4. **For "AI detection"** → Check `core/ai_controller_3stage_fixed.py`
5. **For "hardware control"** → Check `/services/` (new) or `/hardware/` (legacy)
6. **For "autonomous modes"** → Check `/modes/` folder
7. **For "dog identification"** → Check ArUco files and `/live_gui_with_aruco.py`

## ✨ Session Additions (2025-10-27)

### New Capabilities Added:
1. **ArUco Dog Identification** - Individual dog tracking via markers
2. **Bark Detection System** - TFLite emotion classifier integration
3. **Bluetooth ESC Control** - Full gamepad control system
4. **Treat-on-Sit Mode** - Autonomous training with per-dog tracking
5. **WIM-Z Branding** - Updated from TreatBot to WIM-Z platform

### Files Created This Session:
- `/live_gui_with_aruco.py` - ArUco-enabled GUI
- `/tests/test_bark_quiet_training.py` - Bark training test
- `/services/control/bluetooth_esc.py` - Bluetooth control
- `/modes/treat_on_sit.py` - Autonomous training mode

---

*This structure document is the authoritative reference for file locations in the WIM-Z project.*