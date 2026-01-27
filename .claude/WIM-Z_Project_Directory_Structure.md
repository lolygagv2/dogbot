# WIM-Z Project Directory Structure
*Last Updated: January 27, 2026*

## Project Status
**Build Phase:** COMPLETE - All systems operational, in live testing phase

---

## Active Project Structure

```
/home/morgan/dogbot/   # WIM-Z (Watchful Intelligent Mobile Zen) Robot Platform

   .claude/                         # Claude AI session management
      CLAUDE.md                     # Development rules (DO NOT DELETE)
      DEVELOPMENT_PROTOCOL.md       # Development workflow rules
      WIM-Z_Project_Directory_Structure.md  # THIS FILE
      resume_chat.md                # Session history
      product_roadmap.md            # Project phases & status
      development_todos.md          # Priority tasks
      hardware_specs.md             # Hardware configuration
      commands/                     # Session commands
          session_start.md          # Session initialization
          session_end.md            # Session cleanup

   VOICEMP3/                        # Audio files for playback
      talks/                        # Voice commands (19 files)
         sit.mp3                    # "Sit" command
         lie_down.mp3               # "Lie down" command
         stay.mp3                   # "Stay" command
         speak.mp3                  # "Speak" command
         spin.mp3                   # "Spin" command
         crosses.mp3                # "Cross paws" command
         quiet.mp3                  # "Quiet" command
         no.mp3                     # "No" correction
         good_dog.mp3               # Praise audio
         treat.mp3                  # Treat announcement
         dogs_come.mp3              # "Come" command (generic)
         elsa_come.mp3              # "Come" command (Elsa)
         bezik_come.mp3             # "Come" command (Bezik)
         elsa.mp3                   # Elsa greeting
         bezik.mp3                  # Bezik greeting
         dog_0.mp3                  # Unknown dog greeting
         kahnshik.mp3               # Korean command
         kokoma.mp3                 # Korean command
         scooby_intro.mp3           # Fun intro audio
      songs/                        # Music files (12 system + user uploads)
         Wimz_theme.mp3             # WIM-Z theme song
         who_let_dogs_out.mp3       # Fun song
         hungry_like_wolf.mp3       # Fun song
         scooby_snacks.mp3          # Fun song
         cake_by_ocean.mp3          # Fun song
         milkshake.mp3              # Fun song
         yummy.mp3                  # Fun song
         mozart_piano.mp3           # Calming music
         mozart_concerto.mp3        # Calming music
         Ocean Eyes (Astronomyy Remix).mp3
         3LAU - Tokyo feat. XIRA.mp3
         BEST EDM REMIXES OF 2023.mp3
         user/                      # User-uploaded songs (via app)
      wimz/                         # System sounds (21 files)
         WimZOnline.mp3             # Startup announcement
         CoachMode.mp3              # Coach mode announcement
         SilentGuardianMode.mp3     # Silent Guardian announcement
         ManualMode.mp3             # Manual mode announcement
         MissionMode.mp3            # Mission mode announcement
         IdleMode.mp3               # Idle mode announcement
         Wimz_standby.mp3           # Standby announcement
         Wimz_charging.mp3          # Charging detected
         Wimz_lowpower.mp3          # Low battery warning
         BatteryLow.mp3             # Battery low alert
         Wimz_BatteryLow.mp3        # Battery low (alt)
         Wimz_hot.mp3               # Temperature warning
         Wimz_errorlogs.mp3         # Error alert
         Wimz_recording.mp3         # Recording started
         Wimz_saved.mp3             # Recording saved
         Wimz_missioncomplete.mp3   # Mission complete
         busy_scan.mp3              # Scanning busy
         door_scan.mp3              # Door scan
         hi_scan.mp3                # Hi scan
         progress_scan.mp3          # Progress scan
         robot_scan.mp3             # Robot scan

   core/                            # Core system components
      bus.py                        # Event bus (pub/sub messaging)
      state.py                      # System state manager
      store.py                      # SQLite database persistence
      safety.py                     # Safety monitor (battery/temp/CPU)
      behavior_interpreter.py       # Pose detection wrapper
      dog_tracker.py                # Dog presence + ArUco ID
      bark_frequency_tracker.py     # Bark threshold tracking
      bark_store.py                 # Bark event storage
      weekly_summary.py             # Weekly report generation (NEEDS REWORK)
      mission_scheduler.py          # Mission auto-scheduling (NEEDS REWORK)
      ai_controller_3stage_fixed.py # 3-stage AI pipeline
      motor_command_bus.py          # Motor command routing
      camera_mode_controller.py     # Camera mode management
      camera_positioning_system.py  # Camera positioning
      dog_database.py               # Dog profile storage
      event_publisher.py            # Event publishing utilities
      pose_detector.py              # Pose detection core
      treat_dispenser_robot.py      # Dispenser control
      hardware/                     # Hardware abstraction
         i2c_bus.py                 # I2C communication
         motor_controller_polling.py
         proper_pid_motor_controller.py
         servo_controller.py
      vision/                       # Vision modules
         camera_manager.py          # Unified camera interface
         detection_plugins/
            aruco_detector.py       # ArUco marker detector

   voices/                          # Custom voice recordings (per dog)
      default/                     # Default dog voice overrides
      {dog_id}/                    # Per-dog custom voices (e.g., come.mp3, name.mp3)

   services/                        # Service layer
      perception/
         detector.py                # AI detection service (YOLOv8)
         bark_detector.py           # Bark detection (TFLite + bandpass)
      motion/
         motor.py                   # Motor control service
         pan_tilt.py                # Pan/tilt servo service
      reward/
         dispenser.py               # Treat dispenser service
      media/
         led.py                     # LED control (165 NeoPixels)
         sfx.py                     # Sound effects service
         usb_audio.py               # USB audio playback (pygame)
         voice_manager.py           # Custom voice management
         photo_capture.py           # Photo capture with HUD overlay
         push_to_talk.py            # PTT two-way audio
         video_recorder.py          # Video recording service
      cloud/
         relay_client.py            # Cloud relay WebSocket client
      streaming/
         webrtc.py                  # WebRTC video streaming
         video_track.py             # Video track for WebRTC
      control/
         xbox_controller.py         # Xbox controller service
         bluetooth_esc.py           # Bluetooth ESC gamepad
         gamepad.py                 # Gamepad input (placeholder)
         gui.py                     # GUI monitoring service
      power/
         battery_monitor.py         # Battery + charging detection
      input/
         gamepad.py                 # Input handling
      ui/
         gui.py                     # UI service

   orchestrators/                   # High-level coordination
      coaching_engine.py            # Trick coaching (retry logic, custom voice)
      mode_fsm.py                   # Mode state machine
      mission_engine.py             # Formal mission execution
      reward_logic.py               # Reward decision engine
      sequence_engine.py            # Celebration sequences
      program_engine.py             # Training programs

   missions/                        # Mission JSON definitions
      sit.json                     # Quick sit mission (2 stages)
      come_and_sit.json            # Come + sit mission (4 stages)
      sit_training.json            # Extended sit training
      sit_sustained.json           # Sustained sit training
      down_sustained.json          # Sustained down training
      sit_and_speak.json           # Sit + speak combo
      speak_morning.json           # Morning speak training
      speak_afternoon.json         # Afternoon speak training
      alert_training.json          # Alert bark training
      bark_prevention.json         # Bark prevention mission
      stop_barking.json            # Stop barking training
      comfort_scared.json          # Comfort scared dog
      morning_chill.json           # Morning calm period
      morning_quiet_2hr.json       # Extended morning quiet
      afternoon_down_3.json        # Afternoon down training
      afternoon_sit_5.json         # Afternoon sit training
      evening_calm_transition.json # Evening calm transition
      evening_settle.json          # Evening settle mission
      night_quiet_90pct.json       # Night quiet enforcement
      quiet_progressive.json       # Progressive quiet training
      train_sit_daily.json         # Daily sit training
      rules_engine.py              # Mission rules engine

   modes/                           # Autonomous operation modes
      silent_guardian.py            # Bark quiet training
      treat_on_sit.py               # Auto treat-on-sit

   api/                             # REST API
      server.py                     # FastAPI server (all endpoints)
      ws.py                         # WebSocket server (TODO)

   configs/                         # Configuration files
      modes.yaml                    # Mode definitions
      trick_rules.yaml              # Trick definitions + thresholds
      rules/
         silent_guardian_rules.yaml # SG configuration
      sequences/
         celebrate.yaml             # Celebration sequence
         startup.yaml               # Startup sequence
         shutdown.yaml              # Shutdown sequence
      policies/                     # Policy configurations

   config/                          # Robot-specific configuration
      robot_config.yaml             # DO NOT MODIFY without permission
      robot_profiles/
         treatbot.yaml              # Treatbot profile
         treatbot2.yaml             # Treatbot2 profile
      config_loader.py              # Config loading utilities
      settings.py                   # Settings management

   ai/                              # AI models and classifiers
      models/
         dogdetector_14.hef         # Detection model (Hailo)
         dogpose_14.hef             # Pose model (Hailo)
         dog_bark_classifier.tflite # Bark classifier
         emotion_mapping.json       # Emotion labels
         config.json                # Model config
      bark_classifier.py            # Bark emotion classifier

   audio/                           # Audio processing
      bark_buffer.py                # Circular audio buffer

   data/                            # Runtime data
      treatbot.db                   # SQLite database
      dogbot.db                     # Alternative database

   captures/                        # Photo captures
      photo_*.jpg                   # 4K photos (4056x3040)
      snapshot_*.jpg                # AI stream snapshots (640x640)

   reports/                         # Generated reports
      weekly_report_*.md            # Weekly summaries

   tests/                           # Test files
      integration/
         test_10_gates_validation.py
      hardware/
         leds_v3.py
      vision/
         test_camera_*.py
      test_behavior_fusion.py
      test_bark_quiet_training.py
      test_bark_classifier.py

   docs/                            # Documentation (DO NOT DELETE)
      IR_DOCKING_SYSTEM.md          # IR beacon guide (DROPPED)
      *.md                          # Other reference docs

   Archive/                         # Obsolete files (DO NOT USE)
      ai/                           # Old AI implementations
      vision/                       # Old vision code
      core/                         # Old core files

   Entry Points
      main_treatbot.py              # Main WIM-Z autonomous system
      xbox_hybrid_controller.py     # Xbox controller subprocess
      live_gui_detection.py         # Real-time detection GUI
      live_gui_with_simple_modes.py # GUI with modes
      live_gui_with_aruco.py        # GUI with ArUco markers
      detect_aruco_id.py            # ArUco detection utility
      test_3stage_fixed.py          # Working AI test
```

---

## File Status Legend

| Symbol | Meaning |
|--------|---------|
| (no mark) | Active and working |
| (NEEDS REWORK) | Implemented but needs fixes |
| (TODO) | Planned but not implemented |
| (DROPPED) | Feature cancelled |

---

## Key Files by Function

### Main System
| Purpose | File |
|---------|------|
| Entry point | `main_treatbot.py` |
| Xbox controller | `xbox_hybrid_controller.py` |
| API server | `api/server.py` |

### Mode Handlers
| Mode | File |
|------|------|
| Coach | `orchestrators/coaching_engine.py` |
| Silent Guardian | `modes/silent_guardian.py` |
| Mode FSM | `orchestrators/mode_fsm.py` |

### AI/Detection
| Purpose | File |
|---------|------|
| Dog detection | `services/perception/detector.py` |
| Bark detection | `services/perception/bark_detector.py` |
| Pose interpretation | `core/behavior_interpreter.py` |
| Dog tracking | `core/dog_tracker.py` |

### Configuration
| Purpose | File |
|---------|------|
| Trick definitions | `configs/trick_rules.yaml` |
| SG rules | `configs/rules/silent_guardian_rules.yaml` |
| Robot config | `config/robot_config.yaml` (PROTECTED) |

### Audio
| Type | Location |
|------|----------|
| Voice commands | `VOICEMP3/talks/` |
| Music | `VOICEMP3/songs/` |
| System sounds | `VOICEMP3/wimz/` |

---

## Audio File API

Play audio via REST API:
```bash
# Play voice command
curl -X POST http://localhost:8000/audio/play/file \
  -H "Content-Type: application/json" \
  -d '{"filepath": "/talks/good_dog.mp3"}'

# Play system sound
curl -X POST http://localhost:8000/audio/play/file \
  -H "Content-Type: application/json" \
  -d '{"filepath": "/wimz/CoachMode.mp3"}'

# Play song
curl -X POST http://localhost:8000/audio/play/file \
  -H "Content-Type: application/json" \
  -d '{"filepath": "/songs/who_let_dogs_out.mp3"}'
```

---

## Protected Files (DO NOT MODIFY)

- `notes.txt` - User's personal notes
- `config/robot_config.yaml` - Robot hardware config
- `docs/` - Reference documentation
- `.claude/CLAUDE.md` - Development rules
- `.claude/hardware_specs.md` - Hardware specifications

---

## Cleanup Notes

### Consolidated (No longer duplicated)
- Vision code now in `core/vision/` only
- Hardware abstraction in `core/hardware/`
- Audio files organized in `VOICEMP3/` subfolders

### Legacy Folders (Can be archived)
- `/vision/` - Root level (moved to core/vision)
- `/hardware/` - Old implementations (moved to services)

---

*This structure document is the authoritative reference for file locations in the WIM-Z project.*
