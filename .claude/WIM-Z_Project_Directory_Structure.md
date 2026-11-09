# WIM-Z Project Directory Structure
*Last Updated: February 2, 2026*

## Project Status
**Build Phase:** COMPLETE - Build 40 implemented, awaiting validation

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
      talks/                        # Voice commands
         default/                   # Default voice commands (shared)
            sit.mp3                 # "Sit" command
            lie_down.mp3            # "Lie down" command
            stay.mp3                # "Stay" command
            speak.mp3               # "Speak" command
            spin.mp3                # "Spin" command
            crosses.mp3             # "Cross paws" command
            quiet.mp3               # "Quiet" command
            no.mp3                  # "No" correction
            good.mp3                # Praise audio
            treat.mp3               # Treat announcement
            come.mp3                # "Come" command (generic)
            good_dog.mp3            # Good dog praise
         dog_{id}/                  # Per-dog custom voices (uploaded via app)
            name.mp3                # Custom name recording
            come.mp3                # Custom "come" command
            ...                     # Other custom commands
      songs/                        # Music files
         default/                   # Default songs (shared)
            Wimz_theme.mp3          # WIM-Z theme song
            who_let_dogs_out.mp3    # Fun song
            hungry_like_wolf.mp3    # Fun song
            scooby_snacks.mp3       # Fun song
            cake_by_ocean.mp3       # Fun song
            ...                     # Other default songs
         dog_{id}/                  # Per-dog uploaded songs (via app)
            custom_song.mp3         # User-uploaded songs
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
      mission_scheduler.py          # Persisted mission timers: armed at boot, 5 s tick, re-arm/catch-up, last_result from bus (2026-11-07)
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
         notification_service.py    # AWS SNS SMS/push notifications (BUILD 41)
      streaming/
         webrtc.py                  # WebRTC video streaming
         video_track.py             # Video track for WebRTC
      control/
         xbox_controller.py         # Xbox controller INPUT service (reads /dev/input/jsX)
         controller_manager.py      # In-app BT pairing brain (scan/pair/trust/reconnect) — owns the bond only
         bluetooth_esc.py           # Bluetooth ESC gamepad
         gamepad.py                 # Gamepad input (placeholder)
         gui.py                     # GUI monitoring service
      power/
         battery_monitor.py         # Battery + charging detection
         wimz_power_button.py       # Soft power button watcher (root, installed to /usr/local/bin — NOT OTA). Release <5s = shutdown; hold 5s = NETWORK CANCEL → POST /system/network-cancel (fallback flag /run/wimz/net-cancel) raises the sticky AP.
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
         treatbot1.yaml             # Treatbot1 profile (robot_id: treatbot1)
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
         test_scheduler_restart_fires_live.py  # LIVE, restarts treatbot: one-shot 2 min out fires <=30 s after its instant across the restart; mission_run matched by run_id.
         test_missions_setting_live.py         # LIVE: POST /settings missions -> coach mode run -> idle -> mission_run (15 checks).
         test_missions_push_ws_live.py         # LIVE: relay settings_updated{missions} replayed on /ws/local, schedule_sync 40 ms later: ids/reasons/removal/per-dog (19 checks).
         test_coach_events_live.py             # LIVE (2026-11-08): coach activation -> one coach_session {start,end,cues_given,successes,treats}; run_trick -> trick_attempt source=run_trick matching the tool reply, no extra coach_session.
         test_10_gates_validation.py
      hardware/
         leds_v3.py
         beam_watch.sh              # Live GPIO7 through-beam transition monitor (chute sensor alignment)
         test_dispenser_beam_health.py # Beam health (2026-10-20): miss tracker -> beam_suspect, blocked-at-rest check, cancelled ladder never flags empty, suspect/blocked beam = ONE rotation on motor truth. No GPIO; standalone.
         test_power_button_hold.py  # Power-button hold classification: release<5s shutdown, hold 5s network cancel, 15s stuck (2026-09-25). No GPIO; standalone.
         test_xbox_read_event_closed_file.py # Xbox read_event on a closed/stale fd takes the 5 s reopen path, other errors rate-limited (28k-line journal flood 2026-10-24). System python3; standalone.
      network/
         test_wifi_ap_flag_recovery.py  # AP-state regression (stale _in_ap_mode stranded relay, 2026-09-04). No hardware; pytest absent in env_new so runs standalone.
         test_wifi_network_cancel.py    # Network cancel: raise_local_ap sticky + breadcrumb, /run/wimz/net-cancel flag consumption (2026-09-25). Standalone.
         test_captive_portal_diag_log.py # Captive portal: password-reveal markup + GET /log ring buffer captures failed-connect reason (2026-10-12). Mocked WiFiManager, localhost uvicorn; standalone.
        test_wifi_first_connect.py    # First connect at a new location (2026-10-23): try_connect_known scans first and only brings up in-range/hidden saved profiles; connect_in_background answers at once, records the outcome, restarts the AP on failure; save_credentials no longer deletes a missing profile. Standalone, no nmcli.
      vision/
         test_camera_*.py
      streaming/
         test_abr_hysteresis.py     # ABR anti-flap: 16:9 tiers, REMB floor band, step-down debounce, post-down hold (2026-10-03). Standalone.
        test_ptt_mic_mute.py          # PTT playback keeps the WebRTC mic muted for the whole clip + 0.7 s tail, drains stale frames on unmute (2026-10-23). No audio hardware; standalone.
      core/
         test_mission_scheduler_timers.py   # Scheduler (2026-11-07, no hardware): next_fire math once/daily/weekly+grace+missed, one-shot fires after simulated restart, results via run_id, cancel, schedule_mission/sync/cancel tools, relay missions setting (shapes, tz, per-dog replace, reasons), coach mode run.
         test_dog_tracker_dedupe.py # One box per dog: persistence re-match keeps id_method, generic-twin cleanup, ArUco tag 0 (2026-10-03). Standalone.
         test_dog_tracker_behavior_label.py # Behavior label lands on EVERY tracker entry for the animal (index, name, bbox overlap) so the drawn box never blinks (2026-10-24). Standalone.
      perception/
         test_camera_variant_profile.py   # Daytime camera profile follows the module fitted NOW (libcamera Model 'imx708_wide' vs 'imx708_wide_noir' -> yaml standard_/noir_day_profile). Standalone, 6 PASS (2026-10-24).
      api/
         test_ws_night_mode_local.py # /ws/local night-vision contract (2026-10-24): night_mode_state on connect, set_night_mode_override + aliases, night_mode_status pull, worker-thread push. Uses the singleton server; standalone.
         test_ws_command_result_local.py # CONTRACT v2 on /ws/local (2026-11-06): command_id -> command_result, settings/settings_pull + settings_state, initial_status.remote_settings, mark_departure. Standalone.
      cloud/                        # Relay CONTRACT v2 (2026-11-06) — all standalone, no hardware
         test_relay_outbox_ordered_flush.py   # Durable outbox: seq order, mid-batch failure resume, cross-process durability, prune cap, pump POST /events batches of 200 + stop-on-failure/backoff, HMAC headers.
         test_behavior_events_derivation.py   # bark_episode (gap/severity/peak_db/gate), intervention kind/result/episode_id, settle 300 s, pacing yes/no, treat source + command_id, mission_run vocab, status fields, departure method=app.
         test_remote_settings_apply.py        # treat_schedule -> TreatPolicy, sg_sensitivity level/min_episode_s/escalate_after_s -> SG profile + tracker, music_plan/quiet_hours/notify_rules stored, single-key push w/ version, persist + load_cached, per-key rejection.
         test_treat_limit_any_source.py       # treat cap from ANY source (2026-11-08): extract_treat_limits() field sniffing, RemoteSettings.set_treat_limits() -> policy+dispenser+persist+PUT write-back to relay, stale relay push deferred while pending, cap 0 blocks all.
         test_behavior_events_derivation.py   # + mode_change segments (2026-11-07): {mode,start,end,duration_s,next_mode}, contiguous, shutdown close, no terminal row.
         test_behavior_events_derivation.py   # + trick_attempt rows (coach/mission/run_trick, result clamp) and coach_session aggregation (counts, treats, run_trick-only activation skipped, close on stop) (2026-11-08).
         test_tool_dispatcher_single_flight.py # command_result shape, unknown/busy/timeout/exception, slot held until handler returns, preemptive standby, every call logged, relay interception only with command_id.
         test_trick_runner_sequence.py        # run_trick: success (one treat), no_dog/wrong_pose/timeout, invalid cue, SG busy unless force, engine hold/release, mode restored even on dispenser exception.
      reward/
         test_treat_policy_windows.py # TreatPolicy: midnight-wrap windows, m=0 blocks auto allows owner, cooldown 120/m, daily cap blocks all, max_per_hour rolling, default_multiplier, reason classes.
      audio/
         test_bark_classifier_runtime.py # Bark emotion model runs on ai_edge_litert (~25 MB) not full tensorflow (~505 MB) (2026-10-24 lag RCA). Standalone.
      media/
         test_photo_colour_order.py # Saved photos keep camera colours: picamera2 RGB888 is already BGR, the old RGB2BGR made a tan dog blue (2026-10-24). Fake detector/store; standalone.
         test_sound_registry_override.py # Sound ids <group>/<stem>: override in talks/_custom wins, factory never written, reset/reset_all, traversal ids rejected, ffmpeg-missing clean failure (2026-11-06). Temp dirs; standalone.
      motion/
         test_pan_tilt_framing.py   # Coach framing: no top/bottom limit cycle, fit check, one step per new bbox, target selection (2026-10-03). Fake servo; standalone.
      power/
         test_battery_soc_curve.py  # 4S Li-ion SoC table endpoints/midpoints/monotonic (2026-10-03). Standalone.
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
  -d '{"filepath": "/talks/good.mp3"}'

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

## Added 2026-07-05 (reliability + data architecture session)
- `core/data/` — WIMZ spec data store package (spec v0.3): `ids.py` (UUIDv7),
  `schema.py` (verbatim spec DDL — change the spec first!), `wimz_store.py`
  (WimzStore singleton, /data/wimz.db, batched writes)
- `utils/sd_notify.py` — zero-dep systemd notifier (READY/WATCHDOG/STOPPING)
- `scripts/wimz_alive.py` + `wimz-alive.service` — early-boot LED sign of life
- `scripts/backfill_wimz.py` — one-time idempotent history backfill into wimz.db
- `avahi/wimz.service` — DNS-SD advert (_wimz._tcp:8000), installed to /etc/avahi/services/
- `tests/data/test_wimz_store.py` — incl. schema-vs-spec fidelity test
- `tests/diagnostics/test_fps_pipeline_isolation.py` — §1A TEST 1-6 benchmarks
- `treatbot.service` — now Type=notify + watchdog (unit and code deploy TOGETHER)
- `/data/` (filesystem, not repo) — wimz.db + media tree per spec §3

## OTA update system (added 2026-08-31)
- `VERSION` — release tag (date-style, e.g. 2026.08.3); read by `core/version.py` → sw_version in telemetry//health
- `core/version.py` — single source of truth for sw_version (reads VERSION, cached per process)
- `services/system/ota_manager.py` — start_update gates + update_status forwarding + terminal-event emission
- `scripts/ota/wimz_updater.py` — stdlib-only updater; INSTALLED COPY at /home/morgan/wimz/updater/ (sync manually after edits)
- `scripts/ota/bootstrap-ota-layout.sh` — one-time per-unit migration to versioned layout
- `scripts/ota/manual-update.sh` — git-update an ALREADY-OTA'd unit (pull --ff-only, reinstall power-button watcher keeping relay polarity, refresh updater, restart, verify OTA layout); `--verify-only` for checks. Added 2026-09-25.
- `scripts/ota/make-release.sh` — release cut: --bump --upload
- `scripts/systemd/wimz-updater.{path,service}` — request-file trigger + root oneshot
- RUNTIME LAYOUT: /home/morgan/dogbot is a SYMLINK → /home/morgan/wimz/current → releases/<version>/; per-unit data (data, VOICEMP3/talks+songs, env_new, logs, state, captures, photos, recordings, .env, .git→shared/dotgit, .claude session files) lives in /home/morgan/wimz/shared/ and is symlinked into each release. See .claude/OTA_ROBOT_IMPLEMENTATION.md.

## 2026-09-01 additions (data refactor complete)
- core/data/legacy_views.py — wimz-backed REST views for /events/dog* + /sg/sessions/recent (Phase 4)
- scripts/ingest_fleet_history.py — fleet history consolidation → /home/morgan/wimz/fleet_archive/wimz_fleet.db (clone-dedupe, log mining)
- scripts/fleet_evidence_pack.py — canned analysis CSV exports from the fleet archive
- .claude/WIMZ_DATA_REFACTOR_DESIGN_2026-09.md — refactor design + review + phase log (COMPLETE)
- data/archive/ — decommissioned dogbot.db + missions.db (Phase 4)
- /home/morgan/wimz/fleet_archive/ — OUTSIDE releases: wimz_fleet.db + fleet_sources tarball + evidence_pack/
- DEPRECATED (dormant, no importers): core/dog_database.py, core/bark_store.py, missions/ DB layer
