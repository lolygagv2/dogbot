# WIM-Z Resume Chat Log

## Session: 2026-05-02 — Soft-Latch Power Button + Graceful Shutdown

**Goal:** Wire a Pololu #2809 Mini Pushbutton Power Switch to a 4-wire illuminated button so a single press triggers a clean Pi shutdown that ends with a hardware power cut.
**Status:** ✅ COMPLETE — committed `c6b42c8`, pushed to `origin/main`

---

### Hardware Wiring Reference (record for future self)
- **Pololu #2809 (SV)** between battery and load. Hardware-driven LED via Pololu VOUT through 1kΩ → button LED+; LED- to GND. No GPIO involvement for the LED.
- **Button switch:** one wire to Pololu pin A (also tapped to Pi GPIO20 = pin 38); other to GND. Push-on-only configuration.
- **GPIO20 (pin 38):** input, internal pull-up, falling edge = press. Watcher service.
- **GPIO26 (pin 37):** output, pulsed HIGH 500ms at end of shutdown to drop the Pololu latch and cut battery power.
- **GPIO21 (pin 40):** UNUSED — explicitly do not configure.

### What Was Built

**Watcher (boot → press detection):**
- `scripts/wimz_power_button.py` → installed at `/usr/local/bin/wimz_power_button.py` (root, 0755)
- Uses `gpiozero.Button(20, pull_up=True, bounce_time=0.05)`
- On press: logs warning, runs `sudo shutdown -h now`, then `wait_for_release()` so a stuck button doesn't loop
- `wimz-power-button.service`: simple/Restart=on-failure/User=root, `After=multi-user.target`, `WantedBy=multi-user.target` — currently **active (running)**

**Shutdown latch killer (only fires during shutdown sequence):**
- `scripts/wimz_poweroff_pulse.sh` → `/usr/local/bin/wimz_poweroff_pulse.sh` (root, 0755)
- Bash + sysfs (`/sys/class/gpio/export`, suppress already-exported error, wait for sysfs node, set `out`, `1`, sleep 0.5, `0`). Sysfs chosen instead of gpiozero because Python is a heavy dep at this point in the shutdown sequence.
- `wimz-poweroff-pulse.service`: oneshot, `DefaultDependencies=no`, `Before=shutdown.target reboot.target halt.target`, `Requires=shutdown.target`, `RemainAfterExit=yes`, `WantedBy=halt.target reboot.target shutdown.target`. Symlinks confirmed in all three `*.target.wants/`.

### Repo Convention Followed
- Existing pattern is `*.service` units at repo root (alongside `treatbot.service`, `xbox-controller.service`, `wifi-provision.service`) and helper scripts in `scripts/`. Followed it; no new top-level `systemd/` or `deploy/` directory.

### Files Added
- `wimz-power-button.service`
- `wimz-poweroff-pulse.service`
- `scripts/wimz_power_button.py` (executable)
- `scripts/wimz_poweroff_pulse.sh` (executable)

### Verification Performed This Session
- `systemctl status wimz-power-button.service` → active, log line `Power button watcher armed on GPIO20 (pull-up, falling edge)` confirmed
- `systemctl status wimz-poweroff-pulse.service` → loaded, dead (correct — fires at shutdown)
- All four enable symlinks created (`multi-user`, `halt`, `reboot`, `shutdown`)
- `treatbot.service` unaffected, still active

### NOT Tested This Session
- Actual button press → graceful shutdown → power cut → re-press → re-power loop. User declined test mid-session (would require physical power-off). **First action next session: validate the full physical loop.**

### Operational Notes / Gotchas
- Watcher is **live right now**. Any press of the button on this robot will immediately initiate shutdown.
- To pause the watcher (e.g. for hardware probing): `sudo systemctl stop wimz-power-button.service`. Re-enable with `start`.
- After a button-triggered shutdown, on next boot you can audit with: `journalctl -b -1 -u wimz-poweroff-pulse.service`

### Next Session
1. **Physical validation** — press button, time graceful shutdown, confirm GPIO26 pulse fires and Pololu drops power; verify re-press re-latches and Pi reboots.
2. If pulse doesn't fire reliably, suspect ordering vs `shutdown.target` — check `journalctl -b -1` for unit ordering.
3. Consider whether to also surface power-button events in robot telemetry (e.g. log a "user-initiated shutdown" event so the app can distinguish from crash).

---

## Session: 2026-04-27 — IMX708 Camera Swap + Coach WS Cleanup

**Goal:** Get new IMX708 Wide camera working; clean up coach mode WS protocol; verify auto-detection across robot variants
**Status:** COMPLETE

---

### What Was Accomplished

#### 1. IMX708 Camera Brought Online
**Problem:** New Camera Module 3 Wide was physically connected but `rpicam-hello` reported "No cameras available". WebRTC video track starving.

**Root cause:** `/boot/firmware/config.txt` had hardcoded `dtoverlay=imx500` which claimed I2C address `0x1a` at boot, blocking IMX708 (same I2C address). Kernel logged `Failed to register i2c client imx708 at 0x1a (-16)` (EBUSY).

**Fix:** Commented out `dtoverlay=imx500` in boot config. With `camera_auto_detect=1` already present, libcamera now picks up whichever sensor is plugged in. Reboot required.

**Result:** IMX708 Wide detected as `imx708_wide`, AI pipeline running at 14.9 FPS, coach video saved successfully on first test session.

**Important — applies to all 5 robot variants:** Each robot's `config.txt` has the same hardcoded `dtoverlay=imx500` line. To make any robot swap-ready, that line must be commented out (one-line edit per robot, requires reboot).

#### 2. Coach Mode WS Protocol Cleanup
**Problem:** App's "Stop Coaching" button did nothing while home EXIT button worked. User expected `stop_coach` to tear down coach mode.

**Root cause:** Cloud-relay `stop_coach` handler at `main_treatbot.py:1501` was a no-op + lying success ack. Comment claimed it relied on `set_mode(idle)` following — which Build 38 removed. Same pattern for `start_coach`.

**Fix:** Deleted both `start_coach` and `stop_coach` handlers from cloud relay path (`main_treatbot.py:1491-1509`) and local WS path (`api/ws.py:611-630`). Single teardown route now: `set_mode(idle)` → `mode_fsm` → mode-change handler at `main_treatbot.py:1786` → `coaching_engine.stop()`. Mirrors Silent Guardian (no SG stop command exists either).

**App-side (Build 93):** Replaces `stop_coach` with `set_mode(idle)`, listens for `mode_changed` instead of `coach_stopped`. Also dropping `coach_set_behaviors` (no robot handler) and any in-feature SG stop.

#### 3. Repo Hygiene
- `data/robot_state.json` removed from git (runtime state — `_current_dog_id` mutates on every dog selection). Added to `.gitignore`.

#### 4. Camera-Aware Architecture Verified
Confirmed auto-detection wiring is in place from earlier `2bad8c7` commit:
- `services/perception/camera_detect.py` reads sensor model from `Picamera2.global_camera_info()`
- `core/ai_controller_3stage_fixed.py:237-239` picks behavior model file at AI controller init
- `scripts/capture_behavior_sequences.py` tags captured `.npz` filenames with camera type
- Fallback chain per camera: `behavior_<camera>.ts` → `behavior_shared.ts` → `behavior_14.ts`

---

### Commits This Session
- `fa6cea5` — refactor: Remove vestigial start_coach/stop_coach WS handlers
- `79ab7dc` — chore: Stop tracking data/robot_state.json (runtime state)

(plus `954fb42` from prior session was pushed at the start of this one)

---

### Decisions Recorded for App Team
- **Coach teardown:** unify on `set_mode(idle)` (single path).
- **`select_dog` vs `force_dog`:** genuinely separate — `select_dog` is global voice routing, `force_dog` is a coach demo override that *renames* visible dogs (wrong primitive for "coach my selected dog"). App should NOT auto-fire `force_dog` on coach entry. Default ArUco-first / longest-visible targeting is fine.
- **`coach_set_behaviors`:** no robot handler exists; chip wall is read-only mirror of `tricks_available` from `coaching_started` event.
- **Future primitive needed (only if user complains about wrong-dog targeting):** clean `pin_session_dog` WS command that filters eligibility without renaming.

---

### Camera Swap Workflow (Confirmed Working)
1. `sudo systemctl stop treatbot.service`
2. Power off Pi
3. Swap camera CSI ribbon
4. Power on
5. `journalctl -u treatbot -f` → look for `Camera sensor model: '…'` log line

Detection runs at every service start. Behavior model fallback handles missing camera-specific `.ts` files gracefully.

---

### Camera Performance Notes (For LSTM Retraining Decision)
At 640×640 1:1 output, libcamera center-crops the sensor. IMX708 native is 16:9, IMX500 native is 4:3 — so IMX708 actually loses *more* horizontal FOV to the center crop than IMX500 did (~40% vs ~25%). For AI pipeline:
- ✅ IMX708 wins on autofocus (VCM + contrast detection, hill-climbing on sharpness, center-weighted)
- ✅ IMX708 slightly cleaner in low light
- ❌ Wide-angle FOV is wasted at 640×640 — only visible in manual full-res mode
- ❌ Distance/coverage not improved for AI

Decision (logged): **stay with current 1:1 center crop for retraining capture session**. Don't add letterbox FOV change at the same time as the close-up→distant retrain — would compound variables and make smaller dogs even smaller in frame, worsening the existing problem. Revisit aspect/geometry after the first retrain validates.

---

### Next Steps
1. **User:** Capture LSTM training data (paired side-by-side or alt-tab SSH on two robots)
2. **User:** Train on Blackwell PC (`scripts/train_behavior_lstm.py --augment medium`)
3. **User:** Drop trained `behavior_imx500.ts` + `behavior_imx708.ts` (or `behavior_shared.ts`) into `ai/models/`
4. **User:** Update `ai/models/config.json` `behaviors` list to include `"speak"` (currently `["stand","sit","lie","spin"]` — training script has 5)
5. **User:** Flip `_force_geometric=False` at `core/ai_controller_3stage_fixed.py:176` to re-enable LSTM
6. **User:** Apply boot-config fix (`dtoverlay=imx500` → commented out) to the other 4 robots so any can host either camera type

---

## Session: 2026-04-25 — Dog Tracking Deduplication Fix

**Goal:** Fix duplicate bounding boxes (both "Dog" and "Elsa" on same physical dog)
**Status:** COMPLETE

---

### What Was Accomplished

#### Bug Fix: Duplicate Detection Boxes
**Root Cause:** When ArUco visibility is intermittent, both an ArUco-identified entry (e.g., "elsa" at marker_id=42) and a generic entry ("dog_0" at -1000) coexisted in `last_known_positions`, causing two boxes to render.

**Not a regression** - gap from Build 38 (commit 3119962) which added generic entries but never implemented deduplication.

**Fix Applied** (`core/dog_tracker.py`):
1. When ArUco detected → delete any generic entry for that detection index
2. Before creating generic entry → check for overlapping ArUco-identified dog via IoU (>0.3) and reuse that identity
3. In `get_tracked_dogs()` → skip generic entries overlapping ArUco boxes (safety net)

**New helper methods:** `_find_overlapping_tracked_dog()`, `_bbox_iou()`

### Commits
- `0bd343f` — fix: Deduplicate dog tracking to prevent duplicate bounding boxes

### Also This Session
- Gimbal calibration API added (commit `966b6b3`)
- Xbox controller docs updated

### Next Session
- Restart treatbot to apply fix: `sudo systemctl restart treatbot`
- Test with Elsa to verify single box renders
- Lifecycle handler for clean disconnect (interrupted investigation)

---

## Session: 2026-04-25 — Gimbal Calibration API

**Goal:** Add configurable gimbal limits and API endpoints
**Status:** COMPLETE

---

### What Was Accomplished

1. **Gimbal limits now configurable** in `config/robot_profiles/treatbot1.yaml` under `camera:` section
2. **pan_tilt.py updated** to load limits from robot config at startup
3. **New API endpoints added:**
   - `GET /camera/gimbal` - View current limits
   - `POST /camera/gimbal/calibrate` - Adjust limits at runtime with optional save
4. **Xbox controller docs updated** (`XBOX_CONTROLLER_USAGE.md`) with current button mapping (RT=Good, RB=No, etc.)

### Files Modified
- `config/robot_profiles/treatbot1.yaml` — Added gimbal config params
- `services/motion/pan_tilt.py` — Load limits from config
- `api/server.py` — New gimbal calibration endpoints
- `XBOX_CONTROLLER_USAGE.md` — Updated button mapping

### Config Parameters Added
```yaml
camera:
  pan_min: 10
  pan_max: 200
  tilt_min: 20
  tilt_max: 160
  pan_center: 110
  tilt_center: 90
  coach_pan_min: 55
  coach_pan_max: 145
  coach_tilt_min: 25
  coach_tilt_max: 85
```

---

## Session: 2026-04-25 — WiFiManager Race Condition Fix

**Goal:** Investigate and fix system freezes within 2 minutes of launch on internet
**Status:** COMPLETE

---

### What Was Accomplished

#### 1. Root Cause Identified
- System was freezing (hard lock, required power cycle) within ~2 minutes of launch
- Initial wrong diagnosis: LED/SPI bus hammering from rapid pattern changes
- **Actual cause:** Race condition in WiFiManager — multiple threads (WiFi monitor + relay client) creating separate WiFiManager instances and calling nmcli/pgrep concurrently
- Kernel wireless driver deadlocked when hit from multiple threads simultaneously

#### 2. Fix Implemented (c64a160)
- Added `get_wifi_manager()` singleton factory function with global lock
- Added `_op_lock` RLock to WiFiManager to serialize all wireless operations
- Wrapped critical methods: `is_ap_mode()`, `get_connection_status()`, `try_connect_known()`, `start_hotspot()`, `start_demo_hotspot()`, `stop_hotspot()`
- Updated main_treatbot.py and relay_client.py to use singleton
- Fixed WiFi monitor bug: moved thread start to after `self.running=True` (was exiting immediately)

#### 3. CLAUDE.md Updated
- Added "Embodied Context" notice — Claude Code runs directly ON the robot, not external

### Files Modified
- `services/network/wifi_manager.py` — Singleton + thread locks (+277/-236 lines)
- `main_treatbot.py` — Use singleton, fix WiFi monitor start order
- `services/cloud/relay_client.py` — Use singleton (2 locations)
- `.claude/CLAUDE.md` — Added embodied context notice

### Commit
`c64a160` — fix: WiFiManager race condition causing system freeze

### Next Session
- Restart treatbot service to apply fix: `sudo systemctl restart treatbot.service`
- Monitor for 5+ minutes to verify no freezes
- Check logs: `journalctl -u treatbot.service -f`

### Important Notes
- The crash happened specifically "on internet" because relay was active, increasing likelihood of race
- WiFi monitor thread was also broken (exited immediately) — now fixed
- Crashes correlated with LED pattern changes were coincidental, not causal

---

## Session: 2026-04-17 — Documentation Update & Status Review

**Goal:** Update project documentation with current status, verify all systems
**Status:** COMPLETE

---

### What Was Accomplished

#### 1. Product Roadmap Updated (`product_roadmap.md`)
- Changed "Unknown Status" → "Verified Working (April 2026)"
- All App/Relay integration items confirmed working
- All Coach Mode items confirmed working
- All Silent Guardian items confirmed working
- Hardware status confirmed (servo calibration per-unit note added)
- Direct LAN Connection changed from "Deferred" to "IMPLEMENTED"
- Weekly Summary marked as tested (not 100% accurate)
- Mission Scheduler marked as NOT TESTED
- Xbox controller: Added joystick push buttons (center camera, anti-jam)

#### 2. Development TODOs Updated (`development_todos.md`)
- Updated to Build 83 (from Build 40)
- All live testing items marked complete
- Fixed incorrect checkboxes (Mission Scheduler was marked done but not tested)
- Added treat inventory tracking note
- Updated Dropped Features section

#### 3. Hardware Specs Updated (`hardware_specs.md`)
- Treat carousel: MG996R servo → NEMA 17 stepper + TMC2209
- Raspberry Pi 5: Added 4GB variant support note
- GPIO pin mapping: Cleaned up formatting, added TMC2209 pins
- Camera: Added RPi Camera Module 3 Wide as alternative option
- PCA9685 channels: Updated (0=pan, 1=tilt, stepper is separate)
- Treat system: 44 treats (11 × 4 carousels), stepper specs

#### 4. Created LLM Update Document (`LLM_UPDATE_APRIL.md`)
Comprehensive status document covering:
- Reliability metrics (what we have, what we need)
- Performance stats (detection, bark, treats)
- Crash recovery documentation
- WiFi reconnection reliability
- MVP definition (all core features complete)
- Demo breakdown (what works, caution areas, disabled features)
- Current failure points (all resolved)
- Target user (high-income pet owner)
- Data storage confirmation (SQLite: barks, treats, sessions, emotions)

#### 5. Memory Updated
- Direct AP Connection memory updated to "FULLY IMPLEMENTED"

### Key Confirmations from User

**Working Systems:**
- Relay forwarding mission_progress events (not all missions tested)
- Video overlay with AI confidence labels
- Servo tracking checkbox in app
- MP3 upload/download flow
- Bark filter (mostly rejects claps/voice)
- Pose thresholds accurate
- Full coaching session end-to-end
- Silent Guardian bark → intervention → reward
- Escalation and cooldown
- Treat dispenser reliable
- Audio playback consistent
- Servo calibration (needs per-unit tweaking)

**Resolved Issues:**
- WiFi instability → AP fallback + captive portal
- AI misfires → None reported
- Treat jams → NEMA 17 + TMC2209 anti-jam
- Docking → Feature disabled (too complex)

**Not Yet Tested:**
- Mission Scheduler auto-scheduling
- Mission Scheduler time windows
- Weekly Summary accuracy needs work

### Files Modified
- `.claude/product_roadmap.md` - Status updates
- `.claude/development_todos.md` - Build 83 updates
- `.claude/hardware_specs.md` - Hardware corrections
- `.claude/LLM_UPDATE_APRIL.md` - NEW comprehensive status doc
- `~/.claude/projects/.../memory/project_direct_ap.md` - Updated to implemented

### No Code Changes
This was a documentation-only session.

### Next Session
- Consider formal reliability testing for metrics
- Test Mission Scheduler
- Improve Weekly Summary accuracy
- Per-unit calibration documentation for manufacturing

---

## Previous Session: 2026-04-08 — Hailo Driver Patch, Audio Fallback

**Goal:** Diagnose SSH freeze, fix Hailo driver crash, treatbot2 audio/dispenser config
**Status:** FIXED — Hailo driver patched, treatbot running stable

### What Was Accomplished
- Hailo-8 PCIe driver patched (find_vma lock fix)
- Audio ALSA fallback implemented
- TMC2209 UART setup guide created
- Treatbot2 stepper config updated

### Critical Lessons
1. NEVER init hardware at Python module import time
2. kernel.warn_limit can make bugs worse

### System Changes Applied
- `/usr/src/hailo_pci-4.21.0/linux/vdma/memory.c` — patched via DKMS
- Patch files: `patches/hailo_pci_find_vma_fix.patch` + `patches/apply_hailo_fix.sh`
