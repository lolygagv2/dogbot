# WIM-Z Resume Chat Log

## Session: 2026-05-15 — Git sync + new Pi 5 diagnostics (treatbot2)

**Goal:** Pull latest changes, verify new Pi 5 hardware after old one was destroyed (12V GPIO mishap)
**Status:** ✅ Complete

### What Was Accomplished

1. **Git Sync**
   - Pulled 6 commits from origin/main (treatbot3 profiles, charging detector fix, etc.)
   - Fixed stale staged changes in `train_behavior_lstm.py` (4-class alignment, CPU export)
   - Added 174 IMX500 behavior sequences to repo
   - Updated LSTM training guide to remove "speak" (audio-only via bark detector)

2. **Hardware Diagnostics — New Pi 5 (4GB)**
   - Pi 5 Model B Rev 1.1: Healthy (48°C)
   - RAM: 4GB (2.5GB available, swap usage higher than 8GB unit)
   - IMX500 camera: Detected (4056x3040)
   - I2C: PCA9685 servo @ 0x40, ADC @ 0x48
   - USB Audio: Card 0
   - Soft power button: Armed and working

3. **Power Button Note**
   - Service starts ~1.5 min after boot (waits for multi-user.target)
   - User pressed button before service armed — expected behavior, not a bug

### Commits Pushed
- `903b2c0` — fix: Align train_behavior_lstm.py with deployed 4-class model
- `7a37458` — feat: Add IMX500 behavior sequences + LSTM training guide
- `0cbaeaf` — docs: Update LSTM guide — speak removed (audio-only via bark detector)

### Next Session
- No pending tasks

---

## Session: 2026-05-09 — Rebadge cloned Pi as treatbot3 + prep for treatbot4/5

**Goal:** This Pi was cloned from treatbot1's SD card. Give it a unique identity (treatbot3 / wimz_robot_03), pre-create profiles + bootstrap script so future clones for treatbot4 and treatbot5 are one-command operations.
**Status:** ✅ Code/config committed and pushed (`bc3ecb9`). Identity reset script ran successfully on this Pi. **Reboot pending — user will trigger when ready.**

### Hardware delta on units 3-5 (vs 1/2)
- Cytron motor driver + 9V brushed motors (no encoders) — replaces L298N + 6V DFRobot encoder motors. PID disabled because no encoder feedback.
- IMX708 Pi Camera Module 3 Wide — replaces IMX500 used on units 1/2.

### What Was Built

**Repo changes (commit `bc3ecb9`, pushed to origin/main):**
- `config/config_loader.py` — added `treatbot3/4/5` to hostname_map. After hostname change, profile auto-selects, no `/etc/robot_id` needed.
- `config/robot_profiles/treatbot3.yaml`, `treatbot4.yaml`, `treatbot5.yaml` — seeded from treatbot1.yaml. PID disabled, motor multipliers neutral (1.0/1.0), camera rotation 0. HARDWARE_NOTE block flags Cytron driver dependency.
- `scripts/wimz_rebadge.sh` — idempotent post-clone identity reset. Takes one arg (3, 4, or 5). Resets machine-id, regenerates SSH host keys, sets hostname + /etc/hosts, writes new .env with fresh DEVICE_SECRET. Does NOT auto-reboot (so user can read the printed secret).

**System changes on this Pi (via rebadge script):**
- machine-id: `67ec0d9e3e5e47c98d50d87f3684c555` (new, distinct from treatbot1)
- New SSH host keys (ed25519 fingerprint: `SHA256:HtaF5WgUG07LGRQJuEH63X5MewYq9SXGkPJ+yahvl78`) — note: key comment field still says "root@treatbot1" cosmetically because hostname changed AFTER key generation; harmless, will correct on next regeneration
- hostname → `treatbot3`, `/etc/hosts` 127.0.1.1 → `treatbot3`
- `/home/morgan/dogbot/.env` → `DEVICE_ID=wimz_robot_03` + a freshly-generated `DEVICE_SECRET` (stored only in .env, not in git — see file directly)
- Bluetooth pairing database left intact (Xbox controller bond preserved from clone)

### Critical follow-up actions (next session)

1. **Register on Lightsail relay**: `wimz_robot_03` + the secret in `/home/morgan/dogbot/.env` must be added to the relay's allowed-devices list before the robot can connect remotely.
2. **Cytron motor driver** (out of scope this session): `services/motion/motor.py` has no Cytron dispatch. Need a new `core/hardware/motor_controller_cytron.py` plus dispatch wiring before drive control works on units 3/4/5.
3. **Per-unit calibration during bring-up**: motor `left_multiplier`/`right_multiplier`, `camera.rotation`, `pan_center`/`tilt_center`, dispenser `steps_per_slot`. All currently neutral defaults in treatbot3.yaml.
4. **Verify after reboot**:
   ```
   hostname                                    # treatbot3
   journalctl -u treatbot.service -n 80 | grep -iE "profile|robot_id|treatbot3"
   ls /dev/input/js0                           # Xbox controller present
   bluetoothctl devices Connected
   libcamera-hello --list-cameras              # confirm IMX708 detected
   ```

### Cloning to treatbot4 / treatbot5 (future)
After this Pi is fully validated, clone its SD card. On each new clone, first boot:
```bash
sudo bash /home/morgan/dogbot/scripts/wimz_rebadge.sh 4   # or 5
sudo reboot
```
Then register the printed DEVICE_ID/SECRET on Lightsail. That's it.

### Concurrency notes
- treatbot1 stays online. Two separate Xbox controllers (one per robot) — first-come-first-served Bluetooth, no preemptive removal needed.
- Once relay-registered, treatbot3 will appear as a separate device alongside `wimz_robot_01`.
- Laptop SSH known_hosts: `ssh-keygen -R treatbot1.local` and any cached IPs before next SSH from laptop, otherwise host-key-mismatch warning.

---

## Session: 2026-04-27 — Git Pull & Camera Boot Config Docs

**Goal:** Pull latest changes, document camera boot config for fleet
**Status:** COMPLETE

---

### What Was Accomplished

1. **Pulled origin/main** (0806e6a → 954fb42)
   - 35 files updated with significant changes
   - New scripts: capture_behavior_sequences.py, train_behavior_lstm.py
   - New service: camera_detect.py
   - Major updates to main_treatbot.py, api/server.py, wifi_manager.py

2. **Documented Camera Boot Config**
   - Added to `.claude/hardware_specs.md`: camera auto-detect settings
   - `/boot/firmware/config.txt` needs `camera_auto_detect=1` and `#dtoverlay=imx500`
   - Fleet status: treatbot2 fixed, treatbot1/3-5 need edit before camera swap

### Files Modified
- `.claude/hardware_specs.md` — Added camera boot config section

### Commit
- c6e5998 — docs: Add camera boot config for IMX500/IMX708 swaps

### Next Session
- No pending tasks from this session

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
