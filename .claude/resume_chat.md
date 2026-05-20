# WIM-Z Resume Chat Log

## Session: 2026-05-20 — Silent Guardian movement tier + Xbox controller bring-up

**Goal:** Add a physical-movement escalation tier to Silent Guardian; pair a new Xbox controller; diagnose drift + right-motor issues.
**Status:** ✅ All complete. Working tree clean, 2 commits pushed (`04295ee`, `f4c4876`). Battery-smoothing commit `337d92d` also pushed early in session.

### Work completed
1. **Silent Guardian — new Level 3 movement tier** (`04295ee`). Escalation ladder is now 4 levels: verbal → firm verbal → **physical movement** → calming music. Level 3 plays "quiet" then runs 3 cycles of in-place fwd/back/left/right (~400ms each, motors halted between moves, 5s pause between cycles), then waits for the progressive quiet period → reward. `_run_movement_sequence()` drives wheels via `motor_command_bus` (`CommandSource.AUTONOMOUS`), always finishes all cycles (no bark abort), bails on mode shutdown. `max_level` 3→4; music handler renamed `_process_level_3`→`_process_level_music`. No conflict with carousel anti-jam (separate stepper) or treat dispense (serialized on SG loop thread). **App-facing:** `get_status()` now reports `sg_max: 4`.
2. **Xbox controller fixes** (`f4c4876`):
   - **Right motor undrivable via Xbox** — root cause: the Xbox subprocess called `get_motor_bus()`, creating a *second* `ProperPIDMotorController` that fought the main process for motor GPIO lines. Fix: `xbox_controller.py` sets `WIMZ_XBOX_SUBPROCESS=1`; subprocess now drives motors via HTTP API only (main process = sole hardware owner). Standalone mode unaffected.
   - **Connected-but-unresponsive after Bluetooth blip** — a BT link drop recreates `/dev/input/js0`, leaving a stale fd (errno 19); the retry path never reopened it. Added `_reopen_device()` to self-heal.
   - **Reverse "sticking"** — neutral-stop sent a single stop then relied on a 2s heartbeat; a dropped stop left the motor creeping. Now re-sends 0 every ~50ms for a 0.5s window on transition to neutral.
   - Added `_calibrate_stick()` + `controller.xbox.stick_centers` (yaml) for worn/off-center sticks; default 0.0 = no-op.

### Hardware findings / warnings
- **Spare Xbox controller `A8:8C:3E:50:62:70` is JUNK** — left stick Y potentiometer worn out: rests at ~−0.72 and the rest position *wanders* (caused runaway-at-rest). Unstable rest can't be software-calibrated. **Unpaired this session — do not re-pair it.**
- treatbot1 is back on the **original controller `AC:8E:BD:4A:0F:97`**. MAC restored in `xbox_persistent.py` + `fix_xbox_controller.sh`.
- Right motor + encoder verified healthy via direct test (−707 enc counts) — it was never a hardware fault.
- No new files created this session; all changes were edits. Directory structure doc unchanged.

### Next steps
- Restart `treatbot.service` after any reboot (it ran the new code live this session — currently active).
- Coordinate `sg_max: 4` with the Flutter app team (Guardian level display "X/4").
- Silent Guardian Level 3 movement tier verified by code/syntax only — exercise it live when a 4th bark intervention naturally escalates.

## Session: 2026-05-17 — treatbot5 bring-up (cloned from treatbot3)

**Goal:** First-boot bring-up of treatbot5, cloned from treatbot3's SD card. Pull latest, customize per-unit calibration, run drive + gimbal test sequences.
**Status:** ✅ Core bring-up complete. Dispenser test skipped; full vision/AI verification deferred (camera was unplugged mid-session). Working tree changes ready to commit: `config/robot_profiles/treatbot5.yaml`, `.claude/resume_chat.md`.

### Capabilities now working on treatbot5
- **Identity**: hostname `treatbot5`, `DEVICE_ID=wimz_robot_05`, machine-id `f7cfcc5dbb3b4210b526449a5e9eb5d9`. Profile auto-selects via hostname_map.
- **Cloud relay**: connected `wss://api.wimzai.com/ws/device` on first boot — Lightsail already had `wimz_robot_05` registered (no relay-side work needed this session).
- **Battery monitoring**: reads 16.87V at 16.8V bench (0.4% error) with `calibration_factor: 53.33`.
- **Drive train**: Cytron MDD10A live, both wheels respond, forward+reverse verified.
- **Gimbal**: center + limits calibrated via Xbox-stick sweep.

### Per-unit calibration deltas vs treatbot3
1. **Battery divider runs ~1.8% leaner**: factor 53.33 (vs treatbot3's 54.28). Same hardware design, normal resistor tolerance variance.
2. **Right motor wired with reversed polarity** (same as treatbot3): `right_invert: true`. Left motor fine (`left_invert: false`).
3. **Gimbal center is at completely different servo positions**: treatbot5 lives at `pan_center: 64, tilt_center: 89` vs treatbot3's `67, -12`. Pan close but tilt convention reads very differently — both use "higher tilt = looking up" but the absolute neutral pulse on this servo lands at +89 instead of -12. Mounting/servo tolerance.
4. **Tilt range is asymmetric on this unit**: physical limits 57 (down, mechanical wall) to 236 (up). Center at 89 → only 29° of down-tilt headroom but 147° of up-tilt headroom. Locked `tilt_min: 60` (3° safety from wall), `tilt_max: 236`.
5. **Pan range**: `pan_min: -80`, `pan_max: 268` (user-chosen "comfortable max" via Xbox sweep, not necessarily mechanical wall).
6. **`coach_pan_*` / `coach_tilt_*` inherited verbatim from treatbot3** but need re-tuning in actual coach mode — the 30%-inset formula produces tilt range biased upward (away from floor-level dog tracking) given treatbot5's asymmetric tilt range. Flagged in yaml comment.

### Hardware findings
- **Xbox controller (AC:8E:BD:4A:0F:97)** paired fresh — the cloned BlueZ bonding database came over empty (or got cleared) so no pre-pair carried from treatbot3.
- **Camera (IMX708 wide)** was physically unplugged sometime during the gimbal calibration → service hung in startup with V4L2 buffer errors on next restart. Service stopped cleanly at end of session. **Next session: plug camera in, restart service, verify gimbal snaps to (pan=64, tilt=89) and AI pipeline starts cleanly.**

### Drive test sequence executed (wheels lifted, 30% throttle)
1. LEFT only +30 → forward ✅ (`left_invert: false` correct)
2. RIGHT only +30 → BACKWARD ❌ → flipped `right_invert: true` → retest → forward ✅
3. BOTH +30 → straight forward ✅
4. BOTH -30 → straight reverse ✅

### Gimbal calibration sequence (live Xbox sweep, position read via GET /camera/gimbal)
- CENTER: pan=64, tilt=89 (`current_position` reading)
- MAX UP: tilt=236 (locked as tilt_max)
- MAX DOWN: tilt=57 (mechanical wall hit, +3° safety → tilt_min=60)
- MAX LEFT: pan=267.8 (locked → pan_max=268)
- MAX RIGHT: pan=-78.9 (locked → pan_min=-80)

### Pending for treatbot5 (next session)
1. **Re-plug IMX708 camera, restart service, verify** new gimbal snaps to (64, 89) on boot and AI detection pipeline starts. Confirm 4-class behavior model loads.
2. **Dispenser test** — skipped this session. Trigger via API, confirm auger advances. Expect cosmetic "TMC2209 not responding on UART" warning (matches treatbot3).
3. **Field test on-floor** — all drive testing was bench/wheels-lifted.
4. **Coach mode tilt range re-tune** — `coach_tilt_min/max` currently inherited from treatbot3 and likely tracks too high given asymmetric tilt range.
5. **Optional motor calibration tuning**: `left_multiplier`/`right_multiplier` are 1.0/1.0; under load may need bias if robot pulls one direction.

### Commits this session
- Pulled 5 commits from origin/main (392c2cd, 0cbaeaf, 7a37458, 903b2c0, b4e74db) — includes 174 IMX500 training sequences, LSTM training guide, charging-detector fix.
- About to commit: treatbot5.yaml per-device calibration + this resume_chat update.

### Files touched
- MODIFIED: `config/robot_profiles/treatbot5.yaml` (full per-device fill-in: battery factor, Cytron driver dispatch, motor inversion, gimbal calibration)
- MODIFIED: `.claude/resume_chat.md` (this entry)

---

## Session: 2026-05-17 — treatbot4 hardware bring-up

**Goal:** Bring treatbot4 from cloned-from-treatbot1 SD card to a fully-calibrated working unit. Same Cytron + brushed-motor + IMX708 hardware family as treatbot3, but on a different physical Pi/PCB.
**Status:** ✅ COMPLETE (modulo battery, which needs actual battery plugged in to verify). Commit `89efa8f` pushed locally (not yet pushed to origin).

### Capabilities now working on treatbot4
- **Drive train**: Xbox left stick → Cytron MDD10A → both wheels respond, correct directions
- **Camera gimbal**: pan inherited from treatbot3 (pan_center=67), tilt re-calibrated after dead-servo replacement (tilt_center=43)
- **Cloud relay**: connected to `wss://api.wimzai.com/ws/device`; wimz_robot_04 registered Lightsail-side
- **Xbox controller**: paired (28:EA:0B:DB:82:3F), connected via /dev/input/js0
- **Camera**: IMX708 Pi Camera Module 3 Wide, 19.4 FPS detection pipeline

### Hardware findings worth remembering
- **Right motor wired with reversed polarity** — same as treatbot3. `right_invert: true`. Now 2-for-2 on Cytron brushed-motor builds → memory saved (`project_cytron_right_invert.md`) so treatbot5 starts with `true` as expected default.
- **Tilt servo was DOA** — confirmed via swap test (pan ↔ tilt cables swapped, neither chan-1 PCA output nor swapped-onto-chan-0 produced response). User installed a replacement servo mid-session in ~5 min.
- **New tilt servo's mechanical "level" position = code-angle 43°** (vs treatbot3's -12°). Different mounting offset. Symmetric ±50° physical range gives tilt_min=-57, tilt_max=143, with coach-mode 30% inset (3 to 83).
- **Battery sense divider on this PCB is wired the same as treatbot3** — but bench supply is going directly into the Pololu output (downstream of the divider tap), so ADS1115 A0 reads ~-27mV (floating). Calibration_factor 54.28 inherited; will read correctly once battery plugs into the JST.

### Commits this session
- `89efa8f` feat: treatbot4 hardware bring-up — Cytron + camera gimbal + new tilt servo

### Files touched
- MODIFIED: `config/robot_profiles/treatbot4.yaml` (Cytron section, battery cal, tilt re-cal, driver dispatch)

### Pre-existing issues still pending (unchanged from treatbot3 session)
- `services/motion/motor.py:MotorService.initialize()` NameError on ENCODER_MOTOR_AVAILABLE — dead code, not blocking
- TMC2209 dispenser UART warning — defaults match yaml, dispenser works fine
- Battery monitor `-1.48V CRITICAL` logs (test-rig artifact, resolves with battery plugged in)
- `/camera/center` and `/camera/position` API endpoints reference missing methods (would 500 if called)

### Still pending for treatbot4 (next session candidates)
1. **Verify battery calibration** — plug actual battery in, confirm reading ~16.6V at 16.8V (matches treatbot3 pattern)
2. **Treat dispenser test** — TMC2209/NEMA settings inherited from treatbot3, never tested on this unit
3. **Field test outdoors / under-load drive** — all testing this session bench/wheels-up
4. **Re-tune motor `left_multiplier`/`right_multiplier`** if drive feels asymmetric under load
5. **Push commit `89efa8f` to origin** — currently local-only

### Notes for treatbot5 (next clone)
Per the documented procedure, treatbot5 should inherit treatbot3.yaml as starting point. NEW datapoint from treatbot4: right_invert=true should be the **expected default** for these Cytron brushed-motor builds (2/2). Servos may need re-calibration regardless of donor yaml (tilt mounting differed even between treatbot3 and treatbot4).

---

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
