# WIM-Z Resume Chat Log

## Session: 2026-03-26 - Treat Dispenser Calibration, Xbox Motor Attempt (Reverted)
**Goal:** Fix 3 issues: direct AP connection, Xbox carpet movement, treat dispenser degradation
**Status:** PARTIAL — dispenser fixed, AP deferred, Xbox reverted

---

### Problems Solved This Session

| # | Problem | Root Cause | Solution | Status |
|---|---------|------------|----------|--------|
| 1 | Direct phone-to-robot connection doesn't work | Flutter app hardcoded to cloud relay; DNS hijack on AP breaks relay but local API works | Deferred — robot side ready (API on 0.0.0.0:8000), needs Flutter app changes | DEFERRED |
| 2 | Treat dispenser: full rotation initially, then 5-10% movement after ~20 treats, eventually unresponsive | slow_pulse=1544 too close to servo dead band (1500). As friction/temp increased, pulse fell inside dead band. NOT thermal drift (persisted through restarts). | Changed slow_pulse 1544→1560, dispense_duration 0.17→0.10 in treatbot1.yaml. Tested 5x dispenses consistently. | FIXED |
| 3 | Vibrator motor broken | Physical hardware — motor broke off | Added `vibrator_enabled: false` config flag, dispenser skips vibrator init and GPIO calls | FIXED |
| 4 | Xbox controller carpet movement | Attempted: multiplier relocation, carpet minimum scaling, motor_command_bus PID bypass | All changes broke motor control worse — dead center didn't stop, all directions degraded. **Fully reverted.** | REVERTED |

---

### Key Code Changes

**Treat dispenser fix** (3 files, working):
- `config/robot_profiles/treatbot1.yaml`: slow_pulse 1544→1560, dispense_duration 0.17→0.10, vibrator_enabled: false
- `config/config_loader.py`: Added `DispenserConfig.vibrator_enabled` property
- `services/reward/dispenser.py`: Check vibrator_enabled before init/use

**Diagnostic script** (new):
- `tests/hardware/test_treat_servo_diag.py`: Non-interactive PCA9685/servo diagnostic

**Xbox controller** — ALL CHANGES REVERTED:
- `xbox_hybrid_controller.py`: Reverted to HEAD
- `core/motor_command_bus.py`: Reverted to HEAD

### Key Findings (Xbox Controller — For Future Reference)

1. Xbox controller runs as a SUBPROCESS launched by treatbot service (`services/control/xbox_controller.py` → subprocess of `xbox_hybrid_controller.py`)
2. Motor commands flow: xbox subprocess → POST /motor/control → motor_command_bus → PID controller
3. PID controller logs showed Target L=0.0 R=0.0 even during active driving — values not propagating correctly through API→bus��PID chain
4. motor_command_bus divides speed by 100.0 instead of MAX_SPEED (80) when converting to RPM — 20% power loss
5. The treatbot2 fix (multipliers on forward, motor_bus PID bypass) does NOT directly translate to treatbot1 architecture
6. Never kill/restart the Xbox subprocess independently — it's managed by treatbot service

### Treat Servo Diagnostic Results

- PCA9685 at 0x40, I2C working fine
- Pan/tilt servos (ch0, ch1) both working
- Winch servo (ch2) works but 1544us too weak for carousel friction
- 1560us at 0.10s = correct one-slot rotation at full battery (16.58V)
- Battery was at 16.58V (full charge) during testing

### Git Status
- Branch: `main` at `81dea00`
- 0 commits made this session (changes pending)
- Protected files: unchanged
- Xbox controller: fully reverted to HEAD

### Next Steps
1. **Commit treat dispenser fix** (config_loader, treatbot1.yaml, dispenser.py)
2. **Xbox carpet movement** — needs careful investigation of the API→motor_bus→PID value propagation chain before attempting fixes
3. **Direct AP connection** — needs Flutter app changes (robot side ready)
4. **Backward power on carpet** — related to Xbox issue, PID controller may need tuning
5. **Replace vibrator motor** when hardware available

---

## Session: 2026-03-25 - Battery Fix, Motor Speed, Treat Counter Push, Local AP Control
**Goal:** Fix battery meter showing 98% instead of ~60%, increase motor speed for carpet, push uncommitted treat/dispenser code for treatbot2, verify local AP control path
**Status:** COMPLETE

---

### Problems Solved This Session

| # | Problem | Root Cause | Solution | Status |
|---|---------|------------|----------|--------|
| 1 | Battery meter showing ~98% at ~60% real charge | 3 endpoints used wrong formula `voltage/16.8*100` instead of correct `(voltage-12.0)/4.8*100` | Fixed formula in `api/ws.py`, `api/server.py`, `services/cloud/relay_client.py` | FIXED |
| 2 | Robot struggles on carpet | Xbox controller hardcoded MAX_SPEED=50, MAX_RPM=90 despite config saying 80/110 | Updated defaults in `xbox_hybrid_controller.py` to match `treatbot1.yaml` (80/110) | FIXED |
| 3 | Treatbot2 crashing: `No module named 'services.logging'` | `services/logging/` directory not in git | Pushed `dog_event_logger.py` + `__init__.py` to git | FIXED |
| 4 | Treatbot2 missing treat counter, dispenser improvements | 17 source files with accumulated changes never committed | Committed all: treat counter persistence, vibrator motor, anti-jam, dog profiles, SG config, video recording, etc. | FIXED |
| 5 | Resume chat log 1 month behind (last entry Feb 26) | Previous sessions didn't run /session_end | Reconstructed Mar 12 and Mar 22 sessions from git history | FIXED |

---

### Key Code Changes

**Battery percentage fix** (`3eb3c2b` — 4 files):
- `api/ws.py:968`, `services/cloud/relay_client.py:317`, `api/server.py:5342`: Changed `voltage/16.8*100` to `(voltage-12.0)/4.8*100`
- `xbox_hybrid_controller.py`: MAX_SPEED 50→80, MAX_RPM 90→110

**Accumulated features push** (`204322f` — 17 files, +1109/-116):
- `services/reward/dispenser.py`: Treat counter (SQLite persistence), vibrator motor (GPIO16), anti-jam wiggle
- `core/store.py`: Dog event storage/query (+203 lines)
- `core/dog_profile_manager.py`: Cloud profile sync with ArUco mapping
- `modes/silent_guardian.py`: Runtime config updates, enhanced bark detection
- `services/media/video_recorder.py`: High-res recording with AI pause
- `main_treatbot.py`: Major updates (+222 lines)
- Plus: coaching_engine, mission_engine, detector, pan_tilt, sequence_engine, config files

**Missing module push** (`81dea00` — 2 files):
- `services/logging/__init__.py` + `services/logging/dog_event_logger.py`

### Local AP Control Investigation
- Confirmed: Robot can be controlled via phone without internet
- Phone connects to `WIMZ-xxxx` AP (password: `wimzsetup`)
- Treatbot API accessible at `192.168.4.1:8000` (runs in parallel with provisioning)
- App's server address field can point to `192.168.4.1:8000`
- Just don't submit WiFi credentials in captive portal — AP stays up

### Git Status
- Branch: `main` at `81dea00`
- 3 commits made this session
- Protected files: unchanged

### Next Steps
1. Test battery meter accuracy in app (should show ~58% at 14.8V nominal)
2. Test motor speed on carpet with Xbox controller (MAX_SPEED now 80)
3. Xbox controller full button test
4. Treatbot2: `git pull` to get all pushed changes
5. Test local AP control: phone → WIMZ AP → app at 192.168.4.1:8000

---

## Session: 2026-03-22 - PTT Queuing, Mode Sync & Silent Guardian Stability
**Goal:** Fix app-robot communication bugs and Silent Guardian mode dropping to IDLE
**Status:** COMPLETE

---

### Problems Solved This Session

| # | Problem | Root Cause | Solution | Status |
|---|---------|------------|----------|--------|
| 1 | Silent Guardian randomly reverting to IDLE | Multiple unprotected code paths (mission complete, battery critical, emergency clear, Xbox disconnect) all defaulted to IDLE | Added SG protection gate in `set_mode()` — blocks SG→IDLE unless triggered by explicit user action; mission engine restores pre-mission mode; Xbox disconnect restores pre-manual mode | FIXED |
| 2 | App mode indicator out of sync on connect | Robot sent aliases ("guardian"/"training") but app expected raw names ("silent_guardian"/"coach") | Fixed all 4 outbound mode maps to send raw internal names; added 1s delay before mode sync on `user_connected`; added inbound aliases for backward compat | FIXED |
| 3 | Second PTT audio message dropped | `push_to_talk.py` rejected new audio while first was playing ("Already playing audio") | Replaced with queue worker that processes items sequentially; always sends `audio_played` ack even on error | FIXED |
