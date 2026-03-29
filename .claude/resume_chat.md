# WIM-Z Resume Chat Log

## Session: 2026-03-29 - Stepper Motor Treat Dispenser (TMC2209 + NEMA 17)
**Goal:** Replace servo dispenser with stepper motor (NEMA 17 + TMC2209 driver)
**Status:** PARTIAL — Dispenser works, refill works, jam detection FAILED

---

### What Works
- **Single treat dispense** — tap LB, motor steps 137 microsteps, treat drops. Reliable.
- **Refill mode** — hold LB 5+ seconds, continuous fast stepping, release = instant stop.
- **UART communication** — TMC2209 responds, current/microstepping configured via UART.
- **Motor control** — STEP/DIR/EN all working, CW direction confirmed, calibrated.

### What Does NOT Work
- **Jam detection via StallGuard** — COMPLETELY non-functional at our speed.
  - SG_RESULT = 0 in BOTH jammed and unjammed states. Identical register values.
  - Tested: StealthChop mode, SpreadCycle mode, all SGTHRS values (0-200), all speeds (0.5ms-6ms), TCOOLTHRS=0xFFFFF.
  - CS_ACTUAL = 31 (max) always — no current headroom for load measurement.
  - TSTEP identical in jammed vs unjammed (~8540).
  - Root cause: motor speed too low (~3 RPM) for StallGuard back-EMF sensing.
  - **DIAG pin would show same result** — same StallGuard circuit.

### Hardware Setup (Final Working)
| Signal | Board Pin | GPIO (BCM) | TMC2209 Pin |
|--------|-----------|------------|-------------|
| STEP | 32 | GPIO 12 | Pin 7 (STEP) |
| DIR | 36 | GPIO 16 | Pin 8 (DIR) |
| EN | 18 | GPIO 24 | Pin 1 (EN) |
| UART TX | 8 | GPIO 14 (1K resistor) | Pin 4 (PDN_UART) |
| UART RX | 10 | GPIO 15 (direct) | Pin 4 (PDN_UART, spliced with TX) |
| VCC_IO | Pi Pin 17 (3.3V) | — | Pin 7 (VCC_IO) |

### Teyleten TMC2209 V2.0 Board Key Facts
- **UART is on Pin 4 by default** (not Pin 5 as initially assumed)
- Pin 5 is the alternate UART (requires solder bridge mod)
- VCC_IO (Pin 7) MUST be powered externally — no onboard regulator
- Single-wire UART: Pi TX (1K resistor) and Pi RX both splice to Pin 4
- MS1 pulled HIGH on board → 32x microstepping hardware default
- UART mstep_reg_select overrides to 8x

### TMC2209 Configuration (Working)
```
GCONF:      0x00000081  (I_scale_analog=1, mstep_reg_select=1)
IHOLD_IRUN: IRUN=31, IHOLD=5, IHOLDDELAY=6
CHOPCONF:   MRES=5 (8x), vsense=0 (high current range)
SGTHRS:     0 (disabled — doesn't work at our speed)
TCOOLTHRS:  0xFFFFF
```

### Calibration
- **Steps per slot:** 137 microsteps at 8x microstepping
- **Step delay:** 0.006s (12ms/step = MEDIUM speed) for dispense
- **Refill speed:** 0.004s (8ms/step = FAST)
- **Direction:** CW = DIR pin HIGH (1)
- **Reverse steps:** 40 (for unjam sequences)

### Files Modified This Session
- `config/pins.py` — Replaced VIBRATOR with STEPPER_STEP/DIR/EN/UART
- `config/config_loader.py` — New DispenserConfig: steps_per_slot, step_delay, irun, ihold, sgthrs, etc.
- `config/robot_profiles/treatbot1.yaml` — Stepper config replacing servo config
- `services/reward/dispenser.py` — Full rewrite: UART→TMC2209→GPIO stepping, anti-jam (L1/L2), refill mode, emergency_stop
- `api/server.py` — Added: `/treat/refill/step`, `/treat/refill/stop`, `/treat/stop`, `/treat/refill`
- `xbox_hybrid_controller.py` — LB: tap=dispense, hold 5s=refill with instant stop
- `tests/hardware/test_stepper_dispenser.py` — Interactive test/calibration script

### Xbox LB Button Behavior
- **Tap** (< 5s): Dispenses one treat immediately on press
- **Hold** (≥ 5s): Refill mode — continuous fast stepping, release = instant stop
- Refill uses `/treat/refill/step` and `/treat/refill/stop` endpoints
- `_step()` checks abort/stop flags every single step (~8ms response time)

### Anti-Jam Status (NOT WORKING — needs alternative approach)
- Code exists: `_rotate_carousel()` calls `_check_stall()` after stepping, escalates L1→L2
- `_check_stall()` reads DRV_STATUS bit 31 — always returns False during stepping
- Anti-jam L1 (5s gentle reverse-forward) and L2 (3s aggressive shake) code is in place
- Manual unjam endpoint: `POST /treat/unjam`
- Emergency stop: `POST /treat/stop`
- **Problem:** No way to detect jams. StallGuard produces identical readings jammed vs unjammed.

### Next Steps (Jam Detection)
Options to explore:
1. **Physical sensor** — IR break-beam at drop hole (most reliable, needs hardware)
2. **Audio detection** — onboard mic detects clicking pattern of stalled motor
3. **No auto-detection** — manual unjam only, acceptable for now
4. **Current sensing external** — separate current sensor on motor wire
5. **Encoder/position feedback** — verify motor actually rotated

### Git Status
- Branch: `main` at `b5cd6e8`
- 0 commits this session
- All changes uncommitted: dispenser.py, pins.py, config_loader.py, treatbot1.yaml, server.py, xbox_hybrid_controller.py, test_stepper_dispenser.py

---

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
