# WIM-Z Resume Chat Log

## Session: 2026-01-07 ~11:55-12:55
**Goal:** Fix Xbox controller RB photo button and mode cycling issues
**Status:** ✅ Fixes Implemented (Testing Pending)

### Work Completed:

#### 1. Diagnosed RB Photo Button Issues - ✅ FIXED
- **Problem:** RB button wasn't taking photos; buttons were switching to MANUAL mode
- **Root Cause:** `notify_manual_input()` was called on EVERY button press in `process_button()`, which triggered mode switch to MANUAL
- **Fix:** Removed global `notify_manual_input()` from `process_button()` - joystick/triggers already have it in `process_axis()`

#### 2. Fixed Subprocess Logging - ✅ FIXED
- **Problem:** Xbox controller subprocess logs went to PIPE and were never read (invisible)
- **Fix:** Removed `stdout=subprocess.PIPE, stderr=subprocess.PIPE` from subprocess.Popen
- **Added:** `-u` flag for unbuffered Python output
- Now controller logs appear in `journalctl -u treatbot`

#### 3. Updated take_photo() Logic - ✅ FIXED
- **Problem:** take_photo() always tried 4K first, fell back to snapshot
- **Fix:** Now explicitly checks current mode FIRST:
  - MANUAL mode → 4K photo (camera released)
  - Other modes → Snapshot from AI stream (640x640)
- Added `_get_current_mode()` helper that queries `/mode` API

#### 4. Fixed Mode Cycle Sync - ✅ FIXED
- **Problem:** `current_mode_index` was never synced with actual system mode
- **Fix:** `cycle_mode()` now queries actual mode before incrementing
- Changed to blocking API request for mode changes

#### 5. Fixed api_request_blocking() Timeout Parameter - ✅ FIXED
- **Problem:** `take_photo()` passed `timeout=8` but `api_request_blocking()` didn't accept it
- **Fix:** Added `timeout` parameter to both `api_request_blocking()` and `_api_request_sync()`

### Files Modified:
- `xbox_hybrid_controller.py`:
  - Removed `notify_manual_input()` from `process_button()` (line ~1229)
  - Added `_get_current_mode()` helper method
  - Rewrote `take_photo()` to check mode explicitly
  - Updated `cycle_mode()` to sync with actual mode
  - Fixed `api_request_blocking()` to accept timeout parameter
  - Fixed `_api_request_sync()` to accept timeout parameter
- `services/control/xbox_controller.py`:
  - Removed PIPE capturing from subprocess
  - Added `-u` flag for unbuffered output

### Key Behavior Changes:
1. **Button presses no longer auto-switch to MANUAL mode**
2. **RB in COACH/SILENT_GUARDIAN** → Takes 640x640 snapshot from AI stream
3. **RB in MANUAL** → Takes 4K photo (4056x3040)
4. **Mode cycling** → Now correctly syncs with actual system mode before cycling
5. **Controller logs** → Now visible in systemd journal

### User Requirements Confirmed:
1. RB in COACH/SILENT_GUARDIAN → Take snapshot, STAY in current mode
2. RB in MANUAL → Take 4K photo
3. No manual mode timeout when Xbox controller connected

### Next Session:
1. **TEST:** Mode cycling with SELECT button
2. **TEST:** RB photo in different modes
3. **TEST:** Verify photos saved correctly
4. Consider committing changes after testing

### Important Notes/Warnings:
- **Testing pending:** User needs to test SELECT and RB buttons
- **Photo locations:** `/home/morgan/dogbot/captures/photo_*.jpg` (4K), `snapshot_*.jpg` (640x640)
- **Logs now visible:** Use `journalctl -u treatbot -f` to see Xbox controller output

---

## Session: 2026-01-07 ~05:00-06:00
**Goal:** Fix Xbox controller freeze/lock issues, camera photo system
**Status:** ✅ Complete

### Work Completed:

#### 1. Motor Safety Fixes - ✅ FIXED
- **Problem:** Controller freeze caused motors to keep running (dangerous!)
- **Root Cause:** `set_motor_pwm_direct()` didn't update safety tracking variables
- **Fix:** Added `motors_should_be_stopped` and `last_nonzero_command_time` tracking in open-loop mode
- Motors now auto-stop after 1 second if controller freezes

#### 2. Event Bus Rate Limiting - ✅ FIXED
- **Problem:** Rapid button presses (LED toggle spam) could freeze controller
- **Fix:** Made `notify_manual_input()` non-blocking with 100ms rate limit
- Prevents thread spam on rapid button presses

#### 3. Camera Photo System - ✅ IMPLEMENTED
- **Problem:** RB button didn't take photos (camera busy, mode issues)
- **Fix:**
  - Detector now releases camera when entering MANUAL mode
  - Detector re-acquires camera when leaving MANUAL mode
  - Added `/camera/photo` endpoint for 4K photos (4056x3040)
  - Added `/camera/snapshot` endpoint for quick captures from AI stream (640x640)
  - Xbox RB button tries 4K first, falls back to snapshot

#### 4. Per-Robot Camera Config - ✅ IMPLEMENTED
- **Problem:** Different robots have cameras mounted at different orientations
- **Fix:** Added `camera.rotation` config to robot profiles
  - `treatbot.yaml`: 90° clockwise
  - `treatbot2.yaml`: 0° (no rotation)
- Created `config/config_loader.py` with `CameraConfig` class
- Detector reads rotation from config

### Photos Save To:
- `/home/morgan/dogbot/captures/photo_*.jpg` (4K)
- `/home/morgan/dogbot/captures/snapshot_*.jpg` (640x640)

---

## Session: 2026-01-06 ~17:00-18:45
**Goal:** Fix motor calibration - binary on/off → gradual speed control
**Status:** ✅ RESOLVED

### Work Completed:

#### 1. Motor PWM Control - ✅ FIXED
- **Problem:** Motors responded as on/off (binary) instead of gradual speed
- **Root Cause:** WIRING ERROR - GPIO pins were connected wrong
- **User fixed the hardware wiring**

---

## Session: 2026-01-04 00:30 - 01:15
**Goal:** Fix hardware issues on treatbot2 - Battery monitor, servos, LEDs
**Status:** ✅ Partially Complete

### Work Completed:
- Battery Monitor (ADS1115) - FIXED
- Servo/PCA9685 I2C Conflict - FIXED
- LEDs - Hardware issue (GPIO25 working, physical wiring needs check)
