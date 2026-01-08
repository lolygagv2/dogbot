# WIM-Z Resume Chat Log

## Session: 2026-01-08 ~Morning
**Goal:** Debug error audio spam + Fix dog detection + Add Xbox Guide button trick cycling
**Status:** ✅ Complete

### Work Completed:

#### 1. Fixed Constant "Error" Audio Spam - ✅ FIXED
- **Problem:** Error audio playing every 60 seconds during normal operation
- **Root Cause:** Temperature at 70-72°C triggering WARNING alerts (threshold was 70°C)
- **Fixes Applied:**
  - Raised `temp_warning` from 70°C to 76°C in `core/safety.py`
  - Changed error audio to only play for CRITICAL alerts (not WARNING)
  - Added 30-second startup grace period for CPU alerts (100% at startup is normal)
  - Added separate `_play_hot_audio()` using `wimz_hot.mp3` for temperature warnings

#### 2. Fixed Speak Trick False Positives - ✅ FIXED
- **Problem:** Non-bark sounds triggering speak trick reward
- **Fix:** Added 50% minimum confidence filter in coaching engine's `_on_audio_event()`

#### 3. Fixed Charging Detection - ✅ FIXED
- **Problem:** Threshold too high (0.3V) - wouldn't detect charging at 50%+ battery
- **Fix:** Lowered voltage increase threshold from 0.3V to 0.05V in `battery_monitor.py`

#### 4. Fixed Slow Dog Detection - ✅ FIXED
- **Problem:** 30+ seconds to detect dog in clear view
- **Fix:** Lowered confidence threshold from 0.7 to 0.5 in `robot_config.yaml`

#### 5. Replaced Stillness with Time-Based Detection - ✅ FIXED
- **Problem:** 50px stillness requirement too strict for excited small dogs near camera
- **Fix:** Removed stillness tracking entirely, replaced with 3.5 second time-in-view confirmation
- Removed: `dog_still_start`, `last_dog_position`, `_update_dog_position()`
- Updated: `_state_waiting_for_dog()` uses time since first seen

#### 6. Added Consistent Trick Thresholds - ✅ FIXED
- **Problem:** `reward_logic.py` had hardcoded 0.7 confidence, didn't match `trick_rules.yaml`
- **Fix:** Now imports `BehaviorInterpreter` and uses its thresholds for consistency

#### 7. Added Xbox Guide Button Trick Cycling - ✅ IMPLEMENTED
- **Behavior:** Press Guide button (button 8) to cycle tricks in coach mode
- **Cycle:** sit → down → crosses → spin → speak → sit...
- **Features:**
  - Only works in coach mode (ignores other modes)
  - Sets forced trick for next coaching session
  - Plays trick name as audio feedback
  - 1-second cooldown between presses
- **Uses existing API:** `/coaching/force_trick/{trick}`

### Files Modified:
- `core/safety.py` - Temperature thresholds, startup grace period, hot audio
- `config/robot_config.yaml` - Detection confidence 0.7 → 0.5
- `orchestrators/coaching_engine.py` - Removed stillness, added bark confidence filter
- `services/power/battery_monitor.py` - Charging threshold 0.3V → 0.05V
- `api/server.py` - LED reset on recording timeout
- `orchestrators/reward_logic.py` - Uses BehaviorInterpreter thresholds
- `xbox_hybrid_controller.py` - Added Guide button trick cycling

### Next Session:
1. Test Guide button trick cycling in coach mode
2. Verify temperature warnings use wimz_hot.mp3 instead of error audio
3. Test charging detection with new 0.05V threshold
4. Verify dog detection is faster with 0.5 confidence threshold

### Important Notes:
- Pi 5 + Hailo-8 normally runs 65-75°C under AI load (now allowed)
- Guide button only works in coach mode
- Available tricks for cycling: sit, down, crosses, spin, speak

---

## Session: 2026-01-07 ~21:00-22:30
**Goal:** Debug Coach mode audio issues + Add WIM-Z audio feedback system
**Status:** ✅ Complete
**Commit:** 98a4ff11

### Work Completed:

#### 1. Diagnosed Coach Mode Audio Bug - ✅ FIXED
- **Problem:** Dog heard "bezik" greeting, then 10s silence, then "no no no" - no trick cue played
- **Root Cause:** BehaviorInterpreter config failed to load due to init order bug
  - `_load_trick_rules()` tried to use `self.confidence_thresholds` before it was defined
  - Fell back to DEFAULT rules which lacked `audio_command` field
  - Trick "down" tried to play "down.mp3" (doesn't exist) instead of "lie_down.mp3"
- **Fix 1:** Moved `confidence_thresholds` definition BEFORE `_load_trick_rules()` call
- **Fix 2:** Added `audio_command` to all default trick rules with correct filenames

#### 2. Added WIM-Z Audio Feedback System - ✅ IMPLEMENTED
| Feature | File | Trigger |
|---------|------|---------|
| Charging audio | `Wimz_charging.mp3` | Voltage rises 0.3V+ over 15s |
| Low power audio | `Wimz_lowpower.mp3` | Battery < 12.0V |
| Error audio | `Wimz_errorlogs.mp3` | Safety warnings/critical alerts |
| Mission complete | `Wimz_missioncomplete.mp3` | Formal mission success |
| Recording start | `Wimz_recording.mp3` | Xbox Start button (1st press) |
| Recording saved | `Wimz_saved.mp3` | Xbox Start button (2nd press) |

### Files Modified:
- `core/behavior_interpreter.py` - Fixed init order + default audio_commands
- `core/safety.py` - Added error audio alerts
- `services/power/battery_monitor.py` - Added charging detection
- `orchestrators/mission_engine.py` - Added mission complete audio
- `api/server.py` - Updated recording audio paths
- `main_treatbot.py` - Updated low battery audio path
- Added 8 new MP3 files to `VOICEMP3/wimz/`

### Next Session:
1. Restart treatbot to apply changes
2. Test Coach mode - verify trick audio plays correctly
3. Test charging detection by plugging in charger
4. Test Xbox recording with Start button

### Important Notes:
- Treatbot service needs restart: `sudo systemctl restart treatbot`
- Coach mode available tricks: sit, down, crosses, spin, speak (stand excluded)
- Charging cooldown: 5 minutes between announcements
- Error audio cooldown: 60 seconds

---

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
