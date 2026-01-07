# WIM-Z Resume Chat Log

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

### Files Modified:
- `core/hardware/proper_pid_motor_controller.py` - Motor safety in open-loop mode
- `xbox_hybrid_controller.py` - Non-blocking events, photo fallback logic
- `api/server.py` - Camera photo/snapshot endpoints, Xbox detection
- `services/perception/detector.py` - Camera release/reacquire, config rotation
- `config/config_loader.py` (new) - CameraConfig class
- `config/robot_profiles/treatbot.yaml` - Added camera.rotation: 90
- `config/robot_profiles/treatbot2.yaml` - Added camera.rotation: 0

### Files Archived:
- `Archive/servo_control_module.py`
- `Archive/treat_dispenser_robot.py`

### Commits:
- `b6112bd6` - feat: Motor safety, camera system, and per-robot config
- `060c86f1` - Merge with remote (synced mode_fsm.py fix)

### Photos Save To:
- `/home/morgan/dogbot/captures/photo_*.jpg` (4K)
- `/home/morgan/dogbot/captures/snapshot_*.jpg` (640x640)

### Next Session:
1. Test motor safety on original treatbot after git pull
2. Verify camera rotation is correct on treatbot (90° should be right)
3. Fine-tune PID parameters if encoder issues resolved

### Important Notes/Warnings:
- **treatbot sync required:** Run `git pull origin main && sudo systemctl restart treatbot`
- **Camera rotation:** treatbot=90°, treatbot2=0° - verify after testing
- **Motor safety:** 1-second stale command detection now active in open-loop mode

---

## Session: 2026-01-06 ~17:00-18:45
**Goal:** Fix motor calibration - binary on/off → gradual speed control
**Status:** ✅ RESOLVED

### Work Completed:

#### 1. Motor PWM Control - ✅ FIXED
- **Problem:** Motors responded as on/off (binary) instead of gradual speed
- **Root Cause:** WIRING ERROR - GPIO pins were connected wrong:
  - ENA was on pin 31, should be pin 33 (GPIO13)
  - ENB was on pin 33, should be pin 35 (GPIO19)
  - Encoder A2/B2 wires were also swapped
- **User fixed the hardware wiring**

#### 2. Code Improvements Made:
- Rewrote `set_motor_speeds()` in `xbox_hybrid_controller.py`:
  - Proper joystick → PWM mapping (20-70% range)
  - Removed double-clamping that caused binary jumps
  - Multipliers now applied correctly after PWM calculation
- Updated `proper_pid_motor_controller.py`:
  - Added `open_loop_mode` flag
  - Removed PWM_MIN clamping in `set_motor_pwm_direct()`
- Updated `treatbot2.yaml`: left_multiplier=0.5, right_multiplier=1.0

### Key Solutions:
- **Config auto-loading:** Uses hostname detection (treatbot2 → treatbot2.yaml)
- **PWM mapping:** Joystick (0-1) maps to PWM (20-70%), then multiplier applied
- **Open-loop mode:** Bypasses PID when USE_PID_CONTROL=false

### Files Modified:
- `core/hardware/proper_pid_motor_controller.py` (+105 lines)
- `xbox_hybrid_controller.py` (+144 lines)
- `config/robot_profiles/treatbot2.yaml` (motor calibration)

### Files Created:
- `tests/hardware/test_simple_joystick_motor.py`
- `tests/hardware/test_joystick_gpio_direct.py`

### Commit: 0d944455 - fix: Motor PWM control - gradual speed instead of binary on/off

### Next Session:
1. Fine-tune motor multipliers (left=0.5, right=1.0) if balance still off
2. Test encoders now that wiring is fixed (right encoder was reading 0)
3. Consider re-enabling PID control once encoders verified working

### Important Notes/Warnings:
- **WIRING:** All motor/encoder wires now in correct positions per pins.py
- **Multipliers:** Left motor is overpowered (1.6Ω), right is weak (4.5Ω, defective)
- **Right motor replacement:** Still on order

---

## Session: 2026-01-04 00:30 - 01:15
**Goal:** Fix hardware issues on treatbot2 - Battery monitor, servos, LEDs
**Status:** ✅ Partially Complete

### Work Completed:

#### 1. Battery Monitor (ADS1115) - ✅ FIXED
- Created `services/power/battery_monitor.py` - ADS1115 ADC monitoring
- Created `core/hardware/i2c_bus.py` - Shared I2C bus singleton with RLock
- Calibration: 15.15V battery = 3.843V at ADC (factor: 3.942)
- Voltage divider: 10k + 2.2k resistors
- API endpoints: `/battery/status`, `/battery/voltage`
- Working: 15.13V reading confirmed

#### 2. Servo/PCA9685 I2C Conflict - ✅ FIXED
- **Root Cause:** MODE2 register INVRT bit was set (inverting all PWM outputs)
- **Fix:** Added `pca.reset()` and `pca.mode2 = 0x04` to servo initialization
- Updated `core/hardware/servo_controller.py` to use shared I2C bus
- Both battery monitor and servos now work together on shared I2C

#### 3. LEDs (NeoPixels + Blue LED) - ❌ HARDWARE ISSUE
- GPIO25 confirmed outputting HIGH via pinctrl
- No light visible = hardware wiring issue on treatbot2
- User will check physical connections

### Key Solutions:
- **I2C Bus Sharing:** Created singleton with `threading.RLock()` for thread-safe access
- **PCA9685 Fix:** `pca.mode2 = 0x04` ensures OUTDRV=1, INVRT=0
- **Install Script:** Created `install_treatbot2_updates.sh` for replicating on other device

### Files Created:
- `core/hardware/i2c_bus.py` (new) - Shared I2C bus singleton
- `services/power/battery_monitor.py` (new) - Battery monitoring service
- `services/power/__init__.py` (new) - Package init
- `install_treatbot2_updates.sh` (new) - Installation script for other device

### Files Modified:
- `core/hardware/servo_controller.py` - Use shared I2C, add MODE2 fix
- `api/server.py` - Add battery API endpoints
- `services/media/led.py` - Added Silent Guardian patterns (from previous session)

### Important Notes/Warnings:
- **PCA9685 MODE2:** If servos stop working, check if INVRT bit got set again
- **LEDs on treatbot2:** Confirmed software working, hardware disconnected
- **Pi 5 GPIO:** Uses lgpio (not RPi.GPIO), gpiochip4 is the main chip
