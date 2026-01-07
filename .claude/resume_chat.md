# WIM-Z Resume Chat Log

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
