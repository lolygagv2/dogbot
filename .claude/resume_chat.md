# WIM-Z Resume Chat Log

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

### Next Session:
1. User to check LED hardware wiring on treatbot2 (GPIO25 → Blue LED, GPIO10 → NeoPixels)
2. Test full TreatBot startup with all fixes
3. Run install script on original treatbot device

### Important Notes/Warnings:
- **PCA9685 MODE2:** If servos stop working, check if INVRT bit got set again
- **LEDs on treatbot2:** Confirmed software working, hardware disconnected
- **Pi 5 GPIO:** Uses lgpio (not RPi.GPIO), gpiochip4 is the main chip
