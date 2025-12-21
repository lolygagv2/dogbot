# WIM-Z Resume Chat Log

## Session: 2025-12-20 21:00 - 22:15
**Goal:** Fix Auto-Start Control Issues - Motors Getting Stuck, System Freezing
**Status:** ✅ COMPLETE

### ✅ Problems Solved This Session:

#### 1. **GPIO Initialization Timing (Motor Controller)**
- **Problem**: "GPIO busy" errors when Xbox controller tried to initialize motors
- **Root Cause**: `ProperPIDMotorController.__init__()` claimed GPIO pins immediately, blocking fallback controllers
- **Fix**: Deferred GPIO initialization to `start()` method instead of `__init__()`
- **Files**: `core/hardware/proper_pid_motor_controller.py`
  - Added `_initialize_gpio()` method
  - Added `_cleanup_gpio()` method
  - GPIO only claimed when `start()` called

#### 2. **Motor Watchdog Re-enabled**
- **Problem**: Watchdog was disabled (999999 timeout) - motors could run forever
- **Fix**: Set `watchdog_timeout = 2.0` seconds
- **Added**: 1-second "stale command" detection for additional safety

#### 3. **Blocking Device Read (CRITICAL)**
- **Problem**: `device.read(8)` blocked forever with no timeout - caused total system freeze
- **Root Cause**: Joystick device opened in blocking mode, no select/timeout
- **Fix**: Added `select.select()` with 100ms timeout before read
- **File**: `xbox_hybrid_controller.py` line 612

#### 4. **Excessive Logging Causing I/O Bottleneck (CRITICAL)**
- **Problem**: ~150+ log lines/second choking the system
- **Root Causes**:
  - 3 `print()` calls on every joystick axis event
  - 2 `logger.info()` on every motor command (10/sec)
  - 4 `logger.info()` on every PWM apply (50/sec × 2 motors)
- **Fix**: Removed all excessive debug output
- **Files**: `xbox_hybrid_controller.py`, `core/hardware/proper_pid_motor_controller.py`

#### 5. **RT Trigger Not Working**
- **Problem**: Right trigger did nothing
- **Root Cause**: RT only added 20% boost, not primary throttle
- **Fix**: Made RT the primary throttle (0-100% speed control)
- **Control**: Left stick = direction, RT (hold) = throttle

#### 6. **Emergency Stop Pin Numbers Wrong**
- **Problem**: `global_emergency_stop()` had wrong pin list [17,27,22,23,24,25]
- **Fix**: Corrected to [17,18,27,22,13,19] (IN1,IN2,IN3,IN4,ENA,ENB)
- **Added**: Multiple fallback methods (gpiozero → lgpio → gpioset)

#### 7. **Motor Power Underpowered**
- **Problem**: Motors very weak, not responsive
- **Root Causes**:
  - `MAX_RPM = 60` (was reduced 50% from 120)
  - Feedforward gain at 0.6 (too conservative)
- **Fix**:
  - `MAX_RPM = 100` (moderate for responsive control)
  - Feedforward gain = 0.9

#### 8. **LED Initialization Disabled**
- **Problem**: "LEDs not initialized" errors
- **Root Cause**: LED controller self-disabled in `__init__`
- **Fix**: Re-enabled `_initialize_blue_led()` and `_initialize_neopixels()`
- **File**: `core/hardware/led_controller.py`

#### 9. **LED Mode Pattern Case Sensitivity**
- **Problem**: LED patterns always showing "idle" regardless of mode
- **Root Cause**: Mode names uppercase in FSM but lowercase in pattern map
- **Fix**: Added both uppercase and lowercase mode names to mapping
- **File**: `services/media/led.py`

### 📁 Files Modified:
| File | Changes |
|------|---------|
| `core/hardware/proper_pid_motor_controller.py` | GPIO deferred init, watchdog 2s, stale command detection, removed excessive logging |
| `xbox_hybrid_controller.py` | Added select() timeout, RT as throttle, removed debug prints, fixed emergency stop pins |
| `core/hardware/led_controller.py` | Re-enabled LED initialization |
| `services/media/led.py` | Fixed mode pattern case sensitivity |

### 🔧 Key Technical Solutions:

1. **Blocking I/O Fix**:
   ```python
   ready, _, _ = select.select([self.device], [], [], 0.1)  # 100ms timeout
   if ready:
       event_data = self.device.read(8)
   ```

2. **Deferred GPIO Pattern**:
   ```python
   def __init__(self):
       self._gpio_initialized = False  # Don't claim GPIO here

   def start(self):
       if not self._gpio_initialized:
           self._initialize_gpio()  # Claim GPIO only when starting
   ```

3. **Stale Command Detection**:
   ```python
   if not self.motors_should_be_stopped:
       if current_time - self.last_nonzero_command_time > 1.0:
           # Force stop if no fresh movement command in 1 second
           self.left.target_rpm = 0
           self.right.target_rpm = 0
   ```

### 🎮 Updated Xbox Controls:
| Control | Function |
|---------|----------|
| Left Stick | Direction (forward/back/turn) |
| **RT (hold)** | **Throttle (required to move)** |
| Right Stick | Camera pan/tilt |
| A | Emergency Stop |
| B | Stop Motors |
| X | Blue LED toggle |
| LT | Cycle NeoPixel modes |
| Y | Play sound |
| LB | Dispense treat |
| RB | Take photo |

### 📊 Performance Improvements:
- **Logging reduced**: ~150 lines/sec → ~1 line/sec (PID status every 1s)
- **Watchdog**: 999999s → 2s timeout
- **Response**: No more system freezes from I/O blocking
- **Motor power**: ~2.5x improvement (RPM 60→100, feedforward 0.6→0.9)

### ⚠️ Important Notes for Next Session:
- System stable after logging fixes
- RT trigger now required to move (safety feature)
- Motors stop within 1-2 seconds if controller crashes
- All services running: API, Xbox, Bark Detection, LEDs, Audio

### 🚀 Current System Status:
- ✅ treatbot.service running
- ✅ Xbox controller subprocess active
- ✅ API server on http://localhost:8000
- ✅ Bark detection classifying audio
- ✅ LED system initialized (165 NeoPixels + Blue LED)
- ✅ Audio system working (USB audio plughw:2)

---

## Session: 2025-12-19 05:45 - 06:20
**Goal:** Fix Audio Recording Double-Trigger and Related Issues
**Status:** ✅ COMPLETE

### ✅ Problems Solved This Session:

#### 1. **Microphone Capture Volume at 0%**
- **Problem**: Recorded audio files had no sound (mean volume -84 dB = silence)
- **Root Cause**: USB mic capture volume was at 0% in ALSA mixer
- **Fix**: `amixer -c 2 sset 'Mic' 100% cap` + `sudo alsactl store`
- **Result**: Recorded audio now has proper levels (mean -53 dB)

#### 2. **LED Modes Not Changing During Recording**
- **Problem**: Fire/chase LED modes not activating during recording
- **Root Cause**: Code was calling `_neopixels.set_mode()` but `_neopixels` is raw NeoPixel object without `set_mode()` method
- **Fix**: Changed all 4 occurrences to use `get_led_controller().set_mode()` instead
- **File**: `api/server.py` (lines 2465-2606)

#### 3. **Recording Double-Trigger (CRITICAL)**
- **Problem**: Recording flow ran twice - beep+fire+record+playback+chase, then immediately beep+fire again
- **Root Cause**: **TWO Xbox controller processes running simultaneously!**
  - Process 1: Spawned by `main_treatbot.py` as morgan user
  - Process 2: Spawned by `xbox-controller.service` as root user
  - Both received button events and made API calls
- **Fix**: Disabled standalone `xbox-controller.service` since treatbot already manages the controller internally
- **Command**: `sudo systemctl disable xbox-controller.service`

#### 4. **Added Server-Side Recording Lock**
- Added `in_progress` flag to `_recording_state` dictionary
- Set `True` at start of recording, `False` when done/error
- Server rejects duplicate `/audio/record/start` requests
- Status endpoint now returns `in_progress` for client-side checking

### 📁 Files Modified:
| File | Changes |
|------|---------|
| `api/server.py` | Fixed LED mode calls, added `in_progress` lock, error handling |
| `xbox_hybrid_controller.py` | Added debug logging, `in_progress` check |
| `services/media/usb_audio.py` | (already had SDL_AUDIODRIVER fix from earlier) |

### 🔧 Services Configuration Change:
| Service | Before | After |
|---------|--------|-------|
| `treatbot.service` | enabled | enabled (spawns Xbox controller internally) |
| `xbox-controller.service` | enabled | **DISABLED** (duplicate - was causing double-triggers) |

### ⚠️ IMPORTANT: Auto-Start Configuration
- **Only `treatbot.service` should be enabled for auto-start**
- Treatbot spawns the Xbox controller as a subprocess via `services/control/xbox_controller.py`
- Having both services enabled causes duplicate button handling

### 🎮 Recording Flow (Now Working):
1. Press Start → beep + fire LED + 2 sec recording
2. Automatic playback of recording
3. Chase LED mode for 10 seconds (waiting for confirmation)
4. Press Start again → saves to `VOICEMP3/talks/custom_*.mp3` + plays "good dog" + rainbow LED

### 🔍 Debugging Technique Used:
```bash
# Found duplicate processes with:
ps aux | grep xbox
# Showed two processes (morgan + root) receiving same button events

# Found parent process:
ps -ef | grep <PID>
# Showed morgan process spawned by main_treatbot.py (PID 48592)
```

### 📝 Next Session Notes:
- Audio recording feature fully working
- Mic volume persisted via `alsactl store`
- Only treatbot.service needed for full functionality

---

## Session: 2025-12-19 04:45 (Updated 05:17)
**Goal:** Remote Demo Setup - Auto-start services + LED upgrades + Audio recording feature
**Status:** ✅ COMPLETE (with post-reboot fixes)

### ✅ Work Completed:

#### 1. **NeoPixel LED System Upgrade (165 LEDs)**
- Updated LED count from 75 → 165 (Adafruit 332 LED/m silicone bead strip)
- Changed GPIO from 12 → 10 (SPI MOSI for Pi 5)
- Added 3 new patterns: `gradient_flow`, `chase`, `fire`
- Fixed API server to handle new patterns in `_safe_animation_loop()`
- **Files Modified:**
  - `config/settings.py` - LED count, brightness, GPIO pin
  - `config/pins.py` - NEOPIXEL = 10
  - `api/server.py` - New pattern handlers
  - `core/hardware/led_controller.py` - New LEDMode enum values + patterns
  - `services/media/led.py` - New pattern methods
  - `xbox_hybrid_controller.py` - Added new modes to led_modes list

#### 2. **Auto-Start Services for Remote Demo**
- Enabled `treatbot.service` for boot auto-start
- Enabled `xbox-controller.service` for boot auto-start
- **On boot flow:**
  - treatbot.service starts → API server + Mode FSM (autonomous)
  - xbox-controller.service starts → waits for controller
  - Xbox input detected → Mode FSM switches to MANUAL mode

#### 3. **Dynamic Audio File Discovery**
- Updated `xbox_hybrid_controller.py` to dynamically scan VOICEMP3/talks and VOICEMP3/songs
- New files added to folders are auto-discovered on controller startup
- Removed hardcoded SOUND_TRACKS list

#### 4. **NEW: Audio Recording Feature (Start Button)**
- **Button 7 (Start/Menu ☰)** records new talk audio
- **Workflow:**
  1. Press Start → Beep, LED fire, record 2 sec via USB mic
  2. Playback automatically
  3. Press Start again within 10s → Save to VOICEMP3/talks/custom_YYYYMMDD_HHMMSS.mp3
  4. No second press → Recording discarded
- **API Endpoints Added:**
  - `POST /audio/record/start` - Start recording
  - `POST /audio/record/confirm` - Save recording
  - `POST /audio/record/cancel` - Discard recording
  - `GET /audio/record/status` - Check pending state

#### 5. **Audio Volume Fix**
- System volume was at 20%, boosted to 95%
- Saved with `sudo alsactl store`

### 📁 Files Modified This Session:
| File | Changes |
|------|---------|
| `config/settings.py` | LED count 75→165, GPIO 10 |
| `config/pins.py` | NEOPIXEL = 10 |
| `api/server.py` | New LED patterns + audio recording endpoints |
| `core/hardware/led_controller.py` | New patterns + LEDMode enum |
| `services/media/led.py` | New pattern methods |
| `xbox_hybrid_controller.py` | Dynamic audio, recording button, new LED modes |
| `tests/hardware/test_led_165.py` | NEW - LED test script |

### 🔧 Services Status:
| Service | Enabled | Auto-Start |
|---------|---------|------------|
| treatbot.service | ✅ | On boot |
| xbox-controller.service | ❌ | DISABLED (duplicate) |

### 🎮 Xbox Controller Button Map (Updated):
| Button | Function |
|--------|----------|
| A | Emergency stop |
| B | Emergency stop (safety) |
| X | Toggle LED mode |
| Y | Play treat sound |
| LB | Dispense treat |
| RB | Take photo |
| LT | Cycle LED modes |
| RT | Speed control (throttle) |
| **Start (☰)** | **Record new talk audio** |
| D-pad Left | Cycle songs |
| D-pad Right | Cycle talks |
| D-pad Down | Play queued audio |
| D-pad Up | Stop audio |

#### 6. **Post-Reboot Fixes (05:07)**
After first reboot test, fixed two issues:

**Audio Not Working:**
- Root cause: pygame wasn't using USB audio device
- Fix: Added `os.environ['SDL_AUDIODRIVER'] = 'alsa'` and `os.environ['AUDIODEV'] = 'plughw:2,0'` before pygame import
- File: `services/media/usb_audio.py`

**Blue LED Flicker (X Button):**
- Root cause: Double-trigger on button press causing ON→OFF sequence
- Fix: Added 500ms cooldown to `toggle_led()` function
- File: `xbox_hybrid_controller.py`

### ⚠️ Notes for Next Session:
- All post-reboot issues resolved
- First recorded audio saved: `custom_20251219_043322.mp3`
- USB mic is card 2 (`hw:2,0`)

---

## Session: 2025-12-17 05:30
**Goal:** Fix Xbox controller DC motor control - restore proper closed-loop PID system
**Status:** ✅ COMPLETE - Major breakthrough achieved!

### 🎯 Primary Mission: Restore Proper Closed-Loop PID Control

**User Request:** "You gave up on closed-loop control. That was the entire point of encoder motors. The encoders ARE working - we proved 1,286 counts in 3 seconds. The PID system should work now. RE-ENABLE CLOSED-LOOP CONTROL"

### ✅ Work Completed:

#### 1. **Critical Hardware Configuration Fixed**
- **Fixed Encoder PPR**: Corrected from 660 to 341 (DFRobot FIT0521: 11 PPR × 34:1 gearbox)
- **Verified Encoder Functionality**: Confirmed 1,286 counts in 3 seconds (exceeds 740-count benchmark)
- **Proper Pin Mapping**: GPIO4/23 (left), GPIO5/6 (right) verified correct

#### 2. **Implemented Proper PID Motor Controller**
- **Created**: `/home/morgan/dogbot/core/hardware/proper_pid_motor_controller.py`
- **Features**:
  - 2000Hz encoder polling (faster than 1190Hz encoder frequency)
  - 50Hz PID control loop
  - Conservative gains: Kp=0.3, Ki=0.02, Kd=0.005
  - Anti-windup protection with integral reset on zero crossing
  - Target ramping for smooth acceleration (300 RPM/second max)
  - Quadrature decoding for direction detection
  - Moving average RPM calculation (10 samples)
  - Feedforward + feedback control

#### 3. **Safety & PWM Configuration**
- **PWM Limits**: 25-50% (safe for 6V motors on 14V system)
- **Watchdog System**: Emergency stop protection
- **Encoder Verification**: Real-time debug logging shows target vs actual RPM

#### 4. **System Integration**
- **Xbox Controller**: `USE_PID_CONTROL = True` enabled
- **Motor Command Bus**: Updated to use ProperPIDMotorController
- **Legacy Compatibility**: Maintained existing API for backward compatibility

---
