# WIM-Z Resume Chat Log

## Session: 2026-01-26 (Robot 02 - Bug Fix Marathon)
**Goal:** Critical bug fixes for app integration, audio, mode management, PTT two-way audio
**Status:** ✅ Complete

---

### Work Completed This Session

#### Bug 1: Random Sounds Playing in IDLE Mode
- **Problem:** Robot playing "No", "Good dog", "Do you want a treat" when idle
- **Root Cause:** `RewardLogic._on_audio_event()` processed bark events in ALL modes
- **Fix:** Added mode check in `orchestrators/reward_logic.py` to skip IDLE/MANUAL modes
- **Additional:** Added call stack tracing to `usb_audio.py` `play_file()` for future debugging

#### Bug 2: Mode Reverts to IDLE from MANUAL
- **Problem:** App sets Manual mode, reverts to Idle after 120s timeout
- **Root Cause:** `ModeFSM._evaluate_manual_transitions()` has 120s timeout, only reset by Xbox controller
- **Fix:** Cloud command handler in `main_treatbot.py` now publishes `manual_input_detected` event on every cloud command while in MANUAL mode, resetting the timeout
- **Import Fix:** Added `publish_system_event` to imports from `core.bus` (was causing NameError blocking ALL commands)

#### Bug 3: Volume Command Handler
- **Fix:** Added `set_volume` (0.0-1.0) and `audio_volume` (0-100) handlers in cloud command handler

#### Bug 4: Photo Handler Verification
- **Verified:** `/camera/snapshot` and `/camera/photo_hud` endpoints work correctly
- **Note:** No `/command` REST endpoint exists - use specific endpoints

#### Missing Cloud Command Handlers
- Added `audio_volume` (app sends 0-100 scale)
- Added `take_photo` → `/camera/snapshot`
- Added `call_dog` → plays `{name}_come.mp3` or fallback `dog_0.mp3`

#### Photo Send-Back to App (CRITICAL)
- **Problem:** Photo captured but never sent back to app via relay
- **Fix:** After snapshot capture, base64 encodes JPEG and sends via `relay_client.send_event('photo', {...})`

#### Voice Upload 422 Fix
- **Problem:** `/voices/upload` returned 422 because `dog_id` was required but app didn't send it
- **Fix:** Made `dog_id` optional (defaults to `"default"`) in Pydantic model and cloud handler

#### WAV Audio Format Support
- **PTT play:** Already handled WAV correctly (skips ffmpeg, plays directly)
- **Voice upload:** Fixed `voice_manager.py` to auto-detect format from file header (RIFF=WAV, ID3=MP3) and convert non-MP3 to MP3 via ffmpeg for consistent storage

#### PTT Playback "Device Busy" Fix (CRITICAL)
- **Problem:** `aplay` failed because USBAudio service held the device via pygame
- **Fix:** Replaced direct `aplay` calls with routing through `USBAudioService.play_file()` via pygame. PTT stops current audio first (priority playback)

#### PTT Listen Feature (Two-Way Audio)
- **Problem:** `record_audio()` blocked the asyncio event loop for 5+ seconds
- **Fix:** Wrapped in `run_in_executor()` in relay client handler
- **Fix:** Changed `hw:2,0` to `plughw:2,0` for proper sample rate conversion (16kHz)
- **Verified:** Full pipeline works: record → ffmpeg encode → base64 → send

#### Warning LED Pattern
- Added `warning` pattern (orange/red flash) to LED service
- Fixed infinite loop: auto-stops after 3 seconds

#### Fire LED Race Condition (LOW PRIORITY)
- **Problem:** Two pattern threads could write to pixels simultaneously
- **Root Cause:** `_stop_current_pattern()` join timeout (1s) too short, then `_stop_pattern.clear()` re-enabled old thread
- **Fix:** Increased join timeout to 2s with retry

---

### Files Modified
| File | Changes |
|------|---------|
| `main_treatbot.py` | Added import, cloud command handlers (volume, photo, call_dog, upload_voice), manual timeout reset, photo send-back |
| `orchestrators/reward_logic.py` | Added IDLE/MANUAL mode check on audio events |
| `services/media/usb_audio.py` | Added call stack tracing to play_file logging |
| `services/media/push_to_talk.py` | Fixed `hw:` → `plughw:` for recording, replaced aplay with USBAudio routing |
| `services/media/voice_manager.py` | Auto-detect audio format, convert WAV→MP3 for storage |
| `services/media/led.py` | Added `warning` pattern (3s auto-stop), fixed pattern thread race condition |
| `services/cloud/relay_client.py` | Fixed async blocking in audio_request handler with run_in_executor |
| `api/server.py` | Made `dog_id` optional in VoiceUploadRequest |

---

### System Status (Robot 02)
| Component | Status |
|-----------|--------|
| LEDs | ✅ Working (warning pattern added) |
| Audio | ✅ Fixed (no random sounds in IDLE) |
| PTT Play | ✅ Fixed (routed through pygame, no device busy) |
| PTT Listen | ✅ Fixed (async, correct sample rate) |
| Mode FSM | ✅ Fixed (manual mode persists with app) |
| Photo | ✅ Fixed (sends back to app via relay) |
| Voice Upload | ✅ Fixed (accepts WAV, defaults dog_id) |
| Volume | ✅ Added (both 0-1 and 0-100 scales) |

---

### Next Session Tasks
1. Test full app flow end-to-end (PTT talk + listen, photo, mode changes)
2. Monitor logs for any remaining random audio in IDLE mode
3. App-side: Fix iOS audio recording (empty M4A files - 28 bytes header only)
4. App-side: Handle incoming `photo` event and display in UI
5. App-side: Add remote debug logging (RemoteLogger)
6. Consider committing all changes

---

### Important Notes/Warnings
- **Uncommitted changes:** 8+ source files modified - needs commit
- **App audio recording broken:** iOS sends 28-byte M4A headers with no audio content
- **`=1.6.0` file** in project root is a pip artifact - safe to delete
- **Audio stack trace logging** is verbose - may want to reduce to DEBUG level after debugging

---

## Session: 2026-01-24 03:30-04:30 EST (Robot 02)
**Goal:** LED initialization fix, continued from previous session
**Status:** ✅ Complete

---

### Work Completed This Session

#### 1. LED Initialization Fix (Main Issue)
- **Problem:** `NeoPixel initialization failed: No module named 'adafruit_raspberry_pi5_neopixel_write'`
- **Root Cause:** Robot 02 was missing the Pi5-specific NeoPixel library
- **Fix:** Updated Adafruit-Blinka from GitHub (8.62.0 → 8.69.0)
- **Result:** Automatically pulled in `Adafruit-Blinka-Raspberry-Pi5-Neopixel-1.0.0rc2`
- **Verification:** LEDs now initialize correctly:
  ```
  Blue LED initialized on GPIO25
  NeoPixel ring initialized: 165 LEDs on GPIO10
  LedService - INFO - LED pattern started: idle
  ```

---

### Key Solutions

1. **Pi5 NeoPixel Library Missing**: The `neopixel_write.py` module tries to import `adafruit_raspberry_pi5_neopixel_write` on Pi5 boards. This package wasn't installed on Robot 02. Updating Adafruit-Blinka from GitHub automatically resolved the dependency.

---

### Package Changes
```
Adafruit-Blinka: 8.62.0 → 8.69.0
+ Adafruit-Blinka-Raspberry-Pi5-Neopixel-1.0.0rc2 (new)
```

---

### Files Modified
- No code changes this session - package installation only

---

### System Status (Robot 02)
| Component | Status |
|-----------|--------|
| LEDs | ✅ Working (165 NeoPixels + Blue LED) |
| Hailo AI | ✅ Working (4.21.0) |
| Camera | ✅ Working (detection FPS: 15.0) |
| Motors | ✅ Ready |
| Audio | ✅ Working |
| Relay | ✅ Connected |
| WebRTC | ✅ Ready |

---

### Next Session Tasks
1. Test app connection with video streaming
2. Verify relay server handles new message types (robot_connected, heartbeat, command_ack)
3. Test dispense_treat command from app
4. Verify mode change notifications reach app

---

### Important Notes/Warnings
- **numpy downgraded** to 1.26.4 - OpenCV may warn but works fine
- **Hailo wheel location**: `/home/morgan/dogbot/hailov2/hailort-4.21.0-cp311-cp311-linux_aarch64.whl`
- **Robot 02 differences**: treatbot2.yaml has different motor calibration, PID disabled

---

## Previous Session: 2026-01-24 02:30-03:00 EST (Robot 02)
**Goal:** Relay heartbeat, WebRTC video fix, Hailo setup, cloud commands
**Status:** ✅ Complete

### Work Completed
1. **Relay Client Heartbeat** - Added robot_connected, heartbeat, command_ack, robot_disconnecting
2. **WebRTC Video Fix** - Camera now independent of AI, detection runs for WebRTC without Hailo
3. **Hailo SDK Fix** - Installed matching wheel (4.21.0), enabled system site-packages
4. **Pan/Tilt Log Spam** - Fixed idle mode centering check, changed to DEBUG level
5. **LED Cloud Commands** - Added fallback path to /led/pattern endpoint
6. **Cloud Dispatch** - Added dispense_treat command and mode change notifications

### Commits
| Hash | Description |
|------|-------------|
| `20c88e93` | fix: Add dispense_treat command + mode change notifications |
| `b10e67f5` | fix: LED cloud commands use fallback controller like Xbox |
| `667fb7da` | fix: Stop pan_tilt log spam in idle mode |
| `5015d17a` | fix: Camera capture works without Hailo AI |
| `f7b29f44` | feat: Add heartbeat and connection status messages to relay client |

---

## Previous Session: 2026-01-23 (Robot 01 → Robot 02 Sync)
**Goal:** Git Sync
**Status:** ✅ Complete
- Synced 87 files from Robot 01 to Robot 02
- Committed as `63ddf137`
