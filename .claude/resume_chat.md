# WIM-Z Resume Chat Log

## Session: 2026-01-24 02:30-03:00 EST (Robot 02)
**Goal:** Relay heartbeat, WebRTC video fix, Hailo setup, cloud commands
**Status:** ✅ Complete

---

### Work Completed This Session

#### 1. Relay Client Heartbeat & Connection Status
- **Added `robot_connected`** - Sent on WebSocket connect with device_id and version
- **Added `heartbeat`** - Sent every 30 seconds with device_id and timestamp
- **Added `command_ack`** - Sent after each command with success/error status
- **Added `robot_disconnecting`** - Sent on clean shutdown before WebSocket close
- **File:** `services/cloud/relay_client.py`

#### 2. WebRTC Video Fix (Camera without Hailo)
- **Issue:** WebRTC connected but video was null - "Video frame 0: None" forever
- **Cause:** `DetectorService.initialize()` returned False when Hailo unavailable
- **Cause:** `start_detection()` required `ai_initialized`, not just camera
- **Fix:** Camera now independent of AI - detection loop runs for WebRTC even without Hailo
- **Files:** `services/perception/detector.py`, `main_treatbot.py`

#### 3. Hailo Python SDK Installation
- **Issue:** "Hailo platform not available" despite hardware working
- **Cause:** `python3-hailort` 4.20.0 (apt) didn't match `hailort` 4.21.0 (manual)
- **Cause:** venv had `include-system-site-packages = false`
- **Fix:** Enabled system site-packages in venv config
- **Fix:** Installed matching wheel: `hailov2/hailort-4.21.0-cp311-cp311-linux_aarch64.whl`
- **Note:** numpy downgraded 2.2.6 → 1.26.4 (models unaffected - .hef/.ts are independent)

#### 4. Pan/Tilt Log Spam Fix
- **Issue:** "Camera centered" logged every 100ms in idle mode
- **Cause:** Idle handler checked `abs(pan - 90)` but center is actually 100
- **Fix:** Changed to check `abs(pan - self.center_pan)`
- **Fix:** Changed log level from INFO to DEBUG
- **File:** `services/motion/pan_tilt.py`

#### 5. LED Cloud Commands Fix
- **Issue:** Cloud `/led/pattern` failed with "LEDs not initialized" while Xbox worked
- **Cause:** Xbox uses `get_led_controller()` with fallback to DirectNeoPixelController
- **Cause:** Cloud used `get_led_service().set_pattern()` with no fallback
- **Fix:** `/led/pattern` now tries LedService first, then falls back to direct controller
- **File:** `api/server.py`

#### 6. Cloud Command Dispatch
- **Added `dispense_treat`** to command dispatch table (was only `treat`)
- **Added mode change notifications** - app now receives status_update on mode changes
- **File:** `main_treatbot.py`

---

### Commits This Session

| Hash | Description |
|------|-------------|
| `20c88e93` | fix: Add dispense_treat command + mode change notifications |
| `b10e67f5` | fix: LED cloud commands use fallback controller like Xbox |
| `667fb7da` | fix: Stop pan_tilt log spam in idle mode |
| `5015d17a` | fix: Camera capture works without Hailo AI |
| `f7b29f44` | feat: Add heartbeat and connection status messages to relay client |

---

### Key Solutions

1. **Hailo SDK version mismatch**: Runtime 4.21.0 but Python bindings 4.20.0 → Install matching wheel
2. **Camera/AI independence**: Detection loop should run for WebRTC even without AI
3. **Dual LED paths**: Xbox and Cloud used different code paths with different fallback behavior
4. **Idle mode centering bug**: Hardcoded 90 instead of actual center positions (100, 55)

---

### Files Modified
- `services/cloud/relay_client.py` - Heartbeat, connection status messages
- `services/perception/detector.py` - Camera works without AI
- `main_treatbot.py` - Detection startup, cloud commands, mode notifications
- `services/motion/pan_tilt.py` - Fixed log spam and centering logic
- `api/server.py` - LED endpoint fallback
- `env_new/pyvenv.cfg` - Enable system site-packages

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

## Previous Sessions

### Session: 2026-01-23 (Robot 01 → Robot 02 Sync)
**Goal:** Git Sync
**Status:** ✅ Complete
- Synced 87 files from Robot 01 to Robot 02
- Committed as `63ddf137`

