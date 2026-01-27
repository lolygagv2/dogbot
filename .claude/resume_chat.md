# WIM-Z Resume Chat Log

## Session: 2026-01-27 (Build 24 - 5 Fixes)
**Goal:** Build 24 - Custom voice coaching, song uploads, photo HUD, missions, camera control
**Status:** Complete

---

### Work Completed This Session

#### FIX 2: Custom Dog Voice in Coaching Engine
- **Problem:** Coaching engine hardcoded paths to `/VOICEMP3/talks/`, ignoring custom voice recordings
- **Fix:** `_play_audio()` now tries `play_command(command, dog_id)` when session has dog_id, falls back to direct file path
- **Fix:** `_state_greeting()` and `_state_retry_greeting()` check custom voice via `play_command()` before fallback
- **File:** `orchestrators/coaching_engine.py`

#### FIX 3: User Upload Songs
- **Problem:** No way for users to upload custom songs via the app
- **Fix:** `_build_playlist()` now merges system songs + `songs/user/` songs (prefixed as `user/filename.mp3`)
- **Fix:** Added `list_songs()` method with source info (system vs user)
- **Fix:** Created `/VOICEMP3/songs/user/` directory
- **API:** `GET /music/list`, `POST /music/upload`, `DELETE /music/user/{filename}`
- **Cloud:** `upload_song`, `delete_song`, `list_songs`
- **Files:** `services/media/usb_audio.py`, `api/server.py`, `main_treatbot.py`

#### FIX 4: Photo HUD Overlay
- **Problem:** `take_photo` used `/camera/snapshot` (no HUD overlay)
- **Fix:** Changed to `POST /camera/photo_hud` which draws bounding boxes, names, timestamps
- **Fix:** HUD defaults to on, app can send `with_hud: false`; base64 data returned directly (no file read)
- **File:** `main_treatbot.py`

#### FIX 5: Missions (App-Triggered)
- **Problem:** MissionEngine existed but no mission definitions or cloud routing
- **Fix:** Created `missions/sit.json` (2 stages: wait_for_dog, wait_for_sit)
- **Fix:** Created `missions/come_and_sit.json` (4 stages: call_dog, wait_for_dog, command_sit, wait_for_sit)
- **Cloud:** `start_mission`, `cancel_mission`, `mission_status`, `list_missions`
- **Relay events:** `mission_progress`, `mission_complete`, `mission_stopped` (matches app listener names)
- **Bus forwarding:** Internal `mission.started`/`.completed`/`.stopped` events now forwarded to relay
- **Files:** `missions/sit.json`, `missions/come_and_sit.json`, `main_treatbot.py`

#### FIX 8: Camera Manual Control
- **Problem:** App drive screen couldn't suppress auto-tracking/scanning
- **Fix:** Added `_manual_camera_control` flag to PanTiltService
- **Fix:** `set_manual_camera(active)` method; `_control_loop()` skips auto-tracking when flag is active
- **API:** `POST /camera/manual_control`
- **Cloud:** `camera_control` command
- **Files:** `services/motion/pan_tilt.py`, `api/server.py`, `main_treatbot.py`

#### Mission Event Type Fix (mid-session correction)
- **Problem:** App listens for `mission_progress`, `mission_complete`, `mission_stopped` but robot emitted `mission_update`
- **Fix:** Changed cloud command replies and added bus event forwarding to use correct event types

---

### Files Modified
| File | Changes |
|------|---------|
| `orchestrators/coaching_engine.py` | `_play_audio()`, `_state_greeting()`, `_state_retry_greeting()` use custom voice via `play_command()` |
| `services/media/usb_audio.py` | `_build_playlist()` merges system+user songs, added `list_songs()` |
| `services/motion/pan_tilt.py` | Added `_manual_camera_control` flag, `set_manual_camera()`, control loop check |
| `api/server.py` | Added `POST /camera/manual_control`, `GET /music/list`, `POST /music/upload`, `DELETE /music/user/{filename}` |
| `main_treatbot.py` | Photo HUD, mission commands, song commands, camera control, mission bus forwarding |
| `missions/sit.json` | New mission definition |
| `missions/come_and_sit.json` | New mission definition |

---

### New API Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| POST | `/camera/manual_control` | Toggle auto-tracking suppression |
| GET | `/music/list` | List system + user songs |
| POST | `/music/upload` | Upload song (base64) |
| DELETE | `/music/user/{filename}` | Delete user song |

### New Cloud Commands
| Command | Params | Relay Event |
|---------|--------|-------------|
| `upload_song` | `filename`, `data` | `music_update` |
| `delete_song` | `filename` | `music_update` |
| `list_songs` | none | `music_update` |
| `start_mission` | `mission`, `dog_id` | `mission_progress` |
| `cancel_mission` | none | `mission_stopped` |
| `mission_status` | none | `mission_progress` |
| `list_missions` | none | `mission_progress` |
| `camera_control` | `active` | none (fire-and-forget) |

---

### Commit
`c719251b` - feat: Build 24 - custom voice coaching, song uploads, photo HUD, missions, camera control

---

### Remaining Uncommitted Changes (pre-existing)
- `services/control/xbox_controller.py` -- modified from previous session
- `=1.6.0` -- pip artifact in project root (safe to delete)
- Various snapshot JPGs in `captures/`

---

### Next Session Tasks
1. Test custom voice playback in coaching mode with a dog that has custom recordings
2. Test song upload/delete/list from app
3. Test photo HUD -- verify bounding boxes on returned image
4. Test `start_mission sit` from app -- verify mission engine starts
5. Test camera_control on/off from drive screen
6. Consider committing `xbox_controller.py` changes
7. Clean up `=1.6.0` pip artifact file

---

### Important Notes/Warnings
- **Mission event types:** App expects `mission_progress`, `mission_complete`, `mission_stopped` -- robot now emits these correctly
- **`xbox_controller.py` still uncommitted** from previous session
- **`=1.6.0` file** in project root is a pip artifact -- safe to delete
- **User songs directory:** `/home/morgan/dogbot/VOICEMP3/songs/user/` created and ready

---

## Session: 2026-01-27 (Build 23 - WebRTC + Safety Fix)
**Goal:** Fix WebRTC session accumulation, mode revert on disconnect, startup error audio
**Status:** Complete

---

### Work Completed This Session

#### FIX 1: Safety Monitor Startup Grace Period
- **Problem:** Error audio (`Wimz_errorlogs.mp3`) playing on every restart — CPU at 97-100% triggers CRITICAL alert after 45s grace period expires while Pi 5 is still loading Hailo model
- **Fix:** Extended CPU and memory startup grace period from 45s to 90s in `core/safety.py`

#### FIX 2: WebRTC Mode Revert on Disconnect (CRITICAL)
- **Problem:** Mode reverts to IDLE when ANY WebRTC session closes, even if app is still connected (e.g., during session replacement)
- **Root Cause:** `_handle_webrtc_close()` in `relay_client.py` called `_handle_app_disconnect()` which set mode to IDLE from MANUAL
- **Fix:** Removed `_handle_app_disconnect()` entirely. Mode now ONLY changes via explicit `set_mode` command from the app. WebRTC session lifecycle is independent of mode state.

#### FIX 3: WebRTC Connection Tracking Logs
- **Added:** `[WEBRTC]` prefixed logs throughout `webrtc.py` showing active connections after every session create/cleanup
- **Covers:** `create_peer_connection`, `create_offer`, `_cleanup_connection`, connection state changes, ICE state changes

---

### Files Modified
| File | Changes |
|------|---------|
| `core/safety.py` | CPU/memory startup grace: 45s → 90s |
| `services/cloud/relay_client.py` | Removed `_handle_app_disconnect()`, WebRTC close no longer reverts mode |
| `services/streaming/webrtc.py` | Added `[WEBRTC]` connection tracking logs, "mode unchanged" notes |

---

### Commit
`413e50c6` - fix: WebRTC single session enforcement, no mode revert on disconnect, safety grace period

---

### Remaining Uncommitted Changes (pre-existing, not from this session)
- `services/control/xbox_controller.py` — modified (from previous session)
- `services/streaming/webrtc.py` — had pre-existing uncommitted motor API fallback changes (included in this commit)
- `=1.6.0` — pip artifact in project root (safe to delete)
- Various snapshot JPGs in `captures/`

---

### Next Session Tasks
1. Test WebRTC reconnection — verify mode persists when app reconnects
2. Test startup — verify no error audio on clean boot (90s grace period)
3. Monitor `[WEBRTC] Active connections:` logs to confirm single-session enforcement
4. Consider committing remaining `xbox_controller.py` changes
5. Clean up `=1.6.0` pip artifact file

---

### Important Notes/Warnings
- **Mode no longer auto-reverts on WebRTC disconnect** — app must explicitly send `set_mode` to change mode
- **Xbox controller disconnect still reverts MANUAL→IDLE** (in `main_treatbot.py:1180-1185`) — this is correct behavior
- **`xbox_controller.py` still uncommitted** from previous session
- **`=1.6.0` file** in project root is a pip artifact — safe to delete

---

## Session: 2026-01-26 (Robot 02 - 3 Critical Fixes)
**Goal:** PTT async playback, manual mode timeout, startup command guard
**Status:** ✅ Complete

---

### Work Completed This Session

#### FIX 1: PTT Timeout (CRITICAL)
- **Problem:** `/ptt/play` blocked synchronously waiting for audio to finish; HTTP timeout at 5s killed requests for clips >5s
- **Root Cause:** `play_audio()` in `push_to_talk.py` called `usb_audio.wait_for_completion(timeout=30)` while holding lock
- **Fix:** Moved conversion + playback into a background daemon thread (`PTTPlayback`). Method returns `{"success": True, "message": "Audio playback started"}` immediately. `_playing` flag cleared in thread's `finally` block.

#### FIX 2: Manual Mode Timeout
- **Problem:** Manual mode timeout too short for app usage
- **Fix:** Changed `manual_timeout` from `120.0` (2 min) to `300.0` (5 min) in `orchestrators/mode_fsm.py`
- **Note:** Activity reset already works — `_handle_cloud_command` publishes `manual_input_detected` on every app command in MANUAL mode

#### FIX 3: Stale Commands on Startup
- **Problem:** On restart, relay server sends buffered commands from previous session (music plays, treats dispense)
- **Fix:** Added 5-second startup grace period in `_handle_cloud_command` — any cloud commands arriving within `_startup_grace_period` are logged and ignored
- **Additional:** Clear relay client `_message_queue` and event bus history on startup

---

### Files Modified
| File | Changes |
|------|---------|
| `services/media/push_to_talk.py` | `play_audio()` now runs in background thread, returns immediately |
| `orchestrators/mode_fsm.py` | `manual_timeout`: 120 → 300 seconds |
| `main_treatbot.py` | Added startup grace period (5s), clear relay queue + event bus history on boot |

---

### Commit
`685b7e22` - fix: PTT async playback, manual mode 5min timeout, startup command guard

---

### Remaining Uncommitted Changes (pre-existing, not from this session)
- `services/control/xbox_controller.py` — modified (from previous session)
- `services/streaming/webrtc.py` — modified (from previous session)
- `=1.6.0` — pip artifact in project root (safe to delete)
- Various snapshot JPGs in `captures/`

---

### Next Session Tasks
1. Test PTT play from app — verify audio plays without timeout errors
2. Test manual mode persistence — verify 5-minute timeout works with app
3. Test restart behavior — verify stale commands are ignored
4. Consider committing remaining `xbox_controller.py` and `webrtc.py` changes
5. Clean up `=1.6.0` pip artifact file

---

### Important Notes/Warnings
- **2 source files still uncommitted** from previous session: `xbox_controller.py`, `webrtc.py`
- **`=1.6.0` file** in project root is a pip artifact — safe to delete
- **PTT `_playing` flag** is now managed by background thread — if thread crashes, flag stays True until next call resets it

---

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
