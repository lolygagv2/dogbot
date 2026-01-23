# WIM-Z Resume Chat Log

## Session: 2026-01-23 03:00-04:30 EST (Latest)
**Goal:** Multiple bug fixes - Audio, Motors, WebSocket, Camera
**Status:** ✅ Complete

---

### Problems Solved This Session

#### 1. Audio Deadlock Fix
- **Issue:** Audio commands timing out after 5 seconds
- **Cause:** `play_next()` calls `play_file()` which both try to acquire same `threading.Lock()`
- **Fix:** Changed to `threading.RLock()` (reentrant lock) in `services/media/usb_audio.py`

#### 2. Bark Detection Mode Control
- **Issue:** Bark detection running in all modes (should only run in SILENT_GUARDIAN, COACH, MISSION)
- **Fix:** Added start/stop logic in `_on_mode_change` in `main_treatbot.py`

#### 3. Music Player Logic Overhaul
- **Issue:** Music auto-played on startup, prev/next triggered playback
- **Fix:** Complete rewrite of music player state tracking:
  - `_music_playing` and `_playlist_track` separate from general audio
  - `audio_next`/`audio_prev` only change index, don't auto-play
  - `audio_toggle` plays current song if stopped, stops if playing
  - Telemetry: `{"audio": {"playing": bool, "track": "song.mp3"}}`

#### 4. Motor Watchdog Too Aggressive
- **Issue:** "Stale movement command" stopping motors mid-drive at 1.0s threshold
- **Fix:** Increased stale command threshold from 1.0s to 2.5s
- **File:** `core/hardware/proper_pid_motor_controller.py`
- Also reduced PID log interval from 1s to 2s

#### 5. WebSocket Reconnection Handling
- **Issue:** "Cannot write to closing transport" errors, log spam on disconnect
- **Fix:** Updated `services/cloud/relay_client.py`:
  - Check `_ws.closed` before sending
  - Message queue (up to 50) for offline buffering
  - Log suppression (only first failure logged)
  - Better exponential backoff: 1s → 2s → 4s → ... max 30s
  - Queue flush after reconnection

#### 6. Duplicate Log Lines
- **Issue:** Every log message appeared twice
- **Fix:** Added `root_logger.handlers.clear()` before adding handlers in `main_treatbot.py`

#### 7. Camera Color Inversion (BGR/RGB)
- **Issue:** Blue door appeared orange in WebRTC stream
- **Fix:** Updated `services/streaming/video_track.py` - Picamera2 RGB888 actually outputs BGR, removed unnecessary RGB→BGR conversion

---

### Files Modified This Session

| File | Change |
|------|--------|
| `services/media/usb_audio.py` | RLock, music player state tracking |
| `main_treatbot.py` | Bark detection mode control, logging fix |
| `core/hardware/proper_pid_motor_controller.py` | Stale threshold 1.0s→2.5s, log interval 1s→2s |
| `services/cloud/relay_client.py` | Connection state check, message queue, backoff |
| `services/streaming/video_track.py` | BGR/RGB color fix |
| `api/server.py` | Updated audio endpoint messages |
| `core/safety.py` | Raised temp thresholds by 10°C |

---

### Resolved Issues
- **Fan hardware** - Original fan was dead, replaced with new fan - now working ✅
- Temperature thresholds were raised by 10°C (can revert if needed now that fan works)

### Verified Working
- ✅ Camera colors correct in app (BGR/RGB fix confirmed)
- ✅ Fan replaced and working
- ✅ Music controls working (prev/next/toggle)

### Next Session Tasks
1. Monitor motor behavior during driving

---

## Session: 2026-01-22 (Earlier)
**Goal:** Dog Identification System + Voice Command Storage + Two-Way Audio PTT
**Status:** Complete

---

### Work Completed This Session

#### 1. Dog Identification System
Created multi-method dog identification with priority system:
- **ARUCO markers** (100% confidence) - Direct marker detection
- **Color matching** (80% confidence) - HSV-based coat color extraction
- **Persistence tracking** - Based on tracking history
- **Unknown fallback** - Generic "Dog" label

**Files Created:**
- `core/dog_profile_manager.py` - Profile management with color extraction

**Files Modified:**
- `core/dog_tracker.py` - Integrated color-based identification
- `core/ai_controller_3stage_fixed.py` - Added dog_id_methods to results
- `core/state.py` - Added dog_name and id_method to DetectionStatus
- `services/perception/detector.py` - Updated state with identification info
- `services/cloud/relay_client.py` - Added profile fetching from cloud

**Telemetry Format:**
```json
{"detection": {"dog_name": "Bezik", "id_method": "aruco", ...}}
```

#### 2. Voice Command Storage
Custom voice recordings per dog for personalized commands:
- Storage: `/home/morgan/dogbot/voices/{dog_id}/{command}.mp3`
- Priority: Custom voice → Default voice fallback

**Files Created:**
- `services/media/voice_manager.py` - Voice file storage and retrieval

**Files Modified:**
- `api/ws.py` - Added upload_voice, list_voices, delete_voice handlers
- `api/server.py` - Added REST endpoints: /voices/*, /audio/play_command
- `services/media/usb_audio.py` - Added play_command() method
- `main_treatbot.py` - Added cloud command handlers for voice

#### 3. Two-Way Audio Push-to-Talk
Real-time audio communication between app and robot:
- Play audio from app through speaker
- Record from USB mic and send back to app
- Supports AAC, MP3, Opus, WAV formats
- Auto-pauses bark detector during recording

**Files Created:**
- `services/media/push_to_talk.py` - PTT service with arecord/aplay

---

### Commit: 6956f5c3 - feat: Dog identification, voice commands, and push-to-talk audio

---

## Session: 2026-01-22 (Earlier)
**Goal:** WiFi Provisioning + App UI Support (NO command, audio cycling)
**Status:** Complete

---

### Work Completed This Session

#### 1. WiFi Provisioning System (AP Mode)
Created `services/network/` module for first-time WiFi setup:
- `wifi_manager.py` - NetworkManager wrapper (nmcli commands)
- `captive_portal.py` - FastAPI server on port 80
- `wifi_provisioning.py` - Main orchestrator
- `templates/setup.html` - Mobile-friendly setup page

**Flow:** No WiFi → Creates "WIMZ-XXXX" AP → User connects → Configures WiFi → Reboots

#### 2. "NO" Command Support
- Added `LEDMode.WARNING` with yellow/purple flash pattern
- Added `Colors.WARNING` (255, 200, 0) amber

#### 3. Audio Track Cycling
- Added `play_next()` / `play_previous()` to USBAudioService
- Playlist built from `/songs/` folder (12 tracks)

---

### Commit: 37144d88 - feat: WiFi provisioning AP mode + NO command + audio cycling

---
