# WIM-Z Resume Chat Log

## Session: 2026-01-22 (Latest)
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

**WebSocket Commands:**
- `{"type": "command", "command": "upload_voice", "name": "sit", "dog_id": "1", "data": "<base64>"}`
- `{"type": "command", "command": "list_voices", "dog_id": "1"}`

#### 3. Two-Way Audio Push-to-Talk
Real-time audio communication between app and robot:
- Play audio from app through speaker
- Record from USB mic and send back to app
- Supports AAC, MP3, Opus, WAV formats
- Auto-pauses bark detector during recording

**Files Created:**
- `services/media/push_to_talk.py` - PTT service with arecord/aplay

**Files Modified:**
- `api/ws.py` - Added audio_message, audio_request handlers
- `api/server.py` - Added REST endpoints: /ptt/*
- `services/cloud/relay_client.py` - Added PTT message handlers
- `main_treatbot.py` - Added ptt_play, ptt_record cloud handlers

**WebSocket Messages:**
- `{"type": "audio_message", "data": "<base64>", "format": "aac"}` - Play audio
- `{"type": "audio_request", "duration": 5, "format": "aac"}` - Record audio

---

### Files Created This Session
- `core/dog_profile_manager.py`
- `services/media/voice_manager.py`
- `services/media/push_to_talk.py`

### Files Modified This Session
- `core/dog_tracker.py`
- `core/ai_controller_3stage_fixed.py`
- `core/state.py`
- `services/perception/detector.py`
- `services/cloud/relay_client.py`
- `services/media/usb_audio.py`
- `api/ws.py`
- `api/server.py`
- `main_treatbot.py`

---

### Key Technical Decisions
1. Used HSV color space for dog coat color classification (lighting-robust)
2. Used arecord/aplay for PTT (reliable USB audio access)
3. Pauses bark detector during PTT recording to free microphone
4. Profile manager is a singleton with thread-safe access

### Next Session Tasks
1. Test dog identification with real dogs
2. Test PTT audio quality
3. Tune color classification thresholds if needed
4. Test voice command uploads from app

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

**Installation:** `sudo ./scripts/install_wifi_provision.sh`

#### 2. "NO" Command Support
- Added `LEDMode.WARNING` with yellow/purple flash pattern
- Added `Colors.WARNING` (255, 200, 0) amber
- `no.mp3` already exists in VOICEMP3/talks/

#### 3. Audio Track Cycling
- Added `play_next()` / `play_previous()` to USBAudioService
- Playlist built from `/songs/` folder (12 tracks)
- Track state: `_current_track`, `_current_index`
- Updated endpoints: `/audio/next`, `/audio/previous`, `/audio/playlist`

#### 4. Audio Telemetry
- Added `audio` object to `/telemetry` endpoint
- Format: `{"audio": {"playing": true, "track": "song.mp3", "looping": false}}`

---

### Commit: 37144d88 - feat: WiFi provisioning AP mode + NO command + audio cycling

---

## Session: 2026-01-21
**Goal:** Cloud Control Architecture - Fix cloud commands and WebRTC
**Status:** Partial - Commands fixed, data receipt bugs remain

---

### Work Completed This Session

#### 1. Cloud Command Handler Fixed (`main_treatbot.py`)
Rewrote `_handle_cloud_command()` to forward commands to local REST API:

| Command | API Endpoint | Body |
|---------|-------------|------|
| `treat` | POST `/treat/dispense` | - |
| `led` | POST `/led/pattern` | `{"pattern": "..."}` |
| `servo` | POST `/servo/pan` / `/servo/tilt` | `{"angle": float}` |
| `audio` | POST `/audio/play` | `{"file": "..."}` |
| `mode` | POST `/mode/set` | `{"mode": "..."}` |
| `motor` | *Ignored* | Goes via WebRTC data channel |

#### 2. WebRTC Data Channel Added (`services/streaming/webrtc.py`)
- Added data channel for low-latency motor control
- Channel name: `"control"`, unreliable mode for speed
- Message format: `{"type": "motor", "left": -1.0 to 1.0, "right": -1.0 to 1.0}`

#### 3. Relay Client Bug Fixed (`services/cloud/relay_client.py`)
**Critical bug found:** Relay sends params in `data` field, not `params` field.
Fixed: `params = data.get('data', {})` instead of `data.get('params', {})`

---

### Important Notes
- Cloud relay message format: `{"type": "command", "command": "xxx", "data": {...}}`
- Params are in `data` field, NOT `params` field

---
