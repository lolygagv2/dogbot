# WIM-Z Resume Chat Log

## Session: 2026-02-02 - Build 41 Documentation & Features
**Goal:** Update documentation, add servo calibration, implement AWS SNS push notifications
**Status:** COMPLETE

---

### Problems Solved This Session

| # | Problem | Solution | Files Modified |
|---|---------|----------|----------------|
| 1 | Servo camera offset (20° right of center) | Adjusted center_pan from 90° to 110° | `services/motion/pan_tilt.py:65-76,88-94` |
| 2 | No push notification system | Created AWS SNS notification service | `services/cloud/notification_service.py` (NEW) |
| 3 | WebSocket "dropped" note confusing | Clarified: Direct LAN WebSocket deferred, relay-based WS working | `.claude/product_roadmap.md` |
| 4 | Documentation outdated | Updated all docs with Build 40 review, unknowns marked | Multiple .claude/*.md files |

---

### Key Code Changes Made

#### 1. Servo Camera Calibration
**File:** `services/motion/pan_tilt.py`
- Changed `center_pan` from 90° to 110° (20° left offset for physical mount)
- Changed `current_pan` startup from 90° to 110°
- Updated scan pattern: 70°, 110°, 150° (instead of 50°, 90°, 130°)

#### 2. AWS SNS Push Notifications
**File:** `services/cloud/notification_service.py` (NEW - 318 lines)
- SMS sending via AWS SNS
- Subscriber management (add/remove/update)
- Event types: mission_complete, bark_alert, low_battery, weekly_summary
- Async broadcast to all subscribers

**API Endpoints Added:** `api/server.py`
- `GET /notifications/health` - Service health check
- `GET/POST /notifications/subscribers` - List/add subscribers
- `GET/PUT/DELETE /notifications/subscribers/{id}` - Manage subscriber
- `POST /notifications/send` - Direct SMS
- `POST /notifications/test` - Test notification

#### 3. Documentation Updates
- `.claude/development_todos.md` - Build 40 review, unknowns, AWS setup guide
- `.claude/product_roadmap.md` - Build history, clarified WebSocket
- `.claude/WIM-Z_Project_Directory_Structure.md` - Added notification_service.py
- `.claude/hardware_specs.md` - Updated software status

---

### User-Confirmed Status (from unknowns)

| Item | Status |
|------|--------|
| Relay forwarding events | ✅ Working |
| Video overlay AI confidence | ✅ Working |
| MP3 upload/download | ✅ Working |
| Bark filter | ✅ Working |
| Pose thresholds | ✅ Working |
| Coach mode end-to-end | ✅ Working |
| Silent Guardian flow | ✅ Working |
| Treat dispenser | ✅ Working |
| Audio playback | ✅ Working |
| Mission scheduler | ✅ Tested |
| Servo tracking checkbox | ❓ Unknown |
| Servo calibration | ❓ Unknown (just adjusted) |
| Weekly summary | ❌ Not tested |

---

### Commit
`391cb00e` - feat: Build 41 - servo calibration, AWS SNS notifications, doc updates

---

### Next Steps
1. Install boto3: `pip install boto3`
2. Configure AWS credentials: `aws configure`
3. Test notification health: `curl http://localhost:8000/notifications/health`
4. Add subscriber and test SMS
5. Test weekly summary system
6. Verify servo calibration (20° left adjustment)

---

## Session: 2026-02-02 - Build 40 Implementation
**Goal:** Fix Build 39 test failures - mission field names, AI display, servo tracking, coach events
**Status:** COMPLETE

---

### Problems Solved This Session

| # | Problem | Root Cause | Solution | Files Modified |
|---|---------|------------|----------|----------------|
| 1 | Mission stuck on "Initializing" | Field name mismatch | Changed `mission_name`→`mission_id`, `stage`→`stage_number` | `orchestrators/mission_engine.py`, `main_treatbot.py` |
| 2 | AI confidence not showing ("sit 34%") | `update_dog_behavior()` never called | Added bridge call in detector | `services/perception/detector.py:778` |
| 3 | Servo tracking checkbox broken | `tracking_enabled=False` by default | Auto-enable in COACH mode | `services/motion/pan_tilt.py:228` |
| 4 | MP3 upload 413 error | Relay sends relative URL | Construct full URL, save to dog folder | `main_treatbot.py:1137-1190` |
| 5 | No coach progress events | Events not implemented | Added coach_progress/coach_reward | `orchestrators/coaching_engine.py` |
| 6 | /missions 404 | Endpoint missing | Added GET /missions endpoint | `api/server.py:955-972` |

---

### Key Code Changes Made

#### 1. Mission Progress Field Names (P0-R1)
**Files:** `orchestrators/mission_engine.py`, `main_treatbot.py`
- Fixed 7 `mission_progress` events + 1 `mission_complete` event
- `mission_name` → `mission_id`
- `stage` → `stage_number`
- Added `action` field to all events

#### 2. AI Confidence Display (P0-R2)
**File:** `services/perception/detector.py:778`
- Added call to `update_dog_behavior()` after behavior detection
- Bridges behavior data to dog_tracker for video overlay display

#### 3. Servo Tracking Auto-Enable (P0-R3)
**File:** `services/motion/pan_tilt.py:228`
- Auto-enables tracking when entering COACH mode
- Added debug logging to `set_tracking_enabled` handler in main_treatbot.py

#### 4. Download Song URL Fix (P1-R4)
**File:** `main_treatbot.py:1137-1190`
- Constructs full URL from relay's relative path (`https://api.wimzai.com{url}`)
- Saves to dog-specific folder (`VOICEMP3/songs/{dog_id}/`)
- Extracts params from `data` field

#### 5. Coach Progress Events (P1-R5)
**File:** `orchestrators/coaching_engine.py`
- Added `get_relay_client` import
- `coach_progress` events: greeting, command, watching (periodic ~500ms)
- `coach_reward` event on success

#### 6. Missions REST Endpoint (P2-R6)
**File:** `api/server.py:955-972`
- Added `GET /missions` endpoint
- Returns mission catalog for app browser

---

### Commit
`79dc7b8c` - fix: Build 40 - mission field names, AI display, tracking, coach events

---

### Testing Checklist for Build 40
1. [ ] Start mission → Relay logs show `status=` and `stage_number=` (not old field names)
2. [ ] Enter coach mode → Log shows "Auto-enabling tracking for COACH mode"
3. [ ] Video overlay → Shows "sit 34%" confidence labels
4. [ ] MP3 upload → Robot receives `download_song` and fetches file
5. [ ] `GET /missions` → Returns mission catalog

---

### Documentation Created
- `.claude/POST39ROBO.md` - Build 39 test analysis
- `.claude/BUILD40_ROBOT_CLAUDE.md` - Build 40 instructions (from user)
- `.claude/BUILD40_CURRENT_STATE.md` - Implementation status

---

## Session: 2026-02-01 - Build 38 Critical Fixes
**Goal:** Fix critical Build 37 failures (video overlay, bounding boxes, dog ID, servo tracking, MP3 upload)
**Status:** COMPLETE

---

### Problems Solved This Session

| # | Problem | Solution | Files Modified |
|---|---------|----------|----------------|
| 1 | Video overlay shows "IDLE" during missions | Use thread-safe `get_mission_status()` | `services/streaming/video_track.py:285-316` |
| 2 | AI bounding boxes not showing on WebRTC | Store unidentified dogs with generic IDs | `core/dog_tracker.py:185-198` |
| 3 | "Everything is Elsa" dog ID bug | Raised YOLO conf 0.5→0.6, display "Dog" for unknown | `core/ai_controller_3stage_fixed.py:555`, `core/dog_tracker.py:387-391` |
| 4 | Servo tracking disabled | Implemented gentle nudge tracking mode | `services/motion/pan_tilt.py:38,211-261` |
| 5 | MP3 upload crashes WebSocket (5MB) | New `download_song` command via HTTP URL | `api/ws.py:318-603`, `main_treatbot.py:1137-1191` |

---

### Key Code Changes Made

#### 1. Video Overlay Race Condition Fix
**File:** `services/streaming/video_track.py:285-316`
- Changed from direct `active_session` access to thread-safe `get_mission_status()` call
- Now correctly shows mission state instead of "IDLE"

#### 2. Bounding Box Fix
**File:** `core/dog_tracker.py:185-198`
- Unidentified dogs now stored with generic IDs (`dog_0`, `dog_1`, etc.)
- Allows bounding boxes to be drawn even without ArUco identification

#### 3. Dog Identification Fix
**Files:** `core/ai_controller_3stage_fixed.py:555`, `core/dog_tracker.py:387-391`
- Raised YOLO confidence threshold from 0.5 to 0.6
- Display "Dog" instead of wrong specific names for unidentified detections

#### 4. Nudge Servo Tracking
**File:** `services/motion/pan_tilt.py`
- Added `_edge_stable_since` initialization (line 38)
- Replaced disabled `_handle_coach_mode()` with nudge tracking (lines 211-261)
- Edge zone: outer 25% of frame
- Stability delay: 500ms before moving
- Max speed: 2 degrees/second
- Safe limits: pan 55-145°, tilt 25-85°
- Works in COACH and MISSION modes

#### 5. MP3 Download via HTTP
**Files:** `api/ws.py`, `main_treatbot.py`
- New `download_song` command: `{"command": "download_song", "url": "https://...", "filename": "my_song.mp3"}`
- Robot downloads file directly via HTTP (60s timeout, 20MB max)
- Avoids WebSocket message size crash

---

### New API Command
```json
{"command": "download_song", "url": "https://example.com/song.mp3", "filename": "my_song.mp3"}
```
Response: `{"type": "download_complete", "filename": "...", "success": true, "size_bytes": ...}`

---

### Commit
`31199624` - fix: Build 38 - video overlay, bounding boxes, nudge tracking, MP3 download

---

### Testing Checklist for Build 38
1. Start mission → video overlay should show mission state (not "IDLE")
2. Dogs without ArUco should have bounding boxes with "Dog" label
3. Wrong dog names should no longer appear (no more "Everything is Elsa")
4. Camera should gently nudge when dog at frame edge for 500ms+
5. Test `download_song` command with real URL

---

### Unresolved Issues
- P1-R6 (Schedule persistence) - Not addressed, existing code needs verification

---

## Session: 2026-01-31 - Build 36 Bug Fixes
**Goal:** Fix critical issues from Build 35 testing - mission name mismatch, video lag, slow AI detection
**Status:** ✅ Complete

---

### Problems Solved This Session

| # | Problem | Solution | Files Modified |
|---|---------|----------|----------------|
| 1 | Mission name mismatch (stay_training not found) | Added MISSION_ALIASES dict for app->robot name mapping | `orchestrators/mission_engine.py` |
| 2 | Video lag 10-30 seconds | Added frame freshness check, skip stale frames >500ms | `services/perception/detector.py`, `services/streaming/video_track.py` |
| 3 | AI detection too slow (3s + 66%) | Reduced to 1.5s + 50% presence requirement | `orchestrators/mission_engine.py`, `orchestrators/coaching_engine.py` |
| 4 | No "Dog" label when ArUco unavailable | Default to "Dog" instead of raw ID | `services/streaming/video_track.py` |

---

### Key Code Changes Made

#### 1. Mission Name Aliases
**File:** `orchestrators/mission_engine.py`
- Added `MISSION_ALIASES` dict after `POSE_TO_TRICK`
- Maps app names to robot missions: `stay_training`->`sit_training`, `Basic Sit`->`sit_training`, etc.
- Updated `start_mission()` to check aliases if direct name not found
- Returns helpful error with available missions list

#### 2. Frame Freshness Check (Video Lag Fix)
**File:** `services/perception/detector.py`
- Added `get_last_frame_with_timestamp()` method returning (frame, timestamp) tuple

**File:** `services/streaming/video_track.py`
- Added `MAX_FRAME_AGE_SEC = 0.5` constant
- Modified `recv()` to use `get_last_frame_with_timestamp()`
- Skips frames older than 500ms to prevent video lag
- Shows "Buffering..." message for stale frames (yellow text)

#### 3. Faster Dog Detection
**Files:** `orchestrators/mission_engine.py`, `orchestrators/coaching_engine.py`
- Reduced `DETECTION_TIME_SEC` from 3.0 to 1.5 seconds
- Reduced `PRESENCE_RATIO_MIN` from 0.66 (66%) to 0.50 (50%)

#### 4. Default Dog Label
**File:** `services/streaming/video_track.py`
- Changed `dog_name = dog_data.get('name', dog_id)` to `dog_name = dog_data.get('name') or 'Dog'`
- Ensures "Dog" appears on bounding box even without ArUco

---

### Issues Documented for APP/RELAY Teams

| Issue | Owner | Notes |
|-------|-------|-------|
| MP3 upload never reaches robot | RELAY | No upload logs at 18:14 - request not proxied |
| Schedule API missing name | APP | App sends `"name": ""` but robot requires non-empty |
| Mission name "Basic Sit" | APP | Should use `sit_training` or fetch from robot API |
| Coach mode stops on screen lock | APP | App may be sending mode change command |

---

### Testing Checklist for Build 36
1. Start mission via app with "stay_training" -> should map to sit_training
2. Video feed should have <500ms lag (no 10-30s delay)
3. Dog detection should trigger within ~2 seconds (not 60+ seconds)
4. Bounding box should show "Dog" if ArUco not visible

---

## Session: 2026-01-31 - Schedule API Fix (Build 35)
**Goal:** Update schedule CRUD API to match app format with dog_id, schedule_id, and type fields
**Status:** ✅ Complete

---

### Problems Solved This Session

| # | Problem | Solution | Files Modified |
|---|---------|----------|----------------|
| 1 | Schedule API missing dog_id | Added dog_id as required field | `core/schedule_manager.py`, `api/server.py` |
| 2 | Schedule API missing type field | Added type: "once" / "daily" / "weekly" | `core/schedule_manager.py`, `api/server.py` |
| 3 | Response uses id instead of schedule_id | Renamed id to schedule_id | `core/schedule_manager.py` |
| 4 | Scheduler doesn't handle type logic | Updated _should_start_mission() for type behavior | `core/mission_scheduler.py` |
| 5 | "once" schedules keep running | Added auto-disable after first execution | `core/mission_scheduler.py` |

---

### Key Code Changes Made

#### 1. Schedule Manager Updates
**File:** `core/schedule_manager.py`
- Added `dog_id` as required field for schedule creation
- Added `type` field: `"once"`, `"daily"`, or `"weekly"` (default: `"daily"`)
- Renamed `id` to `schedule_id` in JSON structure
- For `"weekly"` type, `days_of_week` is required
- Added backward compatibility for legacy schedules with `id` field
- Added `disable_schedule()` method for auto-disabling "once" schedules
- Added `list_schedules_for_dog()` helper method

#### 2. API Server Updates
**File:** `api/server.py`
- Added `dog_id: str` (required) to `ScheduleCreateRequest`
- Added `type: str = "daily"` to `ScheduleCreateRequest`
- Changed `days_of_week` default to empty list `[]`
- Added `dog_id` and `type` as optional fields in `ScheduleUpdateRequest`

#### 3. Mission Scheduler Type Logic
**File:** `core/mission_scheduler.py`
- Updated `_should_start_mission()` to handle type field:
  - `"once"`: Check time window only, no day restriction, auto-disable after run
  - `"daily"`: Check time window only (ignore days_of_week)
  - `"weekly"`: Check both time window and days_of_week
- Added auto-disable logic for "once" type after execution

---

### Updated API Format

**Create Request (POST /schedules):**
```json
{
  "name": "Morning Sit Training",
  "mission_name": "sit_training",
  "dog_id": "1769681772789",
  "type": "weekly",
  "start_time": "08:00",
  "end_time": "12:00",
  "days_of_week": ["monday", "tuesday", "wednesday"],
  "enabled": true,
  "cooldown_hours": 24
}
```

**Response includes `schedule_id` instead of `id`**

---

### Testing Checklist
1. Restart robot
2. Test create with each type via curl
3. Verify scheduler respects type logic
4. Test from app - create schedule should succeed

---

## Session: 2026-01-31 - Build 34 Part 2 Fixes
**Goal:** Fix issues from Build 34 testing - servo limits, mission presence detection, song upload timeout
**Status:** ✅ Complete

---

### Problems Solved This Session

| # | Problem | Solution | Files Modified |
|---|---------|----------|----------------|
| 1 | Servo limits too restrictive for Manual/Xbox | Reverted to full range (10-200 pan, 20-160 tilt), stored coach limits as constants | `services/motion/pan_tilt.py` |
| 2 | Mission presence detection misaligned with coaching | Restored frames_total increment, matched coaching_engine presence ratio logic | `orchestrators/mission_engine.py` |
| 3 | Song upload timeout (5s too short for large files) | Created dedicated httpx client with 60s timeout for upload_song | `main_treatbot.py` |

---

### Key Code Changes Made

#### 1. Servo Limits Reverted
**File:** `services/motion/pan_tilt.py`
- Restored global limits: `pan_limits = (10, 200)`, `tilt_limits = (20, 160)`
- Restored center position: `current_pan = 90`, `current_tilt = 90`
- Added coach-specific constants for future use:
  - `COACH_PAN_LIMITS = (55, 145)`
  - `COACH_TILT_LIMITS = (25, 85)`
- Auto-tracking remains disabled in `_handle_coach_mode()`

#### 2. Mission Engine Presence Detection
**File:** `orchestrators/mission_engine.py`
- Restored `frames_total` increment in mission loop (line 466-468)
- Updated `_state_waiting_for_dog()` to match coaching_engine:
  - Uses `frames_seen / frames_total` for presence ratio
  - Requires `time_elapsed >= DETECTION_TIME_SEC` (3s) AND `presence_ratio >= PRESENCE_RATIO_MIN` (66%)
- Added presence ratio to log output for debugging

#### 3. Song Upload Timeout Fix
**File:** `main_treatbot.py`
- Created dedicated `httpx.Client(timeout=60.0)` for `upload_song` command
- Large MP3 uploads now have 60s timeout instead of 5s default

---

### Commit
`5a9a24b2` - fix: Build 34 Part 2 - servo limits, mission presence, upload timeout

---

### Testing Checklist
1. Xbox controller should have full pan/tilt range (10-200 pan, 20-160 tilt)
2. Coach mode camera stays centered (auto-tracking disabled)
3. Start Sit Training mission → dog detected within 3-5 seconds
4. Mission progresses to COMMAND state after dog eligible
5. Upload a 5MB+ MP3 file → completes without timeout

---

## Session: 2026-01-31 - Build 34 Bug Fixes (Robot Claude)
**Goal:** Fix critical issues from Build 33 testing - mission execution, dog identification, servo control
**Status:** ✅ Complete

---

### Problems Solved This Session

| # | Problem | Solution | Files Modified |
|---|---------|----------|----------------|
| 1 | Mission execution pipeline broken | Fixed presence ratio calculation (was 2%, needed 66%) | `orchestrators/mission_engine.py` |
| 2 | Wrong dog labeled (Elsa for everything) | Made dog ID more conservative, return "Dog" for unknown | `core/dog_tracker.py`, `services/perception/detector.py` |
| 3 | Video overlay shows ???? | Removed emoji chars (OpenCV font issue) | `services/streaming/video_track.py` |
| 4 | Mode not syncing with app | Send `mode_changed` event (not `status_update`) | `main_treatbot.py` |
| 5 | Servo movement too fast/jerky | Reduced PID gains 50%, limited range, disabled auto-track in Coach | `services/motion/pan_tilt.py` |

---

### Key Code Changes Made

#### 1. Mission Execution Pipeline Fix
**File:** `orchestrators/mission_engine.py`
- **Root Cause:** `frames_total` incremented at 10Hz but `frames_seen` only on events (every 5s), making presence ratio ~2% instead of 66%
- **Fix:** Changed to time-based presence detection instead of broken frame ratio
- Added diagnostic logging for mission start and detector status

#### 2. Dog Identification Regression Fix
**Files:** `core/dog_tracker.py`, `services/perception/detector.py`
- **Root Cause:** Persistence rules too aggressive - labeled unidentified dogs as "Elsa"
- **Fix:** Return `None` for unidentified instead of defaulting to last dog
- Reduced proximity matching threshold from 200px to 80px
- Disabled mutual exclusion rule (was labeling wrong dog)
- Display "Dog" for unknown instead of empty string

#### 3. Video Overlay Emoji Fix
**File:** `services/streaming/video_track.py`
- OpenCV FONT_HERSHEY_SIMPLEX doesn't support Unicode emoji
- Replaced all emoji with bracket notation: `[COACH]`, `[MISSION 1/3]`, etc.

#### 4. Mode Synchronization Fix
**File:** `main_treatbot.py`
- Changed event type from `status_update` to `mode_changed`
- Added `locked` and `reason` fields to event

#### 5. Servo Control Safety
**File:** `services/motion/pan_tilt.py`
- Reduced PID gains by 50%
- Reduced max movement from 3.0° to 1.5° per iteration
- Reduced pan limits from 10-200 to 55-145 degrees
- Reduced tilt limits from 20-160 to 25-85 degrees
- **Disabled auto-tracking in Coach mode** until properly tuned
- Slowed scan sweep from 10s to 20s period

---

### New Documentation Created
- `.claude/AI_DETECTION_ANALYSIS.md` - Explains dog detection/identification system, what broke, and fix proposals

---

### Testing Checklist for Build 34
1. Start Sit Training mission → verify robot enters MISSION mode AND AI activates
2. Dog detection shows "Dog" for unidentified (not "Elsa")
3. Mode changes sync between app and robot
4. Camera stays centered in Coach mode (auto-tracking disabled)
5. Video overlay shows clean text (no ????)

---

## Session: 2026-01-31 - API & Mission Integration Implementation
**Goal:** Implement robot-side improvements from API & Mission Integration plan
**Status:** ✅ Complete

---

### Problems Solved This Session

| # | Problem | Solution | Files Modified |
|---|---------|----------|----------------|
| 1 | Build 32 user_connected/user_disconnected events not handled | Added message handlers to relay client | `services/cloud/relay_client.py` |
| 2 | Mission statistics missing from weekly reports | Added `_get_mission_stats()` method | `core/weekly_summary.py` |
| 3 | Mission scheduler not initialized at boot | Added scheduler to main_treatbot orchestrators | `main_treatbot.py` |

---

### Key Code Changes Made

#### 1. Relay Client - User Event Handlers
**File:** `services/cloud/relay_client.py`
- Added `user_connected` handler: Tracks user, marks app connected, requests dog profiles
- Added `user_disconnected` handler: Stops missions/programs, closes WebRTC, publishes event
- Silent Guardian continues running (autonomous mode)

#### 2. Weekly Summary - Mission Stats
**File:** `core/weekly_summary.py`
- Added `_get_mission_stats()` method with:
  - Total missions, completed, stopped, failed counts
  - Success rate calculation
  - Breakdown by mission name
  - Breakdown by day of week
- Included in `generate_weekly_report()` output

#### 3. Mission Scheduler Initialization
**File:** `main_treatbot.py`
- Added import: `from core.mission_scheduler import get_mission_scheduler`
- Added instance variable: `self.mission_scheduler = None`
- Scheduler initialized in `_initialize_orchestrators()`
- Starts disabled (user enables via API)

---

### Confirmed Working (No Changes Needed)
- `DELETE /dogs/{dog_id}` endpoint (already existed)
- `delete_dog` WebSocket command (already existed)
- Mission engine with 21+ missions
- Program engine with 5 presets
- Reports API endpoints
- Scheduler API endpoints

---

### Next Steps for App Team
1. **Mission Browser UI** - Call `GET/POST /missions/*`
2. **Program Browser UI** - Call `GET/POST /programs/*`
3. **Progress Dashboard** - Display stats from `GET /reports/*`
4. **Scheduler UI** - Configure via `/missions/schedule/*`

---

### Important Notes
- Scheduler starts DISABLED - user must enable via `POST /missions/schedule/enable`
- User disconnect stops missions/programs but NOT Silent Guardian (autonomous)
- Mission stats now included in weekly reports alongside bark/reward/coaching stats

---

## Session: 2026-01-30 Evening (Build 32 - Testing Issues Analysis)
**Goal:** Analyze testing session issues (20:44-21:10) and fix robot-side problems
**Status:** ✅ Complete

---

### Issues Analyzed (From 25-min Test Session)

| # | Issue | Owner | Status |
|---|-------|-------|--------|
| 0 | Login/logout error | App/Relay | Documented |
| 1 | Music upload fails | App/Relay | Documented |
| 2 | Music play/stop | - | WORKS |
| 3 | Dog photo refresh | App | Documented |
| 4 | Voice files not deleted on dog delete | Robot | FIXED |
| 5 | Dogs/voices persist after restart | - | WORKS |
| 6 | Dogs shared across users | Relay | Documented |
| 6b | Robot pairing error message | Relay | Documented |
| 7 | User persistence/email validation | Relay | Documented |
| 8a | Mission shows "???? MISSION" | Robot | FIXED |
| 8b | Mission stuck "Waiting for Dog" | Robot | FIXED |
| 9 | Mission AI framework loading | - | Verified OK |
| 10 | Mode status persistence | - | Verified OK |

---

### Robot-Side Fixes Made

#### Fix 1: Mission Events Missing mission_name
**Problem:** App displayed "???? MISSION" because `mission_name` field was missing from events
**Fix:** Added `mission_name` to all `mission_progress` and `mission_complete` events
**File:** `orchestrators/mission_engine.py` (7 locations)

#### Fix 2: Mission Dog Detection Broken
**Problem:** Dogs marked stale after 2s, but detection events come every 5s
**Root Cause:** `STALE_TIMEOUT_SEC = 2.0` (too short for 5s event interval)
**Fix:** Increased to `STALE_TIMEOUT_SEC = 6.0`
**File:** `orchestrators/mission_engine.py:147`

#### Fix 3: Voice Files Not Deleted on Dog Delete
**Problem:** Custom voice folders remained after dog deletion
**Fix:** Added:
- `delete_dog_voices(dog_id)` function in `services/media/voice_manager.py`
- `delete_dog` WebSocket command handler in `api/ws.py`
- `DELETE /dogs/{dog_id}` REST endpoint in `api/server.py`

---

### Files Modified This Session (Build 32)
1. `orchestrators/mission_engine.py` - mission_name in events, stale timeout fix
2. `services/media/voice_manager.py` - delete_dog_voices function
3. `api/ws.py` - delete_dog command handler
4. `api/server.py` - DELETE /dogs/{dog_id} endpoint

### Commit
`d260261c` - fix: Build 32 - mission events, dog cleanup, stale timeout fix

