# WIM-Z Resume Chat Log

## Session: 2026-01-13 ~Late Night
**Goal:** Build 9 launch features for WIM-Z demo/media readiness
**Status:** ✅ Complete (except Instagram testing with real account)

### Work Completed:

#### 1. Silent Guardian Rewrite - ✅ COMPLETE
- **File:** `modes/silent_guardian.py`
- **Change:** Replaced complex escalation system with simple fixed-timing flow
- **New Flow:**
  - 2 barks detected → "QUIET"
  - 5s → "QUIET" again
  - 5s → "treat.mp3"
  - 5s → "QUIET" again
  - 5s → (if no barking) → "good.mp3" + DISPENSE TREAT
- Bark during sequence = restart from beginning
- Removed: escalation levels, dog calling, visibility checks

#### 2. HTML Reports - ✅ COMPLETE
- **`templates/weekly_report.html`** - Pretty weekly report with dark theme
  - Bark stats, treats, coaching sessions, Silent Guardian effectiveness
  - Bar charts for emotions and trick performance
  - Mobile-responsive CSS
- **`templates/dog_profile.html`** - Per-dog media page
  - Dog photo, achievements, stats
  - Progress bars (quiet time, trick success, attention, behavior)
  - 8-week trend chart, photo gallery
- **Endpoints:** `GET /reports/html/weekly`, `GET /reports/html/dog/{dog_id}`

#### 3. Photo Enhancer - ✅ COMPLETE
- **File:** `services/media/photo_enhancer.py`
- **Features:**
  - Auto-enhance (contrast, brightness, saturation, sharpness)
  - Filters: warm, cool, vintage, dramatic, bright
  - Instagram sizing (1080x1080 square crop)
  - Text overlays (dog name, caption)
  - WIM-Z watermark
- **Endpoint:** `POST /photo/enhance`
- **Output:** `/home/morgan/dogbot/captures/enhanced/`

#### 4. LLM Integration (OpenAI) - ✅ COMPLETE
- **File:** `services/ai/llm_service.py`
- **Features:**
  - GPT-4o Vision for photo captions (~$0.002/caption)
  - GPT-4o-mini for weekly narratives and dog personality
  - Caption styles: friendly, funny, inspirational, hashtag
  - Fallback captions when LLM unavailable
- **Endpoints:** `POST /ai/caption`, `POST /ai/summarize`, `POST /ai/personality`
- **Config:** `OPENAI_API_KEY` in `.env` file

#### 5. Instagram Poster - ✅ BUILT (Not Tested)
- **File:** `services/social/instagram_poster.py`
- **Features:**
  - Username/password authentication via instagrapi
  - Session caching to avoid re-login
  - Photo, video/reel, and story posting
  - Default hashtags for dog content
- **Endpoints:** `POST /social/instagram/login`, `POST /social/instagram/post`
- **Note:** Requires real IG credentials to test

#### 6. Caption Tool Web UI - ✅ COMPLETE
- **File:** `templates/caption_tool.html`
- **Features:**
  - Photo grid selection from captures folder
  - Style dropdown (friendly, funny, inspirational, hashtag)
  - Dog name and context inputs
  - Copy caption button
- **Endpoint:** `GET /tools/caption`

#### 7. Static File Serving - ✅ COMPLETE
- Added routes in `api/server.py`:
  - `/photos/{filename}` → `/home/morgan/dogbot/captures/`
  - `/enhanced/{filename}` → `/home/morgan/dogbot/captures/enhanced/`
  - `/tools/photos` → List available photos

### API Endpoints Added (api/server.py):
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reports/html/weekly` | GET | Pretty weekly report HTML |
| `/reports/html/dog/{dog_id}` | GET | Dog profile HTML |
| `/photo/enhance` | POST | Enhance photo with filters/overlays |
| `/ai/caption` | POST | Generate LLM caption for photo |
| `/ai/summarize` | POST | Generate weekly narrative |
| `/ai/personality` | POST | Generate dog personality |
| `/social/instagram/login` | POST | Login to Instagram |
| `/social/instagram/post` | POST | Post to Instagram |
| `/social/instagram/status` | GET | Check IG login status |
| `/tools/caption` | GET | Caption generator web UI |
| `/tools/photos` | GET | List available photos |

### Files Created:
- `templates/weekly_report.html` (340 lines)
- `templates/dog_profile.html` (420 lines)
- `templates/caption_tool.html` (330 lines)
- `services/media/photo_enhancer.py` (280 lines)
- `services/ai/llm_service.py` (358 lines)
- `services/social/instagram_poster.py` (372 lines)
- `.env` (contains OPENAI_API_KEY)
- `/home/morgan/dogbot/captures/enhanced/` directory

### Files Modified:
- `api/server.py` - Added ~300 lines for new endpoints
- `modes/silent_guardian.py` - Complete rewrite with simple flow

### Errors Fixed During Session:
1. **Route conflict:** `/reports/weekly/html` caught by `/reports/weekly/{date}` → Changed to `/reports/html/weekly`
2. **Missing directory:** `captures/enhanced/` didn't exist → Created it
3. **OpenAI quota:** User added billing to account
4. **Curl URL format:** `&` in URL needed quotes → Wrap in single quotes

### Testing Results:
- ✅ Weekly report HTML renders at `http://localhost:8000/reports/html/weekly`
- ✅ Dog profile HTML renders at `http://localhost:8000/reports/html/dog/elsa`
- ✅ LLM captions generate (tested friendly, funny, inspirational styles)
- ✅ Photo enhancement works with filters and overlays
- ✅ Caption tool web UI loads at `http://localhost:8000/tools/caption`
- ⏳ Instagram posting not tested (needs real credentials)
- ⏳ Silent Guardian not tested with live barking dogs

### Next Session:
1. Test Silent Guardian with actual barking dogs
2. Test Instagram posting with real account credentials
3. Discuss HTML report styling improvements (user mentioned later)
4. Consider auto-captioning hooks (after coaching, before IG post)
5. Test video recording with AI overlays for demo footage

### Important Notes:
- LLM costs ~$0.002 per photo caption
- Photo enhancer saves to `captures/enhanced/` with timestamp
- Static files accessible via browser at `/photos/` and `/enhanced/`
- Instagram requires real username/password in config

---

## Session: 2026-01-11 ~Evening
**Goal:** Complete 8-item plan: Mission engine fixes, missions, video recording, bark attribution
**Status:** ✅ Complete

### Work Completed:

#### 1. Mission Engine - 4 Bug Fixes (orchestrators/mission_engine.py)
- **Bug 1:** Fixed `log_reward()` wrong parameters (lines 578-585)
  - Changed from wrong params (`treat_dispensed`, `audio_played`, `lights_activated`)
  - To correct: `success=True`, `treats_dispensed=1`, `mission_name=session.mission.name`
- **Bug 2:** Fixed hardcoded `daily_limit = 10` (line 476-480)
  - Now reads from mission config: `session.mission.config.get('daily_limit', 30)`
- **Bug 3:** Fixed stage advancement bounds - checks if success_event is "DogDetected"
- **Bug 4:** Fixed wrong method call: `execute()` → `execute_sequence()`

#### 2. Created 13 New Mission JSON Files (20 total)
- `morning_quiet_2hr.json` - 2hr quiet period (7am-1pm)
- `morning_chill.json` - Calm morning behavior (8am-12pm)
- `afternoon_sit_5.json` - 5 sits (12pm-7pm)
- `afternoon_down_3.json` - 3 lie downs (12pm-6pm)
- `afternoon_crosses_2.json` - 2 crosses (12pm-7pm)
- `evening_settle.json` - Settle after dinner (5pm-9pm)
- `evening_calm_transition.json` - Wind down (6pm-9pm)
- `night_quiet_90pct.json` - 90% quiet overnight (7pm-4am)
- `speak_morning.json` - Bark on cue AM (8am-12pm)
- `speak_afternoon.json` - Bark on cue PM (2pm-6pm)
- `sit_sustained.json` - Extended 30s sit
- `down_sustained.json` - Extended 30s down
- `quiet_progressive.json` - Progressive quiet (5min → 10min → 15min)

#### 3. Video Recording Service - NEW (services/media/video_recorder.py)
- 640x640 MP4 recording at 15 FPS
- AI overlays: bounding boxes (color-coded), pose keypoints, skeleton, behavior labels
- Recording indicator (red dot) + timestamp overlay
- Methods: `start_recording()`, `stop_recording()`, `toggle_recording()`, `list_recordings()`
- Saves to: `/home/morgan/dogbot/recordings/`

#### 4. Video API Endpoints (api/server.py lines 1837-1914)
- `POST /video/record/start` - Start recording
- `POST /video/record/stop` - Stop recording
- `GET /video/record/status` - Get recording status
- `GET /video/recordings` - List all recordings
- `POST /video/record/toggle` - Toggle recording on/off

#### 5. Xbox Video Recording Integration (xbox_hybrid_controller.py)
- **Short press Start (< 2s):** Audio recording (existing)
- **Long press Start (> 2s):** Toggle video recording
- Added `toggle_video_recording()` method (lines 1352-1378)
- Audio feedback: "Wimz_recording.mp3" / "Wimz_saved.mp3"

#### 6. Bark-Dog Attribution Fix (services/perception/bark_detector.py)
- **Problem:** All 2520 barks in database had dog_id=None
- **Root Cause:** Only attributed when exactly 1 dog visible at bark moment
- **Fix:** Added last known dog tracking (30-second window)
  - Added `_last_known_dog_id`, `_last_known_dog_name`, `_last_known_dog_time`
  - Updated `_on_vision_event()` to track last known dog
  - Updated `_publish_bark_event()` to use last known dog when no dogs visible

### Files Modified:
- `orchestrators/mission_engine.py` - 4 bug fixes (+32 lines)
- `api/server.py` - 5 video endpoints (+92 lines)
- `services/perception/bark_detector.py` - Bark attribution (+28 lines)
- `xbox_hybrid_controller.py` - Long-press video toggle (+59 lines)

### Files Created:
- `services/media/video_recorder.py` - Video recording service (423 lines)
- `missions/*.json` - 13 new mission files
- `/home/morgan/dogbot/recordings/` - Output directory

### Documentation Updated:
- `.claude/product_roadmap.md` - Accurate status for all phases
- `.claude/development_todos.md` - Current task list
- `.claude/WIM-Z_Project_Directory_Structure.md` - Added VOICEMP3 structure

### Testing Results:
- Weekly Summary: Runs but dog_id=None for historical barks (will improve going forward)
- Silent Guardian: 229 sessions, 179 interventions logged
- Database: 1555 barks, 111 rewards in treatbot.db

### Next Session:
1. Restart treatbot: `sudo systemctl restart treatbot`
2. Test video recording via API and Xbox long-press
3. Verify new barks get dog attribution
4. Test mission loading with new JSON files

### Important Notes:
- Bark attribution only works going forward (historical data unchanged)
- 30-second window handles dogs momentarily out of frame
- Video overlay colors: Green=Elsa, Magenta=Bezik, Yellow=Unknown

---

## Session: 2026-01-10 ~Afternoon
**Goal:** Fix coaching engine "green lighting" all tricks + improve bark detection
**Status:** ✅ Complete

### Work Completed:

#### 1. Fixed Threading Race Condition - ✅ FIXED (Previous Part of Session)
- **Problem:** Tricks completed instantly (<2 seconds) without dog actually performing them
- **Root Cause:** Event bus threading model - each callback spawns new thread, causing stale events to execute AFTER state changes
- **Fix:** Added timestamp validation:
  - `BehaviorInterpreter._reset_timestamp` tracks when reset was called
  - `_update_detection()` rejects events with timestamp < reset_timestamp
  - `CoachingEngine._listening_started_at` tracks when bark listening starts
  - `_on_audio_event()` rejects bark events before listening started

#### 2. Added Bandpass Filter for Bark Detection - ✅ FIXED
- **Problem:** Bark detector triggered on any loud sound (claps, voice, HVAC)
- **Fix:** Added 400-4000Hz bandpass filter to `services/perception/bark_detector.py`
- Filters audio BEFORE energy calculation
- Dog barks are primarily 400-4000Hz, so they pass through
- Other sounds (rumble <400Hz, electronic noise >4000Hz) filtered out

#### 3. Raised cross/lie Detection Thresholds - ✅ FIXED
- **Problem:** Sitting dogs triggered "cross" and "lie" false positives (thresholds too low)
- **Fix:** Raised thresholds from 0.60/0.65 to 0.75 in:
  - `configs/trick_rules.yaml` - trick-specific and detection.confidence_overrides
  - `core/behavior_interpreter.py` - default fallback values

### Files Modified:
- `services/perception/bark_detector.py`:
  - Added scipy import for butter/sosfilt
  - Added `_bandpass_filter()` method (400-4000Hz)
  - Applied filter before energy calculation
- `configs/trick_rules.yaml`:
  - down confidence_threshold: 0.65 → 0.75
  - crosses confidence_threshold: 0.60 → 0.75
  - detection.confidence_overrides: lie 0.65 → 0.75, cross 0.60 → 0.75
- `core/behavior_interpreter.py`:
  - Default lie threshold: 0.65 → 0.75
  - Default cross threshold: 0.60 → 0.75

### Verification Tests:
1. **Bark test:** Clap/speak near mic → should NOT trigger speak success
2. **Bark test:** Actual dog bark → should trigger speak success
3. **Pose test:** Dog sitting → should NOT trigger "down" or "crosses"
4. **Pose test:** Dog lying at 0.75+ confidence → should succeed

### Next Session:
1. Restart treatbot: `sudo systemctl restart treatbot`
2. Test bark detection with non-bark sounds vs actual barks
3. Test pose detection with sitting vs lying dogs
4. Verify coaching sessions require actual trick performance

### Important Notes:
- Timestamp validation prevents race condition from threaded event callbacks
- Bandpass filter uses scipy.signal.butter (4th order) + sosfilt
- Higher thresholds for cross/lie require more confident detection

---

## Session: 2026-01-08 ~Afternoon
**Goal:** Implement presence-based detection + retry logic for coaching engine
**Status:** ✅ Complete

### Work Completed:

#### 1. Presence-Based Dog Detection - ✅ IMPLEMENTED
- **Problem:** 10-second ArUco grace period blocked session starts, dog identity flip-flopped
- **New Architecture:** Two parallel processes
  - Process 1: Dog presence timer (3 seconds with 66% in-frame = session starts)
  - Process 2: ArUco identification (runs in background, announces name when found)
- **Changes:**
  - `dogs_in_view` now tracks: `{first_seen, last_seen, frames_seen, frames_total, name}`
  - Sessions start based on PRESENCE, not identity (3s + 66% presence ratio)
  - ArUco identification optional - just gives us the dog's name
  - Late ArUco announcement during WATCHING or RETRY_WATCHING states

#### 2. Retry on First Failure - ✅ IMPLEMENTED
- **Behavior:** Dogs get a second chance if they fail the first attempt
- **Flow:** FAILURE → RETRY_GREETING → RETRY_COMMAND → RETRY_WATCHING → SUCCESS/FINAL_FAILURE
- **New States Added:**
  - `RETRY_GREETING` - Re-greet dog by name
  - `RETRY_COMMAND` - Give trick command again
  - `RETRY_WATCHING` - Watch for trick (same logic as WATCHING)
  - `FINAL_FAILURE` - Plays "no_no_no.mp3", no treat
- **Session tracking:** `DogSession.attempt` field (1 or 2)

#### 3. Removed ArUco Grace Period Blocking - ✅ FIXED
- **Problem:** `dog_tracker.py` waited 10 seconds before assigning default name
- **Fix:** Reduced `aruco_grace_period` from 10.0 to 0.0 seconds
- Coaching engine handles late ArUco identification separately

### Files Modified:
- `orchestrators/coaching_engine.py`:
  - Added retry states to `CoachState` enum
  - Updated `DogSession` with `attempt` field
  - Changed `dogs_in_view` to dict with frames tracking
  - Updated `_on_vision_event()` for late ArUco announcement
  - Updated `_state_waiting_for_dog()` with presence ratio logic
  - Updated `_state_failure()` for retry logic
  - Added: `_state_retry_greeting()`, `_state_retry_command()`, `_state_retry_watching()`, `_state_final_failure()`
  - Updated `_get_dog_name()` to use new structure
- `core/dog_tracker.py`:
  - Reduced `aruco_grace_period` from 10.0 to 0.0 seconds

### New Detection Logic:
| Config | Value | Description |
|--------|-------|-------------|
| detection_time_sec | 3.0 | Time dog must be visible |
| presence_ratio_min | 0.66 | Min percentage in-frame (66%) |
| stale_timeout_sec | 5.0 | Remove dog after this long unseen |

### Next Session:
1. Restart treatbot: `sudo systemctl restart treatbot`
2. Test dog detection timing (should start session after 3s with 66% presence)
3. Test retry on failure (dog should get second chance)
4. Verify late ArUco announcement during WATCHING state
5. Test with multiple dogs (up to 2)

### Important Notes:
- Session starts based on DOG PRESENCE, not ArUco identity
- ArUco is optional - just gives the dog a name
- Dogs get 2 attempts per session before final failure
- "no_no_no.mp3" plays on final failure (after 2 failed attempts)
- "no.mp3" no longer plays (removed on first failure, goes to retry)

---

## Session: 2026-01-08 ~Morning
**Goal:** Debug error audio spam + Fix dog detection + Add Xbox Guide button trick cycling
**Status:** ✅ Complete

### Work Completed:

#### 1. Fixed Constant "Error" Audio Spam - ✅ FIXED
- **Problem:** Error audio playing every 60 seconds during normal operation
- **Root Cause:** Temperature at 70-72°C triggering WARNING alerts (threshold was 70°C)
- **Fixes Applied:**
  - Raised `temp_warning` from 70°C to 76°C in `core/safety.py`
  - Changed error audio to only play for CRITICAL alerts (not WARNING)
  - Added 30-second startup grace period for CPU alerts (100% at startup is normal)
  - Added separate `_play_hot_audio()` using `wimz_hot.mp3` for temperature warnings

#### 2. Fixed Speak Trick False Positives - ✅ FIXED
- **Problem:** Non-bark sounds triggering speak trick reward
- **Fix:** Added 50% minimum confidence filter in coaching engine's `_on_audio_event()`

#### 3. Fixed Charging Detection - ✅ FIXED
- **Problem:** Threshold too high (0.3V) - wouldn't detect charging at 50%+ battery
- **Fix:** Lowered voltage increase threshold from 0.3V to 0.05V in `battery_monitor.py`

#### 4. Fixed Slow Dog Detection - ✅ FIXED
- **Problem:** 30+ seconds to detect dog in clear view
- **Fix:** Lowered confidence threshold from 0.7 to 0.5 in `robot_config.yaml`

#### 5. Replaced Stillness with Time-Based Detection - ✅ FIXED
- **Problem:** 50px stillness requirement too strict for excited small dogs near camera
- **Fix:** Removed stillness tracking entirely, replaced with 3.5 second time-in-view confirmation
- Removed: `dog_still_start`, `last_dog_position`, `_update_dog_position()`
- Updated: `_state_waiting_for_dog()` uses time since first seen

#### 6. Added Consistent Trick Thresholds - ✅ FIXED
- **Problem:** `reward_logic.py` had hardcoded 0.7 confidence, didn't match `trick_rules.yaml`
- **Fix:** Now imports `BehaviorInterpreter` and uses its thresholds for consistency

#### 7. Added Xbox Guide Button Trick Cycling - ✅ IMPLEMENTED
- **Behavior:** Press Guide button (button 8) to cycle tricks in coach mode
- **Cycle:** sit → down → crosses → spin → speak → sit...
- **Features:**
  - Only works in coach mode (ignores other modes)
  - Sets forced trick for next coaching session
  - Plays trick name as audio feedback
  - 1-second cooldown between presses
- **Uses existing API:** `/coaching/force_trick/{trick}`

### Files Modified:
- `core/safety.py` - Temperature thresholds, startup grace period, hot audio
- `config/robot_config.yaml` - Detection confidence 0.7 → 0.5
- `orchestrators/coaching_engine.py` - Removed stillness, added bark confidence filter
- `services/power/battery_monitor.py` - Charging threshold 0.3V → 0.05V
- `api/server.py` - LED reset on recording timeout
- `orchestrators/reward_logic.py` - Uses BehaviorInterpreter thresholds
- `xbox_hybrid_controller.py` - Added Guide button trick cycling

### Next Session:
1. Test Guide button trick cycling in coach mode
2. Verify temperature warnings use wimz_hot.mp3 instead of error audio
3. Test charging detection with new 0.05V threshold
4. Verify dog detection is faster with 0.5 confidence threshold

### Important Notes:
- Pi 5 + Hailo-8 normally runs 65-75°C under AI load (now allowed)
- Guide button only works in coach mode
- Available tricks for cycling: sit, down, crosses, spin, speak

---

## Session: 2026-01-07 ~21:00-22:30
**Goal:** Debug Coach mode audio issues + Add WIM-Z audio feedback system
**Status:** ✅ Complete
**Commit:** 98a4ff11

### Work Completed:

#### 1. Diagnosed Coach Mode Audio Bug - ✅ FIXED
- **Problem:** Dog heard "bezik" greeting, then 10s silence, then "no no no" - no trick cue played
- **Root Cause:** BehaviorInterpreter config failed to load due to init order bug
  - `_load_trick_rules()` tried to use `self.confidence_thresholds` before it was defined
  - Fell back to DEFAULT rules which lacked `audio_command` field
  - Trick "down" tried to play "down.mp3" (doesn't exist) instead of "lie_down.mp3"
- **Fix 1:** Moved `confidence_thresholds` definition BEFORE `_load_trick_rules()` call
- **Fix 2:** Added `audio_command` to all default trick rules with correct filenames

#### 2. Added WIM-Z Audio Feedback System - ✅ IMPLEMENTED
| Feature | File | Trigger |
|---------|------|---------|
| Charging audio | `Wimz_charging.mp3` | Voltage rises 0.3V+ over 15s |
| Low power audio | `Wimz_lowpower.mp3` | Battery < 12.0V |
| Error audio | `Wimz_errorlogs.mp3` | Safety warnings/critical alerts |
| Mission complete | `Wimz_missioncomplete.mp3` | Formal mission success |
| Recording start | `Wimz_recording.mp3` | Xbox Start button (1st press) |
| Recording saved | `Wimz_saved.mp3` | Xbox Start button (2nd press) |

### Files Modified:
- `core/behavior_interpreter.py` - Fixed init order + default audio_commands
- `core/safety.py` - Added error audio alerts
- `services/power/battery_monitor.py` - Added charging detection
- `orchestrators/mission_engine.py` - Added mission complete audio
- `api/server.py` - Updated recording audio paths
- `main_treatbot.py` - Updated low battery audio path
- Added 8 new MP3 files to `VOICEMP3/wimz/`

### Next Session:
1. Restart treatbot to apply changes
2. Test Coach mode - verify trick audio plays correctly
3. Test charging detection by plugging in charger
4. Test Xbox recording with Start button

### Important Notes:
- Treatbot service needs restart: `sudo systemctl restart treatbot`
- Coach mode available tricks: sit, down, crosses, spin, speak (stand excluded)
- Charging cooldown: 5 minutes between announcements
- Error audio cooldown: 60 seconds

---

## Session: 2026-01-07 ~11:55-12:55
**Goal:** Fix Xbox controller RB photo button and mode cycling issues
**Status:** ✅ Fixes Implemented (Testing Pending)

### Work Completed:

#### 1. Diagnosed RB Photo Button Issues - ✅ FIXED
- **Problem:** RB button wasn't taking photos; buttons were switching to MANUAL mode
- **Root Cause:** `notify_manual_input()` was called on EVERY button press in `process_button()`, which triggered mode switch to MANUAL
- **Fix:** Removed global `notify_manual_input()` from `process_button()` - joystick/triggers already have it in `process_axis()`

#### 2. Fixed Subprocess Logging - ✅ FIXED
- **Problem:** Xbox controller subprocess logs went to PIPE and were never read (invisible)
- **Fix:** Removed `stdout=subprocess.PIPE, stderr=subprocess.PIPE` from subprocess.Popen
- **Added:** `-u` flag for unbuffered Python output
- Now controller logs appear in `journalctl -u treatbot`

#### 3. Updated take_photo() Logic - ✅ FIXED
- **Problem:** take_photo() always tried 4K first, fell back to snapshot
- **Fix:** Now explicitly checks current mode FIRST:
  - MANUAL mode → 4K photo (camera released)
  - Other modes → Snapshot from AI stream (640x640)
- Added `_get_current_mode()` helper that queries `/mode` API

#### 4. Fixed Mode Cycle Sync - ✅ FIXED
- **Problem:** `current_mode_index` was never synced with actual system mode
- **Fix:** `cycle_mode()` now queries actual mode before incrementing
- Changed to blocking API request for mode changes

#### 5. Fixed api_request_blocking() Timeout Parameter - ✅ FIXED
- **Problem:** `take_photo()` passed `timeout=8` but `api_request_blocking()` didn't accept it
- **Fix:** Added `timeout` parameter to both `api_request_blocking()` and `_api_request_sync()`

### Files Modified:
- `xbox_hybrid_controller.py`:
  - Removed `notify_manual_input()` from `process_button()` (line ~1229)
  - Added `_get_current_mode()` helper method
  - Rewrote `take_photo()` to check mode explicitly
  - Updated `cycle_mode()` to sync with actual mode
  - Fixed `api_request_blocking()` to accept timeout parameter
  - Fixed `_api_request_sync()` to accept timeout parameter
- `services/control/xbox_controller.py`:
  - Removed PIPE capturing from subprocess
  - Added `-u` flag for unbuffered output

### Key Behavior Changes:
1. **Button presses no longer auto-switch to MANUAL mode**
2. **RB in COACH/SILENT_GUARDIAN** → Takes 640x640 snapshot from AI stream
3. **RB in MANUAL** → Takes 4K photo (4056x3040)
4. **Mode cycling** → Now correctly syncs with actual system mode before cycling
5. **Controller logs** → Now visible in systemd journal

### User Requirements Confirmed:
1. RB in COACH/SILENT_GUARDIAN → Take snapshot, STAY in current mode
2. RB in MANUAL → Take 4K photo
3. No manual mode timeout when Xbox controller connected

### Next Session:
1. **TEST:** Mode cycling with SELECT button
2. **TEST:** RB photo in different modes
3. **TEST:** Verify photos saved correctly
4. Consider committing changes after testing

### Important Notes/Warnings:
- **Testing pending:** User needs to test SELECT and RB buttons
- **Photo locations:** `/home/morgan/dogbot/captures/photo_*.jpg` (4K), `snapshot_*.jpg` (640x640)
- **Logs now visible:** Use `journalctl -u treatbot -f` to see Xbox controller output

---

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

### Photos Save To:
- `/home/morgan/dogbot/captures/photo_*.jpg` (4K)
- `/home/morgan/dogbot/captures/snapshot_*.jpg` (640x640)

---

## Session: 2026-01-06 ~17:00-18:45
**Goal:** Fix motor calibration - binary on/off → gradual speed control
**Status:** ✅ RESOLVED

### Work Completed:

#### 1. Motor PWM Control - ✅ FIXED
- **Problem:** Motors responded as on/off (binary) instead of gradual speed
- **Root Cause:** WIRING ERROR - GPIO pins were connected wrong
- **User fixed the hardware wiring**

---

## Session: 2026-01-04 00:30 - 01:15
**Goal:** Fix hardware issues on treatbot2 - Battery monitor, servos, LEDs
**Status:** ✅ Partially Complete

### Work Completed:
- Battery Monitor (ADS1115) - FIXED
- Servo/PCA9685 I2C Conflict - FIXED
- LEDs - Hardware issue (GPIO25 working, physical wiring needs check)
