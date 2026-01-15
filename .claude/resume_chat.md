# WIM-Z Resume Chat Log

## Session: 2026-01-14 ~23:00
**Goal:** Fix behavior detection false positives + implement pose validation system
**Status:** ✅ Complete

### Problems Solved This Session

1. **CROSS over-detection (Critical Bug)**
   - Cross was being detected at 99-100% confidence constantly (1967/2749 detections at 99%+)
   - Root causes: model bias + loose verification threshold + no confidence capping
   - Fixes applied:
     - Tightened `_verify_cross_pose()` threshold from 0.35 to 0.20
     - Added confidence clamping (`np.clip(probs, 0.0, 0.98)`) to prevent 1.0 outputs
     - Implemented full pose validation pipeline

2. **SPEAK trick false positives**
   - Any ambient sound triggered "Good" for speak trick
   - Root cause: bark gate threshold too low (0.25)
   - Fix: Raised all bark gate thresholds (base: 0.25→0.45, min duration: 30ms→80ms)

3. **No persistent logging**
   - Logs were console-only, couldn't review history
   - Fix: Added rotating file handler to `/home/morgan/dogbot/logs/treatbot.log`

4. **Missing behavior filtering pipeline**
   - No validation between pose estimation and classification
   - Fix: Implemented full `PoseValidator` class with 5 validation checks + temporal voting

### Key Code Changes Made

#### New Files Created
- `services/ai/pose_validator.py` - Full pose validation module (~500 lines)
  - 5 validation checks: blur, keypoint confidence, skeleton geometry, bbox validity, detection quality
  - Temporal voting for stable predictions (requires 3/5 frames to agree)
  - Configurable thresholds via `PoseValidatorConfig`
- `services/ai/__init__.py` - Module exports

#### Modified Files
1. **`core/ai_controller_3stage_fixed.py`**
   - Added PoseValidator import and initialization
   - `_stage3_analyze_behavior()` now accepts frame for validation
   - `_analyze_with_torchscript()`:
     - Validates each pose before classification
     - Skips dogs with invalid poses
     - Applies temporal voting after raw predictions
   - Added control methods: `set_validation_enabled()`, `set_temporal_voting_enabled()`, `reset_validator_stats()`
   - Status includes validator stats
   - Confidence capping: `np.clip(probs, 0.0, 0.98)`
   - Cross verification threshold: 0.35 → 0.20

2. **`core/audio/bark_gate.py`**
   - `base_threshold`: 0.25 → 0.45
   - `thresh_close`: 0.50 → 0.65
   - `thresh_mid`: 0.35 → 0.50
   - `thresh_far`: 0.25 → 0.45
   - `min_bark_duration_ms`: 30 → 80

3. **`main_treatbot.py`**
   - Added rotating file handler (10MB, 5 backups)
   - Logs to `/home/morgan/dogbot/logs/treatbot.log`
   - DEBUG level to file, INFO to console

### Architecture Implemented

```
Frame → Dog Detected → Pose Estimated → VALIDATE POSE → Behavior Model → TEMPORAL VOTE → Output
                                            ↓                                   ↓
                                      Reject if:                         Require 3/5
                                      - blurry (Laplacian <80)           frames to agree
                                      - low keypoint conf (<0.35)        before emitting
                                      - impossible geometry              behavior
                                      - empty detection
                                      - bbox too small (<4000px)
```

### Unresolved Issues / Warnings

1. **Model training issue** - The behavior_14.ts model has inherent cross bias. The validation/filtering is a workaround, but ideally the model should be retrained with balanced data.

2. **Spin detection still weak** - Only 143 total spin detections vs 2749 cross. Temporal detection of rotation is hard. May need dedicated spin detection logic.

3. **Distance variance** - Sit recognition varies by distance. Model wasn't trained with distance variance.

### Next Steps Identified

1. **Test the changes** - Restart system and observe:
   - Cross detection should be much less frequent
   - Speak trick should require actual barks
   - Check logs at `/home/morgan/dogbot/logs/treatbot.log`

2. **Monitor validator stats** - Use `ai_controller.get_status()['validator_stats']` to see rejection rates

3. **Tune thresholds if needed** - Validator config can be adjusted

4. **Consider model retraining** - Long-term fix for cross bias

### Testing Commands

```bash
# View live logs
tail -f /home/morgan/dogbot/logs/treatbot.log

# Filter for validation
grep -i "rejected\|invalid\|unstable" /home/morgan/dogbot/logs/treatbot.log

# Check behavior stats in DB
python3 -c "import sqlite3; conn = sqlite3.connect('/home/morgan/dogbot/data/dogbot.db'); cur = conn.cursor(); cur.execute('SELECT behavior, COUNT(*), AVG(confidence) FROM behavior_events WHERE timestamp > datetime(\"now\", \"-1 hour\") GROUP BY behavior'); print([r for r in cur.fetchall()])"
```

### Important Notes

- **Validation can be toggled off for debugging**: `ai_controller.set_validation_enabled(False)`
- **Temporal voting can be toggled off**: `ai_controller.set_temporal_voting_enabled(False)`
- **Stats can be reset**: `ai_controller.reset_validator_stats()`

---

## Session: 2026-01-14 ~Late Night (03:30-04:00)
**Goal:** Fix video recording overlays, cross detection false positives, dog switching speed
**Status:** ✅ Complete

### Work Completed:

#### 1. Video Recording Fixes - ✅ COMPLETE
**File:** `services/media/video_recorder.py` (major rewrite ~200 lines)
- **Fixed bounding boxes**: Rewrote event handling to accumulate individual detection events
- **Behavior text persistence**: Added 3-second display buffer with fade effect
- **Removed record icon**: Stripped the red "REC" dot from bottom left corner
- **Better labels**: Dog name in large text at top of bbox, behavior + confidence below bbox
- **ArUco visualization**: Diamond markers at ArUco positions with dog name labels

**File:** `services/perception/detector.py`
- Added ArUco marker event publishing for video recorder access

#### 2. Cross Detection False Positives Fix - ✅ COMPLETE
**File:** `core/ai_controller_3stage_fixed.py` (+77 lines)
- Added `_verify_cross_pose()` method that checks if front paws are actually close together
- If model says "cross" but paws are > 35% of body width apart → automatically convert to "lie"

#### 3. Faster Dog Switching - ✅ COMPLETE
**File:** `core/dog_tracker.py` (+33 lines changed)
- Reduced `persistence_time` from 30s to 15s
- Added `clear_dog_tracking()` method for manual clearing

### AI Stack Explained:
```
STAGE 1+2 (Hailo-8): dogpose_14.hef - YOLOv8-pose
  → Single inference gives detection boxes + 24 keypoints

STAGE 3 (CPU): behavior_14.ts - TorchScript temporal model
  → 16 frames of pose history → behavior classification
  → Classes: stand, sit, lie, cross, spin (prob_th = 0.7)
```

---

## Session: 2026-01-14 ~Morning
**Goal:** Training Programs system + Auto video recording in Coach mode
**Status:** ✅ Complete

### Work Completed:

#### 1. Training Programs System - ✅ COMPLETE
- `orchestrators/program_engine.py` - ProgramEngine class (450 lines)
- `programs/puppy_basics.json`, `quiet_dog.json`, `trick_master.json`, `daily_routine.json`, `calm_evening.json`
- 10 API endpoints for program management

#### 2. Auto Video Recording in Coach Mode - ✅ COMPLETE
- Videos record automatically during coaching sessions
- Output: `/home/morgan/dogbot/recordings/`
- Filename: `coach_{dog}_{trick}_{timestamp}.mp4`

---

## Session: 2026-01-13 ~Late Night
**Goal:** Build 9 launch features for WIM-Z demo/media readiness
**Status:** ✅ Complete

- Silent Guardian Rewrite (simple fixed-timing flow)
- HTML Reports (`templates/weekly_report.html`, `dog_profile.html`)
- Photo Enhancer (`services/media/photo_enhancer.py`)
- LLM Integration (`services/ai/llm_service.py`)
- Instagram Poster (`services/social/instagram_poster.py`)
- Caption Tool Web UI (`templates/caption_tool.html`)
- Static file serving for photos

---

## Session: 2026-01-11 ~Evening
**Goal:** Mission engine fixes, missions, video recording, bark attribution
**Status:** ✅ Complete

- 4 bug fixes in mission_engine.py
- 13 new mission JSON files (20 total)
- Video recording service + API endpoints
- Bark-dog attribution fix

---

## Session: 2026-01-10 ~Afternoon
**Goal:** Fix coaching engine "green lighting" + improve bark detection
**Status:** ✅ Complete

- Fixed threading race condition
- Added 400-4000Hz bandpass filter for bark detection
- Raised lie/cross confidence thresholds to 0.75
