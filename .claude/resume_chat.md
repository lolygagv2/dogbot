# WIM-Z Resume Chat Log

## Session: 2026-01-15 ~15:30
**Goal:** Fix behavior detection (lie down, spin, crosses removal)
**Status:** Major progress - spin detection significantly improved, needs more tuning

---

### Problems Solved This Session

#### 1. Crosses Behavior Removed Completely
- **Issue:** "Crosses" trick was unreliable - dogs don't cross paws on command
- **Fix:** Removed from ALL locations:
  - `ai/models/config.json` - removed from behaviors list
  - `configs/trick_rules.yaml` - removed trick definition
  - `core/behavior_interpreter.py` - removed from BEHAVIORS and default tricks
  - `orchestrators/coaching_engine.py` - removed from DEFAULT_TRICKS
  - `xbox_hybrid_controller.py` - removed from trick cycle (was hardcoded!)
  - `programs/trick_master.json` - removed crosses mission
  - `missions/afternoon_crosses_2.json` - moved to archive

#### 2. Lie Down Detection Fixed
- **Issue:** Front-facing dogs in sphinx pose have aspect ratio ~0.85-0.95, but threshold was 0.75
- **Fix:** Updated `services/ai/geometric_classifier.py`:
  - `lie_max_aspect`: 0.75 → 0.95
  - `sit_min_aspect`: 1.05 → 1.15
  - Narrowed stand range to 0.95-1.15
- **Result:** Lie down now works for both dogs

#### 3. Spin Detection - Major Rewrite
- **Issue:** Original spin detection used keypoint orientation which fails during motion blur
- **Root Cause 1:** Fast spins (0.25s) at 10fps = only 2-3 frames of data
- **Root Cause 2:** Motion blur destroys keypoint confidence
- **Root Cause 3:** Temporal voting was overriding spin→stand
- **Root Cause 4:** Dog lands in sit after spin, overwriting spin detection

- **Fixes Applied:**
  1. Rewrote `_check_spin()` in `geometric_classifier.py`:
     - Uses bbox-only metrics (no keypoints needed)
     - Tracks frame-to-frame deltas (aspect ratio changes, center movement)
     - Only needs 4 frames minimum
     - Lowered thresholds for sensitivity

  2. Made spin bypass temporal voting in `ai_controller_3stage_fixed.py`:
     - Spin is instant detection, no voting

  3. Added "spin latch" in `behavior_interpreter.py`:
     - Once spin detected, holds for 2 seconds
     - Won't be overwritten by sit/stand/lie (dog landing after spin)

#### 4. Other Fixes
- **Bark threshold raised:** 0.12 → 0.18 (was too sensitive to small noises)
- **Force button full reset:** Now cancels session, resets FSM, clears all state
- **Sequential trick rotation:** sit → down → spin → speak (no more random)
- **Dog name fallback:** Unknown dogs get generic "dogs_come.mp3" greeting

---

### Key Code Changes

| File | Changes |
|------|---------|
| `services/ai/geometric_classifier.py` | Rewrote spin detection, updated aspect ratio thresholds |
| `core/ai_controller_3stage_fixed.py` | Spin bypasses temporal voting |
| `core/behavior_interpreter.py` | Added spin latch (2s protection), removed crosses |
| `core/audio/bark_gate.py` | Raised bark thresholds |
| `orchestrators/coaching_engine.py` | Full reset on force button, sequential tricks |
| `xbox_hybrid_controller.py` | Removed crosses from hardcoded list |
| `configs/trick_rules.yaml` | Removed crosses, updated thresholds |
| `ai/models/config.json` | Removed crosses from behaviors |

---

### Current Spin Detection Thresholds
```python
# In geometric_classifier.py _check_spin()
total_aspect_change > 0.25  → +0.25
total_aspect_change > 0.45  → +0.20
max_aspect_delta > 0.12     → +0.15
total_center_move > 25      → +0.20
total_center_move > 50      → +0.15
aspect_range > 0.18         → +0.20
aspect_range > 0.35         → +0.15
width_range > 25            → +0.15
threshold = 0.40
```

---

### Unresolved Issues / Next Steps

1. **Spin detection still needs tuning** - Getting improved but not 100%
   - Spin latch added (2s hold) should help
   - May need further threshold adjustments

2. **Monitor for false positives** - Lowered thresholds could trigger on non-spins
   - Watch for false spin detections during sit/down/speak tricks

3. **Test spin latch effectiveness** - Verify dogs doing spin→sit get credit for spin

---

### Test Videos Created
- `recordings/wtf_multispins.mp4` - Multiple spins (before fixes)
- `recordings/10spins.mp4` - 10 spins test
- `recordings/coach_bezik_spin_*.mp4` - Various spin test sessions

---

### Important Notes for Next Session

1. **Spin latch active** - Spin detection holds for 2s after detection
2. **Temporal voting bypassed for spin** - Instant detection
3. **Force button = full reset** - Cancels session, advances to next trick
4. **Trick order is now sequential:** sit → down → spin → speak
5. **Crosses completely removed** - Including from xbox_hybrid_controller.py

---

## Session: 2026-01-15 ~04:30
**Goal:** Fix dog behavior detection - keypoints clustering in chest, false positives/negatives
**Status:** ✅ Critical fix committed and pushed

### CRITICAL FIX: Keypoint Decoding Bug
The root cause of all skeleton issues was found in the keypoint decoding formula:
```python
# WRONG (was clustering keypoints in chest):
kpts = (raw + cell) * stride

# CORRECT (per official Hailo YOLOv8 pose postprocessing):
kpts = (raw * 2 + cell) * stride
```

Fixed in ALL files:
- core/ai_controller_3stage_fixed.py
- core/pose_detector.py
- run_pi.py
- run_pi_1024x768.py

### Other Tuning:
- **Aspect Ratio Thresholds:** Lie < 0.75, Stand 0.75-1.05, Sit > 1.05
- **Bark Detection:** Base threshold 0.12 (was 0.55, ambient ~0.02)
- **Pose Validator:** Geometry checks disabled for front-facing dogs

### Key Insight:
Dogs crowd the robot for treats → expect close-up front-facing views.
Back legs/hips/tail often out of frame at close range.

### Commit: 106d192f - fix: Correct keypoint decoding formula for YOLOv8 pose model

---

## Session: 2026-01-14 ~23:00
**Goal:** Fix behavior detection false positives + implement pose validation system
**Status:** ✅ Complete

### Problems Solved
1. **CROSS over-detection** - Was at 99-100% constantly, fixed with tighter verification
2. **SPEAK false positives** - Raised bark gate thresholds
3. **No persistent logging** - Added rotating file handler
4. **Missing validation pipeline** - Implemented PoseValidator class

---

## Session: 2026-01-14 ~Late Night (03:30-04:00)
**Goal:** Fix video recording overlays, cross detection false positives, dog switching speed
**Status:** ✅ Complete

---

## Session: 2026-01-14 ~Morning
**Goal:** Training Programs system + Auto video recording in Coach mode
**Status:** ✅ Complete

---

## Session: 2026-01-13 ~Late Night
**Goal:** Build 9 launch features for WIM-Z demo/media readiness
**Status:** ✅ Complete

---

## Session: 2026-01-11 ~Evening
**Goal:** Mission engine fixes, missions, video recording, bark attribution
**Status:** ✅ Complete

---

## Session: 2026-01-10 ~Afternoon
**Goal:** Fix coaching engine "green lighting" + improve bark detection
**Status:** ✅ Complete
