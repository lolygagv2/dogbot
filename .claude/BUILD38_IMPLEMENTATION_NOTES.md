# BUILD38 IMPLEMENTATION NOTES

**Date:** 2026-02-01
**Author:** Robot Claude
**Purpose:** Document existing code locations BEFORE writing any code (per BUILD38 instructions)

---

## 1. Where is the coaching_engine's reward function?

**File:** `orchestrators/coaching_engine.py`
**Function:** `_state_success()` (lines 713-733)
**Called After:** Successful trick detection in `_state_watching()`

**What it does:**
1. **LED celebration** - `self.led.celebration_sequence(3.0)` (line 720-721)
2. **Play "good dog" audio** - `self._play_audio('good.mp3')` (line 724)
3. **Wait** - `time.sleep(1.5)` (line 725)
4. **Dispense treat** - `self._dispense_treat()` (line 728)
5. **Log session** - `self._log_session(success=True)` (line 731)

**Related Functions:**
- `_dispense_treat()` (lines 972-984) - Calls `self.dispenser.dispense_treat(dog_id, reason='coaching_reward')`
- `_play_audio()` (lines 935-970) - Tries custom voice via `play_command()`, falls back to file path

---

## 2. Where is the coaching_engine's trick detection/wait function?

**File:** `orchestrators/coaching_engine.py`
**Function:** `_state_watching()` (lines 643-711)

**What it does:**
1. Uses `self.interpreter.check_trick(expected_trick, dog_id=dog_name)` (line 690)
2. Checks `result.completed` for success (line 692)
3. Handles special case for `speak` trick via bark counting (lines 658-687)
4. Detects timeout using `watch_elapsed >= detection_window` (line 706)

**Key Variables:**
- `self.interpreter` = `get_behavior_interpreter()` (line 115)
- Detection window comes from `trick_rules.yaml` via `trick_rules.get('detection_window_sec', self.watch_duration)` (line 655)

---

## 3. Where is the coaching_engine's detection startup code?

**File:** `orchestrators/coaching_engine.py`
**Function:** `start()` (lines 193-239)

**What it does:**
1. Disables AGC for bark detection - `set_agc(False)` (line 200)
2. Resets FSM state - `self.fsm_state = CoachState.WAITING_FOR_DOG` (line 205)
3. Subscribes to vision events - `self.bus.subscribe('vision', self._on_vision_event)` (line 214)
4. Subscribes to audio events - `self.bus.subscribe('audio', self._on_audio_event)` (line 216)
5. Starts engine thread - `self.engine_thread.start()` (line 225)
6. Sets mode - `self.state.set_mode(SystemMode.COACH, ...)` (line 228)

**Note:** Coach mode does NOT directly start the detector - it ASSUMES the detector is already running in `services/perception/detector.py`. The detector runs continuously and publishes events to the bus.

---

## 4. Where is the servo tracking code? Last working commit?

**File:** `services/motion/pan_tilt.py`
**Function:** `_handle_coach_mode()` (lines 211-245)

**CURRENT STATUS:** **DISABLED** (Build 34)

The function currently does:
```python
def _handle_coach_mode(self, dt: float) -> None:
    # BUILD 34: Skip auto-tracking until servo control is fixed
    return  # <-- HARD DISABLED
```

**Last Working Commit:** `c719251b` (Build 24)

In Build 24, `_handle_coach_mode()` did NOT have the early return. It called:
- `_track_target(self.target_position, dt)` when target exists
- `_scan_for_target()` when target lost

**Key Tracking Functions (exist but unused):**
- `_track_target()` (lines 264-299) - PID tracking with smoothing
- `_pid_control()` (lines 301-328) - PID calculation
- `_scan_for_target()` (lines 330-369) - Sweep pattern

**PID Parameters (Build 24 vs Build 34):**
| Parameter | Build 24 | Build 34 |
|-----------|----------|----------|
| pan kp | 0.15 | 0.08 |
| pan ki | 0.005 | 0.002 |
| pan kd | 0.08 | 0.04 |
| tilt kp | 0.10 | 0.05 |

---

## 5. Where is the bounding box drawing code in video_track.py? Last working commit?

**File:** `services/streaming/video_track.py`
**Function:** `_add_overlays()` (lines 145-216)

**Bounding Box Drawing Code (lines 158-175):**
```python
if hasattr(ai, 'dog_tracker') and ai.dog_tracker:
    tracked = ai.dog_tracker.get_tracked_dogs()
    for dog_id, dog_data in tracked.items():
        bbox = dog_data.get('bbox')
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            dog_name = dog_data.get('name') or 'Dog'
            cv2.putText(frame, str(dog_name), (x1, y1 - 25), ...)
```

**PROBLEM IDENTIFIED:** The code checks `ai.dog_tracker.get_tracked_dogs()` but `bbox` is often missing from the returned data.

**Why bbox is missing:**
1. `dog_tracker.get_tracked_dogs()` returns data from `DogTracker` class in `core/dog_tracker.py`
2. The `bbox` field is only populated when detection events include it
3. Detection events from `ai_controller_3stage_fixed.py` may not always include bbox

**Git History:**
- `c89d584e` (Build 36) - Last change, added "Dog" default label
- `aca36f1c` (Build 25) - Earlier change
- `d29777ba` - Original WebRTC implementation

**Root Cause:** Not a video_track.py bug - the `dog_tracker` isn't being updated with bbox data properly.

---

## 6. For Each Task: Import Existing Code or Write New?

### P0-R1: Video Overlay Shows "IDLE"
**Action:** MODIFY existing code
**File:** `services/streaming/video_track.py`
**Change:** Replace `engine.active_session` check with `engine.get_mission_status()` call (thread-safe)

### P0-R2: Mission Uses Coach Pipeline
**Action:** IMPORT existing functions
**Current State:** Mission engine already has similar code copied from coach, but NOT imported
**Fix:** Create shared functions in coaching_engine that mission_engine can import

**Shared Functions Needed:**
```python
# In coaching_engine.py - expose as public methods:
def reward_dog(self, dog_id: str, trick: str) -> None:
    """LED + voice + treat + cooldown - used by both coach and mission"""

def wait_for_trick(self, trick: str, timeout: float, dog_id: str = None) -> TrickResult:
    """Wait for trick detection - wraps interpreter.check_trick with timeout"""
```

### P0-R3: AI Bounding Boxes
**Action:** FIX existing code
**Files:**
- `core/dog_tracker.py` - Ensure bbox is stored in tracked dogs
- `core/ai_controller_3stage_fixed.py` - Ensure bbox is passed to dog_tracker

### P0-R4: Dog Identification (Everything Is Elsa)
**Action:** MODIFY existing code
**File:** `core/dog_tracker.py`
**Change:** Only assign specific name with positive ArUco match, otherwise "Dog"

### P0-R5: Servo Tracking
**Action:** RESTORE from Build 24
**File:** `services/motion/pan_tilt.py`
**Change:** Remove the `return` statement in `_handle_coach_mode()`, restore Build 24 PID values

### P1-R6: Schedule Storage
**Action:** VERIFY/ENHANCE existing code
**Files:** `core/schedule_manager.py`, `core/mission_scheduler.py`
**Status:** Code exists, verify it works correctly

### P1-R7: MP3 Download
**Action:** WRITE NEW code
**Reason:** Current upload_song uses base64 over WebSocket - need new HTTP download approach
**File:** Add handler in `main_treatbot.py` for `download_song` command

---

## Key Code Locations Summary

| Component | File | Line(s) |
|-----------|------|---------|
| Coach reward function | `orchestrators/coaching_engine.py` | 713-733 |
| Coach trick detection | `orchestrators/coaching_engine.py` | 643-711 |
| Coach startup | `orchestrators/coaching_engine.py` | 193-239 |
| Mission reward (duplicated) | `orchestrators/mission_engine.py` | 849-891 |
| Servo tracking (disabled) | `services/motion/pan_tilt.py` | 211-245 |
| Bounding box drawing | `services/streaming/video_track.py` | 158-175 |
| Video overlay status | `services/streaming/video_track.py` | 240-375 |
| Dog tracker | `core/dog_tracker.py` | entire file |

---

## Verification Checklist (Pre-Coding)

- [x] Identified all code locations
- [x] Documented which existing code to import vs write new
- [x] Found the commit where servo tracking worked (c719251b)
- [x] Identified why bounding boxes don't show (bbox not in dog_tracker data)
- [x] Confirmed mission_engine duplicates coach code instead of importing
- [ ] **COMMIT THIS FILE BEFORE ANY CODE CHANGES**

---

*This document created per BUILD38_ROBOT_CLAUDE.md requirements*
