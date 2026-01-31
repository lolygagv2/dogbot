# Build 32 - Test Session Issues Analysis
**Test Date:** 2026-01-30 20:44-21:10 (25 minutes)
**Tested By:** Morgan

---

## Critical Findings

### MEMORY SPIKE DURING TEST (CRITICAL)
Logs from 21:25-21:36 show **memory at 95-99%** causing command rejection:
```
21:25:16 - MEMORY GATE: Rejecting command at 95.3% memory
21:25:29 - Played error audio alert (Wimz_errorlogs.mp3)
21:27:40 - Memory hit 98.9%
```
**Root Cause:** Unknown - needs investigation. May be related to mission mode loading or audio playback.

---

## Issue Breakdown by Side

### ROBOT SIDE (Can Fix)

| # | Issue | Status | Priority |
|---|-------|--------|----------|
| 4 | Voice files not deleted when dog is deleted | TO FIX | HIGH |
| 8a | Mission shows "???? MISSION" | TO FIX | HIGH |
| 8b | Mission stuck on "Waiting for Dog" even when dog visible | TO FIX | HIGH |
| 9 | Mission AI framework loading question | VERIFY | MEDIUM |
| 10 | Mode status single source of truth | VERIFY | MEDIUM |

### APP/RELAY SIDE (Document for App Team)

| # | Issue | Status | Priority |
|---|-------|--------|----------|
| 0 | Login/logout error - can't re-login after sign out | APP SIDE | HIGH |
| 1 | Music upload fails - "disconnected" during upload | RELAY SIDE | HIGH |
| 3 | Dog profile photo - 2nd change doesn't show preview | APP SIDE | MEDIUM |
| 6 | All users share all dogs - should be per-user | RELAY SIDE | HIGH |
| 6b | Robot pairing error should say "Robot unavailable" | RELAY SIDE | LOW |
| 7 | User persistence and email validation | RELAY SIDE | MEDIUM |

### CONFIRMED WORKING

| # | Feature | Status |
|---|---------|--------|
| 2 | Music play/stop | WORKS |
| 4 | Voice commands stick to dog | WORKS |
| 5 | Dogs and voices persist after restart | WORKS (mostly) |

---

## Detailed Analysis

### Issue 4: Voice Files Not Deleted When Dog Deleted
**Location:** `services/media/voice_manager.py`

**Current Behavior:**
- Voice files are stored in `VOICEMP3/talks/dog_{id}/`
- When dog is deleted from app, the folder and files remain
- No cleanup function called on dog deletion

**Fix Required:**
Add `delete_dog_voices(dog_id)` function to VoiceManager that:
1. Deletes entire `VOICEMP3/talks/dog_{id}/` directory
2. Gets called when dog deletion command received from relay

**Affected folders currently on device:**
```
VOICEMP3/talks/dog_1769381269569/ (4 files: good.mp3, name.mp3, no.mp3, treat.mp3)
VOICEMP3/talks/dog_1769441492377/ (7 files: come, down, good, name, no, sit, spin)
VOICEMP3/talks/dog_1769681772789/
VOICEMP3/talks/dog_1769683310861/
VOICEMP3/talks/dog_1769711863860/
```

---

### Issue 8a: Mission Shows "???? MISSION"
**Symptom:** App displays "???? MISSION" text

**Root Cause Analysis:**
1. Robot sends mission events via relay with `mission_name` field
2. App likely receiving null/undefined mission name
3. Check: `mission_engine.py` `_send_mission_status()` at line 904

**Current Code (mission_engine.py:904-919):**
```python
def _send_mission_status(self, status: str, trick: str = None):
    relay.send_event("mission_progress", {
        "status": status,
        "trick": trick,
        "stage": session.current_stage + 1,
        "total_stages": len(session.mission.stages),
        "dog_name": session.dog_name,
        "rewards": session.rewards_given,
    })
```

**Issue:** Missing `mission_name` field in the event!

**Fix:** Add mission name to all mission events:
```python
"mission_name": session.mission.name if session else None,
```

---

### Issue 8b: Mission Stuck on "Waiting for Dog"
**Symptom:** Dog is clearly visible with bounding boxes, but mission doesn't progress

**Root Cause Analysis:**
Mission engine has its own `dogs_in_view` dict that tracks:
- First seen time
- Frames seen / total frames
- Presence ratio

For a dog to be "eligible", it needs:
1. `time_elapsed >= 3.0s` (DETECTION_TIME_SEC)
2. `presence_ratio >= 0.66` (PRESENCE_RATIO_MIN)

**Potential Issues:**
1. Mission engine's `dogs_in_view` not getting populated from vision events
2. `_on_vision_event` handler not receiving `dog_detected` events when in MISSION mode
3. Dog tracker might not be publishing events when mission mode is active

**Check:** Vision events subscribed at line 244:
```python
self.bus.subscribe("vision", self._on_vision_event)
```

The handler `_handle_dog_detected` updates `dogs_in_view` at line 1316-1346.

**Likely Root Cause:**
The `DetectorService` may not be publishing `dog_detected` events when mode is MISSION, or the event bus subscription isn't active.

---

### Issue 9: Mission AI Framework Loading

**Question:** Does mission load the correct AI framework (YOLO + pose)?

**Answer:** YES - Mission uses the same services as coach mode:
- `get_behavior_interpreter()` - for pose detection
- Vision events from `DetectorService` - for dog detection
- These run continuously regardless of mode

**The problem is likely event routing, not model loading.**

---

### Issue 10: Mode Status Persistence

**Current Architecture:**
- `core/state.py` - Single `SystemState` singleton with `_mode` attribute
- Mode changes logged and broadcasted via event bus
- `state.lock_mode()` used during missions to prevent interruption

**Mode State Sources:**
1. `get_state().mode` - Python runtime state
2. API endpoint `/mode` - Returns current mode
3. WebSocket events - Broadcast mode changes

**Potential Issue:** If relay disconnects, app may lose sync with robot mode.

**Recommendation:** Ensure mode is sent on every reconnection and periodically.

---

## APP/RELAY TEAM DOCUMENTATION

### Issue 0: Login/Logout Flow Broken

**User Report:**
1. Logged in successfully
2. Signed out after a few minutes
3. Got error on robot message
4. Could not re-login to app
5. Had to restart app to login again
6. After restart, got same error when connected to robot

**Questions for Relay Team:**
- What error message was displayed?
- Is the device-user session not being cleared on logout?
- Is there a stale token/session issue?

---

### Issue 1: Music Upload Fails

**User Report:**
- Upload appears to start ("uploading...")
- Immediately shows "disconnected/reconnecting"
- File not found after upload

**Current Music Structure:**
```
VOICEMP3/songs/default/  - Default songs
VOICEMP3/songs/dog1/     - Legacy?
VOICEMP3/songs/user/     - User uploads?
```

**Questions:**
- What endpoint is being called for music upload?
- Is there a file size limit?
- Is the upload going to the correct path?

**Suggestion:**
- Add MP3 files directly to `VOICEMP3/songs/` folder
- Add delete capability (music icon → confirm delete)

---

### Issue 3: Dog Profile Photo Refresh

**User Report:**
1. First "change photo" works - preview shows new photo
2. Second "change photo" in same session - uploads but doesn't show preview
3. Requires app restart to see new photo

**Likely Cause:** App caching the photo URL and not refreshing on update.

**Fix (App Side):**
- Add cache-busting query param to photo URL
- Or force ImageView reload after upload completes

---

### Issue 6: Dogs Shared Across All Users

**User Report:**
All users can see all dogs, not just their own dogs.

**Expected Behavior:**
- Each user should only see their own dogs
- Each user should only access their dogs' custom audio
- Dogs should be scoped to user_id

**Current Risk:**
- User A can play audio for User B's dogs
- Privacy/security concern

**Fix (Relay Side):**
- Dogs table needs `user_id` foreign key
- API queries must filter by authenticated user
- Robot receives only dogs for connected user

---

### Issue 6b: Robot Pairing Error Message

**Current:** Shows generic error when another user tries to pair
**Expected:** Should say "Robot unavailable" or "Robot paired to another user"

---

### Issue 7: User Persistence & Email Validation

**Questions:**
1. Are users stored server-side (yes, in Supabase)?
2. Do users persist after restart?
3. Can we add email validation?
4. Can we implement password reset flow?

**Desired Features:**
- Email verification on signup
- Password reset via email
- "Forgot password" flow

---

## Recommended Fix Order

### Robot Side (This Session)
1. **Issue 8a** - Add `mission_name` to mission events (5 min)
2. **Issue 8b** - Debug mission dog detection (investigate)
3. **Issue 4** - Add dog voice cleanup function (15 min)
4. **Issue 10** - Verify mode sync on reconnect (verify)

### App/Relay Side (Document for Team)
1. **Issue 6** - User-scoped dogs (security priority)
2. **Issue 0** - Login/logout flow fix
3. **Issue 1** - Music upload path/debugging
4. **Issue 3** - Photo cache invalidation
5. **Issue 7** - Email validation/password reset

---

## Memory Investigation Needed

The memory spike to 99% during testing (21:25-21:36) needs investigation:
- What was running at that time?
- Was mission mode loading large files?
- Check if pose models are being loaded multiple times
- Review mission audio file sizes
