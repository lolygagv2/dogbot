# Build 39 Post-Test Analysis - Robot Team

*Date: 2026-02-01*
*Test Window: 21:01-21:22 Local / 02:01-02:22 UTC*
*Updated: With App + Relay Answers*

---

## Executive Summary

Build 39 testing revealed **4 critical issues**. Root causes now identified:

| # | Issue | Root Cause | Fix Owner | Status |
|---|-------|------------|-----------|--------|
| 1 | Mission stuck on "Initializing" | **FIELD NAME MISMATCH** | ROBOT | TO FIX |
| 2 | AI confidence not showing | `update_dog_behavior()` never called | ROBOT | TO FIX |
| 3 | Servo tracking checkbox broken | Command received but `tracking_enabled=False` by default | ROBOT | TO FIX |
| 4 | MP3 upload 413 error | FastAPI body size limit | ROBOT | TO FIX |
| 5 | REST /missions 404 | Missing endpoint | RELAY or ROBOT | TO DISCUSS |

---

## Issue 1: Mission Mode - FIELD NAME MISMATCH (ROOT CAUSE FOUND)

### The Problem
The relay log shows:
```
[MISSION] Progress event from wimz_robot_01: status=None stage=None/None mission_type=None
```

All fields are `None` because **robot sends different field names than app expects**.

### What Robot Sends (WRONG)
```json
{
  "type": "mission_progress",
  "action": "started",
  "state": "waiting_for_dog",
  "current_stage": 1,
  "total_stages": 5,
  "mission_name": "sit_training"
}
```

### What App Expects (CORRECT)
```json
{
  "type": "mission_progress",
  "action": "started",
  "status": "waiting_for_dog",
  "stage_number": 1,
  "total_stages": 5,
  "mission_id": "sit_training"
}
```

### Field Mapping Required
| Robot Current | App Expected | Action |
|---------------|--------------|--------|
| `state` | `status` | RENAME |
| `current_stage` | `stage_number` | RENAME |
| `total_stages` | `total_stages` | OK |
| `mission_name` | `mission_id` | RENAME |
| - | `action` | ALREADY SENT |

### Files to Fix
- `orchestrators/mission_engine.py` - All `relay.send_event("mission_progress", {...})` calls
- `main_treatbot.py` - The `start_mission` response handler

---

## Issue 2: AI Confidence Not Showing on Video

### The Problem
Video overlay used to show "sit 34%" but now only shows "Dog" label.

### Root Cause
`update_dog_behavior()` in `core/dog_tracker.py:404` is **NEVER CALLED**.

The behavior detection works (logs show "stand:0.98") but the data never reaches dog_tracker, so video_track.py has no behavior data to display.

### Fix Location
`services/perception/detector.py:766-780` - After publishing behavior event, call:
```python
if hasattr(self.ai, 'dog_tracker') and self.ai.dog_tracker and dog_name:
    self.ai.dog_tracker.update_dog_behavior(dog_name, behavior_name, confidence)
```

---

## Issue 3: Servo Tracking Checkbox

### App Sends (Confirmed)
```json
{
  "command": "set_tracking_enabled",
  "data": {
    "enabled": true
  }
}
```

### Robot Handler (Already Exists)
`main_treatbot.py:1434-1444`:
```python
elif command == 'set_tracking_enabled':
    enabled = params.get('enabled', False) or params.get('data', {}).get('enabled', False)
    pantilt.set_tracking_enabled(enabled)
```

### The Problem
1. `tracking_enabled` defaults to `False` in pan_tilt.py:33
2. The command handler extracts from `params` but app sends in `data`
3. The extraction `params.get('data', {}).get('enabled', False)` should work, but needs verification

### Fix Options
1. **Option A**: Auto-enable tracking when entering COACH mode
2. **Option B**: Add debug logging to confirm command receipt
3. **Option C**: Both

### Recommended Fix
In `services/motion/pan_tilt.py`, auto-enable tracking in COACH mode:
```python
def _handle_coach_mode(self, dt: float) -> None:
    # Auto-enable tracking in coach mode (user can still disable via settings)
    if not self.tracking_enabled:
        self.logger.info("Auto-enabling tracking for COACH mode")
        self.tracking_enabled = True
```

---

## Issue 4: MP3 Upload 413 Error

### The Problem
FastAPI has a default request body size limit. Even after nginx change, still getting 413.

### Fix
Add body size limit middleware to `api/server.py`:

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# At app creation
app = FastAPI(...)

# Add after app creation
@app.middleware("http")
async def limit_upload_size(request: Request, call_next):
    # Allow up to 50MB for audio uploads
    if request.url.path in ["/audio/upload", "/upload_song"]:
        # Check content-length header
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 50 * 1024 * 1024:
            return Response("File too large", status_code=413)
    return await call_next(request)
```

Or configure uvicorn startup with `--limit-request-body 52428800`.

---

## Issue 5: REST /missions 404

### The Problem
App calls `GET /missions` to get mission catalog, but this endpoint doesn't exist.

### App's Expectation
```json
[
  {"id": "sit_training", "name": "Sit Training", "stages": 5},
  {"id": "down_training", "name": "Down Training", "stages": 4}
]
```

### Options
1. **Relay adds endpoint** - Static list or proxy to robot
2. **Robot adds endpoint** - `GET /missions` returns available missions
3. **App uses WebSocket** - `list_missions` command (already exists)

### Recommended
Robot already has `get_available_missions()` in mission_engine. Add REST endpoint in `api/server.py`:
```python
@app.get("/missions")
async def list_missions():
    from orchestrators.mission_engine import get_mission_engine
    engine = get_mission_engine()
    return engine.get_available_missions()
```

---

## Immediate Action Items (Robot Team)

### Priority 0 - CRITICAL (Mission Broken)
- [ ] **Fix mission_progress field names** in `orchestrators/mission_engine.py`
  - `state` → `status`
  - `current_stage` → `stage_number`
  - `mission_name` → `mission_id`

### Priority 1 - HIGH
- [ ] **Fix behavior display** - Call `update_dog_behavior()` in `services/perception/detector.py`
- [ ] **Fix tracking auto-enable** - Auto-enable in COACH mode OR add debug logging
- [ ] **Fix 413 error** - Add body size limit middleware in `api/server.py`

### Priority 2 - MEDIUM
- [ ] **Add /missions endpoint** - REST endpoint for mission catalog
- [ ] **Fix duplicate mission_stopped** - Investigate why sent twice

---

## Code Changes Required

### 1. orchestrators/mission_engine.py - Fix Field Names

Find all `relay.send_event("mission_progress", {...})` calls and update:

```python
# BEFORE (wrong)
relay.send_event("mission_progress", {
    "action": "started",
    "mission_name": mission_name,
    "state": session.state.value,
    "current_stage": session.current_stage,
    "total_stages": len(mission.stages),
})

# AFTER (correct)
relay.send_event("mission_progress", {
    "action": "started",
    "mission_id": mission_name,
    "status": session.state.value,
    "stage_number": session.current_stage + 1,  # 1-indexed for app
    "total_stages": len(mission.stages),
})
```

### 2. services/perception/detector.py - Fix Behavior Display

Around line 773, after `publish_vision_event('behavior_detected', ...)`:

```python
# Update dog tracker with behavior (for video overlay display)
if hasattr(self.ai, 'dog_tracker') and self.ai.dog_tracker and dog_name:
    self.ai.dog_tracker.update_dog_behavior(dog_name, behavior_name, confidence)
```

### 3. services/motion/pan_tilt.py - Auto-enable Tracking

In `_handle_coach_mode()`, after the method signature:

```python
def _handle_coach_mode(self, dt: float) -> None:
    # Note: tracking must be enabled via set_tracking_enabled command or settings
    if not self.tracking_enabled:
        return  # Keep existing behavior, but add logging
```

Or auto-enable:
```python
def _handle_coach_mode(self, dt: float) -> None:
    # Auto-enable if in coach mode (can be disabled via settings)
    if not self.tracking_enabled:
        self.tracking_enabled = True
        self.logger.info("Auto-enabled tracking for COACH mode")
```

### 4. api/server.py - Add /missions Endpoint

```python
@app.get("/missions")
async def list_available_missions():
    """Get list of available mission definitions for the app catalog"""
    from orchestrators.mission_engine import get_mission_engine
    engine = get_mission_engine()
    missions = engine.get_available_missions()
    # Format for app
    return [
        {
            "id": m["name"],
            "name": m.get("description", m["name"]),
            "stages": m.get("stage_count", len(m.get("stages", [])))
        }
        for m in missions
    ]
```

---

## Testing Checklist for Build 40

After implementing fixes:

1. [ ] Start mission via app → Button changes to "Stop", overlay shows stage progress
2. [ ] Video shows "sit 34%" confidence labels on detected dogs
3. [ ] Tracking checkbox enables camera movement in COACH mode
4. [ ] MP3 upload (5MB+) succeeds without 413 error
5. [ ] App mission catalog loads (no 404)

---

*Ready for implementation. Which fix should I start with?*
