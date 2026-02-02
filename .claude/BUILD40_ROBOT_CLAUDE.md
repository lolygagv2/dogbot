# Build 40 - Robot Claude Instructions

**Date:** February 2, 2026
**Based on:** Build 39 test results cross-referenced across all three Claude instances
**Build 39 test window:** 21:01-21:22 local / 02:01-02:22 UTC

---

## STOP — Read This First

This is your **fourth build** trying to fix mission execution and AI display. Builds 34, 37, 38, and 39 all attempted these fixes. The pattern is: you rewrite things that already exist, break what was working, or send data the app can't parse.

**Before writing ANY code, do these two things:**

1. Read the API Contract v1.2 (in the project root or provided separately)
2. Run: `grep -rn "mission_progress\|send_event.*mission" orchestrators/mission_engine.py` and paste the output into a file called `BUILD40_CURRENT_STATE.md` so we can see exactly what field names you're currently using.

---

## Architecture Rules (Non-Negotiable)

- Robot state is authoritative — app displays what robot sends
- Field names in WebSocket events MUST match what the app expects (see API contract)
- Don't rewrite modules that work — import and call them
- HTTP for file transfers, not WebSocket
- Schedules live on the robot

---

## P0-R1: Fix mission_progress Field Names (CRITICAL — ROOT CAUSE FOUND)

### The Problem

All three Claude instances agree: the relay log proves it:
```
[MISSION] Progress event from wimz_robot_01: status=None stage=None/None mission_type=None
```

**FIELD NAME MISMATCH** — you're sending fields the app doesn't recognize.

### What You Send vs What App Expects

| You Send (WRONG)    | App Expects (CORRECT) | Action |
|---------------------|-----------------------|--------|
| `state`             | `status`              | RENAME |
| `current_stage`     | `stage_number`        | RENAME |
| `mission_name`      | `mission_id`          | RENAME |
| `total_stages`      | `total_stages`        | OK     |
| `action`            | `action`              | OK — but verify it's actually being sent |

### What to Do

Find every `send_event("mission_progress", ...)` call in `orchestrators/mission_engine.py` and `main_treatbot.py`. Change the field names:

```python
# EVERY mission_progress event must use EXACTLY these field names:
relay.send_event("mission_progress", {
    "action": "started",           # REQUIRED: "started", "stage_complete", "completed", "failed"
    "mission_id": mission_name,    # NOT "mission_name"
    "status": session.state.value, # NOT "state"
    "stage_number": session.current_stage + 1,  # NOT "current_stage", 1-indexed
    "total_stages": len(mission.stages),
})
```

### Verify After Fix

```bash
grep -n "mission_name\|current_stage\|\"state\"" orchestrators/mission_engine.py
# If any of those old field names appear in a send_event call, you missed one.
```

---

## P0-R2: Fix AI Confidence Display on Video Overlay

### The Problem

Video overlay used to show "sit 34%" but now only shows "Dog" label.

### Root Cause

`update_dog_behavior()` in `core/dog_tracker.py:404` is **never called**. The behavior detection works — logs show "stand:0.98" — but the data never reaches dog_tracker, so video_track.py has no behavior data to display.

### What to Do

In `services/perception/detector.py` around line 773, after publishing the behavior event, add:

```python
# After publish_vision_event('behavior_detected', ...)
if hasattr(self.ai, 'dog_tracker') and self.ai.dog_tracker and dog_name:
    self.ai.dog_tracker.update_dog_behavior(dog_name, behavior_name, confidence)
```

This is a ONE-LINE fix. Don't restructure the detection pipeline. Just bridge the data.

---

## P0-R3: Fix Servo Tracking Toggle

### The Problem

App sends `set_tracking_enabled` with `{"enabled": true}`. Robot has the handler but tracking doesn't engage because `tracking_enabled` defaults to `False`.

### What to Do — Two Changes

**Change 1:** Auto-enable tracking when entering COACH mode:
```python
# In services/motion/pan_tilt.py, _handle_coach_mode()
def _handle_coach_mode(self, dt: float) -> None:
    if not self.tracking_enabled:
        self.logger.info("Auto-enabling tracking for COACH mode")
        self.tracking_enabled = True
    # ... rest of existing logic
```

**Change 2:** Add debug logging to the command handler:
```python
# In main_treatbot.py, set_tracking_enabled handler
elif command == 'set_tracking_enabled':
    enabled = params.get('enabled', False) or params.get('data', {}).get('enabled', False)
    self.logger.info(f"[TRACKING] set_tracking_enabled received: {enabled}")
    pantilt.set_tracking_enabled(enabled)
    if self.relay_client:
        self.relay_client.send_event('tracking_enabled', {'enabled': enabled})
```

---

## P1-R4: Verify download_song Handler Exists

### Context

The MP3 upload chain is: App → Relay (HTTP POST) → Relay stages file → Relay sends `download_song` command to Robot → Robot fetches file via HTTP GET.

The nginx 413 is fixed. The app 422 (missing `device_id`) is being fixed by App Claude. Once that lands, the relay will send you this command:

```json
{
  "type": "command",
  "command": "download_song",
  "data": {
    "url": "/api/music/file/{file_id}",
    "file_id": "uuid-string",
    "filename": "song.mp3",
    "dog_id": "dog_001",
    "size": 3456789
  }
}
```

### What to Verify

```bash
grep -rn "download_song" main_treatbot.py services/ orchestrators/
```

If no handler exists, add one that:
1. Fetches `https://api.wimzai.com{url}` via HTTP GET
2. Saves to `VOICEMP3/songs/{dog_id}/{filename}`
3. Sends `upload_complete` event back: `{"success": true, "filename": filename}`

If the handler exists, confirm it constructs the full URL correctly (relay sends a relative path, robot needs to prepend the relay base URL).

---

## P1-R5: Coach Mode Flow — Emit Proper Events

### The Problem

Coach mode starts and stops correctly. But the robot is NOT sending progress events during coaching, so the app can't show real-time confidence data.

### What to Do

Find where coaching stages already happen in existing code and add `send_event` calls:

```python
# Greeting stage
self.relay_client.send_event('coach_progress', {'stage': 'greeting', 'dog_name': dog_name})

# Command issued
self.relay_client.send_event('coach_progress', {'stage': 'command', 'trick': trick_name})

# Watching (send periodically ~500ms)
self.relay_client.send_event('coach_progress', {'stage': 'watching', 'trick': trick_name, 'confidence': current_confidence})

# Success/reward
self.relay_client.send_event('coach_reward', {'behavior': trick_name, 'dog_name': dog_name, 'confidence': final_confidence})
```

**Do NOT rewrite the coaching engine.** Add send_event calls alongside existing logic.

---

## P2-R6: Add /missions REST Endpoint

### The Problem

App calls `GET /missions` and gets 404.

### What to Do

```python
# In api/server.py
@app.get("/missions")
async def list_available_missions():
    from orchestrators.mission_engine import get_mission_engine
    engine = get_mission_engine()
    missions = engine.get_available_missions()
    return [{"id": m["name"], "name": m.get("description", m["name"]),
             "stages": m.get("stage_count", len(m.get("stages", [])))} for m in missions]
```

---

## DO NOT Do These Things

1. **Do NOT rewrite mission_engine.py from scratch.** Fix the field names, that's it.
2. **Do NOT disable servo tracking.** Fix it so it auto-enables in coach mode.
3. **Do NOT restructure the detection pipeline.** Add the one-line bridge call.
4. **Do NOT change the WebSocket message envelope format.** Only fix field names inside `data`.

---

## Testing Checklist

```bash
# 1. Field names fixed
grep -n "mission_name\|current_stage\|\"state\"" orchestrators/mission_engine.py
# Should return ZERO results in send_event calls

# 2. Behavior bridge exists
grep -n "update_dog_behavior" services/perception/detector.py

# 3. Tracking auto-enable exists
grep -n "Auto-enabling tracking" services/motion/pan_tilt.py

# 4. download_song handler exists
grep -n "download_song" main_treatbot.py

# 5. /missions endpoint exists
grep -n "list_available_missions\|GET.*missions" api/server.py
```

### Live Test Sequence
1. Start mission → Relay logs show `status=` (not `state=`) and `stage_number=` (not `current_stage=`)
2. Enter coach mode → Tracking auto-enables
3. Video overlay → Shows "sit 34%" labels
4. MP3 upload from app → Robot receives `download_song` and fetches file

---

*Build 40 — Fix the field names. Bridge the behavior data. That's 80% of the problems.*
