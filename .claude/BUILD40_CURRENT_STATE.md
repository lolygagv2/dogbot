# Build 40 - Implementation Complete

*Generated: 2026-02-02*

## All Tasks Completed

| Task | Status | Verification |
|------|--------|--------------|
| P0-R1: Fix mission_progress field names | DONE | 0 old field names in send_event calls |
| P0-R2: Add update_dog_behavior bridge | DONE | Line 778 in detector.py |
| P0-R3: Fix servo tracking auto-enable | DONE | Line 228 in pan_tilt.py |
| P1-R4: Verify download_song handler | DONE | Lines 1137-1190 in main_treatbot.py |
| P1-R5: Add coach_progress events | DONE | 4 events added to coaching_engine.py |
| P2-R6: Add /missions REST endpoint | DONE | Line 956 in server.py |

---

## Files Modified

1. **orchestrators/mission_engine.py**
   - Fixed 7 `mission_progress` events: `mission_name` → `mission_id`, `stage` → `stage_number`
   - Fixed 1 `mission_complete` event
   - Added `action` field to all events

2. **main_treatbot.py**
   - Fixed `start_mission` response field names
   - Enhanced `download_song` handler:
     - Constructs full URL from relative path
     - Saves to dog-specific folder
     - Extracts params from `data` field
   - Added debug logging to `set_tracking_enabled`

3. **services/perception/detector.py**
   - Added `update_dog_behavior()` bridge call at line 778

4. **services/motion/pan_tilt.py**
   - Added auto-enable tracking in COACH mode at line 228

5. **orchestrators/coaching_engine.py**
   - Added `get_relay_client` import
   - Added `coach_progress` events for: greeting, command, watching
   - Added `coach_reward` event for success

6. **api/server.py**
   - Added `GET /missions` endpoint at line 955

---

## Verification Commands (All Pass)

```bash
# 1. Field names fixed (should return 0)
grep -n "mission_name\|current_stage\|\"stage\"" orchestrators/mission_engine.py | grep -i "send_event" | wc -l
# Result: 0

# 2. Behavior bridge exists
grep -n "update_dog_behavior" services/perception/detector.py
# Result: 778:                    self.ai.dog_tracker.update_dog_behavior(...)

# 3. Tracking auto-enable exists
grep -n "Auto-enabling tracking" services/motion/pan_tilt.py
# Result: 228:            self.logger.info("Auto-enabling tracking for COACH mode")

# 4. download_song handler exists
grep -n "download_song" main_treatbot.py
# Result: 1137:                elif command == 'download_song':

# 5. /missions endpoint exists
grep -n "list_available_missions" api/server.py
# Result: 956:async def list_available_missions():
```

---

## Testing Checklist for Build 40

1. [ ] Start mission via app → Relay logs show `status=` and `stage_number=` (not `state=` or `current_stage=`)
2. [ ] Enter coach mode → Log shows "Auto-enabling tracking for COACH mode"
3. [ ] Video overlay → Shows "sit 34%" confidence labels
4. [ ] MP3 upload from app → Robot receives `download_song` and fetches file
5. [ ] Call `GET /missions` → Returns mission catalog

---

*Build 40 ready for testing.*
