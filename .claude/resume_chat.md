# WIM-Z Resume Chat Log

## Session: 2026-01-29 (Build 29 - Critical Bug Fixes)
**Goal:** Fix 4 critical issues found during Build 28 testing
**Status:** In Progress - Fixes Applied

---

### Issues & Root Causes Identified

#### ISSUE 1: Mission mode NOT detecting dogs at all
**Symptom:** Mission starts, waits 5 minutes, times out. Zero pose_detected events. Coach mode works fine.
**Root Cause:** In `detector.py:572`, AI processing only runs for SILENT_GUARDIAN and COACH modes:
```python
ai_modes = [SystemMode.SILENT_GUARDIAN, SystemMode.COACH]  # MISSION not included!
```
**Fix:** Added `SystemMode.MISSION` to the `ai_modes` list
**File:** `services/perception/detector.py`

#### ISSUE 2: Voice commands not playing
**Symptom:** `play_voice` commands come in but no audio plays. No logs showing what happened.
**Root Cause:** Insufficient logging in the play_voice handler - couldn't see if path lookup failed or audio playback failed.
**Fix:** Added comprehensive INFO-level logging throughout the play_voice handler and voice_lookup module:
- Logs params received
- Logs path lookup result
- Logs audio service status
- Logs play_file result
**Files:** `main_treatbot.py`, `services/media/voice_lookup.py`

#### ISSUE 3: Motors keep getting disabled by watchdog
**Symptom:** `⏰ Watchdog timeout - no commands received, stopping motors` (repeated)
**Root Cause:** 2-second watchdog timeout. When WebRTC data channel has connectivity issues, commands stop flowing. Need to diagnose WHY commands stop.
**Fix:** Added more detailed logging showing time since last command and open_loop mode status to help diagnose
**File:** `core/hardware/proper_pid_motor_controller.py`

#### ISSUE 4: Random treat dispensing
**Symptom:** Robot dispenses treats randomly when user isn't commanding it
**Root Cause:** `reward_logic.py` processes bark_detected events and triggers treats when:
- Bark emotion is 'alert' or 'attention'
- Confidence >= 55%
- 40% probability roll succeeds
This runs in COACH, SILENT_GUARDIAN, MISSION modes (not IDLE/MANUAL).
**Fix:** Added comprehensive logging to trace exactly when bark rewards are evaluated and why they're granted/denied:
- Logs bark event with mode
- Logs when bark qualifies for reward evaluation
- Logs cooldown checks
- Logs daily limit checks
- Logs probability roll result
**File:** `orchestrators/reward_logic.py`

---

### Files Modified
| File | Changes |
|------|---------|
| `services/perception/detector.py` | Added MISSION to ai_modes list |
| `main_treatbot.py` | Enhanced play_voice/call_dog logging |
| `services/media/voice_lookup.py` | Changed DEBUG to INFO logging |
| `core/hardware/proper_pid_motor_controller.py` | Enhanced watchdog logging |
| `orchestrators/reward_logic.py` | Enhanced bark reward logging |

---

### Testing Required
1. **Mission mode:** Start a mission, verify dog detection works, verify pose detection triggers stage advancement
2. **Voice commands:** Test play_voice from app, check logs for full path trace
3. **Motor watchdog:** Drive via app, check if watchdog still triggers, examine new logs
4. **Bark rewards:** Monitor logs to identify when/why bark rewards trigger, adjust thresholds if needed

---

### Next Steps
1. Test all fixes with `sudo systemctl restart treatbot`
2. Monitor logs: `journalctl -u treatbot -f`
3. If bark rewards too frequent, consider:
   - Increasing confidence threshold (currently 0.55)
   - Reducing probability (currently 0.4)
   - Adding MISSION mode to the skip list

---

### Previous Session: 2026-01-29 (Build 28)
**Completed:**
- Mission mode lock/unlock - prevents mode changes during active missions
- Voice handler consolidation - unified `play_voice` + `call_dog` handlers
- Audio path cleanup - files moved to `talks/default/` and `songs/default/`
- Commit: `f0e6a07e`

---

## Important Notes/Warnings
- **MISSION mode now runs AI:** Detection pipeline is active in MISSION mode
- **Bark rewards can trigger treats:** In COACH/SG/MISSION, barks can trigger treats. Watch logs.
- **Motor watchdog is aggressive:** 2-second timeout may need adjustment based on new logs

---
