# Mode Switching Diagnostic Audit - Findings

**Date:** 2026-02-22
**Auditor:** Claude (Robot Team)
**API Contract:** v1.3

## Executive Summary

This audit investigated intermittent mode switching bugs where:
- Selecting Manual from dropdown would briefly switch then revert to idle
- Pressing Drive would show "Switching to Manual Mode" but robot announces "status idle"

After thorough code path analysis, **three potential root causes** were identified. Two have mitigations in place but could still trigger under specific conditions.

---

## Finding #1: ModeFSM Manual Timeout (POTENTIAL CAUSE)

### Code Location
`orchestrators/mode_fsm.py:203-229`

### Description
The ModeFSM has a 5-minute (300s) manual timeout that automatically returns to the previous mode when no "manual input" is detected. The timeout logic runs on a 1-second polling loop.

```python
self.timeouts = {
    'manual_timeout': 300.0,  # MANUAL -> previous mode (5 minutes no input)
}
```

### How It Could Cause The Bug

1. App sends `set_mode: manual`
2. Robot enters MANUAL mode
3. ModeFSM's `_evaluate_manual_transitions()` runs (every 1 second)
4. If `last_manual_input_time` was not updated, timeout triggers immediately
5. Robot reverts to `pre_manual_mode` (likely IDLE)

### Race Condition Window
The relay_client's `_handle_command()` calls `self.state.set_mode()` but does NOT notify the ModeFSM to update `last_manual_input_time`. The ModeFSM only updates this on:
- `manual_input_detected` system event (from Xbox controller)
- Internal `_transition_to()` calls

**If the app sends set_mode directly without going through ModeFSM, the timeout counter is never reset.**

### Current Mitigation
`main_treatbot.py:843-853`:
```python
# CRITICAL: Reset manual mode timeout when receiving app commands
current_mode = self.state.get_mode()
if current_mode == SystemMode.MANUAL:
    ...
    self.logger.debug("☁️ Reset manual mode timeout (cloud activity)")
```

However, this only runs inside `_handle_cloud_event()` for generic commands - the `set_mode` command in relay_client bypasses this entirely.

### Status: **PARTIALLY MITIGATED**

The mitigation exists but doesn't cover the `set_mode` command path in relay_client.py. A fix would require updating relay_client.py to also reset `mode_fsm.last_manual_input_time` when entering MANUAL mode via app command.

### Risk Level: **MEDIUM**
Could trigger if app sends set_mode command and user doesn't interact for 5 minutes - but unlikely to cause immediate revert.

---

## Finding #2: Controller Disconnect Event Triggers IDLE (CONFIRMED CAUSE)

### Code Location
`main_treatbot.py:1648-1654`

### Description
```python
if event.subtype == 'controller_disconnected':
    current_mode = self.state.get_mode()
    if current_mode == SystemMode.MANUAL:
        self.logger.info("🎮 Xbox controller disconnected - returning to IDLE")
        self.state.set_mode(SystemMode.IDLE, "Controller disconnected")
```

### How It Could Cause The Bug

If the Xbox controller service is running but experiences a brief disconnect (Bluetooth hiccup, USB reconnect, etc.), this triggers an immediate revert to IDLE. This happens unconditionally when in MANUAL mode.

### Sequence of Events
1. App sends `set_mode: manual` to enter Drive mode
2. Robot enters MANUAL mode
3. Xbox controller Bluetooth briefly disconnects (common on Pi)
4. `controller_disconnected` event fires
5. Robot immediately sets mode to IDLE
6. App receives mode_changed event showing IDLE

### Current Mitigation
The ModeFSM has better handling at line 301-308:
```python
elif event.subtype == 'controller_disconnected':
    current_mode = self.state.get_mode()
    if current_mode == SystemMode.MANUAL:
        target_mode = self.pre_manual_mode  # Returns to previous, not always IDLE
```

**However**, the main_treatbot.py handler runs BEFORE the ModeFSM handler processes the same event, causing a race where main_treatbot wins.

### Status: **CONFIRMED BUG**

This is a duplicate handler that conflicts with ModeFSM's more intelligent logic.

### Recommendation
Remove or guard the controller_disconnected handler in main_treatbot.py:1648-1654, or add a check to see if app is connected before reverting.

### Risk Level: **HIGH**
This is the most likely cause of immediate mode reverts, especially when Xbox controller is not in use but the service is running.

---

## Finding #3: Startup Forces IDLE Mode (EXPECTED BEHAVIOR)

### Code Location
`main_treatbot.py:213`

### Description
```python
self.state.set_mode(SystemMode.IDLE, "System starting")
```

### Analysis
This sets IDLE mode during system initialization. This is **expected behavior** - the system should boot in IDLE and wait for user/app commands.

### Status: **NOT A BUG**

This only runs once at startup and cannot cause the reported intermittent revert behavior during normal operation.

---

## Finding #4: WebSocket Reconnection Does NOT Reset Mode

### Code Location
`services/cloud/relay_client.py:825-901`

### Analysis
The reconnection loop in `_reconnect_loop()` and `_connect()` do NOT modify mode state. The only actions on reconnect are:
- Send `robot_connected` announcement
- Request fresh dog profiles
- Flush queued messages

### Status: **NOT A BUG**

WebSocket reconnection is clean and does not interfere with mode state.

---

## Finding #5: State Manager Atomicity (VERIFIED SAFE)

### Code Location
`core/state.py:206-254`

### Analysis
The `set_mode()` method uses `with self._lock:` (RLock) for thread safety:
```python
def set_mode(self, new_mode: SystemMode, reason: str = "", force: bool = False) -> bool:
    with self._lock:
        if new_mode == self.mode:
            return True
        # ... rest of mode change logic
```

All mode changes are atomic within a single lock acquisition. Multiple rapid set_mode calls cannot corrupt state.

### Race Condition Check
If two `set_mode` commands arrive in quick succession:
- First command acquires lock, changes mode, releases lock
- Second command acquires lock, changes mode, releases lock
- **Last writer wins** - this is correct and expected behavior

### Status: **NOT A BUG**

The state manager is properly thread-safe.

---

## Finding #6: ModeFSM pre_manual_mode Persistence (EDGE CASE)

### Code Location
`orchestrators/mode_fsm.py:59-60, 291-308`

### Description
```python
# Track mode before entering MANUAL (to return to it when controller disconnects)
self.pre_manual_mode = SystemMode.IDLE
```

### How It Could Cause Issues
If `pre_manual_mode` is not updated when entering MANUAL via app command (only via ModeFSM's own `_transition_to`), then controller disconnect would revert to IDLE even if user was in COACH before.

### Current Flow
- App sends `set_mode: manual`
- relay_client calls `state.set_mode(MANUAL)` directly
- ModeFSM's `pre_manual_mode` is NOT updated
- Controller disconnects
- ModeFSM sees MANUAL mode, returns to `pre_manual_mode` (which is IDLE)

### Status: **EDGE CASE BUG**

Only affects users who were in a non-IDLE mode before the app sent set_mode:manual.

### Risk Level: **LOW**
Most users will be in IDLE when entering Drive mode, but COACH users could be affected.

---

## Summary of Findings

| # | Finding | Status | Risk | Fix Required |
|---|---------|--------|------|--------------|
| 1 | ModeFSM Manual Timeout | Partial Mitigation | Medium | Yes - relay_client should reset timeout |
| 2 | Controller Disconnect → IDLE | Confirmed Bug | High | Yes - remove duplicate handler |
| 3 | Startup Forces IDLE | Expected | None | No |
| 4 | WebSocket Reconnect | Clean | None | No |
| 5 | State Manager Atomicity | Thread-Safe | None | No |
| 6 | pre_manual_mode Not Updated | Edge Case | Low | Optional |

---

## Recommended Fixes

### Fix 1: Remove Duplicate Controller Disconnect Handler
**File:** `main_treatbot.py:1648-1654`

Either remove this handler entirely (let ModeFSM handle it) or add a guard:
```python
if event.subtype == 'controller_disconnected':
    # Only revert if Xbox controller was the ONLY source of MANUAL mode
    # If app is connected, let app control mode
    if not self.relay_client or not self.relay_client._app_connected:
        current_mode = self.state.get_mode()
        if current_mode == SystemMode.MANUAL:
            self.state.set_mode(SystemMode.IDLE, "Controller disconnected, no app")
```

### Fix 2: Update ModeFSM on App Mode Change
**File:** `services/cloud/relay_client.py:507-528`

After setting MANUAL mode via app command, also update ModeFSM:
```python
if command == 'set_mode':
    # ... existing code ...
    if mode_name == 'manual':
        try:
            from orchestrators.mode_fsm import get_mode_fsm
            fsm = get_mode_fsm()
            fsm.last_manual_input_time = time.time()
            fsm.pre_manual_mode = self.state.get_mode()  # Save current mode
        except:
            pass
```

---

## Why The Bug Is Intermittent

The bug only manifests when:
1. Xbox controller service is running AND
2. Bluetooth experiences a brief disconnect OR
3. User was in MANUAL mode for 5+ minutes without app activity

Under normal conditions (Xbox controller not connected, or app actively sending commands), the bug does not trigger.

---

## Testing Recommendations

1. **Test with Xbox controller service disabled:**
   - `sudo systemctl stop xbox_controller`
   - Verify Drive mode works without reverts

2. **Test with logging enabled:**
   - Add MODE_CHANGE logging as implemented in Task 2
   - Monitor for `controller_disconnected` events

3. **Test 5-minute timeout:**
   - Enter MANUAL mode via app
   - Wait 5 minutes without any commands
   - Verify mode does not auto-revert (if app is connected)

---

*End of Audit*
