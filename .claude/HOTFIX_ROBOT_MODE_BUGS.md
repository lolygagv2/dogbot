# ROBOT CLAUDE — HOTFIX: Mode Switching Bug Fixes

> Priority: HIGH — These fixes address confirmed root causes from the cross-team audit.
> Apply these fixes BEFORE continuing with the v1.3 audio streaming and mode system tasks.

---

## Context

Three audit reports (Robot, App, Relay) were cross-referenced. The relay is clean. The app has a workaround (Build 36 cooldown) that MASKS the robot-side bug but does not fix it. The robot has the actual root causes.

**The app-side 2-second cooldown is hiding your bug.** If a controller disconnect happens AFTER the cooldown window, the user will see the mode revert. These fixes must be applied.

---

## FIX 1: Remove Duplicate Controller Disconnect Handler (CRITICAL)

**File:** `main_treatbot.py` lines 1648-1654

**Problem:** This handler forcibly reverts to IDLE when in MANUAL mode upon any Xbox controller Bluetooth disconnect — even when the user is controlling via the app, not the controller. It also runs BEFORE the ModeFSM's smarter handler, winning the race.

**Current code (REMOVE OR REPLACE):**
```python
if event.subtype == 'controller_disconnected':
    current_mode = self.state.get_mode()
    if current_mode == SystemMode.MANUAL:
        self.logger.info("🎮 Xbox controller disconnected - returning to IDLE")
        self.state.set_mode(SystemMode.IDLE, "Controller disconnected")
```

**Replace with:**
```python
if event.subtype == 'controller_disconnected':
    current_mode = self.state.get_mode()
    if current_mode == SystemMode.MANUAL:
        # Only revert if no app user is connected — app owns mode control when connected
        app_connected = False
        try:
            if self.relay_client and hasattr(self.relay_client, '_app_connected'):
                app_connected = self.relay_client._app_connected
        except Exception:
            pass

        if app_connected:
            self.logger.info("🎮 Xbox controller disconnected - app is connected, keeping MANUAL mode")
        else:
            self.logger.info("🎮 Xbox controller disconnected - no app connected, returning to IDLE")
            self.state.set_mode(SystemMode.IDLE, "Controller disconnected, no app")
```

**If `_app_connected` doesn't exist on relay_client**, check what attribute or method tracks whether an app user is currently connected. Possible alternatives:
- `self.relay_client.is_connected` (relay WebSocket is up)
- Check connection_manager for active app sessions
- A flag set when `user_connected` message is received

The key logic is: **if an app is connected, do NOT revert mode on controller disconnect. Let the app control mode.**

**Testing:**
1. Connect from app, enter Drive/Manual mode
2. If Xbox controller service is running, trigger a Bluetooth disconnect (or `sudo systemctl restart xbox_controller`)
3. Verify robot stays in MANUAL mode
4. Verify app still shows MANUAL mode
5. Disconnect app, enter Manual via Xbox controller, disconnect controller → should revert to IDLE (no app to control it)

---

## FIX 2: Update ModeFSM pre_manual_mode on App Mode Change

**File:** `services/cloud/relay_client.py` — inside the `set_mode` command handler (around lines 507-528)

**Problem:** When the app sends `set_mode manual`, `relay_client.py` calls `state.set_mode(MANUAL)` directly. But the ModeFSM's `pre_manual_mode` is never updated. So if a controller disconnect later triggers ModeFSM's revert logic, it reverts to whatever `pre_manual_mode` was (likely IDLE from init), not the actual mode the user was in before driving.

**Current code** (find the set_mode handler in relay_client.py, likely looks something like):
```python
if command == 'set_mode':
    mode_name = data.get('mode', '')
    # ... validation ...
    self.state.set_mode(new_mode, reason)
```

**Add after the set_mode call:**
```python
if command == 'set_mode':
    mode_name = data.get('mode', '')
    source = data.get('source', 'unknown')
    # ... existing validation and set_mode call ...

    # Update ModeFSM context when entering MANUAL via app
    if mode_name == 'manual':
        try:
            if hasattr(self, 'mode_fsm') and self.mode_fsm:
                self.mode_fsm.last_manual_input_time = time.time()
                # pre_manual_mode should be set to whatever mode was active BEFORE this change
                # The state.get_mode() call above already changed the mode, so we need the previous mode
                # If you have access to previous_mode from the set_mode return or a local var, use that
                self.logger.info(f"☁️ Updated ModeFSM for app-initiated MANUAL mode (source={source})")
        except Exception as e:
            self.logger.warning(f"Could not update ModeFSM: {e}")

    # For ALL app-initiated mode changes, reset manual input time if in manual
    if self.state.get_mode() == SystemMode.MANUAL:
        try:
            if hasattr(self, 'mode_fsm') and self.mode_fsm:
                self.mode_fsm.last_manual_input_time = time.time()
        except Exception:
            pass
```

**Important:** You need to capture `previous_mode` BEFORE calling `state.set_mode()`. Adjust the code order:
```python
previous_mode = self.state.get_mode()  # Capture BEFORE changing
success = self.state.set_mode(new_mode, reason)

if success and mode_name == 'manual':
    try:
        if hasattr(self, 'mode_fsm') and self.mode_fsm:
            self.mode_fsm.pre_manual_mode = previous_mode
            self.mode_fsm.last_manual_input_time = time.time()
            self.logger.info(f"☁️ ModeFSM updated: pre_manual_mode={previous_mode}, timeout reset (source={source})")
    except Exception as e:
        self.logger.warning(f"Could not update ModeFSM: {e}")
```

**If you don't have direct access to `self.mode_fsm`**, find how it's referenced. Check:
- `self.treatbot.mode_fsm`
- Import from `orchestrators.mode_fsm` — `get_mode_fsm()`
- A global/singleton reference

**Testing:**
1. Set robot to Coach mode via app
2. Enter Drive mode via app (set_mode manual, source=drive_enter)
3. Check logs: should see `ModeFSM updated: pre_manual_mode=COACH`
4. Trigger controller disconnect (if applicable)
5. Verify ModeFSM would revert to COACH, not IDLE

---

## FIX 3: Reset Manual Timeout on App Commands (MEDIUM PRIORITY)

**File:** `services/cloud/relay_client.py` — inside the general command handler

**Problem:** The ModeFSM has a 5-minute manual timeout. When controlling via app, regular commands (motor, servo, etc.) go through `_handle_cloud_event()` in main_treatbot.py which resets the timeout. But `set_mode` goes through relay_client directly and doesn't reset it. Additionally, if the user is in manual mode but only watching video (no commands), the 5-minute timeout would fire.

**Fix:** In the relay_client command handler, reset the manual timeout for ANY command received while in MANUAL mode, not just set_mode:

```python
# At the top of the command handler, before processing specific commands:
if self.state.get_mode() == SystemMode.MANUAL:
    try:
        if hasattr(self, 'mode_fsm') and self.mode_fsm:
            self.mode_fsm.last_manual_input_time = time.time()
    except Exception:
        pass
```

**Also consider:** Should the 5-minute manual timeout apply AT ALL when an app is connected? If the user is watching video in Drive mode without sending commands, they're still actively using the robot. Consider disabling the timeout entirely when an app user is connected:

```python
# In ModeFSM._evaluate_manual_transitions() (mode_fsm.py around line 203-229):
# Add check: skip timeout if app is connected
if self._is_app_connected():
    return  # App controls mode, don't auto-timeout

# existing timeout logic follows...
```

**Testing:**
1. Enter Manual mode via app
2. Do NOT send any commands for 5+ minutes (just watch video)
3. Verify robot does NOT revert to IDLE
4. Disconnect app, enter Manual via controller, wait 5 minutes → should revert (no app)

---

## Verification After All Fixes

Run through these scenarios and confirm expected behavior:

| Scenario | Expected Result |
|----------|----------------|
| App sends set_mode manual, controller disconnects | Stays in MANUAL |
| App sends set_mode manual, wait 5+ min, no commands | Stays in MANUAL (app connected) |
| App in Coach, enters Drive, controller disconnects | Stays in MANUAL (app connected) |
| App in Coach, enters Drive, exits Drive | Reverts to COACH (pre_manual_mode) |
| No app, controller enters Manual, controller disconnects | Reverts to IDLE (correct) |
| No app, controller enters Manual, wait 5+ min | Reverts to previous mode (timeout, correct) |
| App disconnects while in Manual | Robot should revert to IDLE after grace period or immediately (your call) |

**Log every mode change with the new format from Task 2:**
```
MODE_CHANGE: COACH -> MANUAL | source=drive_enter | app_ts=2026-02-22T10:00:00Z | server_ts=2026-02-22T10:00:00.123Z
```

---

## Order of Operations

1. Apply Fix 1 (controller disconnect guard) — this is the critical one
2. Apply Fix 2 (pre_manual_mode update)
3. Apply Fix 3 (manual timeout reset)
4. Test all scenarios in verification table
5. Then continue with the v1.3 Task Batch (audio streaming, mode logging, etc.)
