# WIM-Z Resume Chat Log

## Session: 2026-02-26 - WiFi Provisioning Fix & Parallel Boot
**Goal:** Fix broken WiFi provisioning (AP→station transition) and improve boot speed
**Status:** COMPLETE

---

### Problems Solved This Session

| # | Problem | Root Cause | Solution | Files Modified |
|---|---------|------------|----------|----------------|
| 1 | WiFi provisioning fails after submitting hotspot credentials | After killing hostapd, wireless driver stays in AP mode at kernel level — NM scans return nothing for 25s | Cycle wlan0 down/up to force station mode, wait for NM readiness before scanning | `services/network/wifi_manager.py` |
| 2 | Xbox controller blocked until WiFi provisioning completes | `treatbot.service` has `After=wifi-provision.service` + 10s sleep | Removed After dependency, removed sleep — parallel boot | `/etc/systemd/system/treatbot.service` |
| 3 | Slow fallback to AP mode (~30s) when no known WiFi | nmcli timeout 15s + polling loop 15s | Reduced nmcli timeout to 10s, bail on timeout immediately | `services/network/wifi_manager.py` |

---

### Key Code Changes Made

#### 1. WiFi Manager - AP→Station Transition Fix
**File:** `services/network/wifi_manager.py`
- **New `_wait_for_nm_ready()`**: Polls `nmcli device status` until wlan0 shows `disconnected` or `connected`
- **`stop_hotspot()` rewritten**: Cycles `ip link set wlan0 down/up` after killing hostapd to force driver back to station mode, then waits for NM readiness (20s timeout)
- **`save_credentials()`**: Removed redundant 3s sleep (stop_hotspot now handles the wait)
- **`_wait_for_ssid_in_scan()`**: Increased timeout to 30s, added diagnostic logging (shows sample SSIDs on first scan, logs what's visible on last attempts)
- **`try_connect_known()`**: Reduced from ~30s to ~12s worst case (10s nmcli + bail on timeout)

#### 2. Systemd Service - Parallel Boot
**File:** `/etc/systemd/system/treatbot.service` (NOT in git)
- Removed `After=wifi-provision.service` — treatbot starts in parallel
- Removed `ExecStartPre=/bin/sleep 10` — no longer needed
- Kept `Wants=wifi-provision.service` so it still starts

---

### Log Analysis (What Failed)
Timeline from `logs/wifi_provision.log`:
1. User submitted "FriendlyKimchi (2)" credentials via captive portal
2. Hotspot stopped, NM told to take over wlan0
3. Scanned for SSID for 25s (9 attempts) — **never found it**
4. `nmcli device wifi connect` failed: "No network with SSID found"
5. Fell back to AP mode... and NOW cached 47 networks (scans working!)
6. Root cause: wireless driver stuck in AP mode after hostapd killed

---

### Commits Made
- `e5c2999` - fix: WiFi provisioning AP→station transition, parallel boot, faster connect

---

### Treatbot1 Update Instructions Written
Provided user with manual steps for treatbot1:
1. `git pull origin main`
2. Manually edit `/etc/systemd/system/treatbot.service` (remove After=wifi-provision, remove sleep)
3. `sudo systemctl daemon-reload && sudo reboot`

---

### Relay Client Verified
Confirmed relay client has robust reconnection:
- `_reconnect_loop()` with exponential backoff (1s → 30s max)
- Runs forever while `_running=True`
- Auto-flushes queued messages on reconnect
- Safe to start treatbot before WiFi is up

---

### Next Steps
1. Test WiFi provisioning on treatbot1 with the new code
2. Test parallel boot — verify Xbox connects before WiFi
3. Test hotspot→new-network flow end-to-end

---

## Session: 2026-02-25 - Code Review: Xbox vs App Coach Mode & Spin Detection
**Goal:** Compare Xbox controller and app coach mode paths to verify identical code
**Status:** COMPLETE (read-only review, no code changes)

---

### Problems Solved This Session

| # | Question | Finding |
|---|----------|---------|
| 1 | Are VOICEMP3/songs/ root files vs default/ duplicates? | Root files are dead copies (~92MB). All code references `songs/default/`. Safe to delete root mp3s. |
| 2 | Is Xbox coach mode identical to app coach mode? | YES — both go through `POST /mode/set` → `mode_fsm.force_mode()` → `_on_mode_change()` → `coaching_engine.start()` |
| 3 | Is spin trick detection identical for Xbox vs app? | YES — same pipeline: `geometric_classifier._check_spin()` → temporal voting bypass → spin latch → 0.3s hold check |

---

### Key Findings

#### Coach Mode Entry — Identical
- Xbox: `cycle_mode()` → `POST /mode/set {"mode":"coach"}`
- App: relay `command: "mode"` → `POST /mode/set {"mode":"coach"}`
- Both converge at `api/server.py:875` → `mode_fsm.force_mode(COACH)` → event bus → `_on_mode_change()` → `coaching_engine.start()`

#### Spin Detection Pipeline — Identical
1. `geometric_classifier.py:131-134` — `_check_spin()` called FIRST (before pose classification)
2. `_check_spin()` uses bbox temporal analysis: aspect deltas, center movement, width variation over 4-8 frames
3. `ai_controller_3stage_fixed.py:919-922` — Spin bypasses temporal voting (instant motion, not held pose)
4. `behavior_interpreter.py:170-177` — Spin latch: ignores sit/stand/lie for 2s after spin detected
5. `behavior_interpreter.py:126` — Only 0.3s hold required for spin trick
6. Both Xbox and app use `coaching_engine.py:724` → `interpreter.check_trick("spin")`

#### Minor Gap Found
- Xbox `cycle_trick()` calls `reset_session_cooldown` before `force_trick` (xbox_hybrid_controller.py:1540)
- App `force_trick` command does NOT call `reset_session_cooldown`
- Impact: App force_trick may not start new session if cooldown active

#### VOICEMP3/songs/ Duplication
- `songs/` root: 12 files (Feb 2 dates) — NOT referenced by any code
- `songs/default/`: 13 files (Feb 22 dates) — ALL code points here
- `songs/default/` has one extra: `Trancewimz.mp3`

---

### Commits Made
*(None — read-only review session)*

---

### Next Steps
1. Consider deleting root `VOICEMP3/songs/*.mp3` files (~92MB unused duplicates)
2. Consider adding `reset_session_cooldown` to app's `force_trick` handler for parity with Xbox
