# WIM-Z Resume Chat Log

## Session: 2026-05-31 (treatbot5) — fixed phone→robot local-AP demo (3 bugs)

**Robot:** treatbot5
**Status:** ✅ Complete — fixes live on treatbot5 + committed and pushed to `main` (`c5e85b1`).

### Problem
Phone→robot Direct Local AP demo was broken: login worked and commands partly worked, but (1) video showed "connecting" forever, (2) connection felt shaky, (3) after ~2 voice commands the phone dropped and `WIMZ-Demo` AP never came back. Required a power-cycle.

### Root causes (all confirmed in treatbot5 logs, prev boot `-b -1` + `logs/treatbot.log.1`)
1. **AP killed itself the instant a phone associated.** `_wifi_monitor_loop` (main_treatbot.py) tore down the hotspot every `reconnect_interval`=120s to hunt for known wifi, with NO client check. Logs caught it: phone associated 17:31:43, `Stopping hotspot` fired the *same second*, ~38s of no network, then user power-cycled. The rejoin also failed with NM error "A 'wireless' setting is required if no AP path was given."
2. **MJPEG `/video/feed` opened a 2nd `Picamera2()`** while DetectorService already owns the single camera → always failed → "connecting forever." WebRTC can't work on the AP (no STUN/TURN reachable) — that's expected; MJPEG is the local-AP video path.
3. **RelayClient hammered `wss://api.wimzai.com` every 30s** on the AP (no internet) → churn.
   - NOTE: the "92% CPU" SafetyMonitor alert was **Claude Code itself** running on the Pi, NOT a robot load problem. Zero CPU-high alerts during the actual demo.

### Fixes (commits `42b22fc`, `c5e85b1` — now on `main`)
- `main_treatbot.py` — defer AP rejoin while any STA associated; interval 120→300s.
- `services/network/wifi_manager.py` — new `has_associated_stations()` via `iw dev wlan0 station dump` (L2 association — fires immediately, unlike ARP which misses a just-associated phone; this was the Flutter team's catch). `try_connect_known()` now brings saved profiles up by name (`nmcli connection up <name>`) to fix the NM rejoin error.
- `api/server.py` — `/video/feed` streams `detector.get_last_frame_with_timestamp()` (same source as WebRTC video_track) + client-disconnect handling + 10s idle bail-out.
- `services/cloud/relay_client.py` — `_reconnect_loop` skips dialing while `is_ap_mode()` (checked via run_in_executor so the blocking pgrep doesn't stall the asyncio loop).

### Verified
- MJPEG: 45 frames / 1.36 MB in ~2s, valid JPEG. Contract confirmed: `GET /video/feed` → 200, `content-type: multipart/x-mixed-replace; boundary=frame`, no auth.
- Two clean service restarts (relay connect + monitor start, no errors).

### Flutter app (separate repo) shipped Build 115 in parallel
- MJPEG endpoint corrected + probed (`/video/feed` then legacy `/camera/stream`).
- Cleartext HTTP allowed for LAN (Android network_security_config + iOS NSAllowsLocalNetworking) — without it, plain http to 192.168.4.1 is blocked = "connecting forever".
- WebRTC give-up shortened to 6s in local mode → flips to MJPEG fast.

### Next steps / unresolved
1. **Real-phone soak test (NOT yet done):** join `WIMZ-Demo`, drive/stream >5 min. Success = NO `Stopping hotspot` in log while a STA is associated; expect periodic `STA associated on AP (...) — keeping AP up, deferring rejoin`. Couldn't self-test — forcing AP mode cuts the Pi's own wlan0 link.
2. Possible dual-AP-manager interplay (root `wifi_provision.service` + main_treatbot monitor) — watch if issues persist (see commit 48885bd history).
3. Memory updated: `project_direct_ap.md` (was stale "fully working").

---

## Session: 2026-05-30 (treatbot3) — git sync + stash cleanup, committed tb3 per-unit tuning

**Duration:** ~5 min
**Robot:** treatbot3
**Status:** ✅ Complete — clean tree, in sync with origin/main.

### What happened
- `git pull origin main` was blocked: local git is configured for `pull --rebase`, which refuses a dirty tree. Working tree had uncommitted per-unit tuning in `config/robot_profiles/treatbot3.yaml` (carried over uncommitted from the 2026-05-29 session — see entry below).
- Diagnosed before touching anything: the only incoming commit `40fc2ec` (treatbot2 battery divider 3.729→6.0) touches **treatbot2.yaml only** → zero overlap with the local tb3 change. Safe to merge.
- Stashed tb3.yaml → fast-forwarded `c89f448 → 40fc2ec` → `stash pop` restored the change cleanly.
- Per user instruction, **committed** the tb3 tuning as `f1f12b5`:
  - camera `tilt_min` −29 → −35 (extend downward look)
  - dispenser `steps_per_slot` 147 → 150, `step_delay` 0.0060 → 0.0050
- Pushed to origin/main. (Later in session, also pulled treatbot4 commits `3466bad`/`1069191` that landed concurrently — treatbot4.yaml dispenser/tilt tuning + a new battery ADC channel test; clean ff, no overlap.)
- **Cleared all 11 accumulated git stashes** (old WIP/filter-branch cruft going back months). Backed the list up to `/tmp/dropped_stashes_20260530.txt` first.

### State at session end
- Clean working tree, `main` in sync with `origin/main`.
- Protected files untouched (notes.txt, robot_config.yaml, docs/).
- No new files, no structure changes.

### Next session / watch items
- Validate tb3 dispenser tuning live (steps_per_slot 150 / step_delay 0.0050) — confirm reliable single-slot advance without jams. **Service not restarted** this session, so the committed yaml values aren't live on tb3 until `sudo systemctl restart treatbot.service`.
- Confirm camera tilt_min −35 doesn't hit the physical servo floor on tb3.

### Notes
- Local git config is `pull --rebase` → always stash before pulling when per-unit yaml is mid-edit.
- Dropped stashes recoverable for ~2 weeks via `git fsck --unreachable | grep commit` if one turns out to be needed.

---

## Session: 2026-05-30 (treatbot5) — CRITICAL motor runaway fix (dead-man's watchdog) + local-transport plan

**Robot:** treatbot5 (live); safety fix deployed fleet-wide.
**Status:** ✅ Safety fix committed + pushed to origin/main (`c89f448`). Live + watchdog-verified
on **treatbot5 only** (local: `active`, `Motor safety watchdog started (timeout=500ms)`).
⚠️ **treatbot1/2/3/4 NOT yet deployed — Morgan is updating them manually.** Each needs:
`cd /home/morgan/dogbot && git pull --ff-only origin main && sudo systemctl restart treatbot.service`
then confirm the watchdog log line. (Agent fleet-deploy failed: non-interactive ssh runs from
/home/morgan and the snapshot didn't cd into dogbot; service stop also takes ~15-30s.)

### THE critical bug (user nearly lost a robot)
While holding "forward" in the app, the app↔robot link dropped and **the robot kept driving
forward indefinitely** — user had to physically catch it. Root cause: `MotorCommandBus.send_command`
latched the last PWM into the hardware with **no watchdog**. The stop command never arrived on
disconnect, so the motor held forever. The Xbox path only survived via its own 50ms re-send loop,
whose comment "keep the motor-bus watchdog satisfied" referenced a watchdog that **was never
implemented**; the app/WebRTC path had no backstop at all.

### Fix (commit c89f448 — `core/motor_command_bus.py` + `services/streaming/webrtc.py`)
1. **Dead-man's watchdog thread in MotorCommandBus**: forces motors to 0 if no fresh command in
   500ms. App pulses every ~50ms while held (user confirmed) → tolerates ~10 dropped pulses, no
   false stops. Only acts on a stale *movement* command (stale zero already safe; also prevents
   re-firing every tick). Protects ALL sources (app/WebRTC/API/network drop/app crash).
2. **WebRTC teardown stops motors**: `_stop_motors_safety()` from `_cleanup_connection`
   (failed/disconnected/closed) and data-channel `on_close`. Plain-WS path already did this on
   disconnect; the data-channel path carrying motor cmds did not.
- No protocol/signaling changes. Fleet-safe (routes through send_command for both controller types).

### Local-mode transport decision (agreed, NOT yet implemented on robot)
WebRTC video on the LAN is fragile (this session: connected, held ~100s, dropped → "buffering" →
every reconnect timed out). Decision = **HYBRID**: video over **MJPEG** (`GET /video/feed`, already
exists), control over **`/ws/local`** (already used), **WebRTC audio-ONLY** (user requires two-way
audio). Robot TODO: make local WebRTC offer audio-only; confirm /video/feed FPS/quality.
**Deferred** because the Flutter Claude is editing app signaling in parallel — landing a protocol
change mid-rewrite would collide.

### App-side (handed to Flutter Claude, NOT robot bugs)
- Coordination note: `/home/morgan/.claude/plans/flutter-robot-coordination-notes.md` (transport
  contract, must/must-not, endpoints).
- **"Robot 02" shown for Robot 05**: app's own log shows `new=wimz_robot_02`; robot is correctly
  `wimz_robot_05` and **never sends a device-id over /ws/local** (verified `api/ws.py`). Stale
  cached device label in the app — app-side fix. (User flagged as trust/"fraud" risk — high
  priority app-side, but not a robot lie.)
- "Manage Devices" sometimes hangs (likely same device-state path).
- The `100.x` ICE candidate = Tailscale/cellular, harmless (Private Relay confirmed OFF).

### Next session
1. Land audio-only local WebRTC + MJPEG confirmation (after Flutter app signaling settles).
2. Deploy `c89f448` to treatbot1 + treatbot4 when reachable.
3. Reconnect-after-drop robustness (leftover-session blocking new offers — investigate robot half).
4. Pre-existing uncommitted `treatbot5.yaml` gimbal change still in working tree (pan_min -180,
   tilt_min 80, tilt_max 290) — contradicts in-file calibration comments; confirm physically tested
   before committing.

---

## Session: 2026-05-29 (treatbot3) — Gimbal D-pad fix + camera API method-name bugs, fleet-wide deploy + tb2 reconcile

**Duration:** ~ full session
**Robot:** treatbot3 (this unit); changes deployed fleet-wide via origin/main
**Status:** ✅ 3 commits pushed to main (`c863842`, `2e73c35`, `41b834a`) + tb2 dispenser commit (`adabeb6`). All 5 robots pulled + restarted + endpoints verified live. tb2 reconciled off its divergent branch.

### What was reported / asked
1. App gimbal D-pad: pressing "down" snapped the camera UP ~30° and refused to go lower; Center worked. Then user removed app-side clamps and wanted the robot (yaml) to be the sole limit gate.
2. `/camera/center` failed in app with `'PanTiltService' object has no attribute 'center'`.

### Root cause #1 — hardcoded servo angles (gimbal D-pad)
The app D-pad uses the `/servo/pan` + `/servo/tilt` contract endpoints (NOT `/camera/pantilt`, which the drive-screen joystick uses). Old code did `internal = 90 + angle`, clamp pan `0-180` / tilt `45-135`. That assumes a generic 90-centered servo, but each gimbal's physical PWM→angle map differs (servo horn spline, mount offset). treatbot3's level horizon is `tilt_center: -12`; the 45-135 clamp's floor (internal 45) sat ~57° ABOVE level, so the D-pad could never tilt down. PWM scale is fixed in `_angle_to_pulse`: 500–2500µs = −90°..+270° (~5.56µs/°); `pan_min` below −90 is dead (treatbot3 had −135, corrected to −90).

**Fix (`c863842`):** rewrote `/servo/pan` + `/servo/tilt` to `target = pantilt.center_{pan,tilt} + request.angle`, then `move_camera()` which clamps to yaml `pan_limits`/`tilt_limits` via `_move_to_position`. Verified every runtime servo path funnels through that clamp → robot is sole safety gate. Live-tested ±999 offsets clamp exactly to yaml (pan −90..269, tilt −29..238).

### Root cause #2 — nonexistent method names in API (3 endpoints)
`PanTiltService` has NO `center()`, `get_position()`, `set_pan()`, or `set_tilt()`. Real methods: `center_camera(reason=...)`, `get_status()` (→ `current_position` dict), `move_camera(pan=,tilt=)`.
- `2e73c35`: `/camera/center` called `center()` + `get_position()` → fixed.
- `41b834a`: `/camera/position` (GET) called `get_position()`; websocket "camera" command called `set_pan()`/`set_tilt()`/`get_position()` (camera control over WS fully broken) → routed through `move_camera()` + `get_status()`.
These were broken on EVERY unit, not robot-specific.

### Fleet deploy (all via ff-only pulls, never clobbering per-unit yaml)
- SSH reachable as bare `treatbot1..5` (also `.localN`); key `~/.ssh/id_ed25519`. GOTCHA: bare ssh cmd does NOT cd into repo and `cd ... &&` got stripped — use `git -C /home/morgan/dogbot` and absolute grep paths.
- tb1, tb4, tb5: clean fast-forward to `41b834a`, restarted, `/camera/position` + `/camera/center` verified.
- **tb2 reconcile:** was on divergent branch `f900adc` (a local-only per-unit dispenser commit `reverse_steps 40→75`, forked at `c863842`) + uncommitted work. Stashed → `git rebase origin/main` (clean, commit only touched treatbot2.yaml) → pushed rebased commit as `adabeb6` → `stash pop` (its uncommitted server.py center-fix auto-merged away, identical to main). Restarted + verified.

### Left uncommitted intentionally (per user)
- **tb3** `config/robot_profiles/treatbot3.yaml`: per-unit tuning (tilt_min −35, dispenser steps_per_slot 150 / step_delay 0.0050). User said don't worry about it.
- **tb2** `config/robot_profiles/treatbot2.yaml`: had dangerous out-of-range gimbal limits; corrected `tilt_min −180 → −90` and `pan_max 320 → 270` (servo physical floor/ceiling), left `tilt_max 250` / `pan_min −10`. Left UNCOMMITTED + service NOT restarted on tb2 for those yaml values — user will tweak/commit later. Plus tb2 `.claude/resume_chat.md` (scratch).

### Next steps / watch items
- tb2: review + commit the corrected treatbot2.yaml gimbal limits when ready, then restart to apply.
- App D-pad up-travel: app sends ±45 tilt offset, so it can't reach an asymmetric gimbal's full up-range (tilt_max 238). Separate UX item if full up-travel is wanted.
- IDE-buffer revert gotcha recurred: edits to files open in the user's IDE silently reverted off disk twice — re-grep disk after editing an open file before restarting.

---

## Session: 2026-05-28 (late, treatbot5) — R2 bark-spam root-cause + R3/R4 force_trick fixes, pushed to main

**Duration:** ~1.5 hours
**Robot:** treatbot5 (code changes are fleet-wide via origin/main)
**Status:** ✅ Committed + pushed as `a55057f`. Service NOT restarted on TB5 yet — user opted to restart manually. Fleet rollout pending pull + restart.

### What was reported
User noted bark events still flooding the dog history despite the previous session's `3d3c497` speaker-echo suppression fix. Also: app-side Build 106 testing punch list (`.claude/ROBOT_ISSUES_2026-05-28` referenced but file not in repo) listing R3 (coach TTS skips trick prompt) and R4 (coach prompts say generic "Dog").

### Verified the previous fixes are live on TB5
- Local HEAD already at `6f69374` == `origin/main` HEAD. `3d3c497` (bark suppression), `d095c16` (house voice), `b069b22` (Xbox per-dog audio), `a11dffa` (treat count), `d6bcdba` (spin debounce) all pulled.
- `treatbot.service` started at 00:59:10 EDT — AFTER `6f69374` (00:33:37) — so the new code IS running. Speaker-echo suppression confirmed firing in logs (200ms re-extensions during playback).

### Root cause: bark spam ≠ speaker echo (R2)

Pulled 4 false-positive bark events from last 4h logs. Same signature every time:
```
peak=0.37-0.38, duration=1500ms, loudness≈-8.5dB, conf=0.00
classifier said: aggressive (0.34-0.67), notbark ≤ 5%
```

Diagnosis:
1. `BarkGate` (Stage 1) is **pure energy thresholding** — no spectral or ML gating at the entry point. `base_threshold=0.18` is easily passed by talking close to the mic.
2. Spectral filter at `core/audio/bark_detector.py:204` checks 400–1800Hz ratio against `spectral_threshold=0.15` — but speech vowel formants overlap that band heavily, so speech sails through.
3. **ML notbark veto is async and broken anyway.** Veto runs in background thread (`_emotion_worker`), publishing `bark_false_positive` events *after* the bark is already committed and relayed. Even if it ran synchronously, the classifier itself confidently labels human speech as "aggressive" with `notbark < 5%`, so the `notbark > 0.5` gate never triggers.
4. **Smoking gun:** all 4 false positives showed `duration=1500ms` — exactly `max_bark_duration_ms`, the silent cap. Real barks are 100-500ms. The code was capping over-long sustained sound and emitting it as a bark anyway. Speech/music = sustained → always hits the cap.

### Fix shipped (in commit a55057f)

**R2 — bark_gate.py:** Changed cap-and-emit → reject-when-too-long.
- `max_bark_duration_ms: 1500 → 1000`
- Removed `duration = min(duration, max)` cap line
- Added explicit `if duration > cfg.max_bark_duration_ms: return result` with INFO log
- Smoke-tested: would have rejected all 4 of today's false positives

**R3 — trick TTS skipped (coaching_engine.py + xbox_hybrid_controller.py):**
- Root cause: `set_forced_trick()` unconditionally set `_forced_trick_at = time.time()` — this is the "Xbox already played the mp3, skip redundant TTS" flag. App's force_trick path triggered the same flag → engine thought audio was already played → skipped the prompt.
- Fix: `set_forced_trick(trick, dog_id=None, dog_name=None, audio_pre_played=False)`. Only set `_forced_trick_at` when `audio_pre_played=True`.
- Xbox path now passes `?audio_pre_played=1` via query param in `/coaching/force_trick/{trick}`. App paths (WS/REST) leave it False → engine speaks the trick aloud.

**R4 — coach says generic "Dog" (3 handlers + engine):**
- Root cause: WS handler in `main_treatbot.py:1539`, WS handler in `api/ws.py:623`, and REST in `api/server.py:613` all extracted only `trick` from payload, dropping the `dog_id`/`dog_name` fields Build 106 app sends.
- Fix: all three handlers now extract `dog_id` + `dog_name` and pass to `set_forced_trick`. Engine stores them as `_forced_dog_id` / `_forced_dog_name`. `_get_dog_name()` consults the forced name (after `_forced_dog` demo override, before ArUco/select_dog fallbacks).
- Forced dog name cleared when `set_forced_trick(None)` called.

### Commit (pushed to origin/main)
- `a55057f` — fix(coach+bark): R2 bark duration reject, R3 trick TTS, R4 dog name in force_trick

### Files touched (6)
```
 api/server.py                    | 15 ++++++++++++---
 api/ws.py                        |  6 +++++-
 core/audio/bark_gate.py          | 10 +++++-----
 main_treatbot.py                 |  9 +++++++--
 orchestrators/coaching_engine.py | 41 ++++++++++++++++++++++++++++++++++------
 xbox_hybrid_controller.py        |  6 ++++--
```

### Activate + validate (next session / user TODO)
- **Activate on TB5:** `sudo systemctl restart treatbot.service` — not done yet
- **Fleet rollout:** `git pull origin main && sudo systemctl restart treatbot.service` on TB1-4
- **R2 validation:** talk near robot for 2+ s in coach mode → expect `Rejected: too long (Xms — likely speech)` in logs, no bark events emitted/relayed/stored
- **R3 validation:** force a trick from app → robot speaks the trick. Xbox Guide button → still skips engine TTS (Xbox plays it itself, no double "sit sit").
- **R4 validation:** with a profiled dog selected in app, force a trick → robot says the dog's real name. Existing `select_dog` fallback unchanged.

### Open / deferred
- Untracked `.claude/TMC2209_UART_SETUP.md` (per-unit OS notes, intentionally not in git)
- Spectral threshold (`0.15`) and ML classifier mis-labeling speech as bark are still the underlying weakness. Duration-reject fix is necessary but not sufficient if a real bark-length speech burst happens (e.g. a sharp "HEY!"). Follow-up options: raise spectral threshold to 0.25, train classifier on negative human-speech samples, or add a VAD preprocessor.

---

## Session: 2026-05-28 (early hours, treatbot4) — app-testing punch list: 4 issues fixed, pushed to main

**Duration:** ~3 hours
**Robot:** treatbot4 (code changes are fleet-wide via origin/main)
**Status:** ✅ All 4 reported issues from app-testing session committed and pushed. Pull + `sudo systemctl restart treatbot.service` on each robot to activate.

### Issues reported by user, all addressed

1. **False spin reward (~18:09 on TB?)** — dog ran into wall, robot rewarded "spin"
   - Couldn't pull literal frame timeline: log gap on TB4 from 17–19h; user skipped cross-fleet ssh
   - Root cause in `core/behavior_interpreter.py`: when `_last_behavior is None` (after `reset_tracking()`), the first classifier event gets the immediate-accept fast-path. For `spin` (a transient behavior) this means a single misfire frame instantly engages the 2-second spin latch + passes `check_trick` at 0.3s.
   - Fix in `d6bcdba`: added `TRANSIENT_BEHAVIORS = {'spin'}` constant; spin now goes through the 2-frame debounce on first detection. Static poses (sit/lie/stand) keep their fast-path. Smoke-tested all 3 scenarios.

2. **Bark event flood** — speaker echo from non-speak audio classifying as barks
   - Existing `suppress_detection()` mechanism only called from coach speak-trick command audio path. Greetings, "good"/"no", Xbox feedback, mode announcements, SG ladder — none called it.
   - Already-good: YAMNet threshold (0.70), relay-forward throttle (5s in `main_treatbot._forward_event_to_relay`), intensity envelope on event (`peak_energy`/`duration_ms`/`loudness_db`).
   - Fix in `3d3c497`: `usb_audio.play_file()` now spawns a daemon monitor that re-extends `bark_detector.suppress_detection()` every 200ms while pygame is busy, plus 500ms tail. Music excluded. Added `peek_bark_detector_service()` so audio playback doesn't accidentally instantiate the bark detector early.

3. **Voice override audit** — Xbox button MP3s always defaulted, never used uploaded per-dog voice
   - Audit found three orchestrators (coach/mission/SG) and three ws.py voice handlers (play_voice/call_dog/play_command) ALREADY route through `resolve_voice_file`. Only Xbox was broken.
   - Root cause: Xbox runs as a subprocess of main_treatbot (per `services/control/xbox_controller.py:166`). Its `resolve_voice_file()` saw an empty state singleton — `get_aruco_dog_within / session / current_dog` all returned None — so every Xbox-side lookup fell to `/talks/default/*`.
   - Fix in `b069b22`: removed `_get_dog_audio_path`, added `_play_voice_command(name)` that POSTs `{"command": name}` to `/audio/play_command`. Server-side resolution uses the live state. Five call sites rewired (RT "good", B-cycle commands, RB "no", Y "treat", D-pad right "quiet"/"come").

4. **`dispense_treat {count: N}`** — atomic multi-treat from one envelope
   - `dispense_multiple(count, dog_id, reason)` already existed in dispenser. REST handler `/treat/dispense` already routed count>1 to it.
   - Two WS handlers broken: `ws.py:401` ignored count; `ws.py:1421` passed `count` as the FIRST POSITIONAL ARG (`dog_id`) — silently mis-attributed every multi-dispense to a "dog" named "3"/"5"/etc.
   - Fix in `a11dffa`: both WS paths now accept `count` + `dog_id` + `reason`, clamp count to [1, 10], thread multi-dispense off the WS receive loop. REST handler now returns actual dispensed count (was always returning requested count regardless of partial-jam outcomes). Added `import threading` at top of ws.py.
   - App can listen for per-treat `treat_dispensed` bus events for UI progress.

### Follow-up: "no default sounds" clarification

User pushed back on session-end framing that described `/talks/default/*` as the "fallback" — design vision is owner-recorded voices are the norm, defaults are first-boot scaffolding only. Clarification: when a new dog is added but only some commands recorded, every other command should pull from the **first dog the owner added** (their "house voice") before any shipped default plays.

- Fix in `d095c16`: `services/media/voice_lookup.py` now inserts a "first-added dog" step in the resolution chain between per-dog candidates and `/talks/default/`. Folder discovered by scanning `/VOICEMP3/talks/` for `dog_<timestamp>` folders, sorted numerically. Filesystem-smoke-tested on TB4's actual layout — `sit` (in both): first-dog wins; `name` (only first-dog): first-dog; `crosses` (only default): default.
- Memory saved: `project_voice_first_dog_default.md` (the rule) + `feedback_no_default_framing.md` (don't talk about defaults as the norm) + MEMORY.md index entries.

### Commits this session (all pushed to origin/main)
- `3d3c497` — fix(audio): suppress bark detection during all speaker playback
- `d6bcdba` — fix(coach): require 2 consecutive frames before committing 'spin'
- `a11dffa` — feat(treat): accept count param on dispense_treat (1–10) across all paths
- `b069b22` — fix(xbox): route command audio through /audio/play_command for per-dog voice
- `d095c16` — fix(voice): owner's first dog acts as house voice before shipped defaults

### Files touched (7)
```
 api/server.py                        |  9 ++++--
 api/ws.py                            | 46 ++++++++++++++++++++------
 core/behavior_interpreter.py         |  8 ++++-
 services/media/usb_audio.py          | 39 +++++++++++++++++++++++
 services/media/voice_lookup.py       | 46 +++++++++++++++++++++++++-
 services/perception/bark_detector.py |  7 ++++
 xbox_hybrid_controller.py            | 62 +++++++++++-------------------------
```

### Next session / open items
- **Activate the changes:** every fleet member needs `git pull origin main` + `sudo systemctl restart treatbot.service`. None of these fixes are live until the service restarts on that unit.
- **Validation:**
  - Bark flood: trigger coach greeting/"good"/Xbox audio with bark detector active — confirm `"Bark detection suppressed for ...s (speaker echo)"` lines continuously extend, no real bark events during playback.
  - False spin: run a coach session with the actual wall-running scenario or simulate by triggering a single-frame spin via debug. The 2-frame requirement should prevent the false reward.
  - Xbox voice override: upload a custom "good.mp3" for the first dog; press RT; should play the custom not the shipped default. Also test the house-voice fallback by deleting that custom and checking it still uses the first-dog's voice instead of `/talks/default/`.
  - `dispense_treat`: send `{"count": 3}` via WS — should fire 3 dispenses ~1.5s apart, each emit a `treat_dispensed` event.
- **Deferred from Issue 1's asks:**
  - Obstacle-acceleration gate for spin reward (needs depth/ToF or bbox-motion analysis from `dog_tracker`). Cheapest follow-up if false spins recur: bbox translation vs rotation check.
  - Literal frame buffer for 18:09 incident — log gap on TB4, never pulled.
- **Pre-existing untracked items** unchanged from session start (per prior log): `.claude/TMC2209_UART_SETUP.md`, `.claude/nightvisionrobo.md`, `state/`, `tests/hardware/test_battery_adc_channels.py`, plus dirty `config/robot_profiles/treatbot4.yaml`. Not touched this session.

---

## Session: 2026-05-27 (late evening, treatbot4) — sync pull only

**Duration:** ~2 min
**Robot:** treatbot4
**Status:** ✅ Complete — fast-forwarded to `c0931bd`, no work needed

### Work completed
- `git pull origin main`: 3 commits in (`d428393` TB5 DIR/idle/charging fix, `16bbcec` TB5 late-night docs, `c0931bd` night merge). 4 files updated, fast-forward clean, no conflicts.
- All carry-overs from previous TB4 session (service restart + SG validation + TB3 UART) confirmed done by user out-of-band.

### State at session end
- Branch `main` clean (only the 4 known pre-existing untracked files: `.claude/TMC2209_UART_SETUP.md`, `.claude/nightvisionrobo.md`, `state/`, `tests/hardware/test_battery_adc_channels.py`).
- `treatbot.service` active, AI 19.4 FPS, no errors in last hour of logs.
- TB4 in sync with rest of fleet (TB5 dispense direction, idle CPU pause, charging false-positive fix all live in pulled code).

### No commits this session

---

## Session: 2026-05-27 (night, treatbot3) — pull, merge resume_chat conflict, fleet ready for user testing

**Duration:** ~10 min
**Robot:** treatbot3
**Status:** ✅ Working tree clean post-pull. Fleet confirmed in good shape; user moving to user-testing phase next.

### What happened
- `git pull origin main` brought in 4 commits since `7231b80`: `be7c5ff` (tb4 steps_per_slot + Xbox MAC), `0704a89` (tb4 housekeeping log), `d428393` (tb5 dispense direction + idle CPU + charging gate), `16bbcec` (tb5 session log).
- Stash conflict in `.claude/resume_chat.md` between upstream's two new session entries (TB5 late-evening + treatbot4 housekeeping) and the locally-stashed treatbot3 evening entry. Resolved by keeping all three in chronological newest-first order.
- User confirmed TB5 is working great after the late-evening dispense direction + idle CPU + charging gate fixes (`d428393`). No further fixes needed for now.

### Repo changes this session
- `.claude/resume_chat.md` — merge resolution + this session entry.

### Next session
- **User testing phase.** Fleet hardware/firmware is stable across TB1/3/4/5; SG overhaul is live; TB5 dispense is mechanically and electrically correct; idle CPU is sane; battery false-charging closed.
- Carryovers from prior sessions if/when needed:
  1. Verify SG overhaul on a real bark session (sg_escalation rows, profile_json rollups, /sg/sessions/recent, mozart at L4).
  2. Hand `/home/morgan/.claude/plans/flutter-sg-punishment-slider-prompt.md` to Flutter session.
  3. Order new AC600 dongles for TB1/TB2 (ceiling: try the new ones, else stick with built-in WiFi).
  4. TB3 TMC2209 UART setup (last fleet holdout).
  5. `services/network/wifi_manager.py` pgrep spam cleanup.

---

## Session: 2026-05-27 (late evening) — TB5 dispense direction wire fix + idle-mode CPU fix + false-charging gate fix

**Duration:** ~1.5 hours
**Robot:** treatbot5
**Status:** ✅ Shipped commit `d428393` (pushed to origin/main). Detector fix already live (service restarted mid-session). Battery monitor fix needs `systemctl restart treatbot.service` to take effect.

### TL;DR
After carousel reinstall, TB5 dispenser rotated CCW for forward, anti-jam reverse, AND secondary reverse — every direction the same. Initial hypothesis: needed `shaft_invert: true` in TMC2209 config. Confirmed via UART that GCONF bit 4 was indeed set after YAML edit and service restart — but rotation was still all-CCW. Real root cause: **DIR signal wire between Pi GPIO16 (pin 36) and the TMC2209 DIR pin was broken**. Pi was driving HIGH but chip's IOIN.DIR registered LOW. Same failure class as last session's V+ rail (loose-but-continuous wire that fails under load). User reseated the wire, dispenser now works correctly with `shaft_invert: true` left in place. Then chased two more orthogonal issues: idle-mode AI burning CPU at 20 FPS for no reason, and battery monitor false-positives on "Charging detected" right after Xbox driving.

### Diagnostic process (TB5 dispenser)
1. **Symptom**: User reinstalled carousel, dispenser ran but in inverted direction vs other units.
2. **First attempt**: Added `shaft_invert: true` to `treatbot5.yaml`, restarted service. UART readback confirmed GCONF=0x95 with bit 4 set. But rotation didn't change.
3. **Smoking gun**: Read TMC2209 IOIN register while polling Pi GPIO state via `pinctrl get 16`:
   ```
   Pi GPIO16 = hi (output driving HIGH)
   chip IOIN.DIR = 0  (LOW) ← MISMATCH
   chip IOIN.ENN = 1  ← matches Pi GPIO24 (this wire works)
   ```
   Pi was driving HIGH, chip was seeing LOW. Wire is broken/shorted somewhere between them.
4. **User reseated the DIR rail terminal on the breadboard**. Same class of failure as the V+ rail terminal in the last session — a wire that conducts micro-amps for continuity testing but can't carry signal under chip load.
5. **Result**: Dispenser now works correctly with `shaft_invert: true` left as-is. (If TB5 ever swaps to a fresh wiring harness, may need to revisit whether shaft_invert is still required.)

### Idle-mode AI CPU fix (`services/perception/detector.py`)
- AI inference was correctly skipped in IDLE/MANUAL modes (gated at line 750), but the camera capture loop was still spinning at 20 FPS (`time.sleep(0.050)` in the catch-all `else` branch).
- Added explicit `elif run_full_ai: 0.050` and `else: 0.200` (~5 FPS) branches. Frame capture loop now backs off in idle.
- Also fixed misleading heartbeat log: was always saying `[AI] Pipeline active | FPS: X` even when AI was paused. Now says `[AI] Mode=idle — inference paused | frame capture only | FPS: 5.0`.
- **Measured impact**: Python CPU went from ~12-18% to ~3-5% (didn't even appear in `top`) in idle.

### Battery false-charging fix (`services/power/battery_monitor.py`)
- Symptom: "Charging detected: 16.08V (85%), trend=+0.460V" + charging audio fired ~15s after Xbox-driving session ended. Robot was not plugged in.
- Existing motor-idle gate had two holes:
  1. `if cmd.left_speed == 0 and cmd.right_speed == 0: return False` — a stop command (released stick / motors decelerating to zero) instantly disengaged the gate, even though the rail was still rebounding from full-load draw.
  2. Dispenser activity wasn't tracked at all in `_motor_recently_active()` — only drive motors. Stepper pulls ~1.1A peak from the same rail.
- Both fixed: any recent drive command (including stops) keeps the gate engaged for `motor_idle_required_s` (120s). Added dispenser check via `get_dispenser_service().last_dispense_time`.

### Files modified this session
- `config/robot_profiles/treatbot5.yaml` (+1 line, shaft_invert: true)
- `services/perception/detector.py` (+13/-2)
- `services/power/battery_monitor.py` (+19/-8)

### Commits this session (already pushed)
- `d428393` — fix(tb5): dispense direction + idle-mode AI pause + charging false-positive

### Memory implications
- **Broken signal wire is the same failure pattern as broken V+ rail** (loose-but-continuous breadboard terminal). Diagnostic recipe is identical: compare what Pi drives (`pinctrl get N`) vs what chip sees (`IOIN` register over UART). When they disagree, the wire is at fault, not the chip. Worth folding into existing `feedback_tmc2209_uart.md` and/or `feedback_phantom_voltage.md`.
- **`pinctrl get N` reads the drive register, NOT the actual wire voltage**. A shorted wire would still show "op dh | hi" if the GPIO is configured as output driving HIGH. To actually probe the line voltage, you need a multimeter or to read the chip's downstream view.

### Open items (carryover)
1. **`services/network/wifi_manager.py` pgrep spam** — still untouched. Service in client mode shouldn't be checking for hostapd every iteration.
2. **TB3 TMC2209 UART setup** — last holdout in fleet (per evening 2026-05-27 entry).
3. **TB2 dongle work** — same 5 GHz dead pattern as TB1.
4. **Verify battery monitor fix** — restart service, drive around, then stop. Should NOT trigger "Charging detected". For positive test, plug in real charger.

### Service restart needed
The battery_monitor change needs `sudo systemctl restart treatbot.service` to take effect. Detector + dispenser fixes already live.

---

## Session: 2026-05-27 (evening) — treatbot4 housekeeping: pull SG overhaul, commit tuning, dispenser verified

**Duration:** ~30 min
**Robot:** treatbot4
**Status:** ✅ Complete — committed as `be7c5ff`, pushed to origin/main. Service NOT yet restarted (user to restart after filling dispenser).

### Work completed
1. **Pulled origin/main** — 15 commits fast-forwarded cleanly (`fbbff72..7231b80`). No conflicts with local mods. Notable incoming: SG overhaul (`a1bca1c`), tb5 UART fix + tuning adoption (`1a02e95`), tb5 battery recalibration (`0148649`), battery false-charging gate tightened (`7e96d99`), WiFi 5 GHz pinning (`38549f9`), new TMC2209 probe tests.
2. **Battery calibration check (treatbot4)** — verified already done. Current reading 16.52V / 94% / ADC 2.926V, factor 5.638 → 16.50V (internally consistent). DMM-calibrated 2026-05-25 in commit `5f0dacc` (15.43V → 2.7368V → factor 5.638). The 16.52V coincidence with tb5's old over-read is just the natural near-fully-charged reading. **No action needed.**
3. **Committed local tb4 tuning** (`be7c5ff`):
   - `treatbot4.yaml`: `steps_per_slot 137 → 144` (~30.8° → ~32.4°/slot, live-tuned 2026-05-26 against this unit's physical carousel pitch with spreadcycle + irun=31 active)
   - `fix_xbox_controller.sh` + `xbox_persistent.py` default arg: MAC swap to `CC:B0:B3:84:67:FC` (new controller paired on tb4)
   - Note: default MAC in `xbox_persistent.py` will continue to ping-pong per robot until controller MAC moves into per-unit yaml. Out of scope for this session.
4. **Dispenser movement verified** — user confirmed `steps_per_slot=144` "looks great". Going to fill carousel and try a real dispense after restart.

### Still pending / next session
1. **Restart treatbot.service** after filling dispenser — required to pick up: (a) the new SG overhaul code from upstream, (b) the committed `steps_per_slot=144`, (c) the new Xbox MAC default.
2. **Apply 4-item TMC2209 UART fix to treatbot3** — tb5 is now done (commit `1a02e95`); tb3 is the last holdout. Same procedure: VIO + GND + 1K on TX (pin 8) + PDN_UART to chip pin 4.
3. **Validate SG overhaul on tb4** post-restart — bark detection, dog attribution, offline catch-up, BPM fast-escalation. Watch logs for clean startup of the new SG code paths.

### Untracked items still in working tree (all pre-existing, intentional)
- `.claude/TMC2209_UART_SETUP.md` — per-unit OS setup notes, header says "Not in Git" by design
- `.claude/nightvisionrobo.md` — night mode design brief
- `state/night_mode.json` — runtime state; probably should be in `.gitignore`
- `tests/hardware/test_battery_adc_channels.py` — pre-existing test file from May 20

### Commits this session
- `be7c5ff` — chore(treatbot4): live-tuned steps_per_slot + new Xbox controller MAC (pushed)

---

## Session: 2026-05-27 (evening, treatbot3) — pulled SG overhaul, verified battery fixes, memory housekeeping, dispenser physical-test pass

**Duration:** ~20 min
**Robot:** treatbot3 (host this session was run on)
**Status:** ✅ All pulled changes verified live in code. Dispenser physical test passed. User restarted `treatbot.service` to activate the SG overhaul at session close.

### What happened
- `git pull origin main` brought in 9 new commits since `2088503`, including the big Silent Guardian overhaul (`a1bca1c`) and TB5 battery recalibration (`0148649`). Fast-forward, clean.
- Verified the two battery fixes are actually present in code:
  - `services/power/battery_monitor.py:107` — `motor_idle_required_s = 120.0` (was 30)
  - `services/power/battery_monitor.py:274` — trend threshold `0.35` (was 0.20)
  - `config/robot_profiles/treatbot5.yaml:48` — `calibration_factor: 51.29` (was 53.33; 4S DMM read 15.90V vs reported 16.52V)
- User confirmed TB2 WiFi is fixed (band=a pin via `nmcli` + provisioning code patch `38549f9`).
- User clarified the AC600 dongle situation: TB1/TB2's specific dongles have dead 5GHz RF chains. Plan is to order new dongles; if those also fail on those Pi units, accept it as legacy-hardware idiosyncrasy and stick with built-in WiFi rather than rabbit-hole further.
- Dispenser physical movement test passed on treatbot3 (no config changes needed).
- User restarted `treatbot.service` at session end to activate the SG overhaul (mozart path fix, bark dog-attribution fallback to selected dog, `sg_escalation` events on bus, offline catch-up via `GET /sg/sessions/recent`, per-dog SG rollups in `dogs.profile_json`, hourly DBCleanup daemon, BPM fast-escalation config + `POST /sg/config` + relay `sg_config` command).

### Files modified (memory only — outside the repo)
- `~/.claude/projects/-home-morgan-dogbot/memory/project_edimax_wifi_dongles.md` — rewrote. Old version still claimed "USB power/controller fault, software won't fix it" (the retracted theory). New version reflects reality: per-unit dead 5GHz RF chains in the AC600 batch, plan is replacement dongles + built-in WiFi fallback for TB1/TB2.
- `~/.claude/projects/-home-morgan-dogbot/memory/MEMORY.md` — updated index line for the dongle memory.

### Repo changes this session
None. Working tree clean. Nothing to commit.

### Next session
1. Verify SG overhaul on next live SG session (assuming user did the restart): `sg_escalation` rows appear in `dog_events`; `dogs.profile_json['sg']` populates for dogs with non-NULL attribution; `GET /sg/sessions/recent` returns the new session; mozart actually plays at L4.
2. Hand `/home/morgan/.claude/plans/flutter-sg-punishment-slider-prompt.md` to the Flutter session.
3. TB5 bare-motor + loaded dispense tests (after UART fix + treatbot4 tuning adoption).
4. TB5 battery false-charging issue: false-charging gate is fixed fleet-wide and TB5 voltage recalibrated — should be resolved. Confirm on next charge cycle.
5. Order new AC600 dongles for TB1/TB2; ceiling on further dongle debugging is "try the new ones, if they also fail just use built-in."

---

## Session: 2026-05-27 (afternoon) — Silent Guardian overhaul: music fix, dog attribution, offline catch-up, BPM fast-escalation

**Duration:** ~3 hours
**Robot:** treatbot1
**Status:** ✅ Shipped commit `a1bca1c` (pushed to origin/main). Service NOT yet restarted — changes live on next `systemctl restart treatbot.service`.

### Story
Reviewed Silent Guardian session #165 (10:04:51–11:15:36, 157 barks, 33 interventions, 4 treats). User reported "excessive barking before SG acted" and "2 minutes nonstop barking with no escalation to music." Forensic walk through journal logs + SQLite revealed multiple compounding issues that explained the perceived dead air.

### Root causes found
1. **Calming music NEVER played** all session. YAML config pointed to `songs/mozart_piano.mp3` but the file is at `songs/default/mozart_piano.mp3`. Every L4 escalation logged "Calming music file not found" and silently degraded to just playing "quiet.mp3" once. So Level 4 = "say quiet, then wait silently for 2× 20s quiet periods." From outside this looked like the robot stopped responding.
2. **90s GAVE_UP cooldown after every timeout** is dead air — bark events get logged but FSM ignores them. Combined with bug #1, the 10:14:43→10:16:13 window (90s gave-up) + L4 with no music made it feel like 2+ minutes of robot indifference. That was the "2 min nonstop barking with no escalation" period.
3. **All 157 barks logged with `dog_id=NULL`** because ArUco was never visible during bark events and the bark_detector had no fallback to "currently selected dog from the app." User has 2 dogs but neither was attributable.
4. **App showed no events on resume** — relay does NOT buffer for offline phones (`relay_client.py:1265` early-returns if not connected). Plus SG escalations were never published to the bus, so DogEventLogger never wrote them to `dog_events` even for catch-up polling.
5. **"good" without treat** after first treat dispensed — user thought it might be a jam, but it's intentional anti-farming: 10-min cooldown after each treat where verbal praise replaces treat. Confirmed via interventions #195/209/213/215/219 in DB.

### Shipped (commit `a1bca1c`, 7 files, +342/-7)
**Bug fixes:**
- `configs/rules/silent_guardian_rules.yaml` — mozart path → `songs/default/mozart_piano.mp3`
- `modes/silent_guardian.py:125` — `_gave_up_cooldown` 90 → 45 seconds
- `modes/silent_guardian.py:873` — matching in-code fallback string also fixed

**Bark attribution fix:**
- `services/perception/bark_detector.py::_handle_bark_detected` — added final fallback to `state.get_active_dog_id()` so when no ArUco is visible, the app's `select_dog` selection (or recent ArUco/session dog) gets used. ArUco still wins via the existing `bark_event.dog_id` path up top.

**Offline catch-up infrastructure:**
- `modes/silent_guardian.py::_start_intervention` — now publishes `sg_escalation` to the bus → DogEventLogger writes to `dog_events` → also forwarded over the relay socket if app is online
- New `GET /sg/sessions/recent?since=<ts>&limit=20` in `api/server.py` — returns silent_guardian_sessions JOINed with their sg_interventions for app catch-up
- `core/store.py::end_silent_guardian_session` — now also calls `_roll_sg_session_into_dog_profiles(session_id)` which walks the session's interventions grouped by dog_id and updates each dog's `profile_json['sg']` with cumulative counters (sessions_total, interventions_total, quiets_total, treats_total, max_escalation_ever, last_session_id, last_session_ended_at). Long-term per-dog history is now permanent.
- Hourly `DBCleanup` daemon thread in `main_treatbot.py` purges `barks` + `dog_events` > 24h, keeps `sg_interventions` + `silent_guardian_sessions` for 30 days.

**Fast-escalation feature (robot side):**
- New `bark_detection.fast_escalation_bpm` YAML config (default 0 = disabled). When BPM exceeds threshold, `_get_escalation_level()` returns max_level (L4) immediately instead of climbing one per intervention.
- `POST /sg/config` accepts new `fast_escalation_bpm` field
- `_handle_command` in `relay_client.py` now has an `sg_config` command branch — backs the app's "punishment level" slider

### Flutter handoff
Self-contained prompt for app-side slider work saved at:
`/home/morgan/.claude/plans/flutter-sg-punishment-slider-prompt.md`
Covers the relay command shape, slider UX semantics (Off / 10–90 BPM where lower = more aggressive), persistence note (robot doesn't write to disk; app must re-send on reconnect), acceptance criteria.

### Memory implications
- Today's session #165 had `dog_id=NULL` on every intervention/bark → per-dog rollup will skip those (the SQL filter is `WHERE dog_id IS NOT NULL`). Fix is forward-only.
- Robot doesn't persist `fast_escalation_bpm` to YAML — only in-memory. App is source of truth; must re-push on every reconnect.

### Cleanup done
122 cruft files in repo root (all `May 27 10:04:50 ... -INFO- ...` named — shell-paste mishap that turned journalctl lines into filenames containing ~17 bytes of log-fragment each). Deleted in one shot.

### Next session
1. **Restart service to make changes live:** `sudo systemctl restart treatbot.service` (drops current idle override; re-select dog + re-enable SG from app to verify)
2. Watch for `SG_FAST_ESCALATION: BPM X >= Y` log line when sustained barking trips the threshold (only fires if `fast_escalation_bpm > 0`)
3. Verify on next SG session: `sg_escalation` rows appear in `dog_events` table; `dogs.profile_json['sg']` populates for any dog with non-NULL attribution; `GET /sg/sessions/recent` returns the new session
4. Hand the Flutter prompt to the app's Claude session for the slider UI
5. Open question: today's bark detector still requires ArUco OR app `select_dog` for attribution. If user wants 100% attribution coverage with multiple-dog households and no ArUco, may need a "primary dog" config or multi-dog attribution policy. Not built.

---

## Session: 2026-05-27 — treatbot5 UART FIXED + dispenser restored — root cause was unpowered chip

**Duration:** ~2 hours
**Robot:** treatbot5
**Status:** ✅ Complete — UART working, dispenser running on treatbot4 tuning, chopper=spreadcycle confirmed live

### TL;DR
After multiple sessions across May 25-26 chasing UART silence on treatbot5, root cause was finally identified: **the TMC2209 chip was unpowered the entire time.** A loose breadboard rail terminal on the VM (motor power) supply meant battery voltage never reached the chip. An unpowered chip produces EXACTLY the `echo_only` UART pattern (Pi TX drives joined node, Pi RX reads it back, chip sits inert) — electrically indistinguishable from "broken wire to chip" at the bus level. After reseating the rail terminal, UART came up clean on first probe: `chip_replied`, `version=0x21`.

### How we got there
1. **User reported dispenser "fails epically" — no movement, no sound** even with carousel removed (zero load on shaft)
2. **Software side verified firing correctly** — backend logs showed `[TREAT] Dispensing treat #N` with GPIO writes happening
3. **Multimeter on chip V+ pad: 1.1V phantom voltage to GND** — high-impedance reading of a floating pin (EMI pickup), not real low voltage
4. **All wires tested continuous in continuity mode** — wires themselves are fine
5. **Bulk capacitor tested 10MΩ both directions** — cap is healthy, not shorting the rail
6. **User pulled chip + connected to DC bench at 12.8V standalone → pads held 12V perfectly** — chip is innocent
7. **Diagnosis: in-circuit failure was the rail terminal contact** — reseated, V+ came back to battery voltage
8. **UART probe immediately showed `chip_replied`, IOIN=0x21000041, version=0x21** — full success

### Why this took so long across sessions
- Earlier sessions multimetered VIO=3.3V at chip pad, but that was a snapshot in time — connection was flaky (came/went with rework)
- Module swap-tests (treatbot1's chip into treatbot5) didn't fix → we correctly concluded "chip isn't the fault" but wrong implication followed (we assumed wires were the fault, when actually it was the rail terminal upstream of all wires)
- Pi UART loopback proved Pi healthy → also correct, but only proves the Pi end; said nothing about whether the chip was alive
- Continuity testing of wires passed → wires WERE continuous, but a wire's continuity doesn't prove its battery-side end is actually connected to a live source
- Every diagnostic touchpoint was correct in isolation; the pattern just looked exactly like "broken signal wires"

### Config changes applied to treatbot5.yaml
Adopted treatbot4's proven tuning (best-tuned post-UART unit):
- `step_delay: 0.006 → 0.010` — slower stepping = more torque on torque-speed curve
- `reverse_steps: 40 → 70` — stronger anti-jam reverse
- Added `chopper_mode: "spreadcycle"` — full rated torque, now actually applies via working UART

Boot log verification: `TMC2209 configured: IRUN=31, IHOLD=5, 8x microstep, vsense=0, SGTHRS=0, chopper=spreadcycle` ✓

### Major implication for the fleet
**Phantom voltage (~1V) on a power rail is "floating pin," NOT "undervolted."** This is a common pattern after extensive rework — a wire that conducts microamps for continuity testing can fail under chip's milliamp load. Future debugging: probe the source-side end of the supply wire to verify power actually exists at the wire, not just that the wire is end-to-end continuous.

### Memories updated this session
- `feedback_tmc2209_uart.md` — added 2026-05-27 update with "unpowered chip = echo_only" insight + chip-power sanity test procedure
- `feedback_phantom_voltage.md` — NEW: ~1V multimeter readings on rails = floating pin, not undervolt

### Files modified
- `config/robot_profiles/treatbot5.yaml` — dispenser section: step_delay, reverse_steps, chopper_mode

### Pending / next session
1. **Bare-motor dispense test** — confirm Xbox LB now spins motor with full torque
2. **Reinstall carousel** — full loaded dispense test
3. **(Carry over from 2026-05-26)** wifi_manager pgrep spam fix — `services/network/wifi_manager.py` shouldn't poll for hostapd in client mode
4. **(Carry over)** Battery false-charging issue on treatbot5 — still untouched
5. **WiFi dongle situation on TB1/TB2** — see TB1 session entry directly below for full context (AC600 batch has per-unit dead 5 GHz; built-in WiFi being used as workaround)

---

## Session: 2026-05-27 (overnight) — Dongle 5 GHz failure → dual-profile fallback → revert to built-in

**Duration:** ~2 hours
**Robot:** treatbot1 (paralleled with tb2 over SSH)
**Status:** ✅ STABLE on built-in WiFi. Dongles returned to TB3. Dual-profile fallback proven and kept as safety net.

### Story
After tonight's band=a fix on TB1 (committed da3034d earlier), live test of dongle on TB1 failed — `ssid-not-found` from wlan1, 23s scan timeout, no association. Reroll: discovered the **dongle** in TB1 had **zero 5 GHz BSSID visibility** even with the dongle's antenna right next to the AP. Driver advertised full 5 GHz capability; the actual RF chain in this physical unit was dead on 5 GHz. Same AC600 RTL8821CU model works on TB3 (5805 MHz, -33 dBm); the unit in TB1 cannot.

### What we set up on TB1 before giving up on the dongle
1. **Blacklisted brcmfmac** via `/etc/modprobe.d/disable-internal-wifi.conf` — so dongle is the only WiFi radio
2. **Dual-profile fallback** in NetworkManager:
   - `preconfigured` (band=a, autoconnect-priority=10, retries=2) — 5 GHz preferred
   - `preconfigured-24g` (band=bg, autoconnect-priority=5, retries=2) — 2.4 GHz fallback
3. **Re-confirmed wifi-provision.service is enabled** as last-resort AP rescue (60 s threshold)

Booted with dongle attached. Result: NM never tried `preconfigured` (no 5 GHz BSSID visible in scan) → auto-fell-back to `preconfigured-24g` → associated cleanly on 2.4 GHz at -6 dBm, DHCP .211, SSH worked. Multi-profile fallback PROVEN.

### Why we reverted to built-in
User decision: the AC600 batch has per-unit defects, ordering replacement dongles takes time, and built-in WiFi works fine at -17 dBm on 5 GHz with the cage off. Dongles returned to TB3 (which has a working unit). For now TB1 relies on built-in. The blacklist file was removed; dual-profile setup left in place as a dormant safety net.

### Memory updated
`project_wifi_dongle.md` now documents three distinct failure modes:
1. rtw88 deep-LPS beacon storms (fixed 2026-05-21)
2. dual-band BSSID flap (fixed 2026-05-26 with band=a + wifi_manager.py patch)
3. **per-unit dead 5 GHz RF chain in AC600 batch** (NEW — swap dongle, not a software fix)

Plus the dual-profile fallback pattern (recommended for future deployments).

### TB2 follow-up (deferred)
TB2 same dongle batch, same 2.4-only behavior. Same fixes applicable. User said TB2 will be addressed in a separate session — they have full context from this session.

### Commits this session (already pushed)
- `cb8e517` chore: update treatbot2 pan_center calibration (180→134) — rebased onto main from TB2's local
- All TB1 tonight's changes are off-repo (NM connection profiles, modprobe.d files) — no commits needed

### Open items
- Order new dongles (any AC600 batch is suspect; consider MediaTek MT7612U)
- Test each new dongle's 5 GHz reception at the AP location BEFORE committing it to a bot
- Decide whether the chassis cage design needs revisiting (dongle dependency vs built-in penetration)

---

## Session: 2026-05-26 (late) — TB1 SSH-unreachable: ROOT-CAUSED as dual-band BSSID flap, fix shipped

**Duration:** ~1 hour
**Robot:** treatbot1 (paralleled with tb2's own Claude session)
**Status:** ✅ FIXED on TB1. Same fix needs to be applied on TB2.

### The actual cause (resolves the retracted theory from the earlier 2026-05-27 entry)
Not USB power delivery, not bad Edimax dongles, not the rtw88 deep-LPS bug. The real cause: **rtw88 band-flap on dual-band SSIDs**.

"524Pomeranian" is exposed on both 2.4 GHz (`…dc:b0`) and 5 GHz (`…dc:b4`) under one SSID. The rtw88_8821cu dongle keeps re-evaluating and roaming between them (visible in TB1's journal as repeated `authenticate with …b0` ↔ `…b4` events). Every re-association leaves the AP's bridge/forwarding table for the dongle's MAC briefly stale. Symptoms:
- outbound TCP (relay, internet) works fine — chip is awake to transmit
- inbound from LAN (ping, SSH) silently dies — AP forwards to wrong BSSID while chip is on the other radio
- per-Pi-and-location variance: same dongle works on TB3 (signal differential there is large → dongle picks one band and stays), fails on TB1/TB2 (balanced signal → constant flapping)

This is *consistent* with TB2's earlier "dongle was stuck at 2.4 GHz on TB2 but did 5 GHz/ac on TB3" observation — same dongle, different band-flap behavior at different physical locations.

### Fix applied
1. **TB1 home network** — `sudo nmcli connection modify "524Pomeranian" 802-11-wireless.band a` — pins to 5 GHz, stops the flap. No SSH bounce (band setting applies on next re-association). Do **NOT** run `nmcli connection up` after — would needlessly disconnect.
2. **Provisioning code** — `services/network/wifi_manager.py::save_credentials` now auto-pins `band=a` when the scan shows the target SSID has a 5 GHz BSSID. New helper `_ssid_has_5ghz`. Single-band SSIDs left alone so 2.4-GHz-only demo routers still work.

### TB1 currently running on built-in WiFi
User removed the Faraday cage to use built-in WiFi as a workaround while we diagnosed. Built-in is currently at -17 dBm on 5 GHz channel 161. The dongle remains *mandatory* once the cage goes back on — built-in WiFi can't get through the chassis. The `band=a` pin is now in place for when the dongle is re-attached.

### Memory updated
`project_wifi_dongle.md` rewritten to distinguish the two distinct rtw88 failure modes:
- failure mode 1: deep LPS beacon-loss storms (fixed 2026-05-21 with `disable_lps_deep=1`)
- failure mode 2: dual-band BSSID flap (fixed today with `band=a` pin + provisioning patch)

### Next session / TB2 follow-up
1. Apply same fix on TB2: `sudo nmcli connection modify "<ssid>" 802-11-wireless.band a` (skip the `connection up` step)
2. Pull latest main on TB2 to get the `wifi_manager.py` provisioning patch
3. When cage goes back on TB1, verify SSH still works on the dongle

### Commits this session
- `38549f9` fix: pin dual-band SSIDs to 5 GHz on provisioned WiFi networks
- `f177e79` test: add TMC2209 known-good register dump probe

---

## Session: 2026-05-27 — TB1/TB2 SSH-unreachable — RETRACTED dongle theory; pointing at TB2 hardware

**Duration:** ~3 hours + follow-up
**Robots:** treatbot1 + treatbot2 (both broken the same way)
**Status:** ⚠️ DONGLE THEORY RETRACTED. Same dongle that "failed" on TB2 was moved to TB3 and works **perfectly** (5.805 GHz / 802.11ac / -33 dBm / 0 errors / 351 Mb/s). On TB2 the same dongle could only manage 2.4 GHz / 802.11n. Same dongle, same router, same SSID — capability gap is huge. Points at **USB power delivery or USB controller on TB2**, not the dongle. Likely same story on TB1. Cause not yet confirmed; needs the decisive test below.

### Problem statement
TB1 and TB2 both exhibit the same pattern:
- WiFi associated to home network (router shows them online with leased IPs .241 and .211)
- App connects fine via AWS Lightsail relay (outbound from bot works)
- Bluetooth Xbox controller works (separate radio)
- **SSH from laptop times out** (or `Destination Host Unreachable`)
- HDMI shows blank/cursor only — Razer RGB keyboard doesn't enumerate during early boot, so no console login possible

### Key fact that took an embarrassingly long time to surface
**Both TB1 and TB2 use external Edimax USB WiFi dongles** (MAC prefix `90:DE:80`) because the Pi 5's built-in WiFi is unreliable on those units. TB3/4/5 don't have this problem (TB3 uses a Realtek `rtw_8821cu` USB dongle, TB4/5 unknown).

### Theories tested chronologically (most ruled out)
| Theory | Verdict |
|---|---|
| Recent code change in `battery_monitor.py` / `config_loader.py` | WRONG — diffs were trivially benign |
| WebRTC memory leak (def9f08-era fixes) | NOT TESTED — no live shell to measure |
| Auto-AP fallback in `main_treatbot.py:2104-2114` flipping to hostapd after 60s WiFi loss | WRONG — bot was never in AP mode (no WIMZ-* SSID broadcast, app kept working) |
| mDNS / `.local` resolution broken | WRONG — direct-IP SSH also failed |
| Edimax dongle WiFi power-save | WRONG — `wifi.powersave = 2` was already set on TB1 (per-conn) and added to TB2; SD-card fix to add global `/etc/NetworkManager/conf.d/wifi-powersave-off.conf` made no durable difference |
| `hidden=true` on Pi-Imager preconfigured connection + competing saved networks (Gonzaga1, Kazuya, etc.) | PARTIALLY HELPFUL — SD-card fix to `hidden=false` and delete stray `.nmconnection` files made TB1 SSH-able "instantly" after first reboot, then it dropped again after a few minutes |
| `treatbot.service` itself doing something to wlan0 | WRONG — disabled the service AND `wifi-provision.service` on TB2's SD; bot still timed out on SSH after the (re)boot |
| Asymmetric routing (eth0 metric 100 < wlan0 metric 600 → SYN-ACK to wlan0 IP goes out eth0) | RULED OUT — `rp_filter=0` and sshd binds to `0.0.0.0:22`; SSH to wlan0 IP `.211` worked while eth0 was also plugged in |
| Router MAC-aging on quiet WiFi clients (keepalive cron would fix) | NOT TESTED — user gave up before running the experiment |
| **Dongle hardware/firmware/driver in a stuck state** | LEADING THEORY — every test shows wlan0 SSH dies the instant the Ethernet cable is unplugged |

### The decisive observation
Once we got SSH into TB2 via Ethernet (eth0 = 192.168.50.9), all diagnostics looked fine:
- `nft list ruleset` empty
- `wpa_cli status`: `wpa_state=COMPLETED`, freq 2442 (2.4GHz ch 7), 802.11n
- Self-ping wlan0 worked (0.02ms)
- sshd listens on 0.0.0.0:22
- No iptables / nftables blocking
- Routing sane

**But the moment eth0 cable is unplugged, SSH to .211 (wlan0) dies. Every single time.** That's a damning signal — the dongle/driver appears to ride on something Ethernet provides (could be: USB current draw is higher with eth0 active and powers the dongle better; could be NM keepalive behavior; could be coincidence in timing of broadcasts; we don't know).

### What we did to the bots
TB1 (via SD-card surgery):
- Set `hidden=false` on `preconfigured.nmconnection` (was `hidden=true` for a non-hidden network)
- Removed competing networks: Gonzaga1, Kazuya, Trailsidedream
- Left `treatbot.service` enabled — that "worked instantly" on first reboot then degraded

TB2 (via SD-card surgery):
- Removed competing networks: Gonzaga1, Kazuya, Motta studios, OgdenCap, Trailsidedream
- Added `/etc/NetworkManager/conf.d/wifi-powersave-off.conf` (was missing)
- **Disabled `treatbot.service` and `wifi-provision.service`** auto-start
- Still has the WiFi-only failure mode

### Decision → RETRACTION
Initially decided to flash. Then briefly thought "built-in WiFi works → dongle is dead." Then user moved the supposedly-dead TB2 dongle to TB3 and it works **perfectly** (5GHz/802.11ac, -33 dBm, 0 errors, 351 Mb/s).

**The dongle is innocent. TB2 (and likely TB1) has a hardware-level issue** with how it powers/drives USB peripherals. Same brand+model dongle on TB3 negotiates 5GHz 802.11ac; on TB2 it could only negotiate 2.4GHz 802.11n and inbound packets to wlan0 died unless Ethernet was also plugged in (Ethernet plug likely provides marginal additional USB bus stability or triggers driver re-init).

### Decisive test still TODO (cheap, ~10 min)
1. Try the dongle in **each of TB2's 4 USB ports** (2x USB 3.0 blue + 2x USB 2.0 black). If only some ports fail → bad port, use a different one.
2. If all ports fail → put **TB3's SD (cloned or physically swapped) into TB2**. If TB2 with TB3's image still fails → **hardware fault on TB2's USB controller / power circuit; software cannot fix it.** If it works → OS state corruption that survived our partial fixes; remediation is to clone TB3's SD onto TB1/TB2.
3. **Beefier PSU** — official Pi 5 wants 5V/5A USB-C. Underpowered supplies throttle USB current. Cheap thing to try.

Likely outcome (gut): TB1/TB2 have a USB power circuit fault (cap aging, voltage sag under USB load). 5GHz/ac requires more dongle current than 2.4GHz/n, which is why the dongle degrades to /n on these units. Ethernet plug may add just enough current draw / bus activity to keep the USB hub stable.

### What to preserve before flashing
N/A — flash avoided. Switched to built-in WiFi instead.

### Lessons / memories
- **Don't trust mDNS / `.local` hostnames when diagnosing.** Always use the literal IP from the router's DHCP table.
- **Don't trust the router's "online" indicator alone** — the bot can be associated at the WiFi link layer with sshd dead or the dongle in a state where inbound packets fail. Use `ping <ip>` from another LAN host first.
- **When the app works but SSH fails, that's outbound-works/inbound-fails** — a very specific signature. Suspect L2/dongle/driver before the bot's code.
- **Razer RGB keyboards don't enumerate during early Pi boot.** Keep a cheap basic USB keyboard in the toolbox for HDMI rescue.
- **SD-card surgery is the reliable recovery path** when SSH is dead. Pop SD into another Pi via USB reader, mount `/dev/sda2`, edit `/etc/NetworkManager/system-connections/*` and `/etc/systemd/system/multi-user.target.wants/`, sync, unmount, reinsert. ~10 min.
- **The Pi Imager's preconfigured WiFi has `hidden=true` by default** — wrong for any visible home network and causes association flakiness. Always fix to `false`.
- **Stale `.nmconnection` files from prior locations can hijack the bot.** When provisioning a new unit at a new location, audit `/etc/NetworkManager/system-connections/` and delete anything that isn't the current home WiFi.
- **Ethernet always wins.** For new bot bring-up or recovery, plug a cable in first. WiFi-only diagnosis is a trap.

### Files touched
None in the repo. All changes were to TB1 and TB2 SD-card OS state (no code changed in `/home/morgan/dogbot/`).

### Commits this session
- (this commit) — docs: log TB1/TB2 SSH rabbit hole + Edimax dongle memory

---

## Session: 2026-05-26 (evening) — treatbot3 dispenser tuning, treatbot5 motor failure, battery false-charging fix

**Duration:** ~2 hours
**Robot:** treatbot3 (primary) + treatbot5 (via SSH)
**Status:** ✅ treatbot3 tuned and pushed. ⚠️ treatbot5 has a new hardware issue: motor coils appear dead (independent of UART).

### Work completed
1. **TMC2209 UART verified working on treatbot3** — `version=0x21`, IRUN=31, microstep=8, all yaml config now actually applies to the chip.
2. **Spreadcycle enabled on treatbot3** — added `chopper_mode: "spreadcycle"` to `treatbot3.yaml`. Full rated torque vs ~50–70% on stealthchop default. Confirmed live (`chopper=spreadcycle` in init log).
3. **Treatbot3 dispenser tuning, per user**:
   - `steps_per_slot: 137 → 144`
   - `step_delay: 0.006 → 0.010` (slower, more torque per step)
   - `reverse_steps: 100 → 70` (gentler anti-jam)
   - `chopper_mode: "spreadcycle"` (new)
4. **Battery false-charging fix** (fleet-wide code change in `services/power/battery_monitor.py`):
   - `motor_idle_required_s: 30 → 120s` (4S LiPo rebound takes 1–2 min to settle)
   - Trend threshold `0.20V → 0.35V` (legit charging is ~1V/min so 0.35V/25s is well within real detection; passive rebound plateau stays under the gate)
   - Confirmed false charging events on treatbot3 today: 05:26 (trend +0.326V) and 17:37 (trend +0.241V), both with no charger attached.
5. **Battery voltage calibration verified accurate** on treatbot3 — user's DMM read 15.42V; ADS1115 reported `adc_voltage=0.284 × 54.28 = 15.42V` exactly. **No recalibration needed.**

### treatbot5 — new hardware finding
User SSH'd in via `treatbot5.local` from treatbot3. After cranking Vref pot to "max voltage" position (matching treatbot3/4 positions), dispenser is:
- **Silent on every dispense attempt** (no buzz, no whine)
- **Stepper shaft spins freely** by hand — coils NOT energized
- treats #6, #7 logged "successful" at 04:44 but software-log success ≠ physical rotation
- Vref position is correct (matches working units) → pot is not the issue

Most likely now: **dead motor coil(s)**. Independent of the chronic UART silence on treatbot5 — two separate problems.

**User's next step:** spin-test the motor with coil-wire shorting trick:
1. Disconnect 4 motor wires from TMC2209
2. Short A1↔A2, try spinning shaft — should be notably stiff if coil A is live
3. Repeat with B1↔B2
4. If both pairs spin freely when shorted → motor is dead, needs replacement

**User pushed back correctly** on my earlier "motor dispensed 7 treats today" claim — it's a software lie (`_step()` returns success regardless of physical rotation; no encoder feedback). Don't infer "motor worked recently" from dispense logs.

### Commits this session
- `be3ac8a` — docs: Update resume_chat (part 4 + 23/24 merge) + treatbot3 tuning (carried)
- `7e96d99` — **fix: tighten battery false-charging gate + treatbot3 dispenser tuning**

### Pulled this session
- `a199b27` — treatbot5 UART deep-dive session + reusable echo-split probe (`tests/hardware/test_tmc2209_echo_split.py`)
- The deep-dive log corrected the earlier "VIO disconnect" framing; canonical wiring documented in `feedback_tmc2209_uart_wiring.md`.

### Outstanding for next session
1. **Treatbot5 motor diagnosis** — short-coil spin-test to confirm dead motor. If dead, replacement stepper needed. Independent of UART (which is also broken on treatbot5 but is purely a UART-control issue, not dispense-control).
2. **Treatbot5 Vref pot** — confirmed in correct ("max") position now matching treatbot3/4.
3. **TMC2209 UART on treatbot5** — leading theory from deep-dive: clone-board internal routing; pad labeled `PDN_UART` may not actually connect to chip's PDN pin. Side-by-side physical comparison with treatbot1 still TBD.
4. **Treatbot3 spreadcycle test** — user should actually trigger a dispense and confirm the audible whine + improved torque feel.
5. **Treatbot3 blue LED MOSFET swap** — still pending hardware action (logic-level FET).
6. **Other fleet `battery_monitor.py` rollout** — on next `git pull` + service restart, false-charging announcements should stop fleet-wide.

### Lessons / memories
- **Don't infer hardware state from software logs** when there's no closed-loop feedback. Dispenser logs success on STEP-pulse send, not on physical motion. User correctly called this out — should have caught it sooner.
- **Per-yaml tuning IS per-unit** — pushing `treatbot3.yaml` doesn't change behavior on other robots. Only shared code (`battery_monitor.py`) propagates on pull.
- **Test-by-shorting trick** for stepper motors: shorting a coil pair adds back-EMF damping → makes shaft notably stiff to turn. Easy field test for dead windings with no instruments.

---

## Session: 2026-05-26 — treatbot5 UART deep-dive (unresolved), full Pi health audit

**Duration:** ~6+ hours (very long session)
**Robot:** treatbot5
**Status:** ⚠️ Unresolved on the UART question, BUT: comprehensive Pi health audit confirms unit is otherwise healthy and mission-ready. Several memories saved to capture learnings.

### TL;DR — what future-you needs to know up front
- **treatbot5's TMC2209 UART does not work** despite exhaustive diagnostic effort. **Pi UART is provably healthy** (proven via direct GPIO14↔GPIO15 jumper loopback — wrote bytes including null/binary at 4 different baud rates, all came back perfect). The fault is **between the Pi pin header and the TMC2209 chip's UART pin** — specifically the wire→breakout-pad→chip-internal path on treatbot5's specific module(s).
- **It does not matter.** Dispenser runs on hardware defaults (Vref pot + MS1/MS2 strapping); UART has been a nice-to-have, not a must-have, across the entire fleet.
- **Stop chasing UART on treatbot5.** Real next step: physical visual comparison of where treatbot1's UART wire physically lands on its breakout vs where treatbot5's does, with attention to solder bridges near PDN on the module back.
- **treatbot5 yaml is correct as-is.** `microstepping: 8` matches hardware default, no time-bomb.
- **Service restart this session** picked up new gimbal centers from the prior session (`pan_center: 46, tilt_center: 97`) — that carry-over item is done ✓.

### Theories tested and ruled out (in chronological order)
| Theory | Verdict | How disproven |
|---|---|---|
| VIO disconnect (carry-over from part-2 session) | **WRONG** | Multimeter: VIO=3.3V at chip pad |
| TX/RX wires swapped at TMC end | **WRONG** | User re-wired correctly; same result |
| TMC2209 module damaged (chip ESD) | **WRONG** | Swap-test with module pulled from working treatbot1/2 — still silent |
| Factory pin-5 vs pin-4 PDN routing | **RE-OPENED — leading theory** | User said boards "look identical" to treatbot1's, but never verified the wire lands on the same *physical pin position* AND that pin actually routes to the chip's PDN internally. After loopback proved Pi UART is fine, this is the most likely remaining cause. |
| TX series resistor too low (was ~400Ω vs spec 1K) | **PLAUSIBLE BUT WRONG** | Sound physics argument from another LLM; user swapped to 1K — still silent |
| Pyserial/Python software issue | **WRONG** | Shell-only `stty + dd + printf` path returns identical bytes |
| My probe was lying (internal Pi loopback) | **WRONG** | Definitive RX-disconnect test: 0 bytes when wire physically unplugged → probe is honest |
| Bluetooth grabbing ttyAMA0 | **WRONG** | Stopped BT + rmmod hci_uart → no change. (The `hci_uart_bcm serial0-0` line in dmesg refers to a different internal serial node, not ttyAMA0) |
| Comms subsystems (BT/WiFi/etc.) interference | **WRONG** | Strip-down test: NetworkManager off, wpa_supplicant off, audio off, treatbot off, bluetooth off → still `echo_only` |
| Wrong UART device | **WRONG** | Verified `/dev/ttyAMA0` is the GPIO14/15 PL011 (uevent confirms `OF_FULLNAME=/axi/pcie@1000120000/rp1/serial@30000`) |
| Pin function mode | **WRONG** | pinctrl shows GPIO14/15 correctly in alt-4 (UART) mode |
| Process contention | **WRONG** | lsof shows only our probe holding the port |

### Cross-fleet ground truth established
**treatbot1 confirmed working** via chip-level probe + boot log:
- IOIN=0x21000241 (chip version 0x21)
- GCONF=0x00000081
- CHOPCONF=0x05000053
- DRV_STATUS=0xC01F0000

This kills the prior session's "fleet may never have had real UART working" hypothesis. UART CAN work; treatbot5 is genuinely the outlier.

### What treatbot5 IS doing that's wrong (one anomaly)
On every probe, Pi RX reads exactly the bytes Pi TX wrote (4-byte single-wire echo through tied PDN pad) and **nothing else**. Chip never drives the line low to encode a reply. This is anomalous — the same physical chip works on treatbot1.

**THE DECISIVE TEST (added late in session): Direct GPIO14↔GPIO15 jumper loopback.**
- TMC wires disconnected from Pi end
- Single Dupont jumper from Pi pin 8 (GPIO14 TX) directly to Pi pin 10 (GPIO15 RX)
- Wrote `b'HELLO'`, `b'\x55\xAA\x01\x02\x03\xFF\x00LOOPBACK'`, and `b'PI-OK\n'` at 9600/57600/115200/230400 baud
- **Every byte came back perfectly, every baud, every test**
- Pi UART is 100% healthy end-to-end (both TX and RX directions work for arbitrary byte values)

**This kills the earlier hypothesis that GPIO15 was damaged.** It also eliminates Pi-driver/config issues entirely. The fault is genuinely between the Pi pin header and the TMC2209 chip's UART pin — somewhere in the wire→breakout-pad→chip-internal path on treatbot5 specifically.

Most likely remaining cause:
- **The wire on Pi pin 10 (RX) lands on a TMC2209 breakout pad that's labeled "PDN_UART" but isn't actually routed internally to the chip's PDN_UART pin on this specific module.** Multimeter measures pin-to-pad continuity, NOT pad-to-chip-die continuity. The factory text the user found mentions a pin-4/pin-5 jumper-selectable routing on some clone boards.
- treatbot1 likely has the wire on a different physical pin position (or has a solder bridge mod) that treatbot5 lacks — but user said "looks identical," which is a visual claim, not a verified pin-position match.

### Comprehensive Pi health audit (this session) — RESULTS
| System | State |
|---|---|
| Throttling/undervoltage | clean (`throttled=0x0`, no events 7d) |
| Thermal | 49°C (excellent) |
| Power rails | 3V3 at 3.30V, EXT5V at 4.92V — healthy |
| EEPROM firmware | up-to-date (Aug 2025 release) |
| PCIe / Hailo-8 | enumerated, driver loaded, `/dev/hailo0` present |
| I2C bus | PCA9685 (0x40) + ADS1115 (0x48) responsive |
| Camera | DetectorService runs at 14.9 FPS, 0 pipeline errors |
| Storage | 10% used, no FS errors |
| Memory | 5.4G available, no OOM |
| USB | clean |
| Network | connected (WiFi: 524Pomeranian) |
| treatbot.service | clean, no crashes/exceptions in last 24h |
| GPIO function map | all pins in correct alt-modes |

### Cosmetic anomalies worth flagging (NOT faults)
1. **Bluetooth using default MAC** (`BCM: Using default device address (43:45:c0:00:1f:ac)`). BCM firmware didn't load the chip's unique MAC. Side effect of `dtoverlay=disable-wifi` partially disturbing the combo chip's firmware load. Xbox controller pairing still works — not blocking.
2. **wifi_manager log spam every 30s** — `Command failed: pgrep -f hostapd`. The wifi monitor is polling for AP-mode hostapd that isn't running (because robot is in client mode). Code-side fix opportunity, not a failure.
3. **OF overlay memory-leak warnings** for i2c and spi at boot. Cosmetic Pi 5 firmware quirk. Ignore.

### Memories saved this session
- `feedback_tmc2209_uart.md` — canonical wiring spec (1K resistor on TX leg only, PDN pin 4, VIO 3V3, common GND). **Corrects** the earlier "VIO disconnect" theory. Also documents the `echo_only` diagnostic pattern and the RX-disconnect sanity test.
- `project_treatbot1_uart_reference.md` — known-good register reference values for cross-fleet comparison.

### Files created (uncommitted, kept for future use)
- `tests/hardware/test_tmc2209_echo_split.py` — the probe script used throughout. Reusable.
- `tests/hardware/test_tmc2209_dump_regs.py` — created on treatbot1's side for register dump.

### Pending / next session
1. **(Highest-value if pursued) Compare treatbot5's vs treatbot1's UART wire physical landing position.** Now that Pi UART is provably healthy, the only remaining cause of treatbot5 silence is the wire→pad→chip path on the breakout. Have treatbot1's Claude (or yourself) physically inspect: (a) which numbered pin position on the breakout header treatbot1's UART wire lands on, counting from a fixed reference; (b) any solder bridges visible near PDN on the back of treatbot1's module. Then visually compare treatbot5's setup. If different position OR different bridge state → root cause found.
2. **USB-to-serial adapter test** is NO LONGER a useful diagnostic — Pi UART is already proven healthy via loopback. Adapter would now only confirm what loopback already showed.
3. **(Optional) Fix wifi_manager pgrep spam** — code-side: don't poll for hostapd unless in AP mode. `services/network/wifi_manager.py`.
4. **(Optional) Restore BT MAC** — if/when stable Bluetooth identity becomes needed. Unlikely to matter for current usage.
5. **Battery false-charging on treatbot5** still unresolved from prior session. Not touched tonight.
6. **Vref pot crank on treatbot5** — was carry-over item, user mentioned mid-session "I did the trimpot backward (shut off all voltage/max position)". This needs to be re-done in the correct direction to bump dispenser torque. Target Vref ~2.0-2.2V (multimeter to GND).

### Commits this session
- (pending) Update to resume_chat.md (this entry)
- Untracked: `tests/hardware/test_tmc2209_echo_split.py` (kept for future debugging)
- Untracked: `.claude/TMC2209_UART_SETUP.md` (per-unit, not in git by design)

### Critical lessons (saved to memory)
- **Don't trust "UART working" claims in resume_chat without chip-level probe verification.** Most fleet-wide claims through 2026-05-25 were aspirational, scheduled-but-not-executed, or based on the warning being benign. Only `chip_replied` + `version=0x21` from `tests/hardware/test_tmc2209_echo_split.py` counts as proof.
- **The VIO-disconnect theory from part-2 was wrong.** Real canonical wiring is documented in `feedback_tmc2209_uart.md` — 1K resistor on TX leg only, PDN pin 4, VIO=3V3, common GND. Treatbot4's fix yesterday used this canonical wiring.
- **Always do the GPIO14↔GPIO15 jumper loopback test BEFORE concluding "Pi UART is broken".** It's a 30-second physical test that definitively separates "Pi UART hardware issue" from "downstream wire/breakout/chip issue." I spent hours late-session leaning toward "GPIO15 input damaged" — the loopback proved that completely wrong in 5 seconds. The 4-byte single-wire echo through the chip's PDN pad is NOT proof that Pi RX can clock real bytes from a driver — only loopback proves end-to-end UART.
- **Multimeter continuity from Pi pin to a labeled breakout pad ≠ continuity to the chip's silicon pin.** TMC2209 clone breakouts can have selectable internal routing (e.g., the factory text mentioning pin-4 vs pin-5 PDN selection via solder bridge). The pad with the label may not be electrically connected to the chip pin it claims to expose.

---

## Session: 2026-05-25 (part 5) — TMC2209 UART FIXED on treatbot4

**Duration:** ~1.5 hours
**Robot:** treatbot4
**Status:** ✅ UART working — `TMC2209 detected: version=0x21` after 5+ wire iterations

### What was actually wrong
Four separate physical wiring issues, all needed at once:
1. **VIO** wire missing — added 3.3V from Pi pin 1
2. **GND** wire missing — added Pi pin 6 → TMC2209 logic-side GND (not motor GND)
3. **1K resistor on wrong leg** — was on RX (pin 10), moved to TX (pin 8). User's resistor is ~700Ω, close enough.
4. **PDN_UART is at chip pin 4** (textbook StepStick) — earlier in the session I incorrectly inferred pin 5 from a 0V-idle voltage reading and sent user on a wild goose chase. Pin 4 was right all along.

### Verification after fix
`TMC2209 configured: IRUN=31, IHOLD=5, 8x microstep, vsense=0, SGTHRS=0, chopper=spreadcycle` — all yaml settings actually taking effect now.

### Code bug found and fixed
`config/config_loader.py::DispenserConfig` lacked a `chopper_mode` @property and had no `__getattr__` fallback. So `getattr(dispenser, 'chopper_mode', None)` always returned None, and yaml's `chopper_mode: "spreadcycle"` silently fell through to "stealthchop". Added explicit @property. Without this fix the spreadcycle setting would have been silently ignored even after UART came up.

### Yaml time bomb defused
treatbot4.yaml had `microstepping: 4` set during the months UART was broken (had no effect — chip was on hardware default 8x). With UART now working, that setting would have caused 2× over-rotation per dispense. Reverted to `microstepping: 8` in same commit as the UART fix.

### Diagnostic mistakes I made this session — for next session's reference
- Multiple incorrect theories proposed: "PDN HIGH = power-down" (datasheet says UART works regardless), "pin 5 = PDN_UART" (was pin 4), "VIO alone fixes it" (needed 3 more changes too). User correctly called out my batting average. Real fixes were the four wiring items; I should have walked through the full list earlier rather than offering them one-at-a-time as new theories.

### For the user's next moves
1. **Apply same 4-item check to treatbot3 and treatbot5** — likely same wiring pattern, same fix
2. **Before bringing UART up on each:** check yaml `microstepping × steps_per_slot` is consistent (treatbot2 baseline = 8 × 137 = 30.8°/slot)
3. **Test the dispenser** — spreadcycle is now on; expect motor whine + significantly more torque (irun=31 vs the ~16 default that was running before)
4. Vref pot tuning is now optional (only needed if irun=31 + spreadcycle still isn't enough — unlikely)

### Commits this session
(pending) yaml time-bomb fix + chopper_mode config_loader property + this session log

---

## Session: 2026-05-25 (part 4) — treatbot3 TMC2209 UART fixed (wiring corrected)

**Duration:** ~30 min
**Robot:** treatbot3
**Status:** ✅ Complete — UART live, chip detected, yaml config now actually applies.

### Work completed
1. **Pulled origin/main with conflict** in `.claude/resume_chat.md` — stash/pop, resolved by merging the local 2026-05-23/24 treatbot3 LED-MOSFET session into the newest-first order (between 2026-05-24/25 and 2026-05-22). Working tree now has only the local treatbot3.yaml tuning bumps from prior sessions (`coach_tilt_min 29→35`, `dispenser.reverse_steps 40→100`, `right_multiplier 1.0→1.06`).
2. **TMC2209 UART diagnosis on treatbot3** — same boot-time symptom as treatbot4/5 (`TMC2209 not responding on UART — using hardware defaults`). OS-side checks all pass (uart0=on, enable_uart=1, no serial console in cmdline, /dev/ttyAMA0 perms ok, morgan in dialout, treatbot.service SupplementaryGroups includes dialout). yaml time-bomb confirmed *absent* on treatbot3 (already `microstepping: 8`, matches MS1/MS2 hardware strap default — no over-rotation surprise when UART comes alive).
3. **User wired the UART data line correctly** — see notes below. After power cycle: `TMC2209 detected: version=0x21` ✓ + `TMC2209 configured: IRUN=31, IHOLD=5, 8x microstep, vsense=0, SGTHRS=0, chopper=stealthchop`.

### TMC2209 UART wiring — the fleet-wide gotcha
**The systemic mis-wiring across treatbot3/4/5 was the single-wire UART data line, not VIO.** The classic single-wire half-duplex UART hookup is:
- **TMC2209 Pin 4 (PDN_UART)** ← junction of two Pi pins:
  - **Pi Pin 8 (GPIO 14, TXD)** via a **~1kΩ series resistor** (measure end-to-end resistance to confirm ≤1kΩ — this is the "send" leg; the resistor prevents Pi's push-pull TX from clobbering the chip's reply during half-duplex turnaround)
  - **Pi Pin 10 (GPIO 15, RXD)** **direct wire, no resistor** (this is the "listen" leg)
- The wire colors on the harness Morgan was using were ambiguous, which is how all 3 robots got wired the same wrong way. Now corrected on treatbot3.

**Diagnostic signature** of this fault: probing all 4 slave addresses returns clean 4-byte echo (Pi's TX loopback through the shared joined node) with no chip reply (because the joined node never actually reaches PDN_UART on the chip). Looks identical at the bus level to a missing-VIO fault — both produce echo-only — so don't assume which one is wrong just from the symptom. **Check the wire colors and the resistor first** before chasing VIO.

### Outstanding for next session
1. **Apply the same wiring fix to treatbot4 and treatbot5** — Pin 4 of TMC2209 ← Pi Pin 8 (via 1kΩ) + Pi Pin 10 (direct).
   - Before powering on: **check treatbot4's yaml time-bomb** — when last logged it had `microstepping: 4` while the hardware strap default is 8; fixing UART will activate microstepping=4 and cause 2× over-rotation per dispense. Revert to `microstepping: 8` in treatbot4.yaml **before** the wiring fix, or change both atomically.
   - treatbot5.yaml had no `microstepping` field last session — fill it in to match what you want.
2. **Crank Vref pot on each unit if torque feels light** — independent of UART, the Vref pot sets hard current ceiling.
3. **treatbot3 blue LED MOSFET swap** — hardware action still pending (logic-level FET — IRLZ44N / AO3400 / FQP30N06L). Software (commit `e1f3de8`) already drives the path correctly.
4. **(Optional cosmetic)** Add `session_ended` handler to relay_client to silence "Unknown message type" log warnings.

### Commits this session
- `490c980` docs: Update resume_chat (session 2026-05-25 part 4 + 23/24 merge) + treatbot3 tuning

### Correction (per part 5 — written concurrently on treatbot4)
The "data line was the only fault" framing in this part-4 entry is incomplete. The treatbot4 session running in parallel found the actual fleet-wide pattern needs **four** items, all at once: (1) VIO wire, (2) logic-side GND wire, (3) 1kΩ resistor moved to the **TX leg** (Pi Pin 8), not RX, (4) PDN_UART = TMC2209 chip pin 4 (textbook StepStick). Plus a real code bug in `config/config_loader.py::DispenserConfig` — missing `chopper_mode` @property silently dropped yaml's `chopper_mode: "spreadcycle"`. On treatbot3 the user likely fixed several of these at once but only the resistor-leg correction was explicitly called out here. The accurate fleet-wide story is in part 5 above.

---

## Session: 2026-05-25 (part 3) — treatbot5 pull/conflict, Xbox controller swap, gimbal re-center

**Duration:** ~1.5 hours
**Robot:** treatbot5
**Status:** ✅ Complete — committed as `005d78b`, pushed to origin/main

### Work completed
1. **Pull origin/main with conflict** — rebase against `b4f21f3` (later `191ba0c`) hit a conflict in `.claude/resume_chat.md` where the local treatbot5-setup session log overlapped with the incoming treatbot4 night-mode log. Resolved by keeping both entries in newest-first order. New code on this Pi after pull: night mode controller (`5f0dacc`), mood_led LedService routing (`e1f3de8`), tilt_min/dispenser tweaks (treatbot4-only), and a later `feat: night mode kills blue tube` (`191ba0c`).
2. **TMC2209 UART verified on treatbot5** — all OS-side steps from `.claude/TMC2209_UART_SETUP.md` pass (dtparam, cmdline, dialout, /dev/ttyAMA0 perms, SupplementaryGroups). Chip-level probe across all 4 slave addresses returned **echo only** (Pi sees its own TX loopback) — same symptom as treatbot4. Root cause is the same VIO logic-supply wire being disconnected (per part-2 session diagnosis). Dispenser runs on hardware defaults (Vref pot + MS1/MS2 strapping) and is fine for now.
3. **Xbox controller swap on treatbot5** — removed old `AC:8E:BD:4A:0F:97`, paired new `78:86:2E:8C:47:97`. Burned a lot of time chasing a phantom "BlueZ procedure" issue before the actual root cause surfaced in dmesg: `BLE firmware version 5.09, please upgrade for better stability`. The new-out-of-the-box controller had old firmware → xpadneo's welcome-rumble crashed it → infinite reconnect loop (sysfs index raced from .003E to .00BF in seconds). User updated firmware to 5.23 via Xbox Accessories app on Windows. After update, single clean `pair → trust → connect` worked.
4. **Updated `fix_xbox_controller.sh`** — hardcoded MAC swapped to new controller (committed).
5. **Re-centered treatbot5 gimbal** — `pan_center: 64 → 46`, `tilt_center: 89 → 97` (live-read after user physically aimed dead-ahead). Written manually to yaml to avoid the `/camera/calibrate save:true` comment-stripping bug. Not live-reloaded (no `/config/reload`); takes effect next service restart.

### Key learnings (saved to memory)
- **`feedback_xbox_firmware.md`** — When a new Xbox controller won't pair stably (slow-flash LED, BlueZ shows Connected:yes, journal flood of `Error reading event: I/O operation on closed file`, sysfs reconnect-counter racing): FIRST check `dmesg | grep "BLE firmware"`. If xpadneo reports `please upgrade for better stability`, firmware is the blocker — Windows + Xbox Accessories app is the only real fix. Don't waste hours on bluez sequencing.

### Memory cleanup
- Wrote a `feedback_xbox_pairing.md` mid-session claiming `fix_xbox_controller.sh` was the procedure that worked — deleted it immediately when symptoms made clear the script wasn't actually working. Replaced with the firmware-first memory above.

### Procedures confirmed
- **Xbox controller pairing on Pi (with updated firmware):** `bluetoothctl scan on` → wait → `pair MAC` → `trust MAC` → `connect MAC`. Set agent to `NoInputNoOutput` first. `rfkill` cycle the radio if BT state is stuck.
- **Conflict-resolving `resume_chat.md` on rebase:** keep both session blocks, newest first; just delete the conflict markers and add a `---` between sessions.

### Pending / next session
1. **TMC2209 VIO wire** on treatbot5 — same physical fix as treatbot4 (Pi 3V3 → TMC2209 VIO pin). After fix, expect `TMC2209 detected: version=0x21` in dispenser init log instead of the "not responding" warning. Read part-2 session notes about the yaml time-bomb (microstepping mismatch) BEFORE wiring — treatbot5.yaml currently has no `microstepping` field, so it'll use whatever default; check before activating UART.
2. **Crank Vref pot on treatbot5** if dispenser torque feels light.
3. **Restart treatbot.service** to pick up the new gimbal centers (user explicitly said "it's fine" to defer).
4. **Battery false-charging on treatbot5** — still unresolved from prior session; needs per-unit charging-trend threshold tuning.

### Commits this session
- `005d78b` — chore: re-center treatbot5 gimbal + update fix_xbox_controller.sh MAC (pushed)

### Notes
- `.claude/TMC2209_UART_SETUP.md` remains untracked by design (its own header says "Per-Unit, Not in Git"). Treatbot5 passes 100% of its OS-side checks; only the hardware VIO wire is missing — same story as treatbot4.
- Old controller MAC `AC:8E:BD:4A:0F:97` is gone from bluez. If that controller comes back later, it'll need to re-pair from scratch.
## Session: 2026-05-25 (part 4) — treatbot4 VIO fix attempt, UART STILL silent

**Duration:** ~30 min, no code changes
**Robot:** treatbot4
**Status:** ⚠️ Diagnostic still open — VIO alone did not solve UART silence
**Important update for treatbot5 session above:** part-3 (treatbot5) attributed UART silence to the VIO disconnect from part-2's diagnosis. THIS session proves that diagnosis was **incomplete** — wiring VIO on treatbot4 didn't fix it. The actual root cause is still unknown. Don't wire VIO on treatbot5 expecting it to fix UART; do the multimeter test below first.

### What happened
- User wired 3.3V to TMC2209 VIO pin per last session's recommendation
- Re-probed UART after reboot: still silent. Boot log still says `TMC2209 not responding`. Live probe still only sees Pi's own TX loopback.
- User pointed out 2+ other robots have the same silent symptom, ruling out random chip damage / loose solder. This is systematic to their build pattern, not coincidence.

### Where we landed
- Confirmed topology is correct: split is at PDN_UART (labeled at "pin 5" on user's TMC2209 modules — module pinout differs from textbook StepStick), joining Pi pins 8 (TX, through 1K) and 10 (RX, direct)
- Ruled out: OS config, Pi-side wiring, wrong chip pin, VIO disconnect, single damaged chip
- Leading remaining hypothesis: PDN_UART pin sits HIGH at idle on these modules, putting the chip into power-down mode (`pdn_disable=0` is the chip default → PDN_UART idle HIGH = chip off). Adding VIO may have actually activated a module pull-up that holds the line HIGH. treatbot2 working might mean its module lacks that pull-up.

### Open diagnostic (user's next step)
Multimeter from TMC2209 PDN_UART pin to GND with treatbot.service stopped.
- ~3.3V → external pull-up issue; fix by removing module's pull-up jumper or adding stronger external pull-down (~4.7K to GND)
- ~0V → pull-up not the issue; investigate GND reference or chip variant differences

### Memory updated
`project_tmc2209_vio.md` rewritten to reflect that VIO alone wasn't the answer. Leading hypothesis and remaining diagnostic captured.

### No code changes this session — nothing to commit.

---

## Session: 2026-05-25 (part 2) — UART diagnosis, blue-tube night-off, dispenser reality check

**Duration:** ~2 hours
**Robot:** treatbot4 (with implications for whole fleet)
**Status:** ✅ Diagnostic + small fix complete. UART hardware fix is the user's next physical step.

### Major finding
**TMC2209 UART has been silent on treatbot4 (and likely several others) since fleet bring-up — root cause is the VIO logic supply line being disconnected.** STEP/DIR still work via the chip's internal regulator, so motors turn fine; UART transmit needs VIO to drive the line back to the Pi, so the chip echoes nothing. Empirically proven by:
- Direct UART probe across all 4 possible chip addresses → 0/12 reads succeeded (Pi sees own TX loopback, chip silent)
- Side-by-side comparison vs working treatbot2 → OS / cmdline / cmd group / pins.py / dispenser.py all identical, treatbot2 returns chip version 0x21, treatbot4 silent
- Reported by user: "same issue on other treatbots" + "i have the logic wire disconnected, i believed we didn't need it"

### What this means for the yaml
On treatbot4 (and any other unit with VIO disconnected), all of `irun`, `microstepping`, `chopper_mode`, `shaft_invert`, `sgthrs` are **inert** — they describe an intent the chip never receives. The chip runs on hardware defaults (Vref pot + MS1/MS2 pin strapping). The dispenser has been working all along on these defaults; "light torque" is the user's actual mechanical concern.

### Time bomb to be aware of
treatbot4.yaml currently has `microstepping: 4` + `steps_per_slot: 137`. As long as UART stays silent, this is fine (chip ignores microstepping and uses MS1/MS2 default of 8). But the moment the user fixes VIO and UART starts working, `microstepping: 4` will take effect and 137 steps × (1.8°/4) = 61.6° per slot — **2× over-rotation per dispense**. Either revert microstepping to 8 in yaml BEFORE wiring VIO, or update both together.

### Other work this session
1. **Night mode also kills the blue LED tube** (`main_treatbot.py`). Previous callback only stopped the NeoPixel strip; the separate blue tube stayed on. Now calls both `set_pattern('off')` and `ctrl.blue_off()` on day→night.
2. **Confirmed Flutter app's "night mode" toggle is UI-only** — not wired to our `set_night_mode_override` command. Self-contained brief written for whoever updates the Flutter app to send the relay command + listen for `night_mode_state` events.
3. **User reverted treatbot4.yaml `steps_per_slot` 69→137** and bumped `reverse_steps` 50→70. The 69 value was incorrect for the actual (UART-silent) microstepping the chip uses.

### Practical fix for "light torque" (no UART required)
The Vref pot on each TMC2209 module sets the hard current ceiling regardless of UART. Turning it CW raises current. Target ~2.0–2.2V on the Vref pad (multimeter to GND). User has this on their TODO.

### Documented in this session
- `.claude/TMC2209_UART_SETUP.md` (user added) — full per-Pi UART setup guide; treatbot4 already passes 100% of its checks at the OS layer. Only the hardware VIO wire is missing.

### Next session
1. **Wire VIO to Pi 3.3V on each affected robot** — Pi pin 1 (3V3) → TMC2209 VIO pin. Verify after reboot: `TMC2209 detected: version=0x21` in dispenser init log.
2. **Before doing that, fix the yaml time bomb** — change `microstepping: 4 → 8` in treatbot4.yaml to match working treatbot2 geometry, OR plan to update both fields atomically.
3. **Crank Vref pot on treatbot4** — give the carousel more current to handle load.
4. **Wire Flutter app's night-mode toggle to backend** — the relay-command brief from this session.

### Commits this session
- (pending) blue-tube night-off + yaml steps_per_slot/reverse_steps revert

---

## Session: 2026-05-24/25 — treatbot4 night-vision + camera/battery/dispenser tuning

**Duration:** ~6+ hours (long session across the date boundary)
**Robot:** treatbot4
**Status:** ✅ Complete — committed as 5f0dacc, pushed to origin/main

### Problems Solved
1. **NoIR daytime color** — IMX708 Wide NoIR (IR-cut filter removed) had unusable daytime color via AWB (green skin OR magenta blues depending on preset). Manual `ColourGains=(0.887, 1.52)` locked + AWB off + saturation 0.85 + contrast 1.05 is the best achievable in software; fundamental NoIR limitation (green channel dominant, ColourGains only controls R+B). Real fix would be a physical IR-cut filter for daytime use.
2. **Night mode subsystem built** — full implementation per `.claude/nightvisionrobo.md`. Asymmetric hysteresis (5 lux entry, 100 lux exit) was added specifically because IR illuminator pollutes the camera's lux reading; without the high exit threshold the system would oscillate.
3. **Battery showed 148V / 100%** — treatbot4 has the LOW-ratio voltage divider (~5.6:1), not treatbot3's high-ratio (~54:1). Calibrated against multimeter: 15.43V battery → 2.7368V at A0 → factor 5.638. Now reads correctly.
4. **Dispenser stalling** — yaml-side torque was already maxed (irun=31, microstepping=4, slow step_delay). Root cause: TMC2209 was in stealthChop mode (default, ~50-70% torque). Added `chopper_mode` yaml field; setting "spreadcycle" sets GCONF bit 2 for full rated torque. Opt-in per robot so other units unaffected.
5. **`/camera/calibrate` save:true reformats yaml** — discovered this bug mid-session when it stripped comments + sorted keys. Worked around by writing yaml manually. Real fix would be ruamel.yaml in the save path. Avoid `save:true` until that's fixed.
6. **WebRTC stuck reconnecting** — Camera/relay healthy but app never sent SDP answer. Force-quit + relaunch of phone app fixed it (stale peer connection state).

### Key Code Changes
- `modes/night_mode_controller.py` (NEW) — threaded singleton, lux polling, profile switching via `picam2.set_controls()` (preserves WebRTC stream)
- `services/perception/detector.py` — `_apply_saved_calibration` now accepts `awb_enable` + `colour_gains`
- `services/reward/dispenser.py` — `_configure_tmc` reads `chopper_mode` from yaml, sets GCONF bit 2 for spreadCycle
- `services/cloud/relay_client.py` — added `set_night_mode_override` command handler
- `api/server.py` — added `colour_gains`/`awb_enable` to `/camera/calibrate`, plus `GET /night_mode/status` + `POST /night_mode/override`
- `main_treatbot.py` — wired NightModeController into init/start/stop; LED-off callback on day→night
- `config/robot_profiles/treatbot4.yaml` — full daytime camera profile, battery factor 5.638, chopper_mode spreadcycle, dispenser torque tuning

### What Was NOT Done (deferred)
- **Pi onboard ACT/PWR LED dim** — needs a sudoers entry, didn't tackle
- **`tests/hardware/test_battery_adc_channels.py`** — untracked diagnostic, didn't commit (user's call)
- **Xbox MAC files** — `fix_xbox_controller.sh` + `services/control/xbox_persistent.py` still uncommitted (per-robot shared files, intentional)
- **`/camera/calibrate` save:true comment-preservation fix** — known bug, deferred
- **Live yaml reload** — there's no `/config/reload`; all yaml is read-once at service init. User now knows.

### Important Notes / Gotchas for Next Session
- **NoIR daytime color is genuinely a hardware limitation** — don't chase it further in software. Either accept the muted profile or buy a clip-on IR-cut filter.
- **Asymmetric hysteresis (5/100 lux)** is intentional — do not "fix" by making it symmetric, or IR illuminator will cause oscillation.
- **Night mode profile-switching uses `set_controls()` only** — never call `start()`/`stop()`/`configure()` from the night mode path; that would kill WebRTC. The settle pattern is: AE on briefly → 3s settle → read metadata → lock values with AE off.
- **`chopper_mode` field defaults to stealthchop** when absent — safe default for other robots.
- **State file `state/night_mode.json` persists override** across restarts; do not delete unless intentional.
- **cam1 port on treatbot4 Pi 5 is suspect** — camera is wired to **cam0**. Earlier session troubleshooting strongly suggested cam1 port damage. Do not move ribbon to cam1.
- **Service must be restarted to test:** spreadcycle dispenser fix, latest camera profile values, battery factor (already restarted once during session — verified working). User had not done the spreadcycle test by end of session.

### Commit
`5f0dacc — feat: night mode controller + treatbot4 camera/battery/dispenser tuning` (pushed to origin/main)

### Next Session
1. **Verify spreadcycle dispenser** — restart service, listen for motor whine on dispense, confirm `chopper=spreadcycle` in TMC2209 init log
2. **Real-world night mode bench test** — dark room, watch system auto-switch to night, confirm IR illuminator doesn't oscillate the mode
3. **Decide on IR-cut filter** — physical fix for daytime color if dog-training footage quality matters
4. **Pi onboard LED dim** — sudoers entry + writes to `/sys/class/leds/{ACT,PWR}/brightness` if user wants this minor enhancement
---

## Session: 2026-05-23/24 — treatbot3 video diagnosis + blue LED MOSFET hunt

**Duration:** Multi-resume span
**Robot:** treatbot3
**Status:** ✅ Software fix committed; LED is hardware-fault pending physical repair

### Key findings
- **Video "slow"** is app-side, not robot. Robot session setup = sub-second (572ms request→first frame); direct LAN ICE pair sustains 14+fps. The "slow" comes from the app's relay WebSocket flapping at reconnect (5× User connected events in 4s observed) + the app retrying stale session IDs that the robot doesn't recognize. Re-login is the workaround; real fix is Flutter side.
- **UX bug**: app's "unauthorized mode" silently swallows every command with no banner/indicator — wasted ~20min debugging "nothing works" before realizing.
- **`Unknown message type: session_ended`** — cosmetic gap: Lightsail relay sends a session_ended notification the robot has no handler for. Doesn't break anything; could add handler to quiet logs.

### Blue LED diagnostic arc (treatbot3)
**Software issue found and fixed** (commit `e1f3de8`): `mood_led` relay handler called `api.server.blue_led_direct_control()` which made an independent `lgpio.gpio_claim_output(25)` — a SECOND claim on the same pin already owned by `LedController`. Whoever lost the startup race got `'GPIO busy'` silently for the process lifetime, returning False with no logged error. Rerouted through `get_led_service().led.blue_on()/blue_off()` so there's one owner. Cleanly fits `feedback_no_duplicate_handlers`.

**But the LED still doesn't light on treatbot3.** Hardware diagnosis (user opened module up):
- Pin 22 (BCM GPIO 25): clean 3.3V HIGH / 0V LOW ✓
- Vgs at MOSFET silicon pins: clean 3.3V when commanded ON ✓
- Source pin → main GND: continuity ✓
- Drain → LED black wire: continuity ✓
- LED tube itself: passes resistance test ✓
- 12V+ → LED red lead: reaches ✓
- Manually shorting D↔S with probe: LED lights ✓
- **MOSFET channel itself won't conduct despite valid Vgs.** Either dead silicon (failed open) or non-logic-level part (e.g., IRF540/IRF520/IRFZ44N — note the missing "L" — needs 10V Vgs, not 3.3V).

**User action:** swap in a logic-level N-channel MOSFET — IRLZ44N / AO3400 / FQP30N06L are good drop-ins for the modest current the 12V LED tube draws. Once swapped, the existing software (post-`e1f3de8`) should drive it correctly.

### Pulled this session
- `b4e74db → 652a06f` (audio_volume, mood_led handler, treatbot2 gimbal cal, IMX500 LSTM sequences). Ran `scripts/setup_device.sh` (wimz-audio.service, rtw88 conf, WiFi powersave on 4 saved connections).

### Commit pushed
- `e1f3de8` fix: route mood_led relay command through LedService

### Outstanding for next session
1. Swap treatbot3's blue LED MOSFET for logic-level part — then verify mood_led path works end-to-end.
2. (Optional cosmetic) Add `session_ended` handler to relay_client to silence "Unknown message type" warning.
3. (Flutter app, out of scope here) App should: (a) drop stale session IDs after N failed offer/answer rounds instead of hammering for ~2min, (b) surface "unauthorized mode" visibly so users know why commands aren't working.

---

## Session: 2026-05-22 — treatbot5 device setup + power-button diagnosis

**Robot:** treatbot5
**Status:** ✅ Complete. No repo code changes — working tree clean (only this log updated).

### Work completed
1. **Git pull** — `82dbe8b..652a06f` fast-forward, 27 files (+1986/−177). Brought in volume control system (`services/media/volume_manager.py`, `wimz-audio.service`, `apply_saved_volume.py`), adaptive bitrate streaming (`services/streaming/adaptive_bitrate.py`), and device setup tooling (`scripts/setup_device.sh`, `rtw88.conf`, `docs/NEW_ROBOT_SETUP.md`).
2. **Ran `scripts/setup_device.sh`** on treatbot5 — all verification checks passed:
   - `/etc/wimz/` created (owner morgan) — persistent volume state dir
   - `wimz-audio.service` installed + enabled; ran clean (no saved state → applied default 60% via amixer card 2, control 'Speaker')
   - `/etc/modprobe.d/rtw88.conf` installed — disables rtw88 WiFi deep power-save
   - WiFi power-save disabled on all 4 saved connections
3. **Confirmed xpadneo installed** — DKMS `hid-xpadneo v0.9-226-ga16acb0` built for all 4 kernels incl. running `6.12.62+rpt-rpi-2712`. Module auto-loads on Xbox controller BT connect.

### Diagnosis (no fix applied — user said not needed)
- **`reboot` actually powers the robot off.** Root cause: `/lib/systemd/system-shutdown/wimz-killpulse` pulses GPIO26 → Pololu OFF unconditionally. systemd runs system-shutdown hooks for ALL verbs (poweroff/halt/**reboot**/kexec), passing the verb as `$1`. The script ignores `$1`, so a reboot cuts power instead of restarting.
- **Fix (deferred):** wrap the `gpioset` in `case "$1" in poweroff|halt) ... ;; esac`. This hook is NOT in the repo — if fixed later, also add to `scripts/` + `setup_device.sh` for fleet coverage.

### Pending — NOT yet activated
- **WiFi driver change** (`rtw88.conf`) takes effect only after reboot.
- **New pulled code** (volume manager, adaptive bitrate, etc.) loads only after `treatbot.service` restart. Service currently still running old code from May 20 boot.

### Observed this session
- **False "charging" detection on treatbot5** — robot announced/reported charging while NOT plugged in. `battery_monitor._check_charging()` infers charging from a voltage upward trend (motor-idle gated). On treatbot5 this still misfires — needs a per-unit charging threshold/calibration pass. Not fixed this session.

### Next session (TBD — treatbot5 calibration backlog)
1. **Camera/AI verify** — confirm IMX708, gimbal snaps to (pan=64, tilt=89) on boot, AI pipeline + 4-class behavior model load cleanly.
2. **Dispenser test** — never run on treatbot5; trigger auger advance via API.
3. **Motor calibration** — on-floor drive test (all prior testing bench/wheels-lifted); tune `left_multiplier`/`right_multiplier` if it pulls; coach mode tilt re-tune for asymmetric tilt range.
4. **Battery % calibration** — voltage→percent curve and the false-charging detection (`battery_monitor.py`, `_check_charging()`). treatbot5 reads charging while unplugged; needs per-unit voltage-trend threshold tuning. (`calibration_factor` already set to 53.33 last session for the divider, but charging-trend logic is separate.)
---

## Session: 2026-05-21/22 — treatbot2 hardware bring-up + relay mood_led

**Duration:** ~2 hours
**Robot:** treatbot2
**Status:** ✅ Complete

### Problems Solved
1. **WiFi power-save** — Ran setup_device.sh, disabled power-save on all saved networks
2. **Camera ribbon loose** — Identified camera timeout error after reboot, user reseated ribbon
3. **Blue LED not working via app** — Added `mood_led` command handler to relay_client.py (was only in local ws.py)
4. **Gimbal calibration off** — Recalibrated pan (-60/180/320) and tilt (45/90/200) for treatbot2
5. **Xbox controller pairing** — Installed xpadneo driver, paired using `agent on` + `default-agent` sequence
6. **Blue LED hardware** — Diagnosed MOSFET wiring issue (Source wired to GPIO instead of Gate) — user fixed

### Key Code Changes
- `services/cloud/relay_client.py` — Added mood_led command handler for relay path
- `config/robot_profiles/treatbot2.yaml` — Updated gimbal calibration values

### Commit
`46ec876` — feat: relay mood_led command + treatbot2 gimbal calibration

### Hardware Notes (treatbot2)
- Camera ribbon came loose after reboot — check connections on this unit
- Blue LED MOSFET had Source/Gate wires swapped — now fixed by user
- Xbox controller paired successfully (MAC: 28:EA:0B:DB:82:3F)
- xpadneo driver installed via DKMS (not in git — must install per-robot)
- Gimbal pan went to -80 and tore camera ribbon — safe limit set at -60

### Unresolved / Next Steps
- Blue LED toggle in Flutter app needs to send `mood_led` command via relay WebSocket (app-side fix)
- Verify Xbox controller works with treatbot service after reboot
- Consider adding xpadneo install to setup_device.sh for new robots

### Important Notes
- xpadneo is a kernel driver (DKMS), not in git — must be installed separately on each robot
- Xbox pairing requires: `agent on` + `default-agent` BEFORE `pair` command
- JustWorksRepairing was enabled in /etc/bluetooth/main.conf

---

## Session: 2026-05-20 — Silent Guardian movement tier + Xbox controller bring-up

**Goal:** Add a physical-movement escalation tier to Silent Guardian; pair a new Xbox controller; diagnose drift + right-motor issues.
**Status:** ✅ All complete. Working tree clean, 2 commits pushed (`04295ee`, `f4c4876`). Battery-smoothing commit `337d92d` also pushed early in session.

### Work completed
1. **Silent Guardian — new Level 3 movement tier** (`04295ee`). Escalation ladder is now 4 levels: verbal → firm verbal → **physical movement** → calming music. Level 3 plays "quiet" then runs 3 cycles of in-place fwd/back/left/right (~400ms each, motors halted between moves, 5s pause between cycles), then waits for the progressive quiet period → reward. `_run_movement_sequence()` drives wheels via `motor_command_bus` (`CommandSource.AUTONOMOUS`), always finishes all cycles (no bark abort), bails on mode shutdown. `max_level` 3→4; music handler renamed `_process_level_3`→`_process_level_music`. No conflict with carousel anti-jam (separate stepper) or treat dispense (serialized on SG loop thread). **App-facing:** `get_status()` now reports `sg_max: 4`.
2. **Xbox controller fixes** (`f4c4876`):
   - **Right motor undrivable via Xbox** — root cause: the Xbox subprocess called `get_motor_bus()`, creating a *second* `ProperPIDMotorController` that fought the main process for motor GPIO lines. Fix: `xbox_controller.py` sets `WIMZ_XBOX_SUBPROCESS=1`; subprocess now drives motors via HTTP API only (main process = sole hardware owner). Standalone mode unaffected.
   - **Connected-but-unresponsive after Bluetooth blip** — a BT link drop recreates `/dev/input/js0`, leaving a stale fd (errno 19); the retry path never reopened it. Added `_reopen_device()` to self-heal.
   - **Reverse "sticking"** — neutral-stop sent a single stop then relied on a 2s heartbeat; a dropped stop left the motor creeping. Now re-sends 0 every ~50ms for a 0.5s window on transition to neutral.
   - Added `_calibrate_stick()` + `controller.xbox.stick_centers` (yaml) for worn/off-center sticks; default 0.0 = no-op.

### Hardware findings / warnings
- **Spare Xbox controller `A8:8C:3E:50:62:70` is JUNK** — left stick Y potentiometer worn out: rests at ~−0.72 and the rest position *wanders* (caused runaway-at-rest). Unstable rest can't be software-calibrated. **Unpaired this session — do not re-pair it.**
- treatbot1 is back on the **original controller `AC:8E:BD:4A:0F:97`**. MAC restored in `xbox_persistent.py` + `fix_xbox_controller.sh`.
- Right motor + encoder verified healthy via direct test (−707 enc counts) — it was never a hardware fault.
- No new files created this session; all changes were edits. Directory structure doc unchanged.

### Next steps
- Restart `treatbot.service` after any reboot (it ran the new code live this session — currently active).
- Coordinate `sg_max: 4` with the Flutter app team (Guardian level display "X/4").
- Silent Guardian Level 3 movement tier verified by code/syntax only — exercise it live when a 4th bark intervention naturally escalates.

## Session: 2026-05-17 — treatbot5 bring-up (cloned from treatbot3)

**Goal:** First-boot bring-up of treatbot5, cloned from treatbot3's SD card. Pull latest, customize per-unit calibration, run drive + gimbal test sequences.
**Status:** ✅ Core bring-up complete. Dispenser test skipped; full vision/AI verification deferred (camera was unplugged mid-session). Working tree changes ready to commit: `config/robot_profiles/treatbot5.yaml`, `.claude/resume_chat.md`.

### Capabilities now working on treatbot5
- **Identity**: hostname `treatbot5`, `DEVICE_ID=wimz_robot_05`, machine-id `f7cfcc5dbb3b4210b526449a5e9eb5d9`. Profile auto-selects via hostname_map.
- **Cloud relay**: connected `wss://api.wimzai.com/ws/device` on first boot — Lightsail already had `wimz_robot_05` registered (no relay-side work needed this session).
- **Battery monitoring**: reads 16.87V at 16.8V bench (0.4% error) with `calibration_factor: 53.33`.
- **Drive train**: Cytron MDD10A live, both wheels respond, forward+reverse verified.
- **Gimbal**: center + limits calibrated via Xbox-stick sweep.

### Per-unit calibration deltas vs treatbot3
1. **Battery divider runs ~1.8% leaner**: factor 53.33 (vs treatbot3's 54.28). Same hardware design, normal resistor tolerance variance.
2. **Right motor wired with reversed polarity** (same as treatbot3): `right_invert: true`. Left motor fine (`left_invert: false`).
3. **Gimbal center is at completely different servo positions**: treatbot5 lives at `pan_center: 64, tilt_center: 89` vs treatbot3's `67, -12`. Pan close but tilt convention reads very differently — both use "higher tilt = looking up" but the absolute neutral pulse on this servo lands at +89 instead of -12. Mounting/servo tolerance.
4. **Tilt range is asymmetric on this unit**: physical limits 57 (down, mechanical wall) to 236 (up). Center at 89 → only 29° of down-tilt headroom but 147° of up-tilt headroom. Locked `tilt_min: 60` (3° safety from wall), `tilt_max: 236`.
5. **Pan range**: `pan_min: -80`, `pan_max: 268` (user-chosen "comfortable max" via Xbox sweep, not necessarily mechanical wall).
6. **`coach_pan_*` / `coach_tilt_*` inherited verbatim from treatbot3** but need re-tuning in actual coach mode — the 30%-inset formula produces tilt range biased upward (away from floor-level dog tracking) given treatbot5's asymmetric tilt range. Flagged in yaml comment.

### Hardware findings
- **Xbox controller (AC:8E:BD:4A:0F:97)** paired fresh — the cloned BlueZ bonding database came over empty (or got cleared) so no pre-pair carried from treatbot3.
- **Camera (IMX708 wide)** was physically unplugged sometime during the gimbal calibration → service hung in startup with V4L2 buffer errors on next restart. Service stopped cleanly at end of session. **Next session: plug camera in, restart service, verify gimbal snaps to (pan=64, tilt=89) and AI pipeline starts cleanly.**

### Drive test sequence executed (wheels lifted, 30% throttle)
1. LEFT only +30 → forward ✅ (`left_invert: false` correct)
2. RIGHT only +30 → BACKWARD ❌ → flipped `right_invert: true` → retest → forward ✅
3. BOTH +30 → straight forward ✅
4. BOTH -30 → straight reverse ✅

### Gimbal calibration sequence (live Xbox sweep, position read via GET /camera/gimbal)
- CENTER: pan=64, tilt=89 (`current_position` reading)
- MAX UP: tilt=236 (locked as tilt_max)
- MAX DOWN: tilt=57 (mechanical wall hit, +3° safety → tilt_min=60)
- MAX LEFT: pan=267.8 (locked → pan_max=268)
- MAX RIGHT: pan=-78.9 (locked → pan_min=-80)

### Pending for treatbot5 (next session)
1. **Re-plug IMX708 camera, restart service, verify** new gimbal snaps to (64, 89) on boot and AI detection pipeline starts. Confirm 4-class behavior model loads.
2. **Dispenser test** — skipped this session. Trigger via API, confirm auger advances. Expect cosmetic "TMC2209 not responding on UART" warning (matches treatbot3).
3. **Field test on-floor** — all drive testing was bench/wheels-lifted.
4. **Coach mode tilt range re-tune** — `coach_tilt_min/max` currently inherited from treatbot3 and likely tracks too high given asymmetric tilt range.
5. **Optional motor calibration tuning**: `left_multiplier`/`right_multiplier` are 1.0/1.0; under load may need bias if robot pulls one direction.

### Commits this session
- Pulled 5 commits from origin/main (392c2cd, 0cbaeaf, 7a37458, 903b2c0, b4e74db) — includes 174 IMX500 training sequences, LSTM training guide, charging-detector fix.
- About to commit: treatbot5.yaml per-device calibration + this resume_chat update.

### Files touched
- MODIFIED: `config/robot_profiles/treatbot5.yaml` (full per-device fill-in: battery factor, Cytron driver dispatch, motor inversion, gimbal calibration)
- MODIFIED: `.claude/resume_chat.md` (this entry)

---

## Session: 2026-05-17 — treatbot4 hardware bring-up

**Goal:** Bring treatbot4 from cloned-from-treatbot1 SD card to a fully-calibrated working unit. Same Cytron + brushed-motor + IMX708 hardware family as treatbot3, but on a different physical Pi/PCB.
**Status:** ✅ COMPLETE (modulo battery, which needs actual battery plugged in to verify). Commit `89efa8f` pushed locally (not yet pushed to origin).

### Capabilities now working on treatbot4
- **Drive train**: Xbox left stick → Cytron MDD10A → both wheels respond, correct directions
- **Camera gimbal**: pan inherited from treatbot3 (pan_center=67), tilt re-calibrated after dead-servo replacement (tilt_center=43)
- **Cloud relay**: connected to `wss://api.wimzai.com/ws/device`; wimz_robot_04 registered Lightsail-side
- **Xbox controller**: paired (28:EA:0B:DB:82:3F), connected via /dev/input/js0
- **Camera**: IMX708 Pi Camera Module 3 Wide, 19.4 FPS detection pipeline

### Hardware findings worth remembering
- **Right motor wired with reversed polarity** — same as treatbot3. `right_invert: true`. Now 2-for-2 on Cytron brushed-motor builds → memory saved (`project_cytron_right_invert.md`) so treatbot5 starts with `true` as expected default.
- **Tilt servo was DOA** — confirmed via swap test (pan ↔ tilt cables swapped, neither chan-1 PCA output nor swapped-onto-chan-0 produced response). User installed a replacement servo mid-session in ~5 min.
- **New tilt servo's mechanical "level" position = code-angle 43°** (vs treatbot3's -12°). Different mounting offset. Symmetric ±50° physical range gives tilt_min=-57, tilt_max=143, with coach-mode 30% inset (3 to 83).
- **Battery sense divider on this PCB is wired the same as treatbot3** — but bench supply is going directly into the Pololu output (downstream of the divider tap), so ADS1115 A0 reads ~-27mV (floating). Calibration_factor 54.28 inherited; will read correctly once battery plugs into the JST.

### Commits this session
- `89efa8f` feat: treatbot4 hardware bring-up — Cytron + camera gimbal + new tilt servo

### Files touched
- MODIFIED: `config/robot_profiles/treatbot4.yaml` (Cytron section, battery cal, tilt re-cal, driver dispatch)

### Pre-existing issues still pending (unchanged from treatbot3 session)
- `services/motion/motor.py:MotorService.initialize()` NameError on ENCODER_MOTOR_AVAILABLE — dead code, not blocking
- TMC2209 dispenser UART warning — defaults match yaml, dispenser works fine
- Battery monitor `-1.48V CRITICAL` logs (test-rig artifact, resolves with battery plugged in)
- `/camera/center` and `/camera/position` API endpoints reference missing methods (would 500 if called)

### Still pending for treatbot4 (next session candidates)
1. **Verify battery calibration** — plug actual battery in, confirm reading ~16.6V at 16.8V (matches treatbot3 pattern)
2. **Treat dispenser test** — TMC2209/NEMA settings inherited from treatbot3, never tested on this unit
3. **Field test outdoors / under-load drive** — all testing this session bench/wheels-up
4. **Re-tune motor `left_multiplier`/`right_multiplier`** if drive feels asymmetric under load
5. **Push commit `89efa8f` to origin** — currently local-only

### Notes for treatbot5 (next clone)
Per the documented procedure, treatbot5 should inherit treatbot3.yaml as starting point. NEW datapoint from treatbot4: right_invert=true should be the **expected default** for these Cytron brushed-motor builds (2/2). Servos may need re-calibration regardless of donor yaml (tilt mounting differed even between treatbot3 and treatbot4).

---

## Session: 2026-05-15 — Git sync + new Pi 5 diagnostics (treatbot2)

**Goal:** Pull latest changes, verify new Pi 5 hardware after old one was destroyed (12V GPIO mishap)
**Status:** ✅ Complete

### What Was Accomplished

1. **Git Sync**
   - Pulled 6 commits from origin/main (treatbot3 profiles, charging detector fix, etc.)
   - Fixed stale staged changes in `train_behavior_lstm.py` (4-class alignment, CPU export)
   - Added 174 IMX500 behavior sequences to repo
   - Updated LSTM training guide to remove "speak" (audio-only via bark detector)

2. **Hardware Diagnostics — New Pi 5 (4GB)**
   - Pi 5 Model B Rev 1.1: Healthy (48°C)
   - RAM: 4GB (2.5GB available, swap usage higher than 8GB unit)
   - IMX500 camera: Detected (4056x3040)
   - I2C: PCA9685 servo @ 0x40, ADC @ 0x48
   - USB Audio: Card 0
   - Soft power button: Armed and working

3. **Power Button Note**
   - Service starts ~1.5 min after boot (waits for multi-user.target)
   - User pressed button before service armed — expected behavior, not a bug

### Commits Pushed
- `903b2c0` — fix: Align train_behavior_lstm.py with deployed 4-class model
- `7a37458` — feat: Add IMX500 behavior sequences + LSTM training guide
- `0cbaeaf` — docs: Update LSTM guide — speak removed (audio-only via bark detector)

### Next Session
- No pending tasks

---

## Session: 2026-05-09 — Rebadge cloned Pi as treatbot3 + prep for treatbot4/5

**Goal:** This Pi was cloned from treatbot1's SD card. Give it a unique identity (treatbot3 / wimz_robot_03), pre-create profiles + bootstrap script so future clones for treatbot4 and treatbot5 are one-command operations.
**Status:** ✅ Code/config committed and pushed (`bc3ecb9`). Identity reset script ran successfully on this Pi. **Reboot pending — user will trigger when ready.**

### Hardware delta on units 3-5 (vs 1/2)
- Cytron motor driver + 9V brushed motors (no encoders) — replaces L298N + 6V DFRobot encoder motors. PID disabled because no encoder feedback.
- IMX708 Pi Camera Module 3 Wide — replaces IMX500 used on units 1/2.

### What Was Built

**Repo changes (commit `bc3ecb9`, pushed to origin/main):**
- `config/config_loader.py` — added `treatbot3/4/5` to hostname_map. After hostname change, profile auto-selects, no `/etc/robot_id` needed.
- `config/robot_profiles/treatbot3.yaml`, `treatbot4.yaml`, `treatbot5.yaml` — seeded from treatbot1.yaml. PID disabled, motor multipliers neutral (1.0/1.0), camera rotation 0. HARDWARE_NOTE block flags Cytron driver dependency.
- `scripts/wimz_rebadge.sh` — idempotent post-clone identity reset. Takes one arg (3, 4, or 5). Resets machine-id, regenerates SSH host keys, sets hostname + /etc/hosts, writes new .env with fresh DEVICE_SECRET. Does NOT auto-reboot (so user can read the printed secret).

**System changes on this Pi (via rebadge script):**
- machine-id: `67ec0d9e3e5e47c98d50d87f3684c555` (new, distinct from treatbot1)
- New SSH host keys (ed25519 fingerprint: `SHA256:HtaF5WgUG07LGRQJuEH63X5MewYq9SXGkPJ+yahvl78`) — note: key comment field still says "root@treatbot1" cosmetically because hostname changed AFTER key generation; harmless, will correct on next regeneration
- hostname → `treatbot3`, `/etc/hosts` 127.0.1.1 → `treatbot3`
- `/home/morgan/dogbot/.env` → `DEVICE_ID=wimz_robot_03` + a freshly-generated `DEVICE_SECRET` (stored only in .env, not in git — see file directly)
- Bluetooth pairing database left intact (Xbox controller bond preserved from clone)

### Critical follow-up actions (next session)

1. **Register on Lightsail relay**: `wimz_robot_03` + the secret in `/home/morgan/dogbot/.env` must be added to the relay's allowed-devices list before the robot can connect remotely.
2. **Cytron motor driver** (out of scope this session): `services/motion/motor.py` has no Cytron dispatch. Need a new `core/hardware/motor_controller_cytron.py` plus dispatch wiring before drive control works on units 3/4/5.
3. **Per-unit calibration during bring-up**: motor `left_multiplier`/`right_multiplier`, `camera.rotation`, `pan_center`/`tilt_center`, dispenser `steps_per_slot`. All currently neutral defaults in treatbot3.yaml.
4. **Verify after reboot**:
   ```
   hostname                                    # treatbot3
   journalctl -u treatbot.service -n 80 | grep -iE "profile|robot_id|treatbot3"
   ls /dev/input/js0                           # Xbox controller present
   bluetoothctl devices Connected
   libcamera-hello --list-cameras              # confirm IMX708 detected
   ```

### Cloning to treatbot4 / treatbot5 (future)
After this Pi is fully validated, clone its SD card. On each new clone, first boot:
```bash
sudo bash /home/morgan/dogbot/scripts/wimz_rebadge.sh 4   # or 5
sudo reboot
```
Then register the printed DEVICE_ID/SECRET on Lightsail. That's it.

### Concurrency notes
- treatbot1 stays online. Two separate Xbox controllers (one per robot) — first-come-first-served Bluetooth, no preemptive removal needed.
- Once relay-registered, treatbot3 will appear as a separate device alongside `wimz_robot_01`.
- Laptop SSH known_hosts: `ssh-keygen -R treatbot1.local` and any cached IPs before next SSH from laptop, otherwise host-key-mismatch warning.

---

## Session: 2026-04-27 — Git Pull & Camera Boot Config Docs

**Goal:** Pull latest changes, document camera boot config for fleet
**Status:** COMPLETE

---

### What Was Accomplished

1. **Pulled origin/main** (0806e6a → 954fb42)
   - 35 files updated with significant changes
   - New scripts: capture_behavior_sequences.py, train_behavior_lstm.py
   - New service: camera_detect.py
   - Major updates to main_treatbot.py, api/server.py, wifi_manager.py

2. **Documented Camera Boot Config**
   - Added to `.claude/hardware_specs.md`: camera auto-detect settings
   - `/boot/firmware/config.txt` needs `camera_auto_detect=1` and `#dtoverlay=imx500`
   - Fleet status: treatbot2 fixed, treatbot1/3-5 need edit before camera swap

### Files Modified
- `.claude/hardware_specs.md` — Added camera boot config section

### Commit
- c6e5998 — docs: Add camera boot config for IMX500/IMX708 swaps

### Next Session
- No pending tasks from this session

---

## Session: 2026-04-25 — Dog Tracking Deduplication Fix

**Goal:** Fix duplicate bounding boxes (both "Dog" and "Elsa" on same physical dog)
**Status:** COMPLETE

---

### What Was Accomplished

#### Bug Fix: Duplicate Detection Boxes
**Root Cause:** When ArUco visibility is intermittent, both an ArUco-identified entry (e.g., "elsa" at marker_id=42) and a generic entry ("dog_0" at -1000) coexisted in `last_known_positions`, causing two boxes to render.

**Not a regression** - gap from Build 38 (commit 3119962) which added generic entries but never implemented deduplication.

**Fix Applied** (`core/dog_tracker.py`):
1. When ArUco detected → delete any generic entry for that detection index
2. Before creating generic entry → check for overlapping ArUco-identified dog via IoU (>0.3) and reuse that identity
3. In `get_tracked_dogs()` → skip generic entries overlapping ArUco boxes (safety net)

**New helper methods:** `_find_overlapping_tracked_dog()`, `_bbox_iou()`

### Commits
- `0bd343f` — fix: Deduplicate dog tracking to prevent duplicate bounding boxes

### Also This Session
- Gimbal calibration API added (commit `966b6b3`)
- Xbox controller docs updated

### Next Session
- Restart treatbot to apply fix: `sudo systemctl restart treatbot`
- Test with Elsa to verify single box renders
- Lifecycle handler for clean disconnect (interrupted investigation)

---

## Session: 2026-04-25 — Gimbal Calibration API

**Goal:** Add configurable gimbal limits and API endpoints
**Status:** COMPLETE

---

### What Was Accomplished

1. **Gimbal limits now configurable** in `config/robot_profiles/treatbot1.yaml` under `camera:` section
2. **pan_tilt.py updated** to load limits from robot config at startup
3. **New API endpoints added:**
   - `GET /camera/gimbal` - View current limits
   - `POST /camera/gimbal/calibrate` - Adjust limits at runtime with optional save
4. **Xbox controller docs updated** (`XBOX_CONTROLLER_USAGE.md`) with current button mapping (RT=Good, RB=No, etc.)

### Files Modified
- `config/robot_profiles/treatbot1.yaml` — Added gimbal config params
- `services/motion/pan_tilt.py` — Load limits from config
- `api/server.py` — New gimbal calibration endpoints
- `XBOX_CONTROLLER_USAGE.md` — Updated button mapping

### Config Parameters Added
```yaml
camera:
  pan_min: 10
  pan_max: 200
  tilt_min: 20
  tilt_max: 160
  pan_center: 110
  tilt_center: 90
  coach_pan_min: 55
  coach_pan_max: 145
  coach_tilt_min: 25
  coach_tilt_max: 85
```

---

## Session: 2026-04-25 — WiFiManager Race Condition Fix

**Goal:** Investigate and fix system freezes within 2 minutes of launch on internet
**Status:** COMPLETE

---

### What Was Accomplished

#### 1. Root Cause Identified
- System was freezing (hard lock, required power cycle) within ~2 minutes of launch
- Initial wrong diagnosis: LED/SPI bus hammering from rapid pattern changes
- **Actual cause:** Race condition in WiFiManager — multiple threads (WiFi monitor + relay client) creating separate WiFiManager instances and calling nmcli/pgrep concurrently
- Kernel wireless driver deadlocked when hit from multiple threads simultaneously

#### 2. Fix Implemented (c64a160)
- Added `get_wifi_manager()` singleton factory function with global lock
- Added `_op_lock` RLock to WiFiManager to serialize all wireless operations
- Wrapped critical methods: `is_ap_mode()`, `get_connection_status()`, `try_connect_known()`, `start_hotspot()`, `start_demo_hotspot()`, `stop_hotspot()`
- Updated main_treatbot.py and relay_client.py to use singleton
- Fixed WiFi monitor bug: moved thread start to after `self.running=True` (was exiting immediately)

#### 3. CLAUDE.md Updated
- Added "Embodied Context" notice — Claude Code runs directly ON the robot, not external

### Files Modified
- `services/network/wifi_manager.py` — Singleton + thread locks (+277/-236 lines)
- `main_treatbot.py` — Use singleton, fix WiFi monitor start order
- `services/cloud/relay_client.py` — Use singleton (2 locations)
- `.claude/CLAUDE.md` — Added embodied context notice

### Commit
`c64a160` — fix: WiFiManager race condition causing system freeze

### Next Session
- Restart treatbot service to apply fix: `sudo systemctl restart treatbot.service`
- Monitor for 5+ minutes to verify no freezes
- Check logs: `journalctl -u treatbot.service -f`

### Important Notes
- The crash happened specifically "on internet" because relay was active, increasing likelihood of race
- WiFi monitor thread was also broken (exited immediately) — now fixed
- Crashes correlated with LED pattern changes were coincidental, not causal

---

## Session: 2026-04-17 — Documentation Update & Status Review

**Goal:** Update project documentation with current status, verify all systems
**Status:** COMPLETE

---

### What Was Accomplished

#### 1. Product Roadmap Updated (`product_roadmap.md`)
- Changed "Unknown Status" → "Verified Working (April 2026)"
- All App/Relay integration items confirmed working
- All Coach Mode items confirmed working
- All Silent Guardian items confirmed working
- Hardware status confirmed (servo calibration per-unit note added)
- Direct LAN Connection changed from "Deferred" to "IMPLEMENTED"
- Weekly Summary marked as tested (not 100% accurate)
- Mission Scheduler marked as NOT TESTED
- Xbox controller: Added joystick push buttons (center camera, anti-jam)

#### 2. Development TODOs Updated (`development_todos.md`)
- Updated to Build 83 (from Build 40)
- All live testing items marked complete
- Fixed incorrect checkboxes (Mission Scheduler was marked done but not tested)
- Added treat inventory tracking note
- Updated Dropped Features section

#### 3. Hardware Specs Updated (`hardware_specs.md`)
- Treat carousel: MG996R servo → NEMA 17 stepper + TMC2209
- Raspberry Pi 5: Added 4GB variant support note
- GPIO pin mapping: Cleaned up formatting, added TMC2209 pins
- Camera: Added RPi Camera Module 3 Wide as alternative option
- PCA9685 channels: Updated (0=pan, 1=tilt, stepper is separate)
- Treat system: 44 treats (11 × 4 carousels), stepper specs

#### 4. Created LLM Update Document (`LLM_UPDATE_APRIL.md`)
Comprehensive status document covering:
- Reliability metrics (what we have, what we need)
- Performance stats (detection, bark, treats)
- Crash recovery documentation
- WiFi reconnection reliability
- MVP definition (all core features complete)
- Demo breakdown (what works, caution areas, disabled features)
- Current failure points (all resolved)
- Target user (high-income pet owner)
- Data storage confirmation (SQLite: barks, treats, sessions, emotions)

#### 5. Memory Updated
- Direct AP Connection memory updated to "FULLY IMPLEMENTED"

### Key Confirmations from User

**Working Systems:**
- Relay forwarding mission_progress events (not all missions tested)
- Video overlay with AI confidence labels
- Servo tracking checkbox in app
- MP3 upload/download flow
- Bark filter (mostly rejects claps/voice)
- Pose thresholds accurate
- Full coaching session end-to-end
- Silent Guardian bark → intervention → reward
- Escalation and cooldown
- Treat dispenser reliable
- Audio playback consistent
- Servo calibration (needs per-unit tweaking)

**Resolved Issues:**
- WiFi instability → AP fallback + captive portal
- AI misfires → None reported
- Treat jams → NEMA 17 + TMC2209 anti-jam
- Docking → Feature disabled (too complex)

**Not Yet Tested:**
- Mission Scheduler auto-scheduling
- Mission Scheduler time windows
- Weekly Summary accuracy needs work

### Files Modified
- `.claude/product_roadmap.md` - Status updates
- `.claude/development_todos.md` - Build 83 updates
- `.claude/hardware_specs.md` - Hardware corrections
- `.claude/LLM_UPDATE_APRIL.md` - NEW comprehensive status doc
- `~/.claude/projects/.../memory/project_direct_ap.md` - Updated to implemented

### No Code Changes
This was a documentation-only session.

### Next Session
- Consider formal reliability testing for metrics
- Test Mission Scheduler
- Improve Weekly Summary accuracy
- Per-unit calibration documentation for manufacturing

---

## Previous Session: 2026-04-08 — Hailo Driver Patch, Audio Fallback

**Goal:** Diagnose SSH freeze, fix Hailo driver crash, treatbot2 audio/dispenser config
**Status:** FIXED — Hailo driver patched, treatbot running stable

### What Was Accomplished
- Hailo-8 PCIe driver patched (find_vma lock fix)
- Audio ALSA fallback implemented
- TMC2209 UART setup guide created
- Treatbot2 stepper config updated

### Critical Lessons
1. NEVER init hardware at Python module import time
2. kernel.warn_limit can make bugs worse

### System Changes Applied
- `/usr/src/hailo_pci-4.21.0/linux/vdma/memory.c` — patched via DKMS
- Patch files: `patches/hailo_pci_find_vma_fix.patch` + `patches/apply_hailo_fix.sh`
