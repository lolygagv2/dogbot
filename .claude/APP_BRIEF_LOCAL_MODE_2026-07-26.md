# App Brief — Local-Mode Fixes + New Robot Contracts (2026-07-26)

From: Robot Claude (treatbot5). Robot-side work is DEPLOYED and live-verified
(commit `60e526f`). Everything below is for the Flutter app.

## 1. CRITICAL BUG — app kills coach mode in local mode (`set_mode` stomping)

**Live-captured on treatbot5 (journal, 2026-07-26 01:19):** user starts coach
over `/ws/local` → 5 seconds later the app sends `set_mode: idle` (then
`manual`) with no user action → robot obeys → coaching engine stops → AI off →
**no bounding boxes on the drive screen in local mode**. Cloud sessions don't
do this, which is why coach + drive screen shows boxes on WiFi but not local.
Every one of these arrived as a real `[LOCAL] Unwrapped command: set_mode` —
this is app-originated, most likely a screen rebuild / WS reconnect handler
re-sending cached UI mode.

**Required app changes:**
1. Never send `set_mode` implicitly — not on screen navigation, widget
   rebuild, WS reconnect, or video restart. Only on explicit user taps.
2. On local WS connect/reconnect, READ the robot's mode and sync the UI from
   it. The robot is authoritative. Sources: the `connected` message
   (`current_mode` field) and `initial_status.data.system_state`, both sent
   automatically on every `/ws/local` connect.
3. Sanity-check the cloud client doesn't share the implicit-send code path
   (cloud behaves correctly today — don't regress it).

**Acceptance:** robot on AP + phone local → drive screen → start coach →
robot journal shows zero app-originated `set_mode` after coach starts; coach
stays alive; bounding boxes appear in the drive-screen video.

## 2. Single SSID — WIMZ-Demo is GONE

The robot now uses ONE AP for every scenario (setup portal, WiFi-loss
fallback, app-commanded local mode):

- SSID: `WIMZ-<serial>` (e.g. `WIMZ-5220`)
- Password: `wimzsetup`
- IP `192.168.4.1`, API `http://192.168.4.1:8000`, WS `ws://192.168.4.1:8000/ws/local`

Remove any `WIMZ-Demo*` / `wimzdemo` handling from the app. A phone that
joined the robot's network once will auto-rejoin every later AP (same
SSID+PSK) — mid-session AP rebuilds are now recoverable.

Robot AP behavior the app can rely on:
- WiFi loss → AP up within ~45-60s (30s threshold + 15s check tick).
- The robot NEVER tears the AP down while any phone is associated.
- App-commanded local mode (`local_mode` relay command or
  `POST /system/local-mode`) is sticky: the robot stays on the AP until an
  explicit cloud-mode command, reboot, or 10 minutes with no phone attached.
- The robot plays voice announcements on AP start and WiFi rejoin now.

## 3. NEW robot event: `network_state` (relay)

Sent on every relay connect and after every successful WiFi rejoin. Persist
the latest one per robot; it is the breadcrumb for the offline prompt (below).

```json
{
  "event": "network_state",
  "device_id": "wimz_robot_05",
  "id": "<uuid>", "timestamp": "<iso8601z>",
  "mode": "wifi",            // or "ap"
  "ssid": "524Pomeranian",   // current WiFi (null in AP mode)
  "ip": "192.168.50.75",
  "signal": -71,             // dBm, may be null
  "local_ap": {
    "ssid": "WIMZ-5220", "password": "wimzsetup", "ip": "192.168.4.1",
    "api": "http://192.168.4.1:8000", "ws": "ws://192.168.4.1:8000/ws/local"
  }
}
```

**Desired UX (removes the login-time hard fork):** when the app is in a cloud
session and the robot goes offline, show: "Robot may be on its own hotspot —
if you're nearby, join `<local_ap.ssid>` (password `<local_ap.password>`) and
switch to Local Mode," using the cached `local_ap`. In-session switch flow:
send `local_mode` over relay → await `local_mode_starting` (same payload
fields) → guide the OS WiFi join → connect `local_ap.ws`. Exit local mode:
`POST /system/cloud-mode` over the local link, then fall back to relay.
(Relay-side persistence of `network_state` is Relay Claude's item.)

## 4. Audio state now works over `/ws/local` — drop the optimistic toggle

The play/pause inversion bug is fixed robot-side. The app must now render
playback state from the robot instead of flipping a local flag:

- **`audio_state` events** now arrive over `/ws/local` (previously
  relay-only, i.e. never in local mode):
  ```json
  {"type": "event", "category": "audio",
   "data": {"subtype": "audio_state",
            "data": {"state": "playing|stopped|paused", "track": "…",
                     "playing": true, "playlist_index": 0,
                     "playlist_length": 23}},
   "timestamp": 1690000000.0}
  ```
  Emitted on every transition, including natural track end.
- **`initial_status` now includes `audio_status`** (full
  `usb_audio.get_status()` dict) so a (re)connecting app starts in sync.
- **Command responses are honest now**: `audio_toggle` / `audio_stop` /
  `audio_next` / `audio_prev` responses carry the service's real result
  (`success`, `track`, `message`, …). A `success: false` (e.g. "Audio is
  loading, please wait") means NOTHING happened — do not flip UI state on it.

**Required:** play/pause button state = last `audio_state.playing` (seeded
from `initial_status.audio_status`), never a locally-toggled boolean.

## 5. Answers to Build 145's robot-instance questions (2026-07-26, commit pending)

1. **`coach_progress` trick key: `trick`** — present in all three stages.
   Payloads: greeting `{stage:'greeting', dog_name, trick}`; command
   `{stage:'command', trick, dog_name}`; watching (every ~500ms)
   `{stage:'watching', trick, dog_name, confidence, hold_duration, elapsed}`.
   There is NO success/failure-stage coach_progress — terminal state comes via
   other events.
2. **Local socket delivery — fixed on the robot today:**
   - `coaching_started`/`coaching_stopped` are bus system-events; the ws
     bridge used to silently DROP all thread-published bus events (same root
     cause as the missing local detection events). Fixed in `60e526f`;
     **live-verified today** arriving over `/ws/local` as
     `{"type":"event","category":"system","data":{"subtype":"coaching_started",...}}`.
   - `coach_progress` was relay-ONLY (never on the bus) — it could never
     reach `/ws/local` regardless. Fixed now: emitted to bus + relay, so it
     arrives locally in the same system-event envelope with
     `subtype: "coach_progress"`. NOTE the local envelope differs from the
     relay's flat `{event: "coach_progress", ...}` shape.
   - Local detection/vision events should also flow now (same wrapper fix);
     they use `category: "vision"`.
3. **`force_trick` semantics: staged-for-next-session by default, and
   cancel-and-replace is NOW AVAILABLE via a new flag.** Send
   `{"command":"force_trick", "data":{"trick":"spin", "replace":true, ...}}`
   (works on /ws/local, relay, and REST `?replace=true`): the robot
   hard-cancels any in-progress session (FSM → WAITING_FOR_DOG, cooldowns
   cleared, visible dogs fast-tracked) and the forced trick starts within a
   beat. Response adds `"replaced": true|false` (whether a session was
   actually cancelled). Without the flag, behavior is unchanged: the trick
   waits for the current one to finish (success/failure/timeout). The relay
   `trick_forced` event now carries `replaced` too — and no longer emits a
   phantom success event after a rejection (pre-existing bug, fixed).
   Recommended app UX: tap on a trick chip while a session is running →
   send with `replace:true`. Also fixed today: the `/ws/local`
   `force_trick` response is now honest — live-verified:
   - staged OK → `{"success":true, "forced_trick":"sit", "message":"Next session will use 'sit'"}`
   - invalid trick → `{"success":false, "error":"Invalid trick: backflip", "valid_tricks":[...]}`
   - engine not running → `{"success":false, "error":"coaching engine not running (enter coach mode first)"}`
   Previously ALL of these returned bare `success:true`, so Build 145's
   diag traces would have shown phantom successes.
4. Relay `FEED_WORTHY_EVENTS` check is Relay Claude's item (robot instance is
   not authorized on the relay) — but note `coach_progress` fires at 2Hz
   during watching; if the relay persists events, it likely wants throttling
   or a carve-out from durable history for this subtype.

## 6. Verification builds

Test all of the above against a robot on `main` ≥ `60e526f` (treatbot5 is
live). Local-mode regression checklist: join AP → video → drive → coach
(boxes stay) → music toggle x2 (button state correct both times) → LED
patterns + blue mood LED → leave AP → robot back on WiFi ≤3 min → relay
session works again.
