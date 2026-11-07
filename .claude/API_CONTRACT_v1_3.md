# WIM-Z API Contract v1.3

> **Single source of truth** for Robot, Mobile App, and Cloud Relay Server.
> When changing any API, update this file FIRST, then update each project to match.

Last Updated: 2026-02-22
Version: 1.3 - Always-on audio streaming, mode/UX restructure, resolution constraints

---

## Architecture Overview

```
┌─────────────────┐                                    ┌─────────────────┐
│   Mobile App    │◄──── WebRTC Video+Audio (P2P/TURN)►│   WIM-Z Robot   │
│   (Flutter)     │                                    │  (Raspberry Pi) │
└────────┬────────┘                                    └────────┬────────┘
         │                                                      │
         │  Commands/Events                    Commands/Events  │
         │  (WebSocket)                           (WebSocket)   │
         │                                                      │
         └──────────────►┌─────────────────┐◄───────────────────┘
                         │  Cloud Relay    │
                         │  (FastAPI)      │
                         └────────┬────────┘
                                  │
                         ┌────────▼────────┐
                         │ Cloudflare TURN │
                         │ (Credential Gen)│
                         └─────────────────┘
```

**Data Flow:**
- **Commands/Events**: App ↔ Relay ↔ Robot (WebSocket, low bandwidth)
- **Video + Audio**: App ↔ Robot direct via WebRTC (high bandwidth, P2P when possible)
- **Push-to-Talk Audio**: App → Relay → Robot (WebSocket, base64 encoded, app→robot only)
- **TURN Relay**: Cloudflare handles NAT traversal when P2P fails

---

## Mode System

### Mode Definitions

| Mode | Resolution | AI Active | Audio Stream | Orientation | Entry Point |
|------|-----------|-----------|-------------|-------------|-------------|
| `idle` | Standard | No | Mic ON | Portrait | Default / exit from any mode |
| `manual` | High (up to 4K) | No | Mic ON | Landscape | Drive button |
| `silent_guardian` | 640x640 | Yes | Mic OFF | Portrait | Portrait dropdown |
| `coach` | 640x640 | Yes | Mic OFF | Portrait or Landscape | Portrait dropdown or Landscape selector |
| `mission` | 640x640 | Yes | Mic OFF | Landscape | Mission flow → Start Mission |

### Mode Resolution Constraint (NEW in v1.3)

**Hard rule**: Manual mode = high resolution, no AI processing. All AI modes (silent_guardian, coach, mission) = 640x640 with Hailo AI active. These are mutually exclusive — you cannot run AI detection at high resolution.

### Mode Transition Rules (NEW in v1.3)

**Portrait dropdown options**: idle, silent_guardian, coach (3 options only)

**Landscape selector options**: manual, coach, mission (mission only if a mission is loaded)

**Transition behaviors:**
- Entering Drive (landscape) from portrait: App stores current portrait mode as `previous_portrait_mode`. Robot switches to `manual`.
- Exiting Drive (landscape) back to portrait: App restores `previous_portrait_mode` (idle, silent_guardian, or coach). Robot receives `set_mode` with the restored mode.
- Selecting coach/silent_guardian from landscape: App exits landscape, returns to portrait, activates selected mode.
- Exiting mission (complete, failed, or cancelled): App returns to portrait. Restores `previous_portrait_mode`.
- Coach in landscape: Stays in landscape. Robot runs coach AI at 640x640 while user has manual drive controls.

**set_mode command now includes source context:**
```json
{
  "type": "command",
  "command": "set_mode",
  "data": {
    "mode": "manual",
    "source": "drive_enter",
    "timestamp": "ISO8601"
  }
}
```

Source values: `dropdown`, `drive_enter`, `drive_exit_restore`, `mission_start`, `mission_end`, `landscape_selector`

---

## WebRTC Audio Streaming (NEW in v1.3)

### Always-On Robot Microphone

The robot includes a **mono audio track** in the WebRTC PeerConnection alongside the video track. This audio track is always negotiated in the SDP offer/answer — it is never added or removed dynamically.

**Robot-side behavior:**
- In `idle` and `manual` modes: USB mic feeds audio frames to the WebRTC audio track.
- In `silent_guardian`, `coach`, and `mission` modes: Audio track remains in SDP but robot sends silence (mic muted server-side on Pi). No renegotiation needed.
- On mode change: Robot mutes/unmutes the audio track source. No WebRTC renegotiation.

**App-side behavior:**
- Audio playback is controlled entirely client-side via a mute/unmute toggle.
- Mute state persists locally across sessions (SharedPreferences / UserDefaults).
- Default on first use: **muted**.
- The mute toggle does NOT send any command to the robot. It only controls local audio playback.
- When the robot's mic is server-side muted (AI modes), the app receives silence regardless of mute toggle state.

**Audio specifications (WebRTC track):**
- Codec: Opus (WebRTC default)
- Sample rate: 48kHz mono (Opus default, downsampled from 16kHz mic input)
- Bitrate: 32-64 kbps (adaptive)
- Latency: ~100-300ms depending on network

**Adaptive quality:**
- Robot monitors WebRTC connection stats (packet loss, RTT).
- If bandwidth is constrained, audio bitrate is reduced first before touching video.
- If severe congestion, audio track can be temporarily muted server-side to preserve video.

### Push-to-Talk: App → Robot (UNCHANGED)

PTT remains on WebSocket via relay. This is intentional — PTT is intermittent, user-initiated, and does not require a persistent reverse audio track.

**App sends recorded audio:**
```json
{
  "type": "audio_message",
  "data": "<base64_encoded_audio>",
  "format": "aac",
  "duration_ms": 3500
}
```

**Robot plays through speaker and acknowledges:**
```json
{
  "type": "audio_played",
  "success": true
}
```

**Echo handling:** When the robot receives and plays a PTT message, it auto-mutes its outbound WebRTC audio track for the duration of playback to prevent echo/feedback. It resumes feeding the audio track after playback completes.

**PTT Audio Specifications (WebSocket):**
- Format: AAC (preferred) or WAV
- Sample rate: 16kHz mono
- Max duration: 10 seconds
- Max size: ~100KB for 10s AAC

### DEPRECATED: audio_request / audio_message (Robot → App)

The `audio_request` command (app requesting robot mic capture) and the robot→app `audio_message` response are **deprecated** as of v1.3. The always-on WebRTC audio track replaces this functionality. These message types will be removed in v1.4.

---

## WebSocket Commands (App → Robot)

### Core Commands

| Command | Data | Description |
|---------|------|-------------|
| `motor` | `{"left": float, "right": float}` | Set motor speeds (-1.0 to 1.0) |
| `servo` | `{"pan": float, "tilt": float}` | Set camera pan/tilt angles |
| `servo_center` | `{}` | Center camera to default position |
| `dispense_treat` | `{}` | Dispense one treat |
| `led` | `{"pattern": string}` | Set LED pattern |
| `audio` | `{"file": string}` | Play audio file |
| `audio_toggle` | `{}` | Play/pause current audio |
| `audio_volume` | `{"volume": int}` | Set volume (0-100) |
| `audio_next` | `{}` | Skip to next track |
| `audio_prev` | `{}` | Go to previous track |
| `set_mode` | `{"mode": string, "source": string, "timestamp": string}` | Change robot mode (v1.3: added source and timestamp) |

### Photo Capture

| Command | Data | Description |
|---------|------|-------------|
| `take_photo` | `{"with_hud": bool}` | Capture photo, optionally with AI overlay |

**Request:**
```json
{
  "type": "command",
  "command": "take_photo",
  "data": {"with_hud": true}
}
```

**Robot Response (sent back via WebSocket):**
```json
{
  "type": "photo",
  "data": "<base64_encoded_jpeg>",
  "timestamp": "2026-01-26T02:15:00Z",
  "filename": "wimz_20260126_021500.jpg",
  "with_hud": true
}
```

**HUD Overlay includes:**
- Dog bounding box (if detected)
- Dog name label
- Current mode indicator
- Timestamp

### Call Dog

| Command | Data | Description |
|---------|------|-------------|
| `call_dog` | `{"dog_id": string?}` | Play dog's name recording or default call |

**Request:**
```json
{
  "type": "command",
  "command": "call_dog",
  "data": {"dog_id": "dog_001"}
}
```

**Robot behavior:**
1. If custom voice recording exists for dog → play `dog_001.mp3`
2. If no custom recording → play generic `come_here.mp3`
3. LED pattern: attention (blue pulse)

---

## WebSocket Events (Robot → App)

### Status Events

| Event | Description |
|-------|-------------|
| `battery` | Battery level update |
| `status_update` | Mode change or system status |
| `alert` | Safety or system alert |
| `detection` | Dog detected/behavior change |
| `treat` | Treat dispensed confirmation |
| `mission` | Mission progress update |
| `error` | Error occurred |

### Photo Event

```json
{
  "type": "photo",
  "data": "<base64_jpeg>",
  "timestamp": "ISO8601",
  "filename": "wimz_YYYYMMDD_HHMMSS.jpg",
  "with_hud": true
}
```

### Audio Played Event

```json
{
  "type": "audio_played",
  "success": true
}
```

---

## LED Patterns

| Pattern | Description | Colors |
|---------|-------------|--------|
| `rainbow` | Cycling rainbow | Multi |
| `pulse` | Breathing effect | Current color |
| `solid` | Solid color | Current color |
| `chase` | Chasing lights | Current color |
| `fire` | Fire flicker effect | Orange/red |
| `warning` | Alert flash | Orange/red fast flash |
| `attention` | Dog attention getter | Blue pulse |
| `celebration` | Treat celebration | Multi sparkle |
| `off` | LEDs off | None |

---

## REST API Endpoints (Robot Local)

### Health & Status

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server alive check |
| GET | `/telemetry` | Full system status |

### Motor Control

| Method | Endpoint | Request | Description |
|--------|----------|---------|-------------|
| POST | `/motor/speed` | `{"left": float, "right": float}` | Set motor speeds |
| POST | `/motor/stop` | - | Stop all motors |
| POST | `/motor/emergency` | - | Emergency stop |

### Camera & Servos

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/camera/stream` | MJPEG video stream (local) |
| GET | `/camera/snapshot` | Single JPEG frame |
| POST | `/servo/pan` | Set pan angle |
| POST | `/servo/tilt` | Set tilt angle |
| POST | `/servo/center` | Center camera |

### Treat Dispenser

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/treat/dispense` | Dispense one treat |
| POST | `/treat/carousel/rotate` | Rotate carousel |

### LED Control

| Method | Endpoint | Request | Description |
|--------|----------|---------|-------------|
| POST | `/led/pattern` | `{"pattern": string}` | Set LED pattern |
| POST | `/led/color` | `{"r": int, "g": int, "b": int}` | Set RGB color |
| POST | `/led/off` | - | Turn off LEDs |

### Audio

| Method | Endpoint | Request | Description |
|--------|----------|---------|-------------|
| POST | `/audio/play` | `{"file": string}` | Play audio file |
| POST | `/audio/stop` | - | Stop playback |
| POST | `/audio/volume` | `{"level": int}` | Set volume (0-100) |
| POST | `/audio/toggle` | - | Play/pause toggle |
| POST | `/audio/next` | - | Next track |
| POST | `/audio/prev` | - | Previous track |
| GET | `/audio/files` | - | List audio files |
| GET | `/audio/current` | - | Current track info |

### Mode Control

| Method | Endpoint | Request | Description |
|--------|----------|---------|-------------|
| GET | `/mode/get` | - | Get current mode |
| POST | `/mode/set` | `{"mode": string}` | Set mode |

---

## WebRTC Signaling

| Message Type | Direction | Description |
|--------------|-----------|-------------|
| `webrtc_request` | App → Relay | Request video+audio stream |
| `webrtc_credentials` | Relay → App | TURN credentials |
| `webrtc_request` | Relay → Robot | Initiate with credentials |
| `webrtc_offer` | Robot → App | SDP offer (includes video + audio tracks) |
| `webrtc_answer` | App → Robot | SDP answer |
| `webrtc_ice` | Bidirectional | ICE candidates |
| `webrtc_close` | Either | Close stream |

**v1.3 Change:** The SDP offer from the robot now always includes both a video track and an audio track. The app must accept both tracks in the SDP answer.

---

## Message Envelope Format

All WebSocket messages use this envelope:

**App → Relay → Robot (Command):**
```json
{
  "type": "command",
  "command": "<command_name>",
  "data": { ... },
  "device_id": "wimz_robot_01"
}
```

**Robot → Relay → App (Event):**
```json
{
  "type": "<event_type>",
  "device_id": "wimz_robot_01",
  "data": { ... },
  "timestamp": "ISO8601"
}
```

---

## Relay CONTRACT v2 FINAL (2026-11-06) — Agent Summit demo

The relay is the consumer and owns the wire. Robot slice shipped 2026-11-06
(`services/cloud/{relay_outbox,event_emitter,behavior_events,remote_settings,tool_dispatcher}.py`,
`orchestrators/trick_runner.py`, `services/reward/treat_policy.py`,
`services/media/sound_registry.py`). Offline-safe: every piece runs with no relay.

### Events — HTTP `POST /events` (device HMAC headers `X-Device-ID`, `X-Timestamp`, `X-Signature`)
Durable outbox in `wimz.db` (`relay_outbox`, spec v0.8). One pump sends rows in `seq` order in
batches of ≤200. A row leaves the queue only on the relay's per-row verdict: `accepted`/`duplicate`
→ sent, `rejected` → dropped; a 2xx with no per-row detail accepts the batch. ANY whole-batch
failure (network, 4xx incl. 422 validation, 5xx) keeps every row pending and retries with backoff
(1 s → 60 s) — a validation error never destroys queued events. Rows the relay cannot take yet
(dog-scoped type, no `dog_id`) are held pending without blocking the rows behind them. Nothing bypasses the pump,
so ordering survives restarts. Wire row = relay `EventIn` (verified against
`api.wimzai.com/openapi.json` 2026-11-06):
```json
{"device_id": "wimz_robot_05",
 "events": [{"event_id": "<uuid7>", "type": "bark_episode", "dog_id": "<app id, REQUIRED>",
             "ts": "2026-11-06T14:02:11.120Z", "duration_s": 12.5, "device_id": "...",
             "data": {..., "id_method": "aruco|sole_dog|last_dog"}}]}
```
Response `EventBatchResult`: `{"accepted": [event_id], "duplicate": [event_id], "rejected": [{"event_id", "reason"}]}`; 422 = whole batch refused (robot keeps rows).
RESOLVED (relay change 2026-11-07): there is no robot↔dog pairing any more — a robot serves all of its owner's dogs (`paired_dog_ids` in `profiles` now lists every dog; never filter on it). `dog_id` is required on every type EXCEPT `status`, which ships with no `dog_id` key (`data.battery`, `data.mode`, optional `docked`/`light_level`/`treats_remaining`; sent every 60 s and on any mode/dock/treat change). The relay fills a missing `dog_id` only when the owner has exactly one dog, otherwise rejects the row — so dog-scoped rows without a dog are still held on the robot. `/agent/utterance` without `dog_id` falls back to the owner's oldest dog.

| type | data |
|---|---|
| `bark_episode` | `{start, end, bark_count, peak_db, severity: low(<3 barks)\|med(3-9)\|high(>=10 or >60 s), duration_s, image_url?}` — the row `id` is the `episode_id` interventions reference; `image_url` = HUD-off still uploaded via `POST /api/media/upload` when severity is high |
| `intervention` | `{time, sg_level, kind: sound\|voice\|treat\|music\|approach, episode_id, result: quiet\|no_quiet}` (L1 sound, L2 voice, L3 approach, L4 music, treat on quiet) |
| `settle` | `{time, seconds_since_last_intervention, calm_seconds}` (300 s calm after an outcome, once) |
| `pacing` | `{start, end, reversals}` — best-effort bbox-centre oscillation heuristic |
| `treat` | `{time, source: auto\|app\|connector\|agent\|schedule, command_id, reason, behavior, image_url?}` — `command_id` echoes the `dispense_treat` command; `image_url` = HUD-off still captured ~2 s after the dispense and uploaded via `POST /api/media/upload` (the row is emitted after the upload, so it arrives ~2-4 s after the dispense) |
| `music` | `{start, end, track_class: calming\|owner\|other, track}` |
| `mission_run` | `{mission_id, run_id, start, end, result: success\|fail\|aborted, rewards_given}` |
| `mode_change` | `{mode: idle\|silent_guardian\|coach\|mission\|night_sentry\|manual\|photography\|emergency, start, end, duration_s, next_mode, reason}` — one row per completed stay in a mode, sent when that mode ends (relay request 2026-11-07); the open segment is closed on clean shutdown/restart with `next_mode: "shutdown"`; not attributed to a dog |
| `status` | `{battery, mode, treats_remaining, docked, light_level}` — `light_level` = mean camera frame brightness 0-255; every 60 s and on change |
| `departure` / `arrival` | `{time, method: "app"}` — ONLY on app/relay instruction (`mark_departure`/`mark_arrival` on `/ws/local`, `POST /events/departure|arrival`); no sensing |

### Settings — `GET /api/dogs/{dog_id}/settings` (HMAC) on connect (after `profiles`), then WS push
```json
{"type": "settings_updated", "dog_id": "...", "key": "treat_schedule", "value": {...}, "version": 7}
```
Keys: `treat_schedule {daily_cap, default_multiplier, max_per_hour, windows:[{start,end,multiplier}]}`
→ TreatPolicy (dispenser gate: window multiplier scales the 120 s protocol cooldown, m=0 blocks
auto treats, owner taps always pass except at the daily cap, max_per_hour rolling);
`sg_sensitivity {level: low|medium|high, min_episode_s, escalate_after_s}` → SG profile
relaxed|normal|sensitive written into `bark_detection` (threshold/loudness/confidence),
`escalate_after_s` → bark window seconds, `min_episode_s` → bark_episode minimum;
`music_plan {enabled, volume, on_departure{track, delay_s, duration_min}, schedule[]}`;
`quiet_hours` and `notify_rules` stored verbatim. Last good document cached in treatbot.db.
Sounds: `{"type":"sound_updated","sound_id":"default/sit","url":"<asset>|null"}` installs/removes an override.

### Commands — existing command frame + `command_id` → `command_result`
```json
{"type":"command","command":"run_trick","command_id":"c-123","cue":"sit","force":false,"timestamp":...}
{"type":"command_result","command_id":"c-123","command":"run_trick","ok":true,"result":{...},"error":null,"duration_ms":18200}
```
`error` = `{"code","message"}` (unknown_command, busy, timeout, exception, invalid_cue, …). One
command at a time; `standby` preempts. Frames WITHOUT `command_id` keep the legacy `command_ack` path.
Commands: `dispense_treat{source?, dog_id?}`, `set_mode{mode}`, `play_sound{url|sound_id, kind?}`
(agent speech arrives as `play_sound{url, kind:"tts"}`), `start_mission{mission_id}`, `go_to` (not supported),
`standby{announce?}`, `run_trick{cue, force?, retry?, timeout_s?}` → `result: success|no_dog|wrong_pose|timeout|busy|cancelled`,
`take_photo{with_hud}` → `{jpeg_b64, width, height, ...}`, `agent_utterance{duration_s<=30, dog_id?}`
(records WAV via the PTT capture path, multipart `POST /agent/utterance` with HMAC + `dog_id`, plays the returned `audio_url`).
Every command is logged (`[TOOL]` journald + wimz `tool_call` row). Same commands work on `/ws/local`.

Mission scheduler (2026-11-07, persisted timers, armed at boot, 5 s tick — fires within ~5 s of its instant
and within ~30 s across a restart; a slot that comes due while the service is down fires on the first tick
after boot if still inside its grace):
`schedule_mission{mission|mission_name, dog_id, fire_at ISO|epoch (one-shot) | start_time "HH:MM" [+type
once|daily|weekly, days[]], mission_id? (relay id, echoed back), end_time?, grace_s?}` → the schedule_sync row;
`schedule_sync{dog_id?, enabled_only?}` → `{missions:[{mission_id, schedule_id, mission_name, dog_id, type,
enabled, start_time, end_time, days_of_week, fire_at, grace_s, next_fire_at, last_run_at, last_run_end,
last_run_id, last_result, missed_at}], count, scheduler:{enabled, now}, active}`;
`cancel_mission{mission_id}` (schedule_id | relay mission_id | mission_name; `"active"` = whatever is running)
→ `{cancelled: schedule_id|null, stopped_run, run_id}` or `error.code: not_found`.
`last_result`: `starting|running|success|fail|aborted|missed|start_failed:<reason>`; the `mission_run` event for
that run has `run_id == last_run_id` and the same success|fail|aborted vocabulary. Grace = `grace_s`, else the
start_time..end_time window (min 60 s), else 1 h for `fire_at` one-shots; beyond it the slot is `missed`.

Settings key `missions` (relay request 2026-11-07): `{missions:[{id, type, name, enabled, trigger{kind:
schedule|departure|manual, times:["HH:MM"], days:["mon".."sun"]}, duration_s, reward_treats}]}` — arrives by
`settings_updated{key:"missions", dog_id}` or in `GET /api/dogs/{dog_id}/settings` (wrapped `{key,value,version}`;
the robot unwraps every key). Every `trigger.kind == "schedule"` mission becomes one persisted schedule (all its
`times`, `days` mon..sun → weekly, none → daily) **replacing that dog's previous relay set**, applied synchronously
before the apply call returns (so before the next command). `last_run_*` carries over when the same `id` is still
present, so a re-push never re-fires a slot. `times` are the dog's local HH:MM in the `timezone` setting (IANA);
`next_fire_at` is reported in the robot's UTC offset, or `null` when the mission will not fire (disabled, no slot,
or `unsupported` type). Each `schedule_sync` entry carries `{id, enabled, next_fire_at}` plus the fields above
(`id` == the setting's mission id). Type → program: sit→sit, sit_stay|stay→sit_sustained, down|down_stay|place→
down_sustained, recall→come_and_sit, quiet→quiet_progressive; `silent_guardian` and `coach` are **mode
missions**: at fire time `set_mode(type)` for `duration_s` (cap 43200 = 12 h), then idle; `reward_treats` 0;
reported like any run (`mission.started`/`.completed` → `mission_run{mission_id:"coach"|"silent_guardian",
run_id, result, rewards_given:0, kind:"mode", schedule_mission_id:<setting id>, scheduled:true}`, and
`schedule_sync.active = {id, run_id, kind:"mode", ends_at, remaining_s}`); `cancel_mission{mission_id}` ends it
early (`mission_run.result: aborted`). Other scheduled runs also carry `schedule_mission_id` + `schedule_id`.
Push ordering: `settings_updated` is applied inside the awaited handler, so the next frame on the socket (the
relay's `schedule_sync` ~40 ms later) is read only after the set is replaced. Trigger times accepted:
`trigger.times:["HH:MM"]`, `trigger.time:"HH:MM"`, `at`, `"HH:MM:SS"`, `[{time}]`, top-level `times`; days as
`mon..sun` or full names, list or "mon,wed". An entry whose time cannot be parsed, or whose type has no program,
is still listed in `schedule_sync` with `next_fire_at: null` and `reason` (also `unsupported_reason`); disabled
entries carry `reason: "disabled"`. The robot logs the first raw entry of every push (`[SCHED] missions setting
dog=… first={…}`). `/ws/local` accepts the identical `settings_updated` frame (replies `settings_applied`) —
used by `tests/integration/test_missions_push_ws_live.py`.

### Robot local additions
REST `GET/POST /settings`, `GET /tools`, `POST /tools/{name}`, `GET /events/outbox`,
`POST /events/departure|arrival`, `GET /sounds`, `POST /sounds/{group}/{stem}/replace|reset|play`,
`POST /sounds/reset_all`. WS: `settings`, `settings_pull` (+ `settings_state` event, also on connect),
`mark_departure`/`mark_arrival`, `sounds_list`, `sound_replace`, `sound_reset`, `sounds_reset_all`.
Sound ids are `<group>/<stem>` over `VOICEMP3/wimz` (release, immutable) and `talks/default`;
overrides live in `talks/_custom/<group>/<stem>.mp3` (shared, survives OTA). Precedence:
per-dog recording > custom override > factory.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-01-18 | Initial contract |
| 1.1 | 2025-01-18 | Added WebRTC signaling, Cloudflare TURN |
| 1.2 | 2026-01-26 | Added take_photo, audio_message, audio_request, call_dog commands; documented LED patterns and modes |
| 1.3 | 2026-02-22 | Always-on WebRTC audio track (robot→app), deprecated audio_request/audio_message for listening, mode/UX restructure with resolution constraints, mode transition rules with source context, adaptive audio quality |
| 2.0 | 2026-11-06 | Relay CONTRACT v2 FINAL: HTTP POST /events (durable outbox), per-dog settings GET + settings_updated push, command_id → command_result, sound_updated, agent_utterance; robot slice for the Dec 9 AI Summit demo |
