# WIM-Z Volume Control — API Contract

App integration contract for system audio volume. Robot-side verified live
2026-05-21.

## Overview

All volume paths route through a single source of truth — `VolumeManager`
(`services/media/volume_manager.py`):

- **Persists across reboots.** Stored in `/etc/wimz/audio_state.json`; a
  boot service (`wimz-audio.service`) re-applies it before WIM-Z starts.
- **Default 60%** if the state file is missing.
- **Can change outside the app.** The Xbox controller's Share button cycles
  volume `0 → 20 → 40 → 60 → 80 → 100 → 0`. The app's cached value therefore
  goes stale — always reconcile the slider to the telemetry value below.
- Applied to hardware via `amixer`; integer **0–100**.

## Reads

### `GET /audio/volume` (HTTP)

```json
{ "success": true, "volume": 75, "state_file": "/etc/wimz/audio_state.json" }
```

- `volume` — integer 0–100
- `state_file` — informational only
- Error → HTTP 500

### Relay `status` telemetry — `data.volume`

Sent every 5 s while an app user is connected:

```json
{
  "event": "status",
  "device_id": "wimz_robot_01",
  "data": {
    "battery": 87.3,
    "temperature": 51,
    "mode": "idle",
    "is_charging": false,
    "treats_remaining": 11,
    "connection_type": "WAN",
    "volume": 75
  },
  "timestamp": "2026-05-21T21:25:00.000Z"
}
```

- `data.volume` — integer 0–100, or `null` if unavailable.
- **Preferred read for the app** — reconcile the slider to this every 5 s.

## Writes

All three write paths are equivalent — same `VolumeManager`, persisted.

### `POST /audio/volume` (HTTP)

Request:
```json
{ "volume": 75 }
```
Response:
```json
{ "success": true, "volume": 75, "message": "Volume set to 75" }
```

- Use the key **`volume`** (integer 0–100). Out-of-range → HTTP 400.
- Note: a second, shadowed handler at this path expected `{"level": N}` — it
  never runs. Always send `{"volume": N}`.

### Relay command `audio_volume` (cloud)

For setting volume over the cloud relay (no direct HTTP reachability):
```json
{ "command": "audio_volume", "data": { "volume": 75 } }
```
- Canonical key `volume`; `level` accepted as an alias.
- Robot replies with `{ "type": "command_ack", "command": "audio_volume",
  "success": <bool> }`.

### Local WebSocket command `audio_volume`

On the local `/ws` endpoint (AP / LAN mode):
```json
{ "command": "audio_volume", "level": 50 }
```
- Note the historical inconsistency: HTTP POST and the relay command use
  `volume`; this local WebSocket command uses `level`.

## Recommended app flow

1. On connect — read `volume` from the first `status` telemetry event
   (or one `GET /audio/volume`).
2. Every 5 s — reconcile the slider to telemetry `data.volume`.
3. On user change — `POST /audio/volume {"volume": N}` (LAN) or the
   `audio_volume` relay command (cloud). Safe to call on every slider release;
   it is idempotent.
