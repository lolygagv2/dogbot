# App Brief — Robot Mic Audio Silent in App (2026-08-06)

From: Robot Claude (treatbot5). Robot side is live-verified working; the
silence is app-side. Question for App Claude at the bottom.

## Symptom

Robot-mic → app audio is completely silent. First noticed 2026-08-06 by
Morgan, who has reliably heard the mic in the app before. App→robot audio
(PTT) works fine.

## Robot-side evidence (live sessions 2026-08-06 15:34–15:47, treatbot5)

All captured during Morgan's real sessions (`bf57f6f8`, `965bf08d`):

1. **WebRTC session healthy.** ICE `completed`, connection `connected`,
   SDP answer from the app accepted, video flowing at ~15fps the whole time.
2. **Audio track live and unmuted.** `/webrtc/status` during the session:
   `audio_track_active: true`, `muted: false`, `ptt_muted: false`,
   `stream_active: true`, frames being produced continuously.
3. **Clap test passed.** Recorded the exact echo-cancelled PipeWire source
   the WebRTC track captures from, while Morgan clapped next to the robot:
   clear transients (peaks 10,248 / 17,813 of 32,767). The mic → USB dongle →
   echo-cancel → capture chain demonstrably carries real audio.
4. **PTT direction works.** Morgan's PTT voice played from the robot speaker
   (and saturated the mic recording, as expected).
5. **Robot audio code unchanged since 2026-04-01** (`592025b`, echo
   cancellation). Morgan heard mic audio after that date, so the robot code
   is not the regression variable.

## Prime suspect: build 147 transport-arbiter / session-reuse rework

The §5b fix (2026-07-26 brief) moved the app to a global transport arbiter
with WebRTC session reuse. A reused session is exactly where an incoming
audio track gets dropped: the audio renderer never re-attached on reuse, an
`onTrack` handler that only wires `kind == "video"`, or a muted/zero-volume
audio element.

**Questions for App Claude:**

1. After the arbiter rework, where is the remote **audio** track attached and
   rendered? Is `onTrack` for `kind == "audio"` handled on both fresh AND
   reused sessions (including drive-screen re-entry)?
2. Is there any mute/volume state applied to the remote audio element, and
   what iOS `AVAudioSession` category is active during a WebRTC session?
   (`ambient` respects the ringer switch and can silence playback entirely;
   `playAndRecord`/`playback` does not.)
3. In which build did robot-mic audio last verifiably play? If ≤146, the
   build-147 rework is confirmed as the regression point.

**Acceptance:** live session, clap next to the robot, clap is heard in the
app.

## Robot-side quality caveats (real bugs, but NOT the silence)

Two robot-side defects were found while investigating — fixes are queued
robot-side. They degrade quality but still deliver audible audio, so don't
let them muddy the app-side investigation:

- `recv()` pacing in `audio_track.py` under-produces (~44fps vs 50 needed) →
  capture queue jams → audio arrives ~1s stale with ~12% frame drops
  (choppy).
- No software gain on the WebRTC path → very quiet stream (ambient ≈
  −38 dBFS; the bark detector applies 30x gain to this same mic, WebRTC
  applies none).

## FYI: `emergency_stop` contract gap fixed robot-side (2026-08-06)

The app auto-sends `{"command": "emergency_stop"}` on drive-screen
exit/connection loss (contract format, no `params`). Until today the robot's
contract handler didn't implement it — it logged `Unknown contract command`
and did nothing (only the legacy params-format handler had it). Now both
formats route to one shared implementation that halts via the motor command
bus (same path as live drive). Live-verified: the app now gets
`{"success": true, "message": "Emergency stop executed", "via": [...]}`.
No app change required, but if the app was ignoring the previous
`success: false` response, it no longer needs to.
