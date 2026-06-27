# Night Sentry — Demo Mode Implementation Plan (treatbot3)

Status: PLANNED 2026-06-27. Not yet implemented. Resume here.

## Decisions (locked with user)
- **Activation:** Xbox **Select** button = existing mode-cycle. Add NIGHT_SENTRY to the
  `cycle_mode()` rotation (Coach → Silent Guardian → Night Sentry → …). NOT a new button.
- **On activation:** play `/home/morgan/dogbot/VOICEMP3/wimz/WIMZ_nsentry.mp3` (exists, 60630 bytes)
  via the central `_announce_mode` map.
- **Scan motion:** robot BODY drives a small half-circle arc (in-place pivots), pausing at
  waypoints to watch. Uses motor_command_bus (AUTONOMOUS).
- **Detection:** Hailo YOLO. ⚠️ MODEL IS DOG-ONLY — emits `dog_detected` only; no person/
  general-animal class on disk. Demo scope = dog/animal detection unless a new model is added.
- **Alert:** on detection → log + snapshot + upload photo to cloud so app shows it on open.
  Photo delivery REUSES the existing base64 `photo` WS event (main_treatbot.py:1176) — no
  server work needed for the image to display.
- **Night vision:** NoIR + 940nm IR illuminator present; night_mode_controller switches the
  low-light camera profile by Lux automatically.

## Hardware-safety (must honor)
- Motor PWM ≤70% (bus clamps ±70 at motor_command_bus.py:243). Config cap 40–70.
- 50ms min between commands; short bursts (≤0.5s) + halt + ≥80ms settle (SG pattern).
- 500ms dead-man watchdog forces STOP if no fresh command (motor_command_bus.py:180-207).
- Guaranteed halt in try/finally; STOP on mode exit, emergency, controller disconnect.
- Test on a stand at speed 40 / 2 waypoints first; power switch reachable.

## Integration points (file:line)
- Enum: `core/state.py:16` add `NIGHT_SENTRY = "night_sentry"`.
- FSM: `orchestrators/mode_fsm.py:72-80` transitions + `:315` override allow-list.
- Mode lifecycle: `main_treatbot.py:1868` `_on_mode_change` start/stop (mirror SG :1912-1929);
  announce map ~:1910; relay forward SYSTEM branch :790.
- New mode: `modes/night_sentry.py` (template = `modes/silent_guardian.py`; sweep :784-846,
  start/stop :345-432, singleton :1176).
- Xbox: `xbox_hybrid_controller.py:1949` `cycle_mode()` — add night_sentry to rotation.
- Audio: `services/media/usb_audio.py:229` `play_file(abs_path)`.
- Motors: `core/motor_command_bus.py` get_motor_bus():317, create_motor_command,
  CommandSource.AUTONOMOUS; clamp :243; watchdog :180.
- Detection: subscribe bus 'vision' → filter `dog_detected` (detector.py:941, fields
  confidence/dog_id/bbox). ⚠️ REQUIRED: add NIGHT_SENTRY to inference-enabled modes at
  `detector.py:740-742` or NO detections fire.
- Snapshot: `detector.get_last_frame()` (detector.py:566) + cv2.imwrite to captures/, OR
  reuse `services/media/photo_capture.get_photo_capture_service().capture_photo(with_hud=True)`
  → returns base64 `data` + `filepath` (api/server.py:1779).
- Cloud: mode publishes `publish_system_event('sentry_detection', {...,image_b64})`;
  main_treatbot._forward_event_to_relay adds `sentry_detection` branch → `send_event('photo',
  {data:b64})` + `send_event('sentry_detection', {meta})`. 640x640 q90 ≈ 40-110KB b64, OK over WS.
- Config: new `configs/rules/night_sentry_rules.yaml` (confidence_threshold, arc_degrees,
  waypoint_count, motor_speed_pct 40-70, burst_seconds, settle_seconds, dwell_seconds,
  cooldown_seconds). No per-unit yaml needed.

## Phased build (incremental, motor-safe)
- **Phase 0 (no motion):** enum + FSM + rules yaml + mode skeleton (announce+idle loop) +
  main_treatbot lifecycle/announce. Test: /mode/set night_sentry → plays mp3, no movement, clean exit.
- **Phase 1 (detection, no motion):** detector.py inference gating; mode alert pipeline minus
  motion → snapshot to captures/ + log on dog_detected ≥ threshold.
- **Phase 2 (body sweep ⚠️):** _run_sweep copied from SG; stand + speed 40 + 2 waypoints first,
  then floor + full half-circle; verify halts (stop/exit/emergency/disconnect).
- **Phase 3 (cloud + toggle):** relay sentry_detection→photo forward; add to cycle_mode().

## Files
CREATE: modes/night_sentry.py, configs/rules/night_sentry_rules.yaml
MODIFY: core/state.py, orchestrators/mode_fsm.py, main_treatbot.py,
        services/perception/detector.py, xbox_hybrid_controller.py

## Part A (now on tb3): all of the above; photo delivery works via existing `photo` event.
## Part B (separate api.wimzai.com + app deploy): dedicated `sentry_detection` history card.
##         Not needed for the photo to display in the demo.

## OPEN DECISION: detection scope — dog-only (now) vs add person/COCO model (bigger scope).
