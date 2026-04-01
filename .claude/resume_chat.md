# WIM-Z Resume Chat Log

## Session: 2026-04-01 — Multi-System Debug, Local Mode, Coach Fixes
**Goal:** Instance 1 (Robot Pi) — PTT logging, coach mode fixes, AI diagnostics, local mode, treat cleanup
**Status:** COMPLETE — All tasks R1-R6 + follow-ups done, committed, pushed

---

### What Was Done (13 files, +601/-92 lines)

#### R1 — PTT Debug Logging
- `[PTT]` prefix logging at every step: relay receive, base64 decode, queue, ffmpeg convert, playback start/complete
- Files: `push_to_talk.py`, `relay_client.py`, `ws.py`
- `POST /debug/ptt-test` endpoint plays 440Hz test tone through same USBAudio path PTT uses
- **Audit finding:** PTT and coach announcements share single pygame.mixer.music channel — last writer wins, no cross-system queue

#### R2 — Coach Mode & ArUco
- **Startup sequence fixed:** `coaching_engine.start()` now plays `VOICEMP3/wimz/CoachMode.mp3` and waits before activating detection pipeline
- **ArUco hardcoded IDs removed:** `aruco_detector.py` had `{1: "elsa", 2: "bezik"}` (WRONG). Now loads dynamically from dog profiles + config.json fallback
- **ArUco dictionary:** All code confirmed `DICT_4X4_1000` (IDs 0-999). Elsa=315, Bezik=832. Plugin was DICT_6X6_250, fixed.
- **ArUco disabled in Silent Guardian** — SG only needs bark detection
- **Generic greeting fallback:** If no ArUco marker detected after 3s, plays `treat.mp3` instead of silence
- `[COACH]`, `[DOG-ID]`, `[ARUCO]` logging at all transitions
- **Trick audit:** All 3 sources (trick_rules.yaml, coaching_engine, xbox_controller) have identical lists: sit, laydown, come, spin, speak

#### R3 — AI Pipeline Diagnostics
- `[AI]` heartbeat every 5s: FPS, detection count, last class/confidence
- `[AI]` 60s stats: Cat A (pipeline errors), Cat B (below threshold), Cat C (classification fails)
- `GET /debug/ai-status` — JSON with pipeline_running, fps, hailo_status, last_detection, thresholds, frames_total
- `POST /debug/ai-thresholds` — hot-update confidence thresholds at runtime without restart
- Added `get/set_confidence_threshold(s)` methods to `behavior_interpreter.py`

#### R4 — Dog ID Fallback
- Coach + mission modes: if no ArUco marker in 3s → play `treat.mp3` (generic greeting)
- `want_a_treat.mp3` was MISSING → code updated to use existing `treat.mp3`
- **Visual dog recognition audit:** ArUco is primary, color-based identification is secondary fallback (in dog_profile_manager.py). Color matching only works when dogs have unique coat colors.

#### R5 — Treat Carousel Cleanup
- Dead launcher code (servo_control_module.py, treat_loader.py) confirmed NOT in runtime paths
- Removed dead `vibrator_enabled` property from config_loader.py
- `[TREAT]` logging: `Dispensing treat #N of 44 | Remaining: X`
- Stepper dispenser confirmed: TMC2209 UART only, GPIO stepping, no servo code

#### R6 — Local Connection Mode
- Server already binds `0.0.0.0:8000` — accessible on LAN
- `[LOCAL]` IP logging on boot: `ws://<ip>:8000/ws/local and http://<ip>:8000`
- **CRITICAL FIX:** App connects to `/ws` not `/ws/local`. Added WebRTC signaling + PTT handlers to `/ws` (ws.py)
- WebRTC: `webrtc_request` → creates peer connection + STUN-only ICE config + offer; `webrtc_answer`; `webrtc_ice`
- PTT: accepts both `ptt_play` (app command name) and `audio_message` (relay protocol name)
- Command field reads both `data.get("command")` and `data.get("type")` for protocol compatibility
- Pi works without relay (confirmed: 23h uptime during DNS failure)

#### Charging False Positive Fix
- `Wimz_charging.mp3` played on every startup even when NOT charging
- **Root cause:** Method 1 (voltage >= 16.5V) and Method 3 (percentage >= 95%) both false-positive on a fully-charged unplugged battery
- **Fix:** Removed both unreliable methods. Now requires 4+ voltage readings (~20s) showing consistent upward trend before declaring charging. Startup grace period prevents first-boot false positives.

### Key Findings
- **Come trick:** Detects "standing" pose, NOT approach/movement. Dog must be standing 1.5s.
- **Spin trick:** Has real temporal logic — tracks bbox aspect/center deltas over 4-8 frames. Works.
- **Speak trick:** Audio-only (bark detector), NOT vision. High risk at noisy conferences.
- **Detection pipeline:** LSTM behavior model is DISABLED (`_force_geometric=True`). All classification is geometric heuristics based on bbox aspect ratio + keypoint positions.
- **Color-based dog ID** is LIVE code in dog_profile_manager.py — secondary fallback after ArUco.

### Files Modified
- `api/server.py` — debug endpoints, local WS WebRTC+PTT, boot logging
- `api/ws.py` — WebRTC signaling + PTT handlers on /ws, ptt_play command support
- `config/config_loader.py` — removed dead vibrator_enabled
- `configs/modes.yaml` — aruco_enabled: false for silent_guardian
- `core/behavior_interpreter.py` — get/set confidence threshold methods
- `core/vision/detection_plugins/aruco_detector.py` — dynamic mapping, DICT_4X4_1000
- `orchestrators/coaching_engine.py` — startup sequence, greeting fallback, logging
- `services/cloud/relay_client.py` — [PTT] logging
- `services/media/push_to_talk.py` — [PTT] logging throughout chain
- `services/perception/detector.py` — [AI] heartbeat, stats, DOG_MARKERS loading, ArUco per-mode
- `services/power/battery_monitor.py` — charging detection fix (trend-only)
- `services/reward/dispenser.py` — [TREAT] logging
- `xbox_hybrid_controller.py` — (minor, from earlier session)

### Git Status
- Branch: `main` at `7632d90`
- 1 commit this session, pushed to origin
- Untracked: `tests/hardware/test_servo_channel2.py`

---

## Session: 2026-03-29 - Stepper Motor Treat Dispenser (TMC2209 + NEMA 17)
**Goal:** Replace servo dispenser with stepper motor (NEMA 17 + TMC2209 driver)
**Status:** PARTIAL — Dispenser works, refill works, jam detection FAILED

---

### What Works
- **Single treat dispense** — tap LB, motor steps 137 microsteps, treat drops. Reliable.
- **Refill mode** — hold LB 5+ seconds, continuous fast stepping, release = instant stop.
- **UART communication** — TMC2209 responds, current/microstepping configured via UART.
- **Motor control** — STEP/DIR/EN all working, CW direction confirmed, calibrated.

### What Does NOT Work
- **Jam detection via StallGuard** — COMPLETELY non-functional at our speed.
  - SG_RESULT = 0 in BOTH jammed and unjammed states. Identical register values.
  - Root cause: motor speed too low (~3 RPM) for StallGuard back-EMF sensing.

### Next Steps (Jam Detection)
Options to explore:
1. **Physical sensor** — IR break-beam at drop hole (most reliable, needs hardware)
2. **Audio detection** — onboard mic detects clicking pattern of stalled motor
3. **No auto-detection** — manual unjam only, acceptable for now
