# WIM-Z Development TODO List
*Last Updated: September 2026 · Release 2026.08.3 (OTA era — "Build N" retired)*

## Current Status: Release 2026.08.3 — OTA-managed fleet, SG intelligence shipped

**Build Phase:** CORE COMPLETE — hardening + manufacturing prep
**Fleet:** 4 active units (tb1, tb2, tb4, tb5) on one codebase + per-unit profiles, **all on the OTA layout**. tb3 (China) assumed permanently offline — excluded from fleet counts.
**Current Focus:** Freeze RCA, validation of scheduler + summaries, SG live E2E test, data-layer refactor (design phase)

---

## 🔥 CURRENT OPEN ITEMS (September 2026)

### Reliability / Safety (highest priority)
- [ ] **Silent hard-freeze RCA** — recurring lockups with no kernel/software trace (corrupt journal). Confirm power-delivery/brownout vs hang. **Evidence collector armed on treatbot2**: `wimz-power-watch.service` samples throttled/EXT5V/temp/core-V to `logs/power_watch.csv` every 30s, fsync'd, so the last pre-freeze sample survives the power cycle. Next step: read that CSV after the next freeze — EXT5V sag/throttled≠0 before the gap = brownout; rails healthy up to the gap = true hang. Note: units stable since returning from beta (2026-09-01).
- [x] **Power-button SPOF redesign** — SOLVED (Aug 2026): every robot now has a hardware-direct OFF switch, no software in the kill path.
- [x] **RTC batteries installed fleet-wide** (Aug 2026) — cross-boot timestamps are now trustworthy; retire the old "never trust cross-boot timestamps" caveat.
- [x] **treatbot2 dispenser** — root cause was a bad crimp on a stepper coil wire (repaired); the TMC2209 chip was fine.

### Validation (blocking "done" claims)
- [ ] **Mission Scheduler** — validate auto-start, time-window enforcement, once/daily/weekly logic (implemented, never tested)
- [ ] **Weekly Summary** accuracy — verify before it becomes an owner/investor metric
- [ ] **SG live end-to-end test** — summary pull, Loop echo, `sg_status_pull` / `audio_loop` over /ws/local in AP mode, with a real dog
- [ ] **bark_type stamp live verify** — live `bark` events now carry `bark_type`/`bark_label` (stamped in `bark_detector.py`, forwarded in `main_treatbot.py`, 2026-09-01); confirm app renders per-bark labels in live + history feeds

### Silent Guardian design decisions (user's call)
- [x] Expose `treat_eligibility_cooldown` (was hardcoded in `silent_guardian.py`) to profile yaml — now reads `session_limits.treat_eligibility_cooldown`, default 600s
- [x] **Post-cap behavior** (decided: keep intervening, no treats) — after the treat cap, SG now continues with verbal praise + LED and logs each quiet as `treat_given=False` (praise mitigates extinction; post-cap quiets now appear in history). `silent_guardian.py` treat-limit branch.

### Data / ML
- [ ] **Data refactor — KICKED OFF (2026-09-01, design phase)** — reshape all data into human/ML-friendly tables for analytics + continual learning. Robot side owns the design MD; App Claude reviews it against their consumer requirements (per-bark rows, sessions as first-class table, stable IDs + ISO8601 UTC, live event contract untouched, **no backfill of legacy rows in v1** — Morgan's standing decision). Do **not** start coding until the MD is approved.

---

## BUILD 40 VALIDATION CHECKLIST

### ✅ Code Changes Reviewed (Implemented in Build 40)
- [x] Mission field names fixed (`mission_name`→`mission_id`, `stage`→`stage_number`)
- [x] AI confidence display bridge added (`update_dog_behavior()` call)
- [x] Servo tracking auto-enable in COACH mode
- [x] MP3 download URL construction (relay relative path fix)
- [x] Coach progress/reward events added
- [x] GET /missions endpoint added

### ✅ Live Testing Complete
- [x] Mission progress events reaching relay with correct field names
- [x] Video overlay showing "sit 34%" confidence labels
- [x] Servo tracking checkbox working in app
- [x] MP3 upload flow working end-to-end (app → relay → robot)
- [x] Coach mode events visible in app

---

## PRIORITY 1: Unknowns (Need User Input)

### ❓ Coach Mode Status
- [x] Is bark detection filtering working? (claps/voice rejected?)
- [x] Are pose thresholds accurate? (sitting ≠ down/crosses?)
- [x] Full coaching session end-to-end tested?

### ❓ Silent Guardian Status
- [x] Bark → intervention flow working?
- [x] Escalation and cooldown working?

### ❓ App/Relay Integration
- [x] Is relay forwarding events correctly?
- [x] Is app displaying mission progress?
- [x] Are WebRTC video streams stable?

### ✅ Hardware Status
- [x] Servo calibration accurate (needs tweaking per unit)
- [x] Treat dispenser working reliably
- [x] Audio playback consistent

---

## PRIORITY 2: Verified Working (From Recent Builds)

### ✅ Build 40 (Feb 2, 2026)
- [x] Mission field names standardized
- [x] AI detection bridge to video overlay
- [x] Servo tracking auto-enable
- [x] Download song URL construction
- [x] Coach progress events
- [x] GET /missions REST endpoint

### ✅ Build 38 (Feb 1, 2026)
- [x] Video overlay race condition fix
- [x] Bounding boxes for unidentified dogs
- [x] Dog identification conservative defaults ("Dog" label)
- [x] Nudge servo tracking (gentle, 2°/sec max)
- [x] MP3 download via HTTP (not WebSocket)

### ✅ Build 36 (Jan 31, 2026)
- [x] Mission name aliases (stay_training → sit_training)
- [x] Frame freshness check (<500ms)
- [x] Faster detection (1.5s + 50% presence)
- [x] Default "Dog" label when ArUco unavailable

### ✅ Build 35 (Jan 31, 2026)
- [x] Schedule API with dog_id, schedule_id, type fields
- [x] Schedule types: once/daily/weekly
- [x] Auto-disable "once" schedules after execution

### ✅ Build 34 (Jan 31, 2026)
- [x] Mission presence detection fixed
- [x] Dog identification regression fixed
- [x] Video overlay emoji removal
- [x] Mode sync events (mode_changed)
- [x] Servo safety limits

### ✅ Earlier Fixes (Jan 2026)
- [x] Threading race conditions (timestamp validation)
- [x] Bark bandpass filter (400-4000Hz)
- [x] Pose thresholds (0.75 for lie/cross)
- [x] Presence-based detection (3s + 66%)
- [x] Retry on first failure
- [x] WIM-Z audio feedback system

---

## PRIORITY 3: Needs Rework/Testing

### Weekly Summary System (`core/weekly_summary.py`)
**Status:** Tested, not 100% accurate
- [x] Tested with live data (not fully accurate)
- [ ] Verify `generate_weekly_report()` returns accurate data
- [ ] Test API endpoints: `GET /reports/weekly`, `GET /reports/trends`

### Mission Scheduler (`core/mission_scheduler.py`)
**Status:** Implemented, type logic added in Build 35, NOT TESTED
- [ ] Auto-scheduling not yet tested
- [ ] Test time window enforcement
- [ ] Verify missions auto-start correctly

---

## PRIORITY 4: Future Enhancements

### Analytics System
- [ ] Daily summary endpoint
- [ ] Bark frequency trends
- [x] Treat usage stats (treat inventory tracking implemented)
- [ ] Bone score rating (1-5)

### Session Management
- [ ] 8-hour session tracking
- [ ] Session reset at midnight
- [ ] Max 11 treats enforcement

### Photography
- [ ] Burst mode
- [ ] Quality scoring
- [ ] Best photo selection

### Push Notifications — ✅ LIVE via Firebase (FCM/APNs)
- [x] **Shipped and verified 2026-08-30**: Firebase project `wimzpushy`; app slice 2026-07-30, relay slice deployed (relay 409f381, contract `PUSH_NOTIFICATIONS_CONTRACT_2026-07-30.md`); lock-screen banners confirmed with the app closed.
- The old AWS SNS plan in this file was never real — do NOT set up boto3/SNS.
- Panic/SG pushes reuse the generic push pipeline with PANIC text (no per-type FCM mapping; per-type app preference toggles don't gate generic pushes — accepted for v1).
- [ ] v2 (parked until Morgan prioritizes): bark-type-filtered pushes ("Distress barking detected"; mute demand/play) — unblocked by the 2026-09-01 per-bark `bark_type` stamp.

---

## Quick Test Commands
```bash
# Restart service
sudo systemctl restart treatbot

# Monitor logs
journalctl -u treatbot -f | grep -i "mission\|coach\|bark\|pose"

# Check mode
curl http://localhost:8000/mode

# Test missions endpoint
curl http://localhost:8000/missions

# Force coach mode
curl -X POST http://localhost:8000/mode/set -H "Content-Type: application/json" -d '{"mode": "COACH"}'
```

---

## Key Files Reference

| Purpose | File |
|---------|------|
| Main entry | `main_treatbot.py` |
| Mission engine | `orchestrators/mission_engine.py` |
| Coach mode | `orchestrators/coaching_engine.py` |
| Silent Guardian | `modes/silent_guardian.py` |
| Detector | `services/perception/detector.py` |
| Video overlay | `services/streaming/video_track.py` |
| Pan/tilt | `services/motion/pan_tilt.py` |
| Relay client | `services/cloud/relay_client.py` |

---

## Dropped Features

- **IR Navigation/Docking** - Hardware caused Pi startup failures

## ✅ Previously "Dropped" - Now Implemented

- **Direct LAN Connection** - Phone connects directly to robot WiFi (WIMZ-*) without relay

---

*Updated: September 2026 — Release 2026.08.3. Cleared solved hardware items (power switch, RTC, tb2 crimp), corrected push notifications to Firebase (live), fleet counts to 4 active units (tb3 offline), data refactor moved to design phase, added bark_type stamp + SG E2E validation items.*
