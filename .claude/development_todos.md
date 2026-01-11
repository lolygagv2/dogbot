# WIM-Z Development TODO List
*Last Updated: January 11, 2026*

## Current Status: Live Testing & Bug Fixes

**Build Phase:** COMPLETE - All hardware and software systems operational
**Current Focus:** Stabilizing Coach and Silent Guardian modes through real-world testing

---

## PRIORITY 1: Immediate Testing (This Session)

### Coach Mode Verification
After last session's fixes (bark filter + pose thresholds), verify:

- [ ] **Test bark detection filtering**
  - Clap near mic → should NOT trigger speak success
  - Say words loudly → should NOT trigger speak success
  - Actual dog bark → SHOULD trigger speak success

- [ ] **Test pose detection thresholds**
  - Dog sitting → should NOT trigger "down" or "crosses"
  - Dog lying (0.75+ confidence) → SHOULD trigger "down"
  - Dog crossing paws (0.75+ confidence) → SHOULD trigger "crosses"

- [ ] **Test full coaching session**
  - Dog enters frame → greeting plays within 3s
  - Trick command plays correctly
  - Success/failure detected accurately
  - Retry on first failure works
  - Treat dispenses on success

### Silent Guardian Verification
- [ ] **Test bark → intervention flow**
  - 2+ barks in 60s → "quiet" audio plays
  - 10s quiet → reward sequence plays
  - Treat dispenses

- [ ] **Test escalation**
  - Continued barking → escalating commands
  - 90-second timeout → give up
  - 2-minute cooldown before next intervention

### Quick Test Commands
```bash
# Restart service with changes
sudo systemctl restart treatbot

# Monitor in real-time
journalctl -u treatbot -f | grep -i "bark\|coach\|trick\|pose\|speak"

# Check mode
curl http://localhost:8000/mode

# Force coach mode
curl -X POST http://localhost:8000/mode/set -d '{"mode": "COACH"}'
```

---

## PRIORITY 2: Known Issues to Fix

### Threading & Race Conditions
- [x] ~~Tricks completing instantly~~ - Fixed with timestamp validation
- [x] ~~Stale events executing after state changes~~ - Fixed in coaching_engine.py
- [ ] Monitor for any new race condition symptoms

### Audio Issues
- [x] ~~Trick audio not playing~~ - Fixed init order in behavior_interpreter.py
- [x] ~~Error audio spam~~ - Fixed temp thresholds in safety.py
- [ ] Verify all audio files play at correct times

### Detection Accuracy
- [x] ~~Bark triggers on claps/voice~~ - Added 400-4000Hz bandpass filter
- [x] ~~Lie/cross false positives~~ - Raised thresholds to 0.75
- [ ] Verify fixes work in real-world testing

---

## PRIORITY 3: Needs Rework

### Weekly Summary System (`core/weekly_summary.py`)
**Status:** Implemented but untested with live data

- [ ] Verify `generate_weekly_report()` returns accurate data
- [ ] Test `get_behavior_trends(weeks=8)` function
- [ ] Verify `export_report()` creates valid markdown/CSV
- [ ] Test API endpoints: `GET /reports/weekly`, `GET /reports/trends`

### Mission Scheduler (`core/mission_scheduler.py`)
**Status:** Implemented but needs integration testing

- [ ] Test time window enforcement
- [ ] Test day-of-week filtering
- [ ] Verify missions auto-start correctly
- [ ] Test API endpoints: `GET /scheduler/status`, `POST /scheduler/enable`

---

## PRIORITY 4: Future Enhancements

### Analytics System (After Stabilization)
- [ ] `GET /analytics/daily` - Daily summary endpoint
- [ ] `GET /analytics/bark-stats` - Bark frequency trends
- [ ] `GET /analytics/treat-usage` - Treat dispensing stats
- [ ] Bone score rating system (1-5 bones based on behavior)

### Session Management
- [ ] Proper 8-hour session tracking
- [ ] Automatic session reset at midnight
- [ ] Max 11 treats per session enforcement
- [ ] Session history persistence

### Photography Enhancements
- [ ] Burst mode (10 photos in 2 seconds)
- [ ] Quality scoring algorithm
- [ ] Auto-select best 3 photos
- [ ] Optional LLM captioning

---

## Recently Completed

### Session 2026-01-10
- [x] Fixed threading race condition with timestamp validation
- [x] Added 400-4000Hz bandpass filter for bark detection
- [x] Raised lie/cross confidence thresholds to 0.75

### Session 2026-01-08
- [x] Implemented presence-based dog detection (3s + 66% ratio)
- [x] Added retry on first failure (2 attempts per session)
- [x] ArUco identification now optional (sessions start on presence)
- [x] Reduced ArUco grace period from 10s to 0s

### Session 2026-01-08 (Earlier)
- [x] Fixed constant error audio spam (temp threshold 70→76C)
- [x] Added speak trick confidence filter (50% minimum)
- [x] Fixed charging detection threshold (0.3V→0.05V)
- [x] Lowered dog detection confidence (0.7→0.5)
- [x] Replaced stillness with time-based detection (3.5s)
- [x] Added Xbox Guide button trick cycling

### Session 2026-01-07
- [x] Fixed BehaviorInterpreter init order bug
- [x] Added WIM-Z audio feedback system (charging, low power, error, etc.)
- [x] Fixed RB photo button issues
- [x] Fixed mode cycling sync
- [x] Motor safety auto-stop on controller freeze

---

## System Health Checklist

Before testing, verify:
- [ ] `sudo systemctl status treatbot` - Service running
- [ ] Camera feed active (check logs for "Frame" messages)
- [ ] Audio working: `curl -X POST http://localhost:8000/audio/play/file -d '{"filepath": "/talks/good_dog.mp3"}'`
- [ ] Temperature normal: `vcgencmd measure_temp` (should be <76C)

---

## Key Files Reference

| Purpose | File |
|---------|------|
| Main entry | `main_treatbot.py` |
| Coach mode | `orchestrators/coaching_engine.py` |
| Silent Guardian | `modes/silent_guardian.py` |
| Bark detection | `services/perception/bark_detector.py` |
| Pose detection | `core/behavior_interpreter.py` |
| Trick rules | `configs/trick_rules.yaml` |
| Xbox controller | `xbox_hybrid_controller.py` |

---

## Dropped Features

- **IR Navigation/Docking** - Hardware caused Pi startup failures
- **WebSocket Server** - REST API sufficient for current needs

---

*Updated after session start on January 11, 2026*
