# WIM-Z Development TODO List
*Last Updated: February 2, 2026*

## Current Status: Build 40 Complete - Awaiting Testing

**Build Phase:** COMPLETE - All core systems operational
**Current Focus:** Validating Build 40 fixes (mission fields, AI display, tracking, coach events)

---

## BUILD 40 VALIDATION CHECKLIST

### ✅ Code Changes Reviewed (Implemented in Build 40)
- [x] Mission field names fixed (`mission_name`→`mission_id`, `stage`→`stage_number`)
- [x] AI confidence display bridge added (`update_dog_behavior()` call)
- [x] Servo tracking auto-enable in COACH mode
- [x] MP3 download URL construction (relay relative path fix)
- [x] Coach progress/reward events added
- [x] GET /missions endpoint added

### ❓ UNKNOWN - Needs Live Testing
- [x] Mission progress events reaching relay with correct field names
- [ ] Video overlay showing "sit 34%" confidence labels
- [ ] Servo tracking checkbox working in app
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

### ❓ Hardware Status
- [ ] Servo calibration still accurate?
- [x] Treat dispenser working reliably?
- [x] Audio playback consistent?

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
**Status:** Implemented, untested with live data
- [ ] ❓ Has this been tested?
- [ ] Verify `generate_weekly_report()` returns accurate data
- [ ] Test API endpoints: `GET /reports/weekly`, `GET /reports/trends`

### Mission Scheduler (`core/mission_scheduler.py`)
**Status:** Implemented, type logic added in Build 35
- [x] ❓ Has auto-scheduling been tested?
- [x] Test time window enforcement
- [ ] Verify missions auto-start correctly

---

## PRIORITY 4: Future Enhancements

### Analytics System
- [ ] Daily summary endpoint
- [ ] Bark frequency trends
- [ ] Treat usage stats
- [ ] Bone score rating (1-5)

### Session Management
- [ ] 8-hour session tracking
- [ ] Session reset at midnight
- [ ] Max 11 treats enforcement

### Photography
- [ ] Burst mode
- [ ] Quality scoring
- [ ] Best photo selection

### Push Notifications (BUILD 41)
- [x] AWS SNS notification service created (`services/cloud/notification_service.py`)
- [x] API endpoints added (`/notifications/*`)
- [ ] Install boto3: `pip install boto3`
- [ ] Configure AWS credentials (see setup below)
- [ ] Test SMS sending
- [ ] Integrate with mission_complete events
- [ ] Integrate with bark_alert events
- [ ] Integrate with low_battery events

---

## AWS SNS Setup (Push Notifications)
```bash
# 1. Install boto3
pip install boto3

# 2. Configure AWS credentials (choose one method)
# Method A: AWS CLI
aws configure
# Enter: Access Key ID, Secret Key, Region (us-east-1 recommended for SMS)

# Method B: Environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# 3. Test the service
curl http://localhost:8000/notifications/health

# 4. Add a subscriber
curl -X POST http://localhost:8000/notifications/subscribers \
  -H "Content-Type: application/json" \
  -d '{"user_id": "morgan", "phone_number": "+15551234567"}'

# 5. Send test notification
curl -X POST "http://localhost:8000/notifications/test?user_id=morgan"
```

**AWS SMS Sandbox Note:**
New AWS accounts are in SMS sandbox mode. To send SMS:
- Option A: Verify destination phone numbers in AWS Console → SNS → Text messaging
- Option B: Request production access (takes 1-2 days approval)

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
- **WebSocket Server** - REST API sufficient

---

*Updated: February 2, 2026 - Build 40 session start*
