# WIM-Z Resume Chat Log

## Session: 2026-01-21
**Goal:** Cloud Control Architecture - Fix cloud commands and WebRTC
**Status:** Partial - Commands fixed, data receipt bugs remain

---

### Work Completed This Session

#### 1. Cloud Command Handler Fixed (`main_treatbot.py`)
Rewrote `_handle_cloud_command()` to forward commands to local REST API:

| Command | API Endpoint | Body |
|---------|-------------|------|
| `treat` | POST `/treat/dispense` | - |
| `led` | POST `/led/pattern` | `{"pattern": "..."}` |
| `servo` | POST `/servo/pan` / `/servo/tilt` | `{"angle": float}` |
| `audio` | POST `/audio/play` | `{"file": "..."}` |
| `mode` | POST `/mode/set` | `{"mode": "..."}` |
| `motor` | *Ignored* | Goes via WebRTC data channel |

#### 2. WebRTC Data Channel Added (`services/streaming/webrtc.py`)
- Added data channel for low-latency motor control
- Channel name: `"control"`, unreliable mode for speed
- Message format: `{"type": "motor", "left": -1.0 to 1.0, "right": -1.0 to 1.0}`
- Handles both robot-created and app-created data channels

#### 3. Relay Client Bug Fixed (`services/cloud/relay_client.py`)
**Critical bug found:** Relay sends params in `data` field, not `params` field.
```
Raw: {"type": "command", "command": "motor", "data": {"left": -0.57, "right": -0.73}}
```
Fixed: `params = data.get('data', {})` instead of `data.get('params', {})`

---

### Files Modified

| File | Lines Changed | Changes |
|------|---------------|---------|
| `main_treatbot.py` | +267 | Rewrote cloud command handler to forward to API |
| `services/cloud/relay_client.py` | +19/-6 | Fixed params extraction from 'data' field |
| `services/streaming/webrtc.py` | +215 | Added data channel support for motor control |

---

### Unresolved Issues

1. **Data receipt bugs** - User noted "lots of bugs on data receipt" - needs investigation next session
2. **Motor commands** - Currently ignored in cloud handler (should go via WebRTC data channel)
3. **WebRTC video** - Not confirmed working yet

---

### Next Session Tasks

1. Debug data receipt issues in relay_client.py
2. Test WebRTC video streaming end-to-end
3. Test motor control via WebRTC data channel
4. Verify all cloud commands work from app

---

### Important Notes

- Cloud relay message format: `{"type": "command", "command": "xxx", "data": {...}}`
- Params are in `data` field, NOT `params` field
- API contract reference: `API_CONTRACT_v1.1.md`

---

## Session: 2026-01-20
**Goal:** Update relay URL to production server
**Status:** Complete

---

### Work Completed This Session

#### Production Relay URL Update
Changed cloud relay URL from `wss://api.wimz.io/ws/device` to `wss://api.wimzai.com/ws/device`

**Files Modified:**
- `config/robot_config.yaml` - Production relay URL
- `services/cloud/relay_client.py` - Default URLs in dataclass and factory
- `API_CONTRACT_v1.1.md` - Documentation
- `CLAUDE_INSTRUCTIONS_ROBOT.md` - Documentation

---

### Next Session Tasks
1. Set DEVICE_ID and DEVICE_SECRET in .env
2. Enable cloud relay (`cloud.enabled: true`)
3. Test connection to production relay server
4. Verify HMAC authentication works

---

## Session: 2026-01-18
**Goal:** Mode system fixes + API contract compliance
**Status:** Complete

---

### Work Completed This Session

#### 1. Mode System Fixes
- **Fixed mode persistence on Xbox disconnect**: When controller disconnects, now returns to previous mode instead of always SILENT_GUARDIAN
- **Added IDLE mode to Xbox cycling**: Cycle order is now MANUAL → IDLE → COACH → SILENT_GUARDIAN
- **Added audio announcements**: Each mode change plays corresponding MP3 from `/VOICEMP3/wimz/`

#### 2. Bark Detection Sensitivity Reduced (Too Many False Positives)
| Parameter | Before | After | Effect |
|-----------|--------|-------|--------|
| `confidence_minimum` | 0.10 | **0.45** | Only confident detections |
| `loudness_threshold_db` | -35 | **-20** | Ignores quieter sounds |
| `threshold` (bark count) | 2 | **3** | Requires 3 barks to trigger |
| `base_threshold` (energy) | 0.08 | **0.12** | Higher energy required |
| `min_bark_duration_ms` | 100 | **150** | Filters out short clicks |
| `bark_cooldown_ms` | 800 | **1000** | Slower between detections |

#### 3. API Contract Compliance (API_CONTRACT.md)
Added all missing REST endpoints to match contract.

---

## Session: 2026-01-16 ~19:00
**Goal:** Configure Silent Guardian mode - fix audio issues and bark detection tuning
**Status:** Complete

---

## Session: 2026-01-16 ~04:30
**Goal:** Add WebRTC streaming for mobile app (Phase 2 of Flutter app project)
**Status:** Complete - WebRTC service implemented

---

## Session: 2026-01-15 ~15:30
**Goal:** Fix behavior detection (lie down, spin, crosses removal)
**Status:** Major progress - spin detection significantly improved

---

## Session: 2026-01-15 ~04:30
**Goal:** Fix dog behavior detection - keypoints clustering in chest
**Status:** Critical fix committed

---
