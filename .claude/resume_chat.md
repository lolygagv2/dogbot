# WIM-Z Resume Chat Log

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
Added all missing REST endpoints to match contract:
- `/motor/speed`, `/motor/emergency`
- `/camera/stream`, `/servo/pan`, `/servo/tilt`, `/servo/center`
- `/treat/carousel/rotate`
- `/led/pattern`, `/led/color`, `/led/off`
- `/mode/get`
- `/missions`, `/missions/{id}`, `/missions/{id}/start`, `/missions/{id}/stop`, `/missions/active`
- `/telemetry/contract`

Updated WebSocket server (`api/ws.py`):
- Added ping/pong handling (`{"type": "ping"}` → `{"type": "pong"}`)
- Added auth handling (local mode accepts all)
- Added contract command format support (flat JSON)
- Added contract event format broadcasting
- Added periodic status broadcasts (every 5s)

---

### Files Modified

| File | Changes |
|------|---------|
| `orchestrators/mode_fsm.py` | Added `pre_manual_mode` tracking, return to previous mode on disconnect |
| `xbox_hybrid_controller.py` | Added IDLE to cycle, added mode audio announcements |
| `configs/rules/silent_guardian_rules.yaml` | Raised bark detection thresholds |
| `services/perception/bark_detector.py` | Raised BarkGateConfig thresholds |
| `core/audio/bark_gate.py` | Raised default thresholds |
| `modes/silent_guardian.py` | Updated default config thresholds |
| `api/server.py` | Added ~200 lines of contract-compliant endpoints |
| `api/ws.py` | Added contract command/event handling, ping/pong |
| `API_CONTRACT.md` | Updated implementation checklist |

---

### Next Session Tasks

1. Test Silent Guardian with real dogs to verify bark threshold tuning
2. Test IDLE mode cycling on Xbox controller
3. Test API contract endpoints from mobile app
4. Monitor for any remaining false bark triggers

---

### Important Notes

- Bark detection thresholds may need further tuning based on real-world testing
- IDLE mode = standby mode where device listens but doesn't intervene
- WebSocket broadcasts in BOTH legacy AND contract formats for backwards compatibility

---

## Session: 2026-01-16 ~19:00
**Goal:** Configure Silent Guardian mode - fix audio issues and bark detection tuning
**Status:** ✅ Complete

---

### Work Completed

#### 1. Fixed Audio Overlapping Issue
The "qu-qui-quiet" overlapping audio was caused by `_play_audio()` not waiting for completion.

**Fix:** Updated `_play_audio()` to use `wait_for_completion()`:
```python
def _play_audio(self, filename: str, wait: bool = True):
    # ... plays audio ...
    if wait:
        self.audio.wait_for_completion(timeout=5.0)
```

#### 2. Simplified Level 1 Intervention
- **Before:** Played dog name ("elsa.mp3") + "quiet.mp3" with 0.5s delay
- **After:** Just plays "quiet.mp3" - simpler and clearer
- Removed confusing dog name greeting

#### 3. Simplified Level 2 Intervention
- **Before:** 5-step sequence (quiet, quiet, no, come, quiet) with overlapping audio
- **After:** 3-step sequence: quiet → no → quiet (non-overlapping)
- Removed "dogs_come.mp3" which was confusing
- Each audio now plays fully before the next starts

#### 4. Fixed Missing Audio Files
| Issue | Fix |
|-------|-----|
| `good.mp3` didn't exist | Changed to `good_dog.mp3` |
| Calming music file didn't exist | Changed to `mozart_piano.mp3` |

#### 5. Tuned Bark Detection
| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| Bandpass filter | 500-3500Hz | 400-2500Hz | Tighter filter excludes claps, speech |
| Base threshold | 0.12 | 0.08 | Lower to catch quieter barks |
| Min duration | 80ms | 100ms | Excludes clicks, pops |
| Cooldown | 1000ms | 800ms | Faster response |
| Bark count threshold | 3 | 2 | Faster intervention |
| Time window | 60s | 45s | More responsive |

---

### Key Files Modified

| File | Changes |
|------|---------|
| `modes/silent_guardian.py` | Audio wait, simplified L1/L2, fixed audio filenames |
| `services/perception/bark_detector.py` | Bandpass filter 400-2500Hz, lower thresholds |
| `core/audio/bark_gate.py` | Updated default thresholds |
| `configs/rules/silent_guardian_rules.yaml` | Bark detection tuning, calming music path |

---

### Silent Guardian Audio Summary (All Verified)
| When | File | Status |
|------|------|--------|
| Level 1 | `quiet.mp3` | ✅ |
| Level 2 | `quiet.mp3`, `no.mp3` | ✅ |
| Level 3 | `quiet.mp3` + `mozart_piano.mp3` (looped) | ✅ |
| Reward | `good_dog.mp3` | ✅ |

---

### To Activate
```bash
sudo systemctl restart treatbot
```

---

### Next Steps
1. Test Silent Guardian with real dogs
2. Monitor bark detection accuracy (false positives vs missed barks)
3. Fine-tune thresholds based on real-world testing

---

## Session: 2026-01-16 ~04:30
**Goal:** Add WebRTC streaming for mobile app (Phase 2 of Flutter app project)
**Status:** Complete - WebRTC service implemented, needs restart to activate

---

### Work Completed

#### 1. WebRTC Streaming Service Created
New files added to `services/streaming/`:
- `__init__.py` - Package exports
- `video_track.py` - WIMZVideoTrack class (aiortc VideoStreamTrack)
- `webrtc.py` - WebRTCService singleton

**Features:**
- Reads frames from DetectorService via `get_last_frame()`
- Optional AI overlay (bounding boxes, poses, behaviors)
- Configurable FPS (default 15) and bitrate (1.5 Mbps)
- Support for 2 concurrent WebRTC clients
- MediaRelay for efficient multi-client streaming

#### 2. API Endpoints Added to `api/server.py`
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/webrtc/status` | GET | Service status and connection info |
| `/webrtc/offer` | POST | Handle SDP offer, return answer |
| `/webrtc/ice` | POST | Add ICE candidate |
| `/webrtc/close/{session_id}` | POST | Close connection |
| `/ws/webrtc/{session_id}` | WebSocket | Real-time signaling |

#### 3. Config Added to `robot_config.yaml`
```yaml
webrtc:
  enabled: true
  stun_servers: ["stun:stun.l.google.com:19302"]
  turn_servers: []  # Add for cloud relay
  video:
    fps: 15
    bitrate: 1500000
    enable_ai_overlay: true
  max_connections: 2
```

#### 4. Dependencies Installed
- `aiortc==1.14.0` (Python WebRTC)
- System packages: libavdevice-dev, libavfilter-dev, libopus-dev, libvpx-dev

---

## Session: 2026-01-15 ~15:30
**Goal:** Fix behavior detection (lie down, spin, crosses removal)
**Status:** Major progress - spin detection significantly improved

---

### Problems Solved

#### 1. Crosses Behavior Removed Completely
- Removed from ALL locations (config, tricks, coaching engine, xbox controller)

#### 2. Lie Down Detection Fixed
- Updated aspect ratio thresholds in geometric_classifier.py

#### 3. Spin Detection - Major Rewrite
- Uses bbox-only metrics (no keypoints needed)
- Spin bypasses temporal voting (instant detection)
- Added "spin latch" - holds for 2 seconds after detection

---

## Session: 2026-01-15 ~04:30
**Goal:** Fix dog behavior detection - keypoints clustering in chest
**Status:** ✅ Critical fix committed

### CRITICAL FIX: Keypoint Decoding Bug
```python
# WRONG: kpts = (raw + cell) * stride
# CORRECT: kpts = (raw * 2 + cell) * stride
```

---
