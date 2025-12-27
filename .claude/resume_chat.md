# WIM-Z Resume Chat Log

## Session: 2025-12-27 ~04:00-05:00 UTC
**Goal:** Fix coaching mode stability and refactor behavior detection architecture
**Status:** ✅ Complete (architecture created, testing pending)

### Problems Solved This Session:

1. **Coaching Mode Reset Issue**
   - Mode kept resetting from COACH to IDLE
   - Root cause: Temperature critical threshold (75°C) being hit by Pi running Hailo
   - Fix: Raised temp_critical to 82°C, then disabled temp-based mode forcing entirely for testing

2. **Mode Re-entry Bug**
   - Switching modes and returning to coaching didn't reset FSM state
   - Fix: Added state reset in `coaching_engine.start()` method

3. **Behavior Model Inconsistency**
   - Model flip-flopping between sit/cross detection on fluffy Samoyeds
   - Confirmed: Training data IS mean-centered [-0.5, 0.5], -0.5 subtraction is CORRECT
   - Issue is fundamental: fluffy dogs are edge cases for pose detection

### Key Code Changes Made:

#### New Files Created:
- `/core/behavior_interpreter.py` - Layer 1: Behavior detection wrapper
- `/configs/trick_rules.yaml` - Layer 2: YAML config for trick requirements

#### Modified Files:
- `/orchestrators/coaching_engine.py` - Refactored to use new 3-layer architecture
- `/api/server.py` - Added behavior interpreter API endpoints
- `/core/safety.py` - Disabled temp-based mode forcing for testing

### Architecture Created (3-Layer System):

**Layer 1: BehaviorInterpreter** (`/core/behavior_interpreter.py`)
- Tracks detections and hold duration
- `check_trick("sit")` returns if requirements met
- Simple device-level state (not per-dog tracking)

**Layer 2: Trick Rules** (`/configs/trick_rules.yaml`)
- All trick definitions in one place
- Confidence thresholds, hold durations, audio files
- Config-driven, no code changes needed to tune

**Layer 3: Orchestrators** (coaching_engine, mission_engine)
- Call interpreter.check_trick()
- Handle rewards, sessions, cooldowns

### API Endpoints Added:
- `GET /behavior/status` - Current detection state
- `GET /behavior/tricks` - All trick rules from config
- `POST /coaching/force_trick/{trick}` - Force specific trick for testing
- `POST /coaching/reset_cooldowns` - Reset per-dog cooldowns

### Unresolved Issues:
1. **Behavior detection inconsistency on fluffy dogs** - Model limitation, not code bug
2. **Temperature management** - Pi runs hot (79-82°C) during Hailo inference
3. **New architecture untested** - Created but not live-tested yet

### Next Steps:
1. Restart TreatBot to test new 3-layer architecture
2. Test coaching mode with new interpreter
3. Tune confidence thresholds in trick_rules.yaml for fluffy dogs
4. Consider adding temporal smoothing to reduce sit/cross flip-flopping

### Important Notes:
- Temperature-based mode forcing is DISABLED in safety.py for testing
- This should be re-enabled for production (restore line 330)
- Fluffy white Samoyeds are worst-case for pose estimation
