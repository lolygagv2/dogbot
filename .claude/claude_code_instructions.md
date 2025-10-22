# TreatBot Implementation Instructions for Claude Code

## 🎯 Project Context
You are implementing TreatBot - an autonomous AI-powered dog training robot. The hardware is working individually but needs integration into a cohesive system. The core AI pipeline (3-stage detection → pose → behavior) is functional. Hardware controllers (motors, servos, audio, LEDs) work independently. The mission is to create a unified orchestration layer.

## ⚠️ CRITICAL: Preserve Working Code
**DO NOT MODIFY OR REPLACE these working files:**
- `/core/ai_controller_3stage_fixed.py` - The ONLY working AI pipeline
- `/core/hardware/*` - ALL hardware controllers are verified working
- `/models/*.hef` - Both Hailo models are working

## 📁 Current Project Structure

### Working Components (DO NOT BREAK)
```
/core/
  ai_controller_3stage_fixed.py  ✅ 3-stage AI pipeline (WORKING)
  /hardware/                     ✅ All controllers WORKING
    motor_controller.py          - DC motors via L298N  
    servo_controller.py          - PCA9685 servos (pan/tilt/carousel)
    audio_controller.py          - DFPlayer Pro + relay
    led_controller.py            - NeoPixel + blue LED

/models/
  dogdetector_14.hef            ✅ Dog detection (640x640)
  dogpose_14.hef                ✅ Pose estimation (640x640)
```

### Files Needing Integration
```
camera_mode_controller.py        - Mode switching (needs integration)
camera_positioning_system.py     - Pan/tilt tracking (needs integration)
test_mission_with_controls.py    - Testing framework (reference only)
```

## 🏗️ Implementation Plan

### Phase 1: Core Infrastructure
Create the foundational event bus and state management system that all components will use.

**Files to create:**
```python
# /core/bus.py
"""
Event bus for inter-component communication
- Implement publish/subscribe pattern
- Thread-safe message passing
- Event types: VisionEvent, AudioEvent, MotionEvent, SystemEvent
"""

# /core/state.py
"""
Global state manager
- Track system mode (IDLE, DETECTION, VIGILANT, PHOTOGRAPHY)
- Track mission state (active mission, progress)
- Track hardware state (battery, temperature)
- Thread-safe state updates
"""

# /core/store.py
"""
SQLite data persistence
Tables:
- events(id, timestamp, type, payload_json)
- dogs(id, name, profile_json, last_seen)
- rewards(id, timestamp, dog_id, behavior, success)
- telemetry(timestamp, battery_pct, temperature, mode)
"""

# /core/safety.py
"""
Safety monitoring and emergency stops
- Monitor battery voltage (shutdown at 10%)
- Monitor temperature (pause at >70°C)
- Emergency stop handler
- Graceful shutdown sequence
"""
```

### Phase 2: Service Layer
Wrap existing hardware controllers with service interfaces that use the event bus.

**Files to create:**
```python
# /services/perception/detector.py
"""
Wrapper for ai_controller_3stage_fixed.py
- Import AI3StageControllerFixed
- Publish pose events to bus
- Handle mode-based pipeline switching
"""

# /services/perception/pose.py
"""
Process pose keypoints from detector
- Track pose stability (10+ consecutive frames)
- Identify behaviors: sit, down, stand, spin
- Publish VisionEvent.Pose to bus
"""

# /services/motion/pan_tilt.py
"""
Camera tracking using servo_controller
- Import existing ServoController
- Implement PID tracking of detected dogs
- Center dog in frame during DETECTION mode
- Scan pattern for VIGILANT mode
"""

# /services/audio/bark.py
"""
Basic bark detection (placeholder for now)
- Monitor audio amplitude
- Detect bark patterns
- Publish AudioEvent.Bark to bus
"""

# /services/media/sfx.py
"""
Sound effect player using audio_controller
- Import existing AudioController
- Play celebration sounds
- Handle voice commands
- Cache frequently used sounds
"""

# /services/reward/dispenser.py
"""
Treat dispensing service
- Import ServoController for carousel
- Track treats dispensed per dog
- Implement portion control
- Log all dispenses to store
"""
```

### Phase 3: Orchestration Layer
Create the mission engine and reward logic that coordinates services.

**Files to create:**
```python
# /orchestrators/sequence_engine.py
"""
Execute celebration sequences
- Parse sequence YAML files
- Coordinate LED, audio, treat timing
- Handle sequence interruption
- Log sequence completion
"""

# /orchestrators/mission_engine.py
"""
Mission state machine
- Load mission JSON files
- Track mission progress
- Handle mission transitions
- Enforce daily limits
"""

# /orchestrators/mode_fsm.py
"""
Camera mode state machine
- IDLE → DETECTION (on motion)
- DETECTION → VIGILANT (no motion 3s)
- Any → PHOTOGRAPHY (user override)
- Enforce single pipeline rule
"""

# /orchestrators/reward_logic.py
"""
Reward decision engine
- Load reward policies from YAML
- Check behavior + quiet time
- Implement variable ratio schedules
- Track cooldowns between rewards
"""

# /orchestrators/pipeline_manager.py
"""
Manage AI pipeline switching
- Coordinate with mode_fsm
- Start/stop AI pipeline based on mode
- Handle pipeline errors gracefully
"""
```

### Phase 4: Configuration Files
Create the configuration structure for missions, sequences, and policies.

**Files to create:**
```yaml
# /configs/modes.yaml
modes:
  photography:
    resolution: [4056, 3040]
    fps: 10
    ai_enabled: false
    manual_controls: true
    
  ai_detection:
    resolution: [640, 640]
    fps: 30
    ai_enabled: true
    single_frame: true
    
  vigilant:
    resolution: [1920, 1080]
    fps: 15
    ai_enabled: true
    tiling: true
    tiles: 4

# /configs/policies/reward.yaml
policies:
  default:
    behaviors:
      sit:
        min_duration: 10  # seconds
        require_quiet: true
        cooldown: 20  # seconds between rewards
        treat_probability: 0.6  # variable ratio
        sounds: ["good_dog.mp3", "well_done.mp3"]
        led_pattern: "rainbow"
        
      down:
        min_duration: 15
        require_quiet: true
        cooldown: 30
        treat_probability: 0.5
        sounds: ["excellent.mp3"]
        led_pattern: "pulse_green"

# /configs/sequences/celebrate.yaml
name: celebrate
steps:
  - type: parallel
    actions:
      - service: motion
        command: stop
      - service: audio
        command: play
        params:
          sound: "good_dog.mp3"
      - service: led
        command: pattern
        params:
          pattern: "rainbow"
          duration: 3
  - type: wait
    duration: 0.5
  - type: action
    service: treat
    command: dispense
    params:
      count: 1
  - type: wait
    duration: 2
  - type: action
    service: led
    command: "off"

# /missions/train_sit_daily.json
{
  "name": "train_sit_daily",
  "description": "Daily sit training with limits",
  "enabled": true,
  "schedule": "daily",
  "max_rewards": 5,
  "duration_minutes": 30,
  "stages": [
    {
      "name": "detect_dog",
      "timeout": 60,
      "success_event": "VisionEvent.DogDetected"
    },
    {
      "name": "wait_for_sit",
      "timeout": 120,
      "success_event": "VisionEvent.Pose.Sit",
      "min_duration": 10
    },
    {
      "name": "reward",
      "sequence": "celebrate",
      "cooldown": 20
    }
  ],
  "completion": {
    "on_success": "log_training",
    "on_max_rewards": "end_mission",
    "on_timeout": "pause_and_retry"
  }
}
```

### Phase 5: Main Orchestrator
Create the unified main entry point that starts all systems.

**File to create:**
```python
# /main_treatbot.py
"""
Main TreatBot orchestrator - THE definitive entry point

Startup sequence:
1. Initialize safety systems
2. Load configurations
3. Initialize hardware (servos, motors, audio, LEDs)
4. Initialize services (perception, motion, audio, rewards)
5. Start event bus
6. Initialize orchestrators (mission, sequence, mode, reward)
7. Load default mission
8. Start main loop

Main loop:
- Monitor system health
- Process event queue
- Update telemetry
- Check for mode changes
- Handle shutdown signals

Shutdown sequence:
1. Stop active missions
2. Safe position servos
3. Turn off LEDs
4. Stop motors
5. Flush data to store
6. Graceful exit
"""
```

### Phase 6: API Layer (Basic)
Create minimal API for testing and monitoring.

**Files to create:**
```python
# /api/server.py
"""
FastAPI server for REST endpoints
Endpoints:
  POST /mode/set
  GET /mode
  POST /missions/start
  POST /missions/stop
  GET /missions/status
  POST /treat/dispense
  GET /telemetry
  GET /events/recent
"""

# /api/ws.py
"""
WebSocket server for real-time updates
Channels:
  /ws/telemetry - Battery, temperature, mode
  /ws/detections - Dog detections and poses
  /ws/events - System events
"""
```

## 🚀 Implementation Order

1. **Start with Phase 1** - Create core infrastructure (bus, state, store, safety)
2. **Test Phase 1** - Ensure event bus and state management work
3. **Implement Phase 2** - Wrap existing hardware with services
4. **Test Phase 2** - Verify services can control hardware via bus
5. **Implement Phase 3** - Create orchestrators
6. **Add Phase 4** - Configuration files
7. **Create Phase 5** - Main orchestrator
8. **Test full system** - Run basic mission
9. **Add Phase 6** - API for monitoring

## 🔧 Integration Notes

### Using Existing Hardware Controllers
```python
# Example: Wrapping servo controller
from core.hardware.servo_controller import ServoController

class PanTiltService:
    def __init__(self, bus):
        self.servo = ServoController()
        self.servo.initialize()
        self.bus = bus
        
    def track_dog(self, detection):
        # Use existing servo.set_angle() methods
        center_x = detection.center[0]
        # Calculate servo adjustment
        self.servo.set_angle(0, new_pan_angle)
```

### Using Working AI Pipeline
```python
# Example: Wrapping AI controller
from core.ai_controller_3stage_fixed import AI3StageControllerFixed

class DetectorService:
    def __init__(self, bus):
        self.ai = AI3StageControllerFixed()
        self.ai.initialize()
        self.bus = bus
        
    def process_frame(self, frame):
        # Use existing AI pipeline
        dogs, poses, behaviors = self.ai.process_frame(frame)
        
        # Publish to bus
        if behaviors:
            self.bus.publish(VisionEvent.Behavior(behaviors[0]))
```

### Mission System Integration
- Missions are JSON files defining training sequences
- Mission engine reads JSON and manages state transitions
- Each mission stage waits for specific events from the bus
- Rewards are triggered through the sequence engine

### Camera Mode Switching
- Mode FSM monitors motion from motor_controller
- Pipeline manager switches AI on/off based on mode
- Camera resolution changes handled by Picamera2 reconfiguration
- Only one pipeline active at a time (enforced by pipeline_manager)

## 📋 Testing Checkpoints

### Checkpoint 1: Core Systems
- [ ] Event bus can publish/subscribe
- [ ] State manager tracks mode changes
- [ ] SQLite store saves events
- [ ] Safety system monitors battery

### Checkpoint 2: Hardware Services
- [ ] LED service creates patterns
- [ ] Audio service plays sounds
- [ ] Servo service moves camera
- [ ] Treat service dispenses

### Checkpoint 3: AI Integration
- [ ] Detector publishes pose events
- [ ] Behavior detection works (sit, down, stand)
- [ ] Mode switching changes pipeline

### Checkpoint 4: Mission System
- [ ] Sequence engine runs celebrate.yaml
- [ ] Mission engine tracks progress
- [ ] Reward logic enforces cooldowns
- [ ] Daily limits enforced

### Checkpoint 5: Full Integration
- [ ] Dog sits → celebrate sequence triggers
- [ ] Mission completes after 5 rewards
- [ ] API reports system status
- [ ] WebSocket streams events

## 🎯 Success Criteria

The system is complete when:
1. TreatBot autonomously detects a dog sitting
2. Waits for 10 seconds of quiet sitting
3. Triggers the celebration sequence (lights + sound + treat)
4. Logs the event to the database
5. Enforces cooldown before next reward
6. Stops after 5 rewards per day
7. Can be monitored via API

## 💡 Important Implementation Tips

1. **Don't break what works** - The AI pipeline and hardware controllers are functional. Wrap them, don't rewrite them.

2. **Use the event bus** - All communication between services should go through the bus, not direct calls.

3. **Handle failures gracefully** - Each service should handle hardware failures without crashing the system.

4. **Log everything** - Use the store to log all events for debugging and analytics.

5. **Test incrementally** - Test each phase before moving to the next.

6. **Keep it modular** - Each service should be independent and testable.

7. **Follow the architecture** - Maintain separation between HAL, Services, Orchestrators, and API.

## 🔍 Debugging Helpers

Existing debug scripts to reference:
- `test_mission_with_controls.py` - Shows how to use AI controller
- `live_gui_detection.py` - Shows camera integration
- `test_treat_servo.py` - Shows servo control

Log locations:
- System logs: `/var/log/treatbot/`
- Event database: `/data/treatbot.db`
- Debug images: `/tmp/treatbot/`

## 🚨 Common Pitfalls to Avoid

1. **Don't create multiple AI pipelines** - Use only AI3StageControllerFixed
2. **Don't hardcode GPIO pins** - Use the existing hardware controllers
3. **Don't block the event bus** - Keep handlers fast, offload to threads
4. **Don't skip safety checks** - Always check battery and temperature
5. **Don't ignore existing code** - Build on what works

## 🎬 Final Notes

This is a complex system but the hard parts (AI, hardware control) are done. Your job is to create the orchestration layer that brings it all together. Start with the event bus and state management, then gradually add services and orchestrators. Test frequently and incrementally. The goal is a working autonomous training robot, not perfect code. Good luck!
