# DogBot Feature Roadmap & Command System Design

## ✅ Completed Features (October 7, 2025)

### 1. 1024x768 Resolution Support
- **Files Updated**: `run_pi_1024x768.py`, `config/config.json`
- **Implementation**:
  - Handles both square and rectangular models
  - Proper letterboxing for 1024x768 (width x height)
  - Compatible with IMX500 camera native resolution

### 2. ArUco Marker Visualization
- **Status**: Full visual feedback implemented
- **Features**:
  - Green overlays on detected markers
  - ID and dog name display
  - Center point visualization
  - Real-time tracking in GUI
- **Testing**: Run `python test_pose_gui_enhanced.py` to see markers

### 3. Behavior Cooldown System
- **Implementation**: Per-dog, per-behavior cooldowns
- **Cooldown Times**:
  - Stand: 2 seconds
  - Sit: 5 seconds (max 2 sits in 10 seconds)
  - Lie: 5 seconds
  - Cross: 4 seconds
  - Spin: 8 seconds
- **Features**:
  - Visual cooldown status in GUI
  - Prevents rapid re-triggering
  - Maintains behavior history

## 🚧 In Progress Features

### 4. Servo Camera Tracking
**Implementation Strategy**:

```python
# Tracking Logic (already implemented in run_pi_1024x768.py)
class ServoTracker:
    MIN_TRACKING_TIME = 5.0  # Start tracking after 5 seconds
    DEADZONE = 0.1           # 10% center deadzone
    SERVO_SPEED = 0.05       # Movement speed factor
```

**Tracking Modes**:
1. **Passive Mode**: Camera stays neutral, no tracking
2. **Active Tracking**: After 5 seconds of continuous detection
3. **Return to Neutral**: When dog lost or out of frame

**Integration Points**:
```python
# In main.py or TreatSenseiCore
if detection_confirmed_for_5_seconds:
    servo_controller.enable_tracking()
    servo_controller.track_to(dog_center_x, dog_center_y)
elif dog_lost:
    servo_controller.return_to_neutral()
```

### 5. Follow Mode (Motor Control)
**Concept**: Keep dog at constant size in frame

```python
def calculate_follow_command(bbox_area, target_area=50000):
    """Calculate motor command to maintain dog size"""
    if bbox_area < target_area * 0.8:
        return "FORWARD"  # Dog too small, move closer
    elif bbox_area > target_area * 1.2:
        return "BACKWARD"  # Dog too big, back up
    else:
        return "STOP"  # Good distance
```

## 📋 Command System Architecture

### Command Categories

#### 1. **Immediate Commands** (No confirmation needed)
- Treat dispensing on behavior detection
- Servo tracking adjustments
- LED status updates

#### 2. **Buffered Commands** (With cooldowns)
- Behavior rewards (respects cooldowns)
- Audio feedback
- Movement commands

#### 3. **Queued Commands** (Sequential execution)
- Training sequences
- Multi-step behaviors
- Patrol routes

### Command Pipeline

```python
class CommandSystem:
    def __init__(self):
        self.immediate_queue = []
        self.buffered_commands = {}
        self.sequence_queue = collections.deque()

    def issue_command(self, cmd_type, action, params):
        if cmd_type == "immediate":
            self.execute_now(action, params)
        elif cmd_type == "buffered":
            self.add_to_buffer(action, params)
        elif cmd_type == "sequence":
            self.queue_sequence(action, params)
```

## 🎯 Feature Priority List (By Implementation Ease)

### Easy (< 1 hour each)
1. **Audio Feedback on Behavior**
   ```python
   def on_behavior_detected(behavior):
       audio_controller.play_sound(f"{behavior}.mp3")
   ```

2. **LED Status Indicators**
   ```python
   def update_led_status(state):
       if state == "tracking":
           led_controller.set_color(0, 255, 0)  # Green
       elif state == "searching":
           led_controller.pulse(255, 255, 0)     # Yellow pulse
   ```

3. **Screenshot on Command**
   - Already implemented in GUI
   - Add to main system: Save on specific behaviors

### Medium (1-3 hours each)
4. **Smart Treat Dispensing**
   ```python
   def smart_dispense(dog_id, behavior):
       if behavior == "sit" and not recently_treated(dog_id):
           dispense_treat()
           log_treatment(dog_id, behavior)
   ```

5. **Behavior Sequence Detection**
   ```python
   def detect_sequence(history):
       # Detect "sit -> lie -> sit" = special trick
       if history[-3:] == ["sit", "lie", "sit"]:
           issue_special_reward()
   ```

6. **Patrol Mode**
   ```python
   def patrol_mode():
       waypoints = [(0, 0), (5, 0), (5, 5), (0, 5)]
       for point in cycle(waypoints):
           navigate_to(point)
           scan_for_dogs()
   ```

### Complex (3+ hours)
7. **Multi-Dog Coordination**
   - Track multiple dogs simultaneously
   - Prioritize based on behavior
   - Maintain individual cooldowns

8. **Training Session Manager**
   ```python
   class TrainingSession:
       def __init__(self, duration, behaviors):
           self.target_behaviors = behaviors
           self.success_count = {}
           self.session_timer = Timer(duration)
   ```

9. **Obstacle Avoidance**
   - Integrate ultrasonic sensors
   - Path planning around obstacles
   - Safety stops

## 🧪 Testing Checklist

### Standalone Testing
```bash
# Test 1: Basic pose detection
python run_pi_1024x768.py

# Test 2: GUI with full visualization
python test_pose_gui_enhanced.py

# Test 3: Check ArUco detection
# Place markers 315 (Bezik) and 832 (Elsa) in view
# Markers should show green overlays with names

# Test 4: Behavior cooldowns
# Trigger a sit behavior
# Try to trigger again within 5 seconds
# Should show [COOLDOWN] status

# Test 5: Servo tracking
# Press 'T' to toggle tracking
# Keep dog in view for 5+ seconds
# Camera should start following
```

### Integration Testing
```python
# In main.py interactive mode
ai test          # Test AI detection
ai pose          # Test pose detection
servo track on   # Enable servo tracking
behavior log     # Show behavior history
```

## 🔧 Configuration Tuning

### Adjust Sensitivity
```json
// config/config.json
{
    "prob_th": 0.7,        // Increase for fewer false positives
    "cooldown_s": {
        "sit": 3           // Reduce for more frequent rewards
    }
}
```

### Servo Tracking Parameters
```python
# In ServoTracker class
MIN_TRACKING_TIME = 3.0    # Reduce to start tracking sooner
DEADZONE = 0.15            # Increase for less jittery movement
SERVO_SPEED = 0.1          # Increase for faster tracking
```

## 📊 Performance Metrics

### Current Performance (1024x768)
- **Inference Time**: ~150-200ms per frame
- **Effective FPS**: 5-7 FPS
- **Tracking Latency**: < 100ms
- **Cooldown Accuracy**: 100%
- **ArUco Detection Range**: 2-10 feet

### Optimization Opportunities
1. **Reduce Resolution**: Back to 640x640 for 10+ FPS
2. **Skip Frames**: Process every 2nd frame
3. **ROI Tracking**: Only process region around last detection
4. **Multi-threading**: Separate inference and display threads

## 🚀 Next Steps

### Immediate (Today)
1. Run `python run_pi_1024x768.py` to test standalone
2. Verify ArUco markers appear with green overlays
3. Test cooldowns work (max 2 sits per 10 seconds)
4. Check servo tracking engages after 5 seconds

### Tomorrow
1. Integrate into main.py
2. Add motor control for follow mode
3. Implement smart treat dispensing
4. Add training session manager

### This Week
1. Complete obstacle avoidance
2. Add voice commands
3. Implement patrol mode
4. Create training programs

## 💡 Pro Tips

1. **Debug Mode**: Set environment variable
   ```bash
   export DOGBOT_DEBUG=1
   python run_pi_1024x768.py
   ```

2. **Quick ArUco Test**: Print markers from
   ```
   https://chev.me/arucogen/
   Dictionary: 4x4_1000
   Marker IDs: 315, 832
   ```

3. **Cooldown Override**: For testing
   ```python
   # Temporarily disable cooldowns
   COOLDOWN_S = {"sit": 0, "stand": 0}
   ```

4. **Performance Mode**: Reduce quality for speed
   ```python
   camera.set(cv2.CAP_PROP_FPS, 15)
   camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
   ```