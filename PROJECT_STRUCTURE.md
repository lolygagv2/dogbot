# TreatBot Project Structure

## Root Directory Files

### Main Entry Points
- `main.py` - Old main orchestrator (deprecated, replaced multiple times)
- `dogbot_main.py` - Another main version (deprecated)
- `test_mission_with_controls.py` - Latest mission testing with keyboard controls
- `test_white_pom_fixed.py` - White Pomeranian detection test
- `test_fixed_ai.py` - Fixed AI pipeline test

### Utility Scripts
- `dispense_treat.py` - Simple treat dispensing command
- `treat_loader.py` - Interactive carousel loading assistant
- `test_treat_servo.py` - Interactive servo testing menu
- `test_treat_servo_auto.py` - Automated servo test sequence

### GUI & Display
- `live_gui_detection.py` - Original real-time detection GUI
- `live_gui_detection_with_modes.py` - GUI with camera mode switching
- `live_gui_with_simple_modes.py` - Simplified GUI version
- `camera_viewer.py` - Basic camera viewer
- `debug_camera_view.py` - Debug camera display

### Debug Scripts (30+ files)
- `debug_*.py` - Various debugging utilities for detection, camera, AI
- `test_*.py` - Test scripts for different components

## Directory Structure

### `/core/` - Core System Components
**Main Controllers:**
- `ai_controller.py` - Original AI controller
- `ai_controller_3stage.py` - 3-stage pipeline (broken version)
- `ai_controller_3stage_fixed.py` - WORKING 3-stage pipeline
- `treat_dispenser_robot.py` - Unified robot orchestrator (one version)
- `camera_mode_controller.py` - Camera mode switching system
- `camera_positioning_system.py` - Camera servo positioning

**`/core/hardware/` - Hardware Controllers (WORKING)**
- `motor_controller.py` - DC motor control via L298N
- `servo_controller.py` - PCA9685 servo control (pan/tilt/carousel)
- `audio_controller.py` - DFPlayer Pro + relay switching
- `led_controller.py` - NeoPixel ring + blue LED control

**`/core/behavior/` - Behavior Analysis**
- `behavior_analyzer.py` - Analyzes poses to detect behaviors
- `reward_system.py` - Manages treat dispensing logic

**`/core/vision/` - Vision System**
- `camera_manager.py` - Camera management
- `/detection_plugins/` - Detection backend plugins (Hailo, OpenCV, ArUco)

**`/core/utils/` - Utilities**
- `event_bus.py` - Inter-module communication
- `state_manager.py` - System state management

### `/config/` - Configuration Files
- `robot_config.yaml` - Main configuration file
- `pins.py` - GPIO pin definitions
- `settings.py` - System settings and constants

### `/missions/` - Mission System
- Mission definition files (to be created)
- Training sequences

### `/tests/` - Test Files
- `/hardware/` - Hardware component tests
  - `leds_v3.py` - LED patterns and animations
  - `test_servo_pca9685.py` - Servo testing

### `/ai/` - AI Models & Detection
- `dog_detection.py` - Basic dog detection
- `enhanced_dog_detection.py` - Enhanced detection
- `/models/` - Model implementations
  - `HailoDetectionYolo.py` - Hailo YOLO implementation

### `/models/` - HEF Model Files
- `dogdetector_14.hef` - Dog detection model (4.3MB)
- `dogpose_14.hef` - Pose detection model (18.8MB)

### `/data/` - Data Storage
- Detection results
- Training data
- Logs

### `/docs/` - Documentation
- `IR_DOCKING_SYSTEM.md` - Docking system design

### `/.claude/` - Claude-specific Files
- `resume_chat.md` - Session history
- `product_roadmap.md` - Product roadmap
- `development_todos.md` - Development tasks
- `/commands/` - Session management commands
  - `session_start.md` - Session initialization
  - `session_end.md` - Session cleanup

### Directories to Ignore
- `/Unused/` - Old/deprecated code
- `/env_new/`, `/env/` - Python virtual environments
- `/hailoRTsuite/`, `/hailov2/` - Hailo SDK files
- `/__pycache__/` - Python cache files
- `/detection_results/`, `/debug_*/` - Test output directories

## Key Working Components

### ✅ VERIFIED WORKING
1. **3-Stage AI Pipeline** - `core/ai_controller_3stage_fixed.py`
2. **Treat Dispenser** - Carousel servo on channel 2
3. **Hardware Controllers** - All in `/core/hardware/`
4. **Detection Models** - Both HEF models working

### 🔧 PARTIALLY INTEGRATED
1. **Mission System** - Framework exists, needs unified API
2. **LED Celebrations** - Patterns exist, not connected to rewards
3. **Camera Modes** - System built, needs integration

### ❌ NOT INTEGRATED
1. **Web API/Dashboard**
2. **Multi-dog tracking**
3. **Docking system**
4. **Unified celebration sequence**

## Problem Areas

### Multiple Main Files
- `main.py` (old)
- `dogbot_main.py` (old)
- `core/treat_dispenser_robot.py` (current?)
- Need ONE definitive main orchestrator

### Scattered Test Files
- 30+ debug scripts in root
- Test files mixed with production code
- Need organized `/tests/` structure

### Incomplete Integration
- Components work individually
- Not wired together into cohesive system
- Mission system design unclear