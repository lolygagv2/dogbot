# Architectural Decision Record

## ADR-001: Use Class-Based Architecture
- Status: ✅ **IMPLEMENTED**
- Context: Multiple procedural implementations exist
- Decision: Consolidate into TreatDispenserRobot + component classes
- Consequences: Better testability, modularity, maintainability
- Implementation: `core/treat_dispenser_robot.py` as main orchestrator

## ADR-002: YAML Configuration System
- Status: ✅ **IMPLEMENTED**
- Context: GPIO pins, thresholds hardcoded across files
- Decision: Extract to config/robot_config.yaml
- Consequences: Runtime reconfiguration without code changes
- Implementation: Full YAML config system with preserved pin/settings files

## ADR-003: Plugin-Based Detection Architecture
- Status: ✅ **IMPLEMENTED** (2025-09-27)
- Context: Multiple AI detection implementations (Hailo, OpenCV, MediaPipe)
- Decision: Create pluggable detection system with unified interface
- Consequences: Easy to add new detection backends, fallback capability
- Implementation: `core/vision/detection_plugins/` with BaseDetector interface

## ADR-004: Event-Driven Inter-Module Communication
- Status: ✅ **IMPLEMENTED** (2025-09-27)
- Context: Tight coupling between modules, difficulty in testing
- Decision: Implement event bus system for decoupled communication
- Consequences: Loose coupling, easier testing, better maintainability
- Implementation: `core/utils/event_bus.py` with pub/sub pattern

## ADR-005: Centralized State Management
- Status: ✅ **IMPLEMENTED** (2025-09-27)
- Context: State scattered across multiple files, no unified view
- Decision: Create centralized state manager with event notifications
- Consequences: Single source of truth, better monitoring, cleaner debugging
- Implementation: `core/utils/state_manager.py` with thread-safe state tracking

## ADR-006: Preserve Existing Hardware Controllers
- Status: ✅ **IMPLEMENTED** (2025-09-27)
- Context: Working hardware controllers in `core/` directory
- Decision: Preserve all existing hardware controllers without modification
- Consequences: Zero risk to proven functionality, faster integration
- Implementation: Existing controllers imported and used by orchestrator

## ADR-007: Unified Camera Interface
- Status: ✅ **IMPLEMENTED** (2025-09-27)
- Context: Multiple camera implementations (3 different interfaces)
- Decision: Create unified camera manager with backend abstraction
- Consequences: Single camera interface, supports multiple backends
- Implementation: `core/vision/camera_manager.py` with Picamera2/OpenCV support

## ADR-008: Modular Behavior Analysis
- Status: ✅ **IMPLEMENTED** (2025-09-27)
- Context: Behavior detection logic scattered across multiple files
- Decision: Consolidate into unified behavior analyzer with configurable parameters
- Consequences: Consistent behavior detection, easy tuning via config
- Implementation: `core/behavior/behavior_analyzer.py` with temporal analysis

## ADR-009: Automated Reward System
- Status: ✅ **IMPLEMENTED** (2025-09-27)
- Context: Manual treat dispensing, no cooldown management
- Decision: Create automated reward system with configurable rules
- Consequences: Consistent reward timing, prevents overfeeding
- Implementation: `core/behavior/reward_system.py` with cooldown tracking

## ADR-010: Visual Training Interface
- Status: ✅ **IMPLEMENTED** (2025-09-28)
- Context: Need for real-time visual feedback during training sessions
- Decision: Create GUI-based camera viewer with manual controls
- Consequences: Enables precise training control, visual confirmation of behavior
- Implementation: `camera_viewer.py` with tkinter GUI and OpenCV integration

## ADR-011: Graceful Hardware Degradation
- Status: ✅ **IMPLEMENTED** (2025-09-28)
- Context: Hardware initialization failures causing complete system crashes
- Decision: Implement graceful degradation allowing partial system operation
- Consequences: System remains functional even with hardware failures
- Implementation: Individual component error handling in `TreatDispenserRobot`

## ADR-012: Real-time Servo Control
- Status: ✅ **IMPLEMENTED** (2025-09-28)
- Context: Need for manual camera positioning during training sessions
- Decision: Extend manual servo control to work in tracking/training modes
- Consequences: Enables real-time camera adjustment without mode switching
- Implementation: Enhanced `manual_servo()` method with pan/tilt support

## ADR-013: Interactive Training Workflow
- Status: ✅ **IMPLEMENTED** (2025-09-28)
- Context: Training requires real-time feedback and manual intervention
- Decision: Create comprehensive training interface with visual confirmation
- Consequences: Enables systematic dog training with immediate feedback
- Implementation: Combined GUI controls, audio commands, and visual detection overlay

## ADR-014: Servo Range Calibration
- Status: ✅ **IMPLEMENTED** (2025-09-28)
- Context: Servo ranges were incorrect - limited pitch (120°) and inverted pan direction
- Decision: Correct servo ranges to pan: 30°-180° (inverted), pitch: 30°-150°
- Consequences: Full camera coverage including ceiling view, proper directional control
- Implementation: Updated `ServoController` ranges and corrected hardware mapping

## ADR-015: HailoRT Platform Integration
- Status: ✅ **IMPLEMENTED** (2025-09-28)
- Context: Hailo AI detection failing due to incorrect platform usage and firmware errors
- Decision: Use proper HailoRT platform API with correct resource management
- Consequences: Direct access to Hailo hardware, proper model loading, resolved context switch errors
- Implementation: Complete `HailoDetector` rewrite using `hailo_platform.pyhailort.pyhailort`
- Resolution: Fixed imports, API calls, resource cleanup, and bindings usage

## ADR-016: Debug and Calibration Tools
- Status: ✅ **IMPLEMENTED** (2025-09-28)
- Context: Need for systematic debugging when detection systems fail
- Decision: Create comprehensive debug and calibration tool suite
- Consequences: Faster troubleshooting, better system maintenance
- Implementation: `simple_camera_feed.py`, `debug_detection.py`, `test_opencv_detection.py`, `servo_calibration_tool.py`

## ADR-017: Hailo Resource Management Strategy
- Status: ✅ **IMPLEMENTED** (2025-09-28)
- Context: Hailo firmware context switch errors causing detection failures
- Decision: Implement per-frame configuration instead of persistent model configuration
- Consequences: Eliminates firmware control failures, ensures clean resource cleanup
- Implementation: Configure model for each inference, immediate shutdown after detection
- Result: Stable Hailo operation without context switch task status errors

## ADR-018: Live Detection Testing Framework
- Status: ✅ **IMPLEMENTED** (2025-09-28)
- Context: Need to validate actual dog detection performance with fixed Hailo pipeline
- Decision: Create GUI-less testing tools for real-world detection validation
- Consequences: Reliable testing without OpenCV display dependencies
- Implementation: `test_hailo_live_detection.py` and enhanced `camera_viewer_debug.py`
- Features: Frame-by-frame analysis, confidence threshold testing, auto-save detected frames

## ADR-019: Real-time Behavior Detection Integration
- Status: ✅ **IMPLEMENTED** (2025-10-02)
- Context: Need to extend from basic dog detection to behavior analysis for training applications
- Decision: Integrate bounding box aspect ratio analysis for real-time pose detection within AI controller
- Consequences: Enables real-time behavior classification without additional models or processing overhead
- Implementation: Enhanced `core/ai_controller.py` with behavior analysis methods
- Features: Sitting/standing/lying detection, movement analysis, behavior change notifications

## ADR-020: Enhanced Camera Viewer with AI Integration
- Status: ✅ **IMPLEMENTED** (2025-10-02)
- Context: Original camera viewer inadequate for AI detection visualization and robot control
- Decision: Create comprehensive camera viewer application with live AI detection, behavior analysis, and robot controls
- Consequences: Complete user interface for monitoring, testing, and controlling AI-powered robot
- Implementation: `camera_viewer_ai.py` with multi-threaded architecture
- Features: Live detection overlays, behavior alerts, servo controls, robot actions, status monitoring

## ADR-021: Robust Error Recovery Strategy
- Status: ✅ **IMPLEMENTED** (2025-10-02)
- Context: Camera and AI errors causing system crashes and infinite restart loops
- Decision: Implement graceful error handling with manual recovery options instead of automatic restarts
- Consequences: System remains stable under error conditions, user maintains control
- Implementation: Enhanced error handling in camera loop and detection threads
- Features: Error counting, graceful degradation, manual restart controls, status indicators

## ADR-022: Behavior Detection Algorithm Strategy
- Status: ✅ **IMPLEMENTED** (2025-10-02)
- Context: Need simple, efficient method for real-time behavior detection without additional AI models
- Decision: Use bounding box aspect ratio analysis combined with movement tracking for pose detection
- Consequences: Lightweight behavior detection that works with existing YOLO detection pipeline
- Implementation: Aspect ratio thresholds (lying <0.6, sitting 0.8-1.2, standing >1.3) + movement analysis
- Features: Real-time pose classification, behavior change detection, movement pattern analysis

## ADR-023: Adaptive Confidence Threshold Strategy
- Status: ✅ **IMPLEMENTED** (2025-10-02)
- Context: Detection instability caused by confidence scores oscillating around fixed threshold
- Decision: Implement confidence threshold hysteresis with high/low threshold values
- Consequences: Eliminates detection flickering, provides stable multi-dog detection continuity
- Implementation: High threshold (0.35) for new detections, low threshold (0.25) for maintaining existing detections
- Technical Details: `current_conf_threshold` switches between `conf_threshold_high` and `conf_threshold_low` based on detection continuity

## ADR-024: Temporal Smoothing for Detection Stability
- Status: ✅ **IMPLEMENTED** (2025-10-02)
- Context: Rapid on/off detection flickering degrading user experience and system reliability
- Decision: Implement temporal smoothing with detection history analysis across multiple frames
- Consequences: Smooth detection transitions, reduced false positives, improved detection reliability
- Implementation: 5-frame detection history buffer with continuity analysis
- Features: Detection history tracking, continuity frame counting, adaptive threshold adjustment based on recent detection patterns

## ADR-025: Multi-Class Dog Detection Strategy
- Status: ✅ **IMPLEMENTED** (2025-10-02)
- Context: Dogs detected across multiple YOLO class indices (2, 6, 7, 8, 9) with varying reliability
- Decision: Implement primary/secondary class prioritization system with class-specific confidence adjustment
- Consequences: More reliable dog detection across different poses and orientations
- Implementation: Primary classes {2} with 10% lower threshold, secondary classes {6, 7, 8, 9} for backup detection
- Technical Details: `is_primary_class` flag reduces threshold by factor of 0.9 for preferred classes

## ADR-026: Enhanced Behavior Classification Thresholds
- Status: ✅ **IMPLEMENTED** (2025-10-02)
- Context: Original behavior thresholds producing incorrect classifications and missing standing poses
- Decision: Revise behavior thresholds based on real-world detection data analysis
- Consequences: More accurate behavior classification, better standing detection, reduced false classifications
- Implementation: Updated thresholds - lying: 0.0-0.8, sitting: 0.7-1.4, standing: 1.3-3.0 aspect ratios
- Validation: Bounding box size validation (min 30x30, max 600x400) to filter unreasonable detections

## ADR-027: Automated Data Collection with Servo Control
- Status: ✅ **IMPLEMENTED** (2025-10-08)
- Context: Need comprehensive negative training data to address 97.7% false positive rate in pose model
- Decision: Integrate servo control into capture_negatives.py for automated pan/tilt sweeping
- Consequences: Complete room coverage, consistent data collection, reduced manual effort
- Implementation: ServoController integration with 10°-200° pan, 20°-150° pitch ranges
- Technical Details: 0.5s capture interval, configurable step sizes, metadata tracking of angles
- Result: 500 images captured in ~4 minutes covering all room angles systematically

## ADR-028: Headless Operation Support for Data Collection
- Status: ✅ **IMPLEMENTED** (2025-10-08)
- Context: OpenCV GUI not available on headless Raspberry Pi systems (SSH operation)
- Decision: Implement automatic GUI detection and conditional display logic
- Consequences: Script runs seamlessly in both GUI and headless environments
- Implementation: CV2_GUI_AVAILABLE flag with try/catch window creation test
- Features: Auto-disable preview, headless mode notifications, Ctrl+C termination
- Result: Full automation capability for remote/SSH operation without display requirements

## ADR-029: Servo Sweep Pattern for Comprehensive Coverage
- Status: ✅ **IMPLEMENTED** (2025-10-08)
- Context: Manual camera positioning misses angles and creates inconsistent datasets
- Decision: Implement systematic sweep pattern - horizontal pan first, then increment pitch
- Consequences: Uniform angular sampling, complete coverage, reproducible captures
- Implementation: Left-to-right pan (10°-200°), then pitch increment (20°-150°)
- Parameters: Configurable step sizes (default 10° pan, 20° pitch)
- Benefits: No blind spots, consistent intervals, metadata preserves exact angles for analysis
