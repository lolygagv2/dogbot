#!/usr/bin/env python3
"""
Detection service wrapper for ai_controller_3stage_fixed.py
Publishes pose events to event bus
"""

import threading
import time
import logging
import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple

from core.bus import get_bus, publish_vision_event
from core.state import get_state, SystemMode
from core.store import get_store

# Mode-based resolution configuration
# AI modes need 640x640 for Hailo inference
# Non-AI modes can use higher resolution for better video quality
MODE_RESOLUTIONS = {
    SystemMode.IDLE: (1920, 1080),           # No AI, full HD video
    SystemMode.MANUAL: (1920, 1080),          # No AI, full HD video
    SystemMode.SILENT_GUARDIAN: (640, 640),  # AI active
    SystemMode.COACH: (640, 640),            # AI active
    SystemMode.PHOTOGRAPHY: (1920, 1080),     # High res photos
    SystemMode.EMERGENCY: (640, 640),        # Safety mode
    SystemMode.SHUTDOWN: (640, 640),         # Shutting down
}
DEFAULT_RESOLUTION = (640, 640)
from core.ai_controller_3stage_fixed import AI3StageControllerFixed
from config.config_loader import get_config

# Camera imports
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# ArUco detection
ARUCO_DICT = cv2.aruco.DICT_4X4_1000
DOG_MARKERS = {
    315: 'elsa',
    832: 'bezik'
}


class DetectorService:
    """
    Wrapper for AI3StageControllerFixed that publishes to event bus
    Handles mode-based pipeline switching
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.store = get_store()
        self.logger = logging.getLogger('DetectorService')

        # AI Controller
        self.ai = None
        self.ai_initialized = False

        # Camera
        self.camera = None
        self.camera_initialized = False

        # ArUco detector
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # Detection state
        self.running = False
        self.detection_thread = None
        self._stop_event = threading.Event()

        # Camera pause state (for MANUAL mode)
        self._camera_paused = False
        self._last_frame = None  # Store last frame for snapshots
        self._last_frame_time = 0
        self._frame_lock = threading.Lock()

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

        # Rate limiting for detection events (prevent log spam)
        self._last_detection_event_time = 0
        self._detection_event_interval = 5.0  # Minimum seconds between detection events

        # Camera retry logic
        self._camera_error_reason = None
        self._camera_retry_interval = 60  # seconds between retry attempts
        self._last_camera_retry_time = 0

        # Resolution tracking
        self._current_resolution = DEFAULT_RESOLUTION
        self._resolution_change_lock = threading.Lock()
        self._resolution_change_in_progress = False

        # Subscribe to mode changes for resolution switching
        self.state.subscribe('mode_change', self._on_mode_change)

    def initialize(self) -> bool:
        """Initialize AI controller and camera

        Returns True if camera works (for WebRTC), even if AI/Hailo fails.
        AI is optional - WebRTC streaming works without it.
        """
        try:
            # Initialize camera first - this is required for WebRTC
            self._initialize_camera()

            # Initialize AI controller (optional - Hailo may not be available)
            try:
                self.ai = AI3StageControllerFixed()
                success = self.ai.initialize()

                if success:
                    self.ai_initialized = True
                    self.logger.info("AI controller initialized (full pipeline)")
                else:
                    self.logger.warning("AI controller init failed - WebRTC will work, no AI processing")
            except Exception as e:
                self.logger.warning(f"AI controller not available: {e} - WebRTC will work, no AI processing")

            self.state.update_hardware(camera_initialized=self.camera_initialized)

            # Return True if camera works - AI is optional
            if self.camera_initialized:
                return True
            else:
                self.logger.error("Camera initialization failed - no video available")
                return False

        except Exception as e:
            self.logger.error(f"Detector initialization error: {e}")
            return False

    def _initialize_camera(self) -> bool:
        """Initialize camera for frame capture"""
        try:
            if PICAMERA_AVAILABLE:
                self.logger.info("Initializing Picamera2 for detection...")

                # Check if any cameras are available BEFORE creating Picamera2
                available_cameras = Picamera2.global_camera_info()
                if not available_cameras:
                    self.logger.error(
                        "No cameras detected! CSI cameras require a system reboot after "
                        "plugging in. Check ribbon cable connection and run: sudo reboot"
                    )
                    self._camera_error_reason = "no_cameras_detected"
                    return False

                self.logger.info(f"Detected cameras: {available_cameras}")

                # Reset any stale camera instances
                try:
                    import subprocess
                    # Kill any stale libcamera processes that might be holding the camera
                    subprocess.run(['pkill', '-f', 'libcamera'], timeout=2, capture_output=True)
                    time.sleep(0.5)
                except Exception:
                    pass

                self.camera = Picamera2()

                # Use mode-based resolution
                current_mode = self.state.get_mode()
                target_resolution = MODE_RESOLUTIONS.get(current_mode, DEFAULT_RESOLUTION)
                self._current_resolution = target_resolution
                self.logger.info(f"Camera resolution for {current_mode.value}: {target_resolution}")

                config = self.camera.create_preview_configuration(
                    main={"size": target_resolution, "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()

                # Wait for camera to stabilize and verify it's actually working
                time.sleep(0.5)
                try:
                    # Test capture to verify camera is working
                    test_frame = self.camera.capture_array()
                    if test_frame is None:
                        self.logger.error("Camera test capture returned None")
                        return False
                    self.logger.info(f"Camera test capture successful: {test_frame.shape}")
                except Exception as e:
                    self.logger.error(f"Camera test capture failed: {e}")
                    return False

                self.camera_initialized = True
                self.logger.info("Picamera2 initialized for detection")
                return True
            else:
                self.logger.warning("Picamera2 not available, detection disabled")
                return False
        except IndexError as e:
            self.logger.error(
                f"Camera initialization error: {e} - No cameras found. "
                "CSI cameras require a reboot after connecting the ribbon cable."
            )
            self._camera_error_reason = "no_cameras_detected"
            return False
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            self._camera_error_reason = str(e)
            return False

    def _capture_frame(self):
        """Capture a frame from the camera"""
        if not self.camera_initialized or self.camera is None:
            self.logger.debug("Camera not initialized or None")
            return None
        try:
            frame = self.camera.capture_array()

            # Apply rotation from robot config (0, 90, 180, 270 degrees clockwise)
            config = get_config()
            rotation = config.camera.rotation
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # rotation == 0: no rotation needed

            # Store last frame for snapshots (thread-safe)
            with self._frame_lock:
                self._last_frame = frame.copy()
                self._last_frame_time = time.time()

            return frame
        except Exception as e:
            self.logger.error(f"Frame capture error: {e}")
            # Try to recover camera on repeated errors
            self._camera_error_count = getattr(self, '_camera_error_count', 0) + 1
            if self._camera_error_count >= 5:
                self.logger.warning("Multiple camera errors, attempting to reinitialize...")
                self._reinitialize_camera()
                self._camera_error_count = 0
            return None

    def _reinitialize_camera(self):
        """Attempt to reinitialize camera after errors"""
        try:
            # Close existing camera
            if self.camera:
                try:
                    self.camera.stop()
                    self.camera.close()
                except Exception as e:
                    self.logger.warning(f"Error closing camera during reinit: {e}")
                self.camera = None
                self.camera_initialized = False

            # Wait for resources to be released
            time.sleep(0.5)

            # Reinitialize
            if self._initialize_camera():
                self.logger.info("Camera reinitialized successfully")
            else:
                self.logger.error("Camera reinitialization failed")
        except Exception as e:
            self.logger.error(f"Camera reinitialization error: {e}")

    def _release_camera(self) -> None:
        """Release camera for external use (e.g., rpicam-still in MANUAL mode)"""
        if self.camera is not None and self.camera_initialized:
            try:
                self.camera.stop()
                self.camera.close()
                self.camera = None
                self.camera_initialized = False
                self._camera_paused = True
                self.logger.info("📷 Camera released for MANUAL mode")
            except Exception as e:
                self.logger.error(f"Camera release error: {e}")

    def _reacquire_camera(self) -> bool:
        """Re-acquire camera after MANUAL mode"""
        if self._camera_paused:
            self.logger.info("📷 Re-acquiring camera from MANUAL mode...")
            result = self._initialize_camera()
            if result:
                self._camera_paused = False
                self.logger.info("📷 Camera re-acquired successfully")
            return result
        return self.camera_initialized

    def _on_mode_change(self, data: Dict[str, Any]) -> None:
        """Handle mode change for resolution switching"""
        try:
            new_mode_str = data.get('new_mode', '')
            previous_mode_str = data.get('previous_mode', '')

            # Convert string to SystemMode enum
            try:
                new_mode = SystemMode(new_mode_str)
            except ValueError:
                self.logger.warning(f"Unknown mode: {new_mode_str}")
                return

            # Get target resolution for new mode
            target_resolution = MODE_RESOLUTIONS.get(new_mode, DEFAULT_RESOLUTION)

            # Only change if resolution differs
            if target_resolution != self._current_resolution:
                self.logger.info(
                    f"📹 Mode change {previous_mode_str} → {new_mode_str}: "
                    f"Resolution {self._current_resolution} → {target_resolution}"
                )
                self._change_resolution(target_resolution)
            else:
                self.logger.debug(f"Mode change to {new_mode_str}, resolution unchanged at {target_resolution}")

        except Exception as e:
            self.logger.error(f"Mode change handler error: {e}")

    def _change_resolution(self, new_resolution: tuple) -> bool:
        """Safely change camera resolution

        Thread-safe method that:
        1. Stops camera capture
        2. Reconfigures with new resolution
        3. Restarts camera

        Returns True if successful, False otherwise.
        Falls back to previous resolution on failure.
        """
        with self._resolution_change_lock:
            if self._resolution_change_in_progress:
                self.logger.warning("Resolution change already in progress, skipping")
                return False

            self._resolution_change_in_progress = True

        old_resolution = self._current_resolution

        try:
            if not self.camera_initialized or self.camera is None:
                # Camera not initialized, just update target resolution
                self._current_resolution = new_resolution
                self.logger.info(f"📹 Resolution set to {new_resolution} (camera not active)")
                return True

            self.logger.info(f"📹 Changing resolution from {old_resolution} to {new_resolution}...")

            # Pause WebRTC video track before camera reconfig to prevent stale reads
            publish_vision_event('webrtc_pause_requested', {
                'reason': 'camera_reconfig',
                'old_resolution': old_resolution,
                'new_resolution': new_resolution
            })
            time.sleep(0.5)  # Allow WebRTC to pause

            # Stop camera
            try:
                self.camera.stop()
            except Exception as e:
                self.logger.warning(f"Error stopping camera: {e}")

            # Reconfigure
            try:
                config = self.camera.create_preview_configuration(
                    main={"size": new_resolution, "format": "RGB888"}
                )
                self.camera.configure(config)
            except Exception as e:
                self.logger.error(f"Error configuring camera at {new_resolution}: {e}")
                # Try to restore old resolution
                try:
                    config = self.camera.create_preview_configuration(
                        main={"size": old_resolution, "format": "RGB888"}
                    )
                    self.camera.configure(config)
                    self.camera.start()
                    self.logger.warning(f"Restored previous resolution {old_resolution}")
                except Exception as restore_error:
                    self.logger.error(f"Failed to restore resolution: {restore_error}")
                    self.camera_initialized = False
                # Resume WebRTC even on failure
                publish_vision_event('webrtc_resume_requested', {'reason': 'camera_reconfig_failed'})
                return False

            # Restart camera
            try:
                self.camera.start()
                time.sleep(0.3)  # Brief stabilization

                # Verify with test capture
                test_frame = self.camera.capture_array()
                if test_frame is not None:
                    actual_shape = test_frame.shape[:2]  # (height, width)
                    expected_shape = (new_resolution[1], new_resolution[0])  # height, width
                    self.logger.info(f"📹 Resolution changed successfully: {actual_shape}")
                    self._current_resolution = new_resolution
                    return True
                else:
                    self.logger.error("Test capture returned None after resolution change")
                    return False

            except Exception as e:
                self.logger.error(f"Error restarting camera: {e}")
                self.camera_initialized = False
                return False

        finally:
            # Resume WebRTC video track after camera reconfig completes (success or failure)
            publish_vision_event('webrtc_resume_requested', {
                'reason': 'camera_reconfig_complete',
                'resolution': list(self._current_resolution)
            })
            with self._resolution_change_lock:
                self._resolution_change_in_progress = False

    def get_current_resolution(self) -> tuple:
        """Get current camera resolution"""
        return self._current_resolution

    def get_last_frame(self) -> Optional[np.ndarray]:
        """Get last captured frame for snapshots (thread-safe)"""
        with self._frame_lock:
            if self._last_frame is not None:
                return self._last_frame.copy()
        return None

    def get_last_frame_age(self) -> float:
        """Get age of last frame in seconds"""
        with self._frame_lock:
            if self._last_frame_time > 0:
                return time.time() - self._last_frame_time
        return float('inf')

    def _detect_aruco_markers(self, frame) -> List[Tuple[int, float, float]]:
        """Detect ArUco markers in frame and return (marker_id, cx, cy) tuples"""
        markers = []
        try:
            # Convert to grayscale for ArUco detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect markers
            corners, ids, rejected = self.aruco_detector.detectMarkers(gray)

            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    # Get center of marker
                    corner = corners[i][0]
                    cx = float(np.mean(corner[:, 0]))
                    cy = float(np.mean(corner[:, 1]))
                    markers.append((int(marker_id), cx, cy))

                    # Log known dog markers
                    if marker_id in DOG_MARKERS:
                        self.logger.info(f"ArUco marker {marker_id} ({DOG_MARKERS[marker_id]}) at ({cx:.0f}, {cy:.0f})")

        except Exception as e:
            self.logger.error(f"ArUco detection error: {e}")

        return markers

    def is_initialized(self) -> bool:
        """Check if detector service is initialized (camera required, AI optional)"""
        return self.camera_initialized

    def start_detection(self) -> bool:
        """Start detection loop

        Starts frame capture loop. AI processing is optional - loop handles
        modes where AI isn't needed (IDLE/MANUAL just capture for WebRTC).
        """
        if not self.camera_initialized:
            self.logger.error("Camera not initialized - cannot start detection")
            return False

        if self.running:
            self.logger.warning("Detection already running")
            return True

        self.running = True
        self._stop_event.clear()

        self.detection_thread = threading.Thread(
            target=self._detection_loop,
            daemon=True,
            name="DetectionService"
        )
        self.detection_thread.start()

        self.logger.info("Detection started")
        publish_vision_event('detection_started', {}, 'detector_service')
        return True

    def stop_detection(self) -> None:
        """Stop detection loop"""
        if not self.running:
            return

        self.running = False
        self._stop_event.set()

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2.0)

        self.logger.info("Detection stopped")
        publish_vision_event('detection_stopped', {}, 'detector_service')

    def _detection_loop(self) -> None:
        """Main detection loop - runs in all operational modes"""
        self.logger.info("Detection loop started")
        self.last_fps_time = time.time()
        self.frame_count = 0
        self._loop_iteration = 0

        while not self._stop_event.is_set():
            try:
                self._loop_iteration += 1

                # Log diagnostic every 100 iterations (or first 5 for debugging)
                if self._loop_iteration <= 5 or self._loop_iteration % 100 == 0:
                    self.logger.info(f"Detection loop alive: iter={self._loop_iteration}, camera_init={self.camera_initialized}, paused={self._camera_paused}, mode={self.state.get_mode()}")

                # Check mode to determine behavior
                # - SHUTDOWN/EMERGENCY: Stop everything
                # - MANUAL/IDLE: Capture frames for WebRTC, NO AI processing (high res)
                # - SILENT_GUARDIAN/COACH: Full AI processing (640x640)
                current_mode = self.state.get_mode()

                # Modes that completely pause detection
                if current_mode in [SystemMode.SHUTDOWN, SystemMode.EMERGENCY]:
                    time.sleep(0.1)
                    continue

                # Periodic camera retry if not initialized
                if not self.camera_initialized and not self._camera_paused:
                    now = time.time()
                    if now - self._last_camera_retry_time >= self._camera_retry_interval:
                        self._last_camera_retry_time = now
                        self.logger.info("Attempting periodic camera reinitialization...")
                        if self._initialize_camera():
                            self.logger.info("Camera reinitialized successfully on retry")
                            self.state.update_hardware(camera_initialized=True)
                        else:
                            self.logger.warning(
                                f"Camera retry failed. Next attempt in {self._camera_retry_interval}s. "
                                f"Reason: {self._camera_error_reason}"
                            )
                    time.sleep(0.5)
                    continue

                # Capture frame from camera (always, for WebRTC streaming)
                if self._loop_iteration <= 5:
                    self.logger.info(f"About to capture frame...")
                frame = self._capture_frame()
                if frame is None:
                    self.logger.warning(f"No frame captured (camera_init={self.camera_initialized}), waiting...")
                    time.sleep(0.1)
                    continue

                if self._loop_iteration <= 5:
                    self.logger.info(f"Frame captured: {frame.shape}")

                # Determine if AI processing should run
                # AI only runs in SILENT_GUARDIAN and COACH modes
                ai_modes = [SystemMode.SILENT_GUARDIAN, SystemMode.COACH]
                run_ai = current_mode in ai_modes

                dogs = []
                poses = []
                behaviors = []
                dog_assignments = {}
                aruco_markers = []

                if run_ai:
                    # Full AI processing for guardian/coach modes
                    # Detect ArUco markers for dog identification
                    aruco_markers = self._detect_aruco_markers(frame)

                    # Publish ArUco marker event for video recorder
                    if aruco_markers:
                        publish_vision_event('aruco_detected', {
                            'markers': aruco_markers,
                            'timestamp': time.time()
                        }, 'detector_service')

                    # Process frame with dog identification
                    result = self.ai.process_frame_with_dogs(frame, aruco_markers)
                    dogs = result.get('detections', [])
                    poses = result.get('poses', [])
                    behaviors = result.get('behaviors', [])
                    dog_assignments = result.get('dog_assignments', {})
                    dog_id_methods = result.get('dog_id_methods', {})
                else:
                    # No AI in IDLE/MANUAL modes - just capture frames for WebRTC
                    # Frame is already captured and stored via _capture_frame()
                    pass

                # Update FPS
                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.current_fps = self.frame_count / (now - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = now
                    self.logger.info(f"Detection FPS: {self.current_fps:.1f}")

                # CRITICAL: Minimum loop interval to prevent Hailo driver exhaustion
                # Primary rate limiting is in AI controller, this is a safety backstop
                # Target ~10 FPS max to leave headroom for Hailo DMA operations
                time.sleep(0.050)  # 50ms minimum between frames (20 FPS cap)

                # Process results
                if dogs:
                    self._process_detections(dogs, poses, behaviors, dog_assignments, dog_id_methods)
                else:
                    # No dogs detected
                    self.state.update_detection(
                        dogs_detected=0,
                        active_dog_id=None,
                        dog_name="",
                        id_method="",
                        current_behavior="",
                        behavior_confidence=0.0
                    )

                # Small delay to prevent CPU overload
                time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Detection loop error: {e}")
                time.sleep(0.1)

        self.logger.info("Detection loop stopped")

    def _process_detections(self, dogs, poses, behaviors, dog_assignments: Dict[int, str] = None, dog_id_methods: Dict[int, str] = None) -> None:
        """Process detection results and publish events

        Args:
            dogs: List of Detection dataclass objects
            poses: List of PoseKeypoints dataclass objects
            behaviors: List of BehaviorResult dataclass objects
            dog_assignments: Dict mapping detection index to dog name (from ArUco)
            dog_id_methods: Dict mapping detection index to identification method
        """
        if dog_assignments is None:
            dog_assignments = {}
        if dog_id_methods is None:
            dog_id_methods = {}

        # Update state
        num_dogs = len(dogs)
        # Get primary dog name and id method (first detection)
        primary_dog_name = dog_assignments.get(0, "")
        primary_id_method = dog_id_methods.get(0, "")
        self.state.update_detection(
            dogs_detected=num_dogs,
            last_detection_time=time.time(),
            dog_name=primary_dog_name.capitalize() if primary_dog_name else "",
            id_method=primary_id_method
        )

        # Rate limit detection events to prevent log spam
        current_time = time.time()
        should_publish_event = (current_time - self._last_detection_event_time) >= self._detection_event_interval

        # Process each dog (Detection dataclass)
        for i, dog in enumerate(dogs):
            # Use ArUco-identified name if available, otherwise generic ID
            dog_name = dog_assignments.get(i)
            dog_id = dog_name if dog_name else f"dog_{i}"

            bbox = [dog.x1, dog.y1, dog.x2, dog.y2]
            center = dog.center

            # Publish dog detection event (rate limited to prevent spam)
            if should_publish_event:
                publish_vision_event('dog_detected', {
                    'dog_id': dog_id,
                    'dog_name': dog_name,  # Will be None if not identified
                    'confidence': dog.confidence,
                    'bbox': bbox,
                    'center': list(center),
                    'timestamp': current_time
                }, 'detector_service')

                # Log to store (also rate limited)
                self.store.log_event('vision', 'dog_detected', 'detector_service', {
                    'dog_id': dog_id,
                    'dog_name': dog_name,
                    'confidence': dog.confidence,
                    'bbox': bbox
                })

            # Update dog seen in store (always update, just don't spam logs)
            self.store.update_dog_seen(dog_id)

        # Update last event time if we published
        if should_publish_event and num_dogs > 0:
            self._last_detection_event_time = current_time

        # Process poses if available (PoseKeypoints dataclass) - rate limited
        if poses and should_publish_event:
            for i, pose in enumerate(poses):
                # Use ArUco-identified name if available
                dog_name = dog_assignments.get(i)
                dog_id = dog_name if dog_name else f"dog_{i}"
                keypoints = pose.keypoints.tolist() if hasattr(pose.keypoints, 'tolist') else list(pose.keypoints)

                publish_vision_event('pose_detected', {
                    'dog_id': dog_id,
                    'dog_name': dog_name,
                    'keypoints': keypoints,
                    'confidence': pose.detection.confidence if pose.detection else 0.0,
                    'num_keypoints': len(keypoints),
                    'timestamp': current_time
                }, 'detector_service')

        # Process behaviors if available (BehaviorResult dataclass)
        if behaviors:
            for i, behavior in enumerate(behaviors):
                # Use ArUco-identified name if available
                dog_name = dog_assignments.get(i)
                dog_id = dog_name if dog_name else f"dog_{i}"
                behavior_name = behavior.behavior
                confidence = behavior.confidence

                # Update state with primary dog's behavior
                if i == 0:
                    id_method = dog_id_methods.get(i, "")
                    self.state.update_detection(
                        active_dog_id=dog_id,
                        dog_name=dog_name.capitalize() if dog_name else "",
                        id_method=id_method,
                        current_behavior=behavior_name,
                        behavior_confidence=confidence,
                        pose_stable=confidence > 0.7
                    )

                # Publish behavior event
                publish_vision_event('behavior_detected', {
                    'dog_id': dog_id,
                    'dog_name': dog_name,
                    'behavior': behavior_name,
                    'confidence': confidence,
                    'duration': getattr(behavior, 'duration', 0.0),
                    'timestamp': time.time()
                }, 'detector_service')

                # Log behavior to store
                self.store.log_event('vision', 'behavior_detected', 'detector_service', {
                    'dog_id': dog_id,
                    'behavior': behavior_name,
                    'confidence': confidence
                })

    def _get_bbox_center(self, bbox: List[float]) -> List[float]:
        """Get center point of bounding box"""
        if len(bbox) >= 4:
            x1, y1, x2, y2 = bbox[:4]
            return [(x1 + x2) / 2, (y1 + y2) / 2]
        return [320, 240]  # Default center

    def get_status(self) -> Dict[str, Any]:
        """Get detector service status"""
        return {
            'initialized': self.ai_initialized,
            'running': self.running,
            'current_fps': self.current_fps,
            'frame_count': self.frame_count,
            'ai_status': self.ai.get_status() if self.ai else None
        }

    def get_camera_status(self) -> Dict[str, Any]:
        """Get camera-specific status for diagnostics"""
        # Check for available cameras
        available_cameras = []
        if PICAMERA_AVAILABLE:
            try:
                available_cameras = Picamera2.global_camera_info()
            except Exception as e:
                self.logger.warning(f"Could not get camera info: {e}")

        return {
            'initialized': self.camera_initialized,
            'paused': self._camera_paused,
            'available_cameras': available_cameras,
            'cameras_detected': len(available_cameras),
            'error_reason': self._camera_error_reason,
            'picamera_available': PICAMERA_AVAILABLE,
            'last_frame_age': self.get_last_frame_age() if self.camera_initialized else None,
            'current_resolution': self._current_resolution,
            'resolution_change_in_progress': self._resolution_change_in_progress
        }

    def request_camera_reinitialize(self) -> Dict[str, Any]:
        """Request camera reinitialization (public API method)"""
        self.logger.info("Camera reinitialize requested via API")
        self._camera_error_reason = None
        self._reinitialize_camera()
        return self.get_camera_status()

    def cleanup(self) -> None:
        """Clean shutdown"""
        self.stop_detection()
        # Clean up camera
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
                self.logger.info("Camera closed")
            except Exception as e:
                self.logger.error(f"Camera cleanup error: {e}")
            self.camera = None
            self.camera_initialized = False
        if self.ai:
            self.ai.cleanup()
        self.logger.info("Detector service cleaned up")


# Global detector service instance
_detector_instance = None
_detector_lock = threading.Lock()

def get_detector_service() -> DetectorService:
    """Get the global detector service instance (singleton)"""
    global _detector_instance
    if _detector_instance is None:
        with _detector_lock:
            if _detector_instance is None:
                _detector_instance = DetectorService()
    return _detector_instance