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

    def initialize(self) -> bool:
        """Initialize AI controller and camera"""
        try:
            # Initialize camera first
            self._initialize_camera()

            # Initialize AI controller
            self.ai = AI3StageControllerFixed()
            success = self.ai.initialize()

            if success:
                self.ai_initialized = True
                self.logger.info("AI controller initialized")
                self.state.update_hardware(camera_initialized=self.camera_initialized)
                return True
            else:
                self.logger.error("AI controller initialization failed")
                return False

        except Exception as e:
            self.logger.error(f"AI controller initialization error: {e}")
            return False

    def _initialize_camera(self) -> bool:
        """Initialize camera for frame capture"""
        try:
            if PICAMERA_AVAILABLE:
                self.logger.info("Initializing Picamera2 for detection...")
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (640, 640), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                self.camera_initialized = True
                self.logger.info("Picamera2 initialized for detection")
                return True
            else:
                self.logger.warning("Picamera2 not available, detection disabled")
                return False
        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            return False

    def _capture_frame(self):
        """Capture a frame from the camera"""
        if not self.camera_initialized or self.camera is None:
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
            return None

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
        """Check if detector service is initialized"""
        return self.ai_initialized

    def start_detection(self) -> bool:
        """Start detection loop"""
        if not self.ai_initialized:
            self.logger.error("AI controller not initialized")
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

        while not self._stop_event.is_set():
            try:
                # Check if we should be running based on mode
                # AI vision runs in ALL operational modes for:
                # - COACH: Active training (full detection + behavior)
                # - SILENT_GUARDIAN: Dog visibility checks for treat dispensing
                # - IDLE: Background monitoring
                # - PHOTOGRAPHY: Frame capture assistance
                # NOT in: MANUAL (controller takes over), SHUTDOWN, EMERGENCY
                current_mode = self.state.get_mode()
                if current_mode in [SystemMode.MANUAL, SystemMode.SHUTDOWN, SystemMode.EMERGENCY]:
                    # Release camera in MANUAL mode so rpicam-still can use it
                    if not self._camera_paused and self.camera_initialized:
                        self._release_camera()
                    time.sleep(0.1)
                    continue

                # Re-acquire camera if we were paused (leaving MANUAL mode)
                if self._camera_paused:
                    if not self._reacquire_camera():
                        self.logger.error("Failed to re-acquire camera")
                        time.sleep(1.0)
                        continue

                # Capture frame from camera
                frame = self._capture_frame()
                if frame is None:
                    self.logger.warning("No frame captured, waiting...")
                    time.sleep(0.1)
                    continue

                # Detect ArUco markers for dog identification
                aruco_markers = self._detect_aruco_markers(frame)

                # Process frame with dog identification
                result = self.ai.process_frame_with_dogs(frame, aruco_markers)
                dogs = result.get('detections', [])
                poses = result.get('poses', [])
                behaviors = result.get('behaviors', [])
                dog_assignments = result.get('dog_assignments', {})

                # Update FPS
                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.current_fps = self.frame_count / (now - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = now
                    self.logger.info(f"Detection FPS: {self.current_fps:.1f}")

                # Process results
                if dogs:
                    self._process_detections(dogs, poses, behaviors, dog_assignments)
                else:
                    # No dogs detected
                    self.state.update_detection(
                        dogs_detected=0,
                        active_dog_id=None,
                        current_behavior="",
                        behavior_confidence=0.0
                    )

                # Small delay to prevent CPU overload
                time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Detection loop error: {e}")
                time.sleep(0.1)

        self.logger.info("Detection loop stopped")

    def _process_detections(self, dogs, poses, behaviors, dog_assignments: Dict[int, str] = None) -> None:
        """Process detection results and publish events

        Args:
            dogs: List of Detection dataclass objects
            poses: List of PoseKeypoints dataclass objects
            behaviors: List of BehaviorResult dataclass objects
            dog_assignments: Dict mapping detection index to dog name (from ArUco)
        """
        if dog_assignments is None:
            dog_assignments = {}

        # Update state
        num_dogs = len(dogs)
        self.state.update_detection(
            dogs_detected=num_dogs,
            last_detection_time=time.time()
        )

        # Process each dog (Detection dataclass)
        for i, dog in enumerate(dogs):
            # Use ArUco-identified name if available, otherwise generic ID
            dog_name = dog_assignments.get(i)
            dog_id = dog_name if dog_name else f"dog_{i}"

            bbox = [dog.x1, dog.y1, dog.x2, dog.y2]
            center = dog.center

            # Publish dog detection event
            publish_vision_event('dog_detected', {
                'dog_id': dog_id,
                'dog_name': dog_name,  # Will be None if not identified
                'confidence': dog.confidence,
                'bbox': bbox,
                'center': list(center),
                'timestamp': time.time()
            }, 'detector_service')

            # Log to store
            self.store.log_event('vision', 'dog_detected', 'detector_service', {
                'dog_id': dog_id,
                'dog_name': dog_name,
                'confidence': dog.confidence,
                'bbox': bbox
            })

            # Update dog seen in store
            self.store.update_dog_seen(dog_id)

        # Process poses if available (PoseKeypoints dataclass)
        if poses:
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
                    'timestamp': time.time()
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
                    self.state.update_detection(
                        active_dog_id=dog_id,
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