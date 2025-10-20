#!/usr/bin/env python3
"""
CameraManager - Unified camera interface
Consolidates multiple camera implementations from conversation threads
"""

import cv2
import numpy as np
import threading
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from .detection_plugins.base_detector import BaseDetector
from .detection_plugins.hailo_detector import HailoDetector
from .detection_plugins.opencv_detector import OpenCVDetector
from .detection_plugins.aruco_detector import ArUcoDetector
from ..utils.event_bus import EventBus

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    logging.warning("Picamera2 not available, using OpenCV camera")

class CameraManager:
    """
    Unified camera manager that consolidates all camera interfaces
    Supports multiple detection backends with automatic fallback
    """

    def __init__(self, detection_config: Dict[str, Any], event_bus: EventBus):
        self.logger = logging.getLogger('CameraManager')
        self.detection_config = detection_config
        self.event_bus = event_bus

        # Camera setup
        self.camera = None
        self.camera_active = False
        self.capture_thread = None
        self.detection_thread = None

        # Detection systems
        self.detectors: Dict[str, BaseDetector] = {}
        self.active_detector = None

        # Frame processing
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.detection_queue = []
        self.max_queue_size = 5

        # Statistics
        self.frame_count = 0
        self.detection_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()

        # Initialize camera and detectors
        self._initialize_camera()
        self._initialize_detectors()

    def _initialize_camera(self):
        """Initialize camera (Picamera2 or OpenCV fallback)"""
        try:
            if PICAMERA_AVAILABLE:
                self.logger.info("Initializing Picamera2...")
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.logger.info("Picamera2 initialized successfully")
            else:
                self.logger.info("Initializing OpenCV camera...")
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    raise Exception("Failed to open OpenCV camera")
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                self.logger.info("OpenCV camera initialized successfully")

        except Exception as e:
            self.logger.error(f"Camera initialization failed: {e}")
            self.camera = None

    def _initialize_detectors(self):
        """Initialize detection backends with automatic fallback"""
        preferred = self.detection_config.get('preferred_backend', 'hailo')
        fallback = self.detection_config.get('fallback_backend', 'opencv_yolo')

        # Try to initialize preferred detector
        if preferred == 'hailo':
            try:
                self.detectors['hailo'] = HailoDetector(self.detection_config)
                self.active_detector = self.detectors['hailo']
                self.logger.info("Hailo detector initialized successfully")
            except Exception as e:
                self.logger.warning(f"Hailo detector failed: {e}")

        # Initialize fallback detector
        if self.active_detector is None or fallback != preferred:
            try:
                self.detectors['opencv'] = OpenCVDetector(self.detection_config)
                if self.active_detector is None:
                    self.active_detector = self.detectors['opencv']
                    self.logger.info("Using OpenCV detector as primary")
                else:
                    self.logger.info("OpenCV detector available as fallback")
            except Exception as e:
                self.logger.error(f"OpenCV detector failed: {e}")

        # Initialize ArUco detector (always available)
        try:
            self.detectors['aruco'] = ArUcoDetector(self.detection_config)
            self.logger.info("ArUco detector initialized")
        except Exception as e:
            self.logger.warning(f"ArUco detector failed: {e}")

        if self.active_detector is None:
            self.logger.error("No detection backends available!")

    def start_detection(self):
        """Start camera capture and detection"""
        if not self.camera:
            self.logger.error("Cannot start detection - no camera available")
            return False

        if self.camera_active:
            self.logger.warning("Detection already active")
            return True

        try:
            # Start camera
            if PICAMERA_AVAILABLE and hasattr(self.camera, 'start'):
                self.camera.start()

            self.camera_active = True

            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()

            # Start detection thread
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()

            self.logger.info("Camera detection started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start detection: {e}")
            self.camera_active = False
            return False

    def stop_detection(self):
        """Stop camera capture and detection"""
        if not self.camera_active:
            return

        self.logger.info("Stopping camera detection...")
        self.camera_active = False

        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=2)

        # Stop camera
        try:
            if PICAMERA_AVAILABLE and hasattr(self.camera, 'stop'):
                self.camera.stop()
            elif hasattr(self.camera, 'release'):
                self.camera.release()
        except Exception as e:
            self.logger.error(f"Camera stop error: {e}")

        self.logger.info("Camera detection stopped")

    def _capture_loop(self):
        """Continuous frame capture loop"""
        self.logger.debug("Capture loop started")

        while self.camera_active:
            try:
                if PICAMERA_AVAILABLE and hasattr(self.camera, 'capture_array'):
                    frame = self.camera.capture_array()
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        self.logger.error("Failed to capture frame")
                        continue

                # Update latest frame
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                    self.frame_count += 1

                # Add to detection queue (drop old frames if queue full)
                if len(self.detection_queue) < self.max_queue_size:
                    self.detection_queue.append(frame.copy())
                else:
                    self.detection_queue.pop(0)
                    self.detection_queue.append(frame.copy())

                # FPS calculation
                self.fps_counter += 1
                if time.time() - self.last_fps_time >= 1.0:
                    fps = self.fps_counter / (time.time() - self.last_fps_time)
                    self.logger.debug(f"Camera FPS: {fps:.1f}")
                    self.fps_counter = 0
                    self.last_fps_time = time.time()

                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                self.logger.error(f"Capture loop error: {e}")
                time.sleep(0.1)

        self.logger.debug("Capture loop ended")

    def _detection_loop(self):
        """Continuous detection processing loop"""
        self.logger.debug("Detection loop started")

        while self.camera_active:
            try:
                if not self.detection_queue:
                    time.sleep(0.01)
                    continue

                # Get frame for detection
                frame = self.detection_queue.pop(0)

                # Run detection
                if self.active_detector:
                    detections = self.active_detector.detect(frame)
                    self._process_detections(detections, frame)

                # Run ArUco detection if available
                if 'aruco' in self.detectors:
                    aruco_detections = self.detectors['aruco'].detect(frame)
                    self._process_aruco_detections(aruco_detections, frame)

                self.detection_count += 1

            except Exception as e:
                self.logger.error(f"Detection loop error: {e}")
                time.sleep(0.1)

        self.logger.debug("Detection loop ended")

    def _process_detections(self, detections: List[Dict[str, Any]], frame: np.ndarray):
        """Process detection results and emit events"""
        if not detections:
            self.event_bus.publish('no_detections', {
                'timestamp': time.time(),
                'frame_shape': frame.shape
            })
            return

        # Filter for dogs (class_id 16 in COCO)
        dog_detections = [d for d in detections if d.get('class_id') == 16]

        if dog_detections:
            # Find best detection
            best_detection = max(dog_detections, key=lambda x: x.get('confidence', 0))

            # Calculate center point
            bbox = best_detection.get('bbox', [0, 0, 0, 0])
            center_x = bbox[0] + bbox[2] // 2
            center_y = bbox[1] + bbox[3] // 2

            # Emit dog detection event
            self.event_bus.publish('dog_detected', {
                'detection': best_detection,
                'center': (center_x, center_y),
                'dog_count': len(dog_detections),
                'timestamp': time.time(),
                'frame_shape': frame.shape
            })

    def _process_aruco_detections(self, aruco_data: Dict[str, Any], frame: np.ndarray):
        """Process ArUco marker detections"""
        if aruco_data.get('markers'):
            self.event_bus.publish('aruco_detected', {
                'markers': aruco_data['markers'],
                'timestamp': time.time()
            })

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_status(self) -> Dict[str, Any]:
        """Get camera and detection status"""
        return {
            'camera_available': self.camera is not None,
            'camera_active': self.camera_active,
            'active_detector': self.active_detector.__class__.__name__ if self.active_detector else None,
            'available_detectors': list(self.detectors.keys()),
            'frame_count': self.frame_count,
            'detection_count': self.detection_count,
            'detection_queue_size': len(self.detection_queue)
        }

    def switch_detector(self, detector_name: str) -> bool:
        """Switch to a different detection backend"""
        if detector_name not in self.detectors:
            self.logger.error(f"Detector '{detector_name}' not available")
            return False

        self.active_detector = self.detectors[detector_name]
        self.logger.info(f"Switched to {detector_name} detector")
        return True

    def cleanup(self):
        """Clean up resources"""
        self.stop_detection()

        # Cleanup detectors
        for detector in self.detectors.values():
            try:
                if hasattr(detector, 'cleanup'):
                    detector.cleanup()
            except Exception as e:
                self.logger.error(f"Detector cleanup error: {e}")

        self.logger.info("Camera manager cleaned up")