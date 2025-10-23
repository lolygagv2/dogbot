#!/usr/bin/env python3
"""
Detection service wrapper for ai_controller_3stage_fixed.py
Publishes pose events to event bus
"""

import threading
import time
import logging
from typing import Optional, Dict, Any, List

from core.bus import get_bus, publish_vision_event
from core.state import get_state, SystemMode
from core.store import get_store
from core.ai_controller_3stage_fixed import AI3StageControllerFixed


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

        # Detection state
        self.running = False
        self.detection_thread = None
        self._stop_event = threading.Event()

        # Performance tracking
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0

    def initialize(self) -> bool:
        """Initialize AI controller"""
        try:
            self.ai = AI3StageControllerFixed()
            success = self.ai.initialize()

            if success:
                self.ai_initialized = True
                self.logger.info("AI controller initialized")
                self.state.update_hardware(camera_initialized=True)
                return True
            else:
                self.logger.error("AI controller initialization failed")
                return False

        except Exception as e:
            self.logger.error(f"AI controller initialization error: {e}")
            return False

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
        """Main detection loop"""
        self.logger.info("Detection loop started")
        self.last_fps_time = time.time()
        self.frame_count = 0

        while not self._stop_event.is_set():
            try:
                # Check if we should be running based on mode
                current_mode = self.state.get_mode()
                if current_mode not in [SystemMode.DETECTION, SystemMode.VIGILANT]:
                    time.sleep(0.1)
                    continue

                # Process frame using existing AI controller
                dogs, poses, behaviors = self.ai.process_frame(None)  # AI controller handles frame capture

                # Update FPS
                self.frame_count += 1
                now = time.time()
                if now - self.last_fps_time >= 1.0:
                    self.current_fps = self.frame_count / (now - self.last_fps_time)
                    self.frame_count = 0
                    self.last_fps_time = now

                # Process results
                if dogs:
                    self._process_detections(dogs, poses, behaviors)
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

    def _process_detections(self, dogs: List[Dict], poses: List[Dict], behaviors: List[Dict]) -> None:
        """Process detection results and publish events"""

        # Update state
        num_dogs = len(dogs)
        self.state.update_detection(
            dogs_detected=num_dogs,
            last_detection_time=time.time()
        )

        # Process each dog
        for i, dog in enumerate(dogs):
            dog_id = f"dog_{i}"  # Simple ID for now

            # Publish dog detection event
            publish_vision_event('dog_detected', {
                'dog_id': dog_id,
                'confidence': dog.get('confidence', 0.0),
                'bbox': dog.get('bbox', [0, 0, 0, 0]),
                'center': self._get_bbox_center(dog.get('bbox', [0, 0, 0, 0])),
                'timestamp': time.time()
            }, 'detector_service')

            # Log to store
            self.store.log_event('vision', 'dog_detected', 'detector_service', {
                'dog_id': dog_id,
                'confidence': dog.get('confidence', 0.0),
                'bbox': dog.get('bbox', [0, 0, 0, 0])
            })

            # Update dog seen in store
            self.store.update_dog_seen(dog_id)

        # Process poses if available
        if poses:
            for i, pose in enumerate(poses):
                dog_id = f"dog_{i}"

                publish_vision_event('pose_detected', {
                    'dog_id': dog_id,
                    'keypoints': pose.get('keypoints', []),
                    'confidence': pose.get('confidence', 0.0),
                    'num_keypoints': len(pose.get('keypoints', [])),
                    'timestamp': time.time()
                }, 'detector_service')

        # Process behaviors if available
        if behaviors:
            for i, behavior in enumerate(behaviors):
                dog_id = f"dog_{i}"
                behavior_name = behavior.get('behavior', 'unknown')
                confidence = behavior.get('confidence', 0.0)

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
                    'behavior': behavior_name,
                    'confidence': confidence,
                    'duration': behavior.get('duration', 0.0),
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