#!/usr/bin/env python3
"""
AI Controller with 3-Stage Pipeline using Temporal Behavior Model
Stage 1: Dog Detection (dogdetector_14.hef) - 640x640
Stage 2: Pose Estimation (dogpose_14.hef) - 640x640 crops
Stage 3: Behavior Analysis (behavior_14.ts) - Temporal TorchScript model

This version uses the trained temporal model instead of heuristics
"""

import os
import json
import time
import collections
import numpy as np
import cv2
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Check for Hailo availability
try:
    from hailo_sdk_client import ClientRunner
    import hailo_platform as hpf
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("⚠️ Hailo SDK not available - will run in CPU mode")

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Dog detection result"""
    bbox: List[float]
    confidence: float
    dog_id: Optional[str] = None


@dataclass
class Pose:
    """Dog pose estimation result"""
    keypoints: np.ndarray
    confidence: float
    bbox: List[float]
    dog_id: Optional[str] = None


@dataclass
class Behavior:
    """Dog behavior analysis result"""
    behavior: str
    confidence: float
    dog_id: Optional[str] = None


class AI3StageControllerTemporal:
    """
    3-stage AI pipeline with temporal behavior model
    Uses TorchScript model for behavior classification
    """

    def __init__(self, config_path: str = "config/config.json"):
        """Initialize AI controller with temporal model"""
        # Load configuration
        self.config = self._load_config(config_path)

        # Model parameters
        self.input_size = tuple(self.config.get("imgsz", [640, 640]))
        self.T = self.config.get("T", 16)  # Temporal window size
        self.behaviors = self.config.get("behaviors", ["stand", "sit", "lie", "cross", "spin"])
        self.prob_th = self.config.get("prob_th", 0.7)
        self.behavior_cooldowns = self.config.get("cooldown_s", {})

        # ArUco parameters
        self.marker_to_dog = {
            int(d["marker_id"]): str(d["id"])
            for d in self.config.get("dogs", [])
        }
        self.assume_other = self.config.get("assume_other_if_two_boxes_one_marker", True)
        self.camera_rotation_deg = self.config.get("camera_rotation_deg", 90)

        # Model paths
        self.detection_model_path = Path("ai/models") / self.config.get("detect_path", "dogdetector_14.hef")
        self.pose_model_path = Path("ai/models") / self.config.get("hef_path", "dogpose_14.hef")
        self.behavior_model_path = Path("ai/models") / self.config.get("behavior_head_ts", "behavior_14.ts")

        # Hailo models
        self.vdevice = None
        self.detection_network = None
        self.pose_network = None

        # TorchScript behavior model
        self.behavior_model = None

        # Temporal buffers
        self.pose_history = collections.deque(maxlen=self.T)
        self.behavior_ema = {}  # Exponential moving average per dog
        self.last_behaviors = {}
        self.cooldown_timers = {}

        # ArUco detector
        self.aruco_detector = self._setup_aruco()

        # Frame counter
        self.frame_count = 0

        logger.info(f"Temporal AI Controller initialized with config: {self.config}")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {
                "imgsz": [640, 640],
                "behaviors": ["stand", "sit", "lie", "cross", "spin"],
                "prob_th": 0.7
            }

    def _setup_aruco(self):
        """Setup ArUco marker detector"""
        dict_name = str(self.config.get("aruco_dict", "DICT_4X4_1000"))
        dict_const = getattr(cv2.aruco, dict_name)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_const)

        try:
            # Newer OpenCV
            detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        except AttributeError:
            # Older OpenCV
            detector = (aruco_dict, cv2.aruco.DetectorParameters_create())

        return detector

    def initialize(self) -> bool:
        """Initialize models - Hailo and TorchScript"""
        try:
            if HAILO_AVAILABLE:
                # Initialize Hailo models
                logger.info("Initializing Hailo models...")
                self.vdevice = hpf.VDevice()

                # Load detection model
                if self.detection_model_path.exists():
                    detection_hef = hpf.create_hef_from_file(str(self.detection_model_path))
                    detection_config = hpf.VDevice.create_params()
                    detection_config.scheduling_algorithm = hpf.SchedulingAlgorithm.ROUND_ROBIN
                    self.detection_network = self.vdevice.create_infer_model(detection_hef, detection_config)
                    logger.info(f"Loaded detection model: {self.detection_model_path}")
                else:
                    logger.error(f"Detection model not found: {self.detection_model_path}")
                    return False

                # Load pose model
                if self.pose_model_path.exists():
                    pose_hef = hpf.create_hef_from_file(str(self.pose_model_path))
                    pose_config = hpf.VDevice.create_params()
                    pose_config.scheduling_algorithm = hpf.SchedulingAlgorithm.ROUND_ROBIN
                    self.pose_network = self.vdevice.create_infer_model(pose_hef, pose_config)
                    logger.info(f"Loaded pose model: {self.pose_model_path}")
                else:
                    logger.error(f"Pose model not found: {self.pose_model_path}")
                    return False
            else:
                logger.warning("⚠️ Hailo not available - running in CPU-only mode")
                # In CPU mode, we'll use simplified detection/pose

            # Load TorchScript behavior model (CPU)
            if self.behavior_model_path.exists():
                self.behavior_model = torch.jit.load(str(self.behavior_model_path), map_location="cpu").eval()
                logger.info(f"✅ Loaded temporal behavior model: {self.behavior_model_path}")
            else:
                logger.error(f"❌ Behavior model not found: {self.behavior_model_path}")
                return False

            logger.info("All models initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            return False

    def process_frame(self, frame: np.ndarray) -> Tuple[List[Detection], List[Pose], List[Behavior]]:
        """
        Process a single frame through all 3 stages

        Args:
            frame: Input frame (BGR)

        Returns:
            (detections, poses, behaviors)
        """
        self.frame_count += 1

        # Apply camera rotation if needed
        if self.camera_rotation_deg:
            frame = self._rotate_frame(frame, self.camera_rotation_deg)

        # Stage 1: Dog Detection
        detections = self._detect_dogs(frame)

        # Stage 2: Pose Estimation
        poses = []
        if detections:
            for det in detections:
                pose = self._estimate_pose(frame, det.bbox)
                if pose:
                    pose.dog_id = det.dog_id
                    poses.append(pose)

        # Stage 3: Temporal Behavior Analysis with TorchScript
        behaviors = []
        if poses:
            # Update pose history
            self._update_pose_history(poses)

            # Run temporal model
            behaviors = self._analyze_behaviors_temporal()

        return detections, poses, behaviors

    def _detect_dogs(self, frame: np.ndarray) -> List[Detection]:
        """Stage 1: Dog detection using Hailo"""
        if not self.detection_network:
            # In CPU-only mode, create a mock detection for testing
            if not HAILO_AVAILABLE:
                # Create a centered detection for testing
                h, w = frame.shape[:2]
                center_box = [w//4, h//4, 3*w//4, 3*h//4]
                detection = Detection(
                    bbox=center_box,
                    confidence=0.95,
                    dog_id="test_dog"
                )
                return [detection]
            return []

        try:
            # Preprocess for detection
            input_frame = cv2.resize(frame, self.input_size)
            input_frame = input_frame.astype(np.float32) / 255.0
            input_frame = np.expand_dims(input_frame, axis=0)

            # Run inference
            with self.detection_network.get_bindings() as bindings:
                bindings.input().set_buffer(input_frame)
                bindings.run()
                outputs = bindings.output()

            # Parse detections
            detections = self._parse_detections(outputs, frame.shape)

            # Detect ArUco markers for dog ID
            markers = self._detect_markers(frame)
            if markers:
                detections = self._assign_dog_ids(detections, markers)

            return detections

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []

    def _estimate_pose(self, frame: np.ndarray, bbox: List[float]) -> Optional[Pose]:
        """Stage 2: Pose estimation on cropped dog"""
        if not self.pose_network:
            # In CPU-only mode, create mock pose data for testing
            if not HAILO_AVAILABLE:
                # Generate random keypoints for testing temporal behavior (24 dog keypoints)
                keypoints = np.random.randn(24, 3) * 50 + 320  # 24 dog keypoints with x,y,conf
                return Pose(
                    keypoints=keypoints,
                    confidence=0.9,
                    bbox=bbox
                )
            return None

        try:
            # Crop and resize
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            input_crop = cv2.resize(crop, self.input_size)
            input_crop = input_crop.astype(np.float32) / 255.0
            input_crop = np.expand_dims(input_crop, axis=0)

            # Run inference
            with self.pose_network.get_bindings() as bindings:
                bindings.input().set_buffer(input_crop)
                bindings.run()
                outputs = bindings.output()

            # Parse keypoints (9 tensors, 72 channels total)
            keypoints = self._parse_keypoints(outputs)

            if keypoints is not None:
                return Pose(
                    keypoints=keypoints,
                    confidence=0.8,  # Average confidence
                    bbox=bbox
                )

            return None

        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return None

    def _update_pose_history(self, poses: List[Pose]):
        """Update temporal buffer with new poses"""
        # Convert poses to format expected by TorchScript model
        # Shape: (batch_size, num_keypoints, 3)
        pose_data = []
        for pose in poses:
            # Ensure keypoints are in correct shape
            kpts = pose.keypoints
            if kpts.ndim == 1:
                # Reshape flattened keypoints to (num_kpts, 3)
                num_kpts = len(kpts) // 3
                kpts = kpts.reshape(num_kpts, 3)
            pose_data.append(kpts)

        if pose_data:
            # Add to history
            self.pose_history.append(pose_data)

    def _analyze_behaviors_temporal(self) -> List[Behavior]:
        """Stage 3: Temporal behavior analysis using TorchScript model"""
        if not self.behavior_model or len(self.pose_history) < 4:
            return []

        try:
            behaviors = []

            # Prepare temporal sequence (last T frames)
            sequence = list(self.pose_history)[-self.T:]

            # Pad if needed
            while len(sequence) < self.T:
                sequence.insert(0, sequence[0] if sequence else [np.zeros((24, 3))])

            # Convert to tensor
            # Shape: (T, num_dogs, num_keypoints, 3)
            max_dogs = max(len(frame_poses) for frame_poses in sequence)
            tensor_data = []

            for frame_poses in sequence:
                frame_data = []
                for i in range(max_dogs):
                    if i < len(frame_poses):
                        # Take all 24 keypoints but only x,y (drop confidence)
                        pose_data = frame_poses[i][:, :2]  # 24 keypoints, x,y only
                        frame_data.append(pose_data)
                    else:
                        # Pad with zeros for missing dogs (24 keypoints, x,y only)
                        frame_data.append(np.zeros((24, 2)))
                tensor_data.append(frame_data)

            # Convert to PyTorch tensor
            tensor_input = torch.tensor(tensor_data, dtype=torch.float32)

            # Reshape for LSTM: flatten the pose data
            # From (T, num_dogs, 24, 2) to (num_dogs, T, 24*2=48)
            T = len(tensor_data)
            num_dogs = tensor_input.shape[1] if len(tensor_input.shape) > 1 else 1

            # Reshape to (num_dogs, T, features)
            tensor_input = tensor_input.transpose(0, 1)  # (num_dogs, T, 24, 2)
            tensor_input = tensor_input.reshape(num_dogs, T, -1)  # (num_dogs, T, 48)

            # Run TorchScript model
            with torch.no_grad():
                output = self.behavior_model(tensor_input)

            # Parse output (probabilities for each behavior)
            probs = torch.softmax(output, dim=-1).numpy()

            # Handle different output shapes
            if probs.ndim == 1:
                # Single dog output (5 behaviors)
                probs = probs.reshape(1, -1)

            # Get behaviors for each dog
            for dog_idx in range(min(max_dogs, len(probs))):
                dog_probs = probs[dog_idx]
                max_idx = np.argmax(dog_probs)
                max_prob = dog_probs[max_idx]

                if max_prob > self.prob_th:
                    behavior_name = self.behaviors[max_idx] if max_idx < len(self.behaviors) else "unknown"

                    # Check cooldown
                    if self._check_cooldown(behavior_name, dog_idx):
                        behaviors.append(Behavior(
                            behavior=behavior_name,
                            confidence=float(max_prob),
                            dog_id=f"dog_{dog_idx}"
                        ))

                        # Update cooldown
                        self._update_cooldown(behavior_name, dog_idx)

                        logger.info(f"🎯 TEMPORAL BEHAVIOR: {behavior_name} (conf={max_prob:.2f})")

            return behaviors

        except Exception as e:
            logger.error(f"Temporal behavior analysis failed: {e}")
            return []

    def _check_cooldown(self, behavior: str, dog_idx: int) -> bool:
        """Check if behavior is off cooldown"""
        key = f"{dog_idx}_{behavior}"
        if key not in self.cooldown_timers:
            return True

        elapsed = time.time() - self.cooldown_timers[key]
        cooldown_duration = self.behavior_cooldowns.get(behavior, 2.0)

        return elapsed > cooldown_duration

    def _update_cooldown(self, behavior: str, dog_idx: int):
        """Update cooldown timer for behavior"""
        key = f"{dog_idx}_{behavior}"
        self.cooldown_timers[key] = time.time()

    def _rotate_frame(self, frame: np.ndarray, degrees: int) -> np.ndarray:
        """Rotate frame by specified degrees"""
        if degrees == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif degrees == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif degrees == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    def _detect_markers(self, frame: np.ndarray) -> List[Tuple[int, float, float]]:
        """Detect ArUco markers in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        try:
            if isinstance(self.aruco_detector, tuple):
                # Old OpenCV API
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_detector[0], parameters=self.aruco_detector[1])
            else:
                # New OpenCV API
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        except Exception as e:
            logger.error(f"ArUco detection failed: {e}")
            return []

        markers = []
        if ids is not None:
            for corner, marker_id in zip(corners, ids.flatten()):
                pts = corner[0]
                cx = float(pts[:, 0].mean())
                cy = float(pts[:, 1].mean())
                markers.append((int(marker_id), cx, cy))

        return markers

    def _assign_dog_ids(self, detections: List[Detection], markers: List[Tuple[int, float, float]]) -> List[Detection]:
        """Assign dog IDs based on ArUco markers"""
        for marker_id, cx, cy in markers:
            if marker_id in self.marker_to_dog:
                dog_id = self.marker_to_dog[marker_id]

                # Find closest detection to marker
                min_dist = float('inf')
                closest_det = None

                for det in detections:
                    bbox = det.bbox
                    det_cx = (bbox[0] + bbox[2]) / 2
                    det_cy = (bbox[1] + bbox[3]) / 2

                    dist = ((det_cx - cx) ** 2 + (det_cy - cy) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_det = det

                if closest_det:
                    closest_det.dog_id = dog_id

        return detections

    def _parse_detections(self, outputs: np.ndarray, frame_shape: Tuple[int, ...]) -> List[Detection]:
        """Parse detection model outputs"""
        # Implementation depends on your detection model output format
        # This is a placeholder - adapt to your model
        detections = []

        # Example parsing (adjust based on actual model output)
        if len(outputs.shape) == 3:  # (1, num_detections, 6)
            for i in range(outputs.shape[1]):
                conf = outputs[0, i, 4]
                if conf > 0.5:
                    x1, y1, x2, y2 = outputs[0, i, :4]
                    # Scale to frame size
                    h, w = frame_shape[:2]
                    bbox = [x1 * w, y1 * h, x2 * w, y2 * h]
                    detections.append(Detection(bbox=bbox, confidence=float(conf)))

        return detections

    def _parse_keypoints(self, outputs: np.ndarray) -> Optional[np.ndarray]:
        """Parse pose model outputs to keypoints"""
        try:
            # Assuming 72 channels total (24 keypoints * 3 values each)
            if outputs.size >= 72:
                keypoints = outputs.flatten()[:72]
                return keypoints.reshape(24, 3)
            return None
        except Exception as e:
            logger.error(f"Keypoint parsing failed: {e}")
            return None

    def cleanup(self):
        """Cleanup resources"""
        if self.detection_network:
            self.detection_network.release()
        if self.pose_network:
            self.pose_network.release()
        if self.vdevice:
            self.vdevice.release()

        logger.info("Temporal AI Controller cleaned up")


def test_temporal_controller():
    """Test the temporal AI controller"""
    print("🧪 Testing Temporal AI Controller with behavior_14.ts...")

    controller = AI3StageControllerTemporal()

    if not controller.initialize():
        print("❌ Failed to initialize controller")
        return

    print("✅ Controller initialized with temporal model")

    # Test with dummy frame
    test_frame = np.zeros((640, 640, 3), dtype=np.uint8)

    # Process multiple frames to build temporal history
    print("\n📊 Processing frames to build temporal sequence...")
    for i in range(20):
        detections, poses, behaviors = controller.process_frame(test_frame)

        if behaviors:
            print(f"Frame {i+1}: Detected behaviors:")
            for b in behaviors:
                print(f"  - {b.behavior} (conf={b.confidence:.2f})")
        else:
            print(f"Frame {i+1}: Building temporal history...")

    controller.cleanup()
    print("\n✅ Test complete")


if __name__ == "__main__":
    test_temporal_controller()