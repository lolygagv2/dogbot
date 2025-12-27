#!/usr/bin/env python3
"""
3-Stage AI Controller for TreatSensei DogBot - SINGLE MODEL VERSION
Uses dogpose_14.hef for BOTH detection and pose (YOLOv8-pose outputs both)
Stage 1: Dog Detection + Pose from single inference
Stage 2: (Combined with Stage 1 - no separate inference needed)
Stage 3: Behavior Analysis (CPU) - temporal keypoint analysis
"""

import numpy as np
import cv2
import time
import logging
import json
import torch
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from core.dog_tracker import DogTracker
from core.event_publisher import DogEventPublisher
from core.dog_database import DogDatabase
from core.bus import EventBus

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

try:
    import hailo_platform as hpf
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("Warning: Hailo platform not available")

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    """Dog detection result"""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

@dataclass
class PoseKeypoints:
    """Dog pose keypoints result"""
    keypoints: np.ndarray  # Shape: (24, 3) - x, y, confidence
    detection: Detection

@dataclass
class BehaviorResult:
    """Behavior analysis result"""
    behavior: str
    confidence: float
    timestamp: float

class AI3StageControllerFixed:
    """3-Stage AI controller using working HEF direct API"""

    def __init__(self, config_path: str = "ai/models/config.json"):
        """Initialize 3-stage AI controller"""
        self.initialized = False
        self.config_path = config_path
        self.config = None

        # VDevice for model loading
        self.vdevice = None
        self.model_loaded = False

        # Single model path (pose model does both detection + pose)
        self.model_path = None

        # TorchScript behavior model
        self.behavior_model = None
        self.behavior_model_path = None

        # Currently active model components
        self.active_hef = None
        self.active_network_group = None
        self.active_network_group_params = None
        self.active_input_vstreams_params = None
        self.active_output_vstreams_params = None
        self.active_input_info = None
        self.active_output_infos = None

        # Stage 3: Temporal behavior analysis
        self.T = 16  # Temporal window size (frames)
        self.pose_history = deque(maxlen=self.T)  # Stores list of keypoints per frame
        self.behaviors = ["stand", "sit", "lie", "cross", "spin"]  # Behavior classes

        # Frame dimensions for behavior normalization (updated each frame)
        self.current_frame_w = 640
        self.current_frame_h = 640
        self.behavior_cooldowns = {}  # Per-dog cooldowns
        self.cooldown_timers = {}
        self.prob_th = 0.7  # Behavior probability threshold
        self.behavior_confidence_threshold = 0.7  # Alias for behavior publishing

        # Camera settings
        self.camera_rotation = 0  # Fixed at 0 as per your notes
        self.input_size = (640, 640)

        # Load configuration
        self._load_config()

        # Initialize dog tracker with persistence rules
        self.dog_tracker = DogTracker(self.config) if self.config else None

        # Initialize event system and database
        self.event_bus = EventBus()
        self.event_publisher = DogEventPublisher(self.event_bus)
        self.dog_database = DogDatabase()

    def _load_config(self):
        """Load configuration from JSON"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                config_file = Path("ai/models/config.json")

            with open(config_file, 'r') as f:
                self.config = json.load(f)

            logger.info(f"Loaded config: {self.config}")

            # Update settings from config
            self.input_size = tuple(self.config["imgsz"])
            self.behavior_cooldowns = self.config["cooldown_s"]

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            # Use defaults
            self.config = {
                "imgsz": [640, 640],
                "behaviors": ["stand", "sit", "lie", "cross", "spin"],
                "prob_th": 0.7,
                "hef_path": "dogpose_14.hef",
                "detect_path": "dogdetector_14.hef"
            }

    def initialize(self) -> bool:
        """Initialize AI controller - load pose model (Hailo) and behavior model (TorchScript)"""
        if not HAILO_AVAILABLE:
            logger.error("Hailo platform not available")
            return False

        try:
            # Create VDevice
            logger.info("Creating VDevice for single-model inference")
            self.vdevice = hpf.VDevice()
            logger.info("VDevice created successfully")

            # Use pose model for everything (it outputs both detection boxes and keypoints)
            self.model_path = Path("ai/models") / self.config["hef_path"]  # dogpose_14.hef

            # Verify pose model exists
            if not self.model_path.exists():
                logger.error(f"Pose model not found: {self.model_path}")
                return False

            logger.info(f"Using single model for detection+pose: {self.model_path.name}")

            # Load the pose model (Hailo)
            if not self._load_model():
                logger.error("Failed to load pose model")
                return False
            logger.info("Pose model loaded successfully (provides both detection + keypoints)")

            # Load TorchScript behavior model (CPU)
            self.behavior_model_path = Path("ai/models") / self.config.get("behavior_head_ts", "behavior_14.ts")
            if self.behavior_model_path.exists():
                self.behavior_model = torch.jit.load(str(self.behavior_model_path), map_location="cpu").eval()
                logger.info(f"TorchScript behavior model loaded: {self.behavior_model_path.name}")
            else:
                logger.warning(f"Behavior model not found: {self.behavior_model_path}, using CPU heuristics")
                self.behavior_model = None

            # Update config-based settings
            self.T = self.config.get("T", 16)
            self.behaviors = self.config.get("behaviors", ["stand", "sit", "lie", "cross", "spin"])
            self.prob_th = self.config.get("prob_th", 0.7)
            self.behavior_cooldowns = self.config.get("cooldown_s", {})

            self.initialized = True
            logger.info("AI Controller initialized with single-model Hailo + TorchScript behavior")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AI controller: {e}")
            return False

    def _load_model(self) -> bool:
        """Load the pose model (does both detection and pose)"""
        if self.model_loaded:
            return True  # Already loaded

        # Verify VDevice is available
        if self.vdevice is None:
            logger.error("VDevice not available - cannot load model")
            return False

        try:
            # Load HEF
            self.active_hef = hpf.HEF(str(self.model_path))

            # Configure VDevice
            params = hpf.ConfigureParams.create_from_hef(
                hef=self.active_hef,
                interface=hpf.HailoStreamInterface.PCIe
            )

            network_groups = self.vdevice.configure(self.active_hef, params)
            self.active_network_group = network_groups[0]

            # Get stream info
            self.active_input_info = self.active_hef.get_input_vstream_infos()[0]
            self.active_output_infos = self.active_hef.get_output_vstream_infos()

            # Create parameters
            self.active_network_group_params = self.active_network_group.create_params()

            self.active_input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
                self.active_network_group,
                quantized=True,
                format_type=hpf.FormatType.UINT8
            )

            self.active_output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
                self.active_network_group,
                quantized=False,
                format_type=hpf.FormatType.FLOAT32
            )

            self.model_loaded = True
            logger.info(f"Pose model loaded: {self.model_path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load pose model: {e}")
            return False


    def process_frame_with_dogs(self, frame_4k: np.ndarray, aruco_markers: List[Tuple[int, float, float]] = None) -> Dict:
        """
        Process frame with dog identification using ArUco markers

        Args:
            frame_4k: Input frame
            aruco_markers: List of (marker_id, cx, cy) tuples from ArUco detection

        Returns:
            Dict with detections, poses, behaviors, and dog assignments
        """
        # Get basic detections
        detections, poses, behaviors = self.process_frame(frame_4k)

        result = {
            'detections': detections,
            'poses': poses,
            'behaviors': behaviors,
            'dog_assignments': {}
        }

        # Apply dog tracking if available
        if self.dog_tracker and aruco_markers is not None:
            assignments = self.dog_tracker.process_frame(detections, aruco_markers)
            result['dog_assignments'] = assignments

            # Add dog names to behaviors and publish events
            for idx, behavior in enumerate(behaviors):
                if idx in assignments:
                    behavior.dog_name = assignments[idx]
                    dog_name = assignments[idx]

                    # Get marker ID from dog name
                    marker_id = self.dog_tracker.dog_names.get(dog_name, 832)

                    # Publish behavior detection event
                    if behavior.behavior and behavior.confidence > self.behavior_confidence_threshold:
                        self.event_publisher.publish_behavior_detected(
                            dog_name=dog_name,
                            dog_id=marker_id,
                            behavior=behavior.behavior,
                            confidence=behavior.confidence
                        )

                        # Record in database
                        self.dog_database.record_behavior(marker_id, behavior.behavior, behavior.confidence)

                        # If it's a successful "sit", dispense reward and record it
                        if behavior.behavior == "sit" and behavior.confidence > 0.7:
                            self.event_publisher.publish_reward_dispensed(
                                dog_name=dog_name,
                                dog_id=marker_id,
                                behavior="sit",
                                treat_count=1
                            )
                            self.dog_database.record_reward(marker_id, "sit", 1)

        return result

    def get_dog_progress_report(self, marker_id: int) -> str:
        """Generate progress report for a specific dog"""
        return self.dog_database.generate_progress_report(marker_id)

    def process_frame(self, frame_4k: np.ndarray) -> Tuple[List[Detection], List[PoseKeypoints], List[BehaviorResult]]:
        """
        Process full pipeline on 4K frame using single-model inference
        Returns: (detections, poses, behaviors)
        """
        if not self.initialized:
            return [], [], []

        try:
            # Single inference gets both detections and poses
            detections, poses = self._run_combined_inference(frame_4k)

            if not detections:
                return detections, [], []

            # Stage 3: Analyze behaviors from pose history
            behaviors = self._stage3_analyze_behavior(poses)

            return detections, poses, behaviors

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return [], [], []

    def _run_combined_inference(self, frame_4k: np.ndarray) -> Tuple[List[Detection], List[PoseKeypoints]]:
        """Run single inference to get both detections and poses from pose model"""
        detections = []
        poses = []

        try:
            if not self.model_loaded:
                logger.error("Model not loaded")
                return [], []

            # Downsample 4K to 640x640 for inference
            h_4k, w_4k = frame_4k.shape[:2]
            # Store frame dimensions for proper behavior normalization
            self.current_frame_w = w_4k
            self.current_frame_h = h_4k
            frame_640 = cv2.resize(frame_4k, self.input_size)

            # Prepare input (BGR format)
            frame_input = frame_640

            if not self.active_network_group:
                raise RuntimeError("No active network group")

            with self.active_network_group.activate(self.active_network_group_params):
                with hpf.InferVStreams(self.active_network_group,
                                      self.active_input_vstreams_params,
                                      self.active_output_vstreams_params) as infer_pipeline:

                    # Prepare input - ensure UINT8 and correct shape
                    input_tensor = np.expand_dims(frame_input, axis=0).astype(np.uint8)
                    input_name = list(self.active_input_vstreams_params.keys())[0]
                    input_data = {input_name: input_tensor}

                    # Run inference
                    output_data = infer_pipeline.infer(input_data)

                    # Parse BOTH detections and keypoints from pose model output
                    detections, poses = self._parse_pose_model_output(output_data, w_4k, h_4k)

            return detections, poses

        except Exception as e:
            logger.error(f"Combined inference error: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    def _parse_pose_model_output(self, outputs: Dict[str, np.ndarray], orig_w: int, orig_h: int) -> Tuple[List[Detection], List[PoseKeypoints]]:
        """Parse pose model output to get both detections and keypoints"""
        detections = []
        poses = []

        try:
            # Group outputs by scale for 640x640 input
            # Expected scales for 640x640: 80x80, 40x40, 20x20
            scales = {
                '80x80': {'bbox': None, 'kpts': None, 'conf': None},
                '40x40': {'bbox': None, 'kpts': None, 'conf': None},
                '20x20': {'bbox': None, 'kpts': None, 'conf': None}
            }

            # Map outputs based on shape patterns
            for layer_name, output in outputs.items():
                if len(output.shape) != 4:
                    continue

                h, w = output.shape[1], output.shape[2]
                channels = output.shape[3]

                scale_name = None
                output_type = None

                if (h, w) == (80, 80):
                    scale_name = '80x80'
                elif (h, w) == (40, 40):
                    scale_name = '40x40'
                elif (h, w) == (20, 20):
                    scale_name = '20x20'

                if scale_name and channels == 64:
                    output_type = 'bbox'
                elif scale_name and channels == 72:
                    output_type = 'kpts'
                elif scale_name and channels == 1:
                    output_type = 'conf'

                if scale_name and output_type:
                    scales[scale_name][output_type] = output

            # Find detections across all scales
            strides = [8, 16, 32]
            scale_names = ['80x80', '40x40', '20x20']
            scale_factors = (orig_w / 640, orig_h / 640)

            for scale_idx, scale_name in enumerate(scale_names):
                scale_data = scales[scale_name]

                if not all(scale_data[key] is not None for key in ['bbox', 'kpts', 'conf']):
                    continue

                bbox_out = scale_data['bbox']
                kpts_out = scale_data['kpts']
                conf_out = scale_data['conf']

                _, h, w, _ = bbox_out.shape
                stride = strides[scale_idx]

                # Process cells with high confidence
                for i in range(h):
                    for j in range(w):
                        conf_raw = conf_out[0, i, j, 0]
                        conf_raw = np.clip(conf_raw, -500, 500)
                        conf = 1.0 / (1.0 + np.exp(-conf_raw))  # Sigmoid

                        if conf < 0.3:  # Confidence threshold
                            continue

                        # Decode bounding box from DFL format (64 channels = 4 sides * 16 bins)
                        bbox_raw = bbox_out[0, i, j, :]

                        # DFL decode: reshape to (4, 16), softmax, weighted sum
                        bbox_dfl = bbox_raw.reshape(4, 16)
                        # Softmax
                        bbox_dfl_exp = np.exp(bbox_dfl - np.max(bbox_dfl, axis=1, keepdims=True))
                        bbox_probs = bbox_dfl_exp / np.sum(bbox_dfl_exp, axis=1, keepdims=True)
                        # Weighted sum with bins 0-15
                        bins = np.arange(16)
                        dist = np.sum(bbox_probs * bins, axis=1)  # [left, top, right, bottom]

                        # Convert to coordinates
                        cx = (j + 0.5) * stride
                        cy = (i + 0.5) * stride
                        x1 = (cx - dist[0] * stride)
                        y1 = (cy - dist[1] * stride)
                        x2 = (cx + dist[2] * stride)
                        y2 = (cy + dist[3] * stride)

                        # Scale to original frame size
                        x1_scaled = int(x1 * scale_factors[0])
                        y1_scaled = int(y1 * scale_factors[1])
                        x2_scaled = int(x2 * scale_factors[0])
                        y2_scaled = int(y2 * scale_factors[1])

                        # Clamp to frame bounds
                        x1_scaled = max(0, min(x1_scaled, orig_w))
                        y1_scaled = max(0, min(y1_scaled, orig_h))
                        x2_scaled = max(0, min(x2_scaled, orig_w))
                        y2_scaled = max(0, min(y2_scaled, orig_h))

                        # Valid bounding box check
                        if x2_scaled <= x1_scaled or y2_scaled <= y1_scaled:
                            continue

                        detection = Detection(x1_scaled, y1_scaled, x2_scaled, y2_scaled, float(conf))
                        detections.append(detection)

                        # Decode keypoints (72 channels = 24 keypoints * 3 [x, y, conf])
                        kpts_raw = kpts_out[0, i, j, :]
                        kpts = np.zeros((24, 3), dtype=np.float32)

                        for k in range(24):
                            # Keypoint decoding: raw values are offsets from anchor cell
                            # Must add cell position (j, i) before scaling to get pixel coords
                            kpts[k, 0] = (kpts_raw[k * 3] + j) * stride * scale_factors[0]
                            kpts[k, 1] = (kpts_raw[k * 3 + 1] + i) * stride * scale_factors[1]
                            kpts[k, 2] = 1.0 / (1.0 + np.exp(-kpts_raw[k * 3 + 2]))

                        poses.append(PoseKeypoints(kpts, detection))

            # Apply NMS to remove duplicate detections
            if len(detections) > 1:
                detections, poses = self._apply_nms(detections, poses, iou_threshold=0.5)

            return detections, poses

        except Exception as e:
            logger.error(f"Error parsing pose model output: {e}")
            import traceback
            traceback.print_exc()
            return [], []

    def _apply_nms(self, detections: List[Detection], poses: List[PoseKeypoints], iou_threshold: float = 0.5) -> Tuple[List[Detection], List[PoseKeypoints]]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(detections) <= 1:
            return detections, poses

        # Sort by confidence
        indices = sorted(range(len(detections)), key=lambda i: detections[i].confidence, reverse=True)
        keep = []

        while indices:
            current = indices.pop(0)
            keep.append(current)

            remaining = []
            for idx in indices:
                iou = self._compute_iou(detections[current], detections[idx])
                if iou < iou_threshold:
                    remaining.append(idx)
            indices = remaining

        return [detections[i] for i in keep], [poses[i] for i in keep]

    def _compute_iou(self, det1: Detection, det2: Detection) -> float:
        """Compute IoU between two detections"""
        x1 = max(det1.x1, det2.x1)
        y1 = max(det1.y1, det2.y1)
        x2 = min(det1.x2, det2.x2)
        y2 = min(det1.y2, det2.y2)

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = det1.width * det1.height
        area2 = det2.width * det2.height
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _stage3_analyze_behavior(self, poses: List[PoseKeypoints]) -> List[BehaviorResult]:
        """Stage 3: Temporal behavior analysis using TorchScript model"""
        behaviors = []

        if not poses:
            # Still add empty frame to history to maintain temporal consistency
            self.pose_history.append([])
            return behaviors

        try:
            # Store full PoseKeypoints objects (including bboxes) for proper normalization
            self.pose_history.append(poses)

            # Need enough temporal history
            if len(self.pose_history) < self.T:
                return behaviors

            # Use TorchScript model if available
            if self.behavior_model is not None:
                behaviors = self._analyze_with_torchscript(poses)
            else:
                # Fallback to simple heuristics
                current_time = time.time()
                for i, pose in enumerate(poses):
                    behavior = self._classify_pose_heuristic(pose.keypoints)
                    if behavior and self._check_cooldown(behavior, i):
                        behaviors.append(BehaviorResult(behavior, 0.8, current_time))
                        self._update_cooldown(behavior, i)

            return behaviors

        except Exception as e:
            logger.error(f"Stage 3 behavior error: {e}")
            return behaviors

    def _analyze_with_torchscript(self, current_poses: List[PoseKeypoints]) -> List[BehaviorResult]:
        """Analyze behavior using TorchScript temporal model"""
        behaviors = []
        current_time = time.time()

        try:
            # Get temporal sequence from history (now contains full PoseKeypoints objects)
            sequence = list(self.pose_history)

            # Build tensor from sequence: (T, num_dogs, 24, 2)
            max_dogs = max(len(frame_poses) for frame_poses in sequence if frame_poses)
            if max_dogs == 0:
                return behaviors

            tensor_data = []

            for frame_idx, frame_poses in enumerate(sequence):
                frame_data = []
                for i in range(max_dogs):
                    if i < len(frame_poses):
                        pose = frame_poses[i]
                        kpts = pose.keypoints
                        if kpts.shape[0] >= 24 and kpts.shape[1] >= 2:
                            pose_data = kpts[:24, :2].copy()

                            # Normalize keypoints RELATIVE TO BOUNDING BOX (as model was trained)
                            det = pose.detection
                            x1, y1, x2, y2 = det.x1, det.y1, det.x2, det.y2
                            w = max(x2 - x1, 1e-6)
                            h = max(y2 - y1, 1e-6)

                            pose_data[:, 0] = np.clip((pose_data[:, 0] - x1) / w, 0, 1)
                            pose_data[:, 1] = np.clip((pose_data[:, 1] - y1) / h, 0, 1)
                        else:
                            pose_data = np.zeros((24, 2))
                        frame_data.append(pose_data)
                    else:
                        frame_data.append(np.zeros((24, 2)))
                tensor_data.append(frame_data)

            # Convert to tensor: (T, num_dogs, 24, 2)
            tensor_input = torch.tensor(np.array(tensor_data), dtype=torch.float32)

            # Center the input around 0 (training data was mean-centered)
            # Shift from [0, 1] to [-0.5, 0.5]
            tensor_input = tensor_input - 0.5

            # Reshape for model: (num_dogs, T, 48)
            # From (T, num_dogs, 24, 2) to (num_dogs, T, 24*2)
            T_len = len(tensor_data)
            num_dogs = tensor_input.shape[1]

            tensor_input = tensor_input.transpose(0, 1)  # (num_dogs, T, 24, 2)
            tensor_input = tensor_input.reshape(num_dogs, T_len, -1)  # (num_dogs, T, 48)

            # Run TorchScript model
            with torch.no_grad():
                output = self.behavior_model(tensor_input)

            # Apply softmax to get probabilities
            probs = torch.softmax(output, dim=-1).numpy()

            # Handle different output shapes
            if probs.ndim == 1:
                probs = probs.reshape(1, -1)

            # Get behaviors for each dog
            for dog_idx in range(min(max_dogs, len(probs))):
                dog_probs = probs[dog_idx]
                max_idx = np.argmax(dog_probs)
                max_prob = float(dog_probs[max_idx])

                # Debug: log all probabilities periodically
                if dog_idx == 0:
                    if not hasattr(self, '_behavior_log_counter'):
                        self._behavior_log_counter = 0
                    self._behavior_log_counter += 1
                    if self._behavior_log_counter % 15 == 0:  # Every 15 frames
                        prob_str = ", ".join([f"{b}:{p:.2f}" for b, p in zip(self.behaviors, dog_probs)])
                        logger.info(f"Behavior probs: [{prob_str}]")

                if max_prob > self.prob_th:
                    if max_idx < len(self.behaviors):
                        behavior_name = self.behaviors[max_idx]

                        # Check cooldown
                        if self._check_cooldown(behavior_name, dog_idx):
                            behaviors.append(BehaviorResult(behavior_name, max_prob, current_time))
                            self._update_cooldown(behavior_name, dog_idx)
                            logger.info(f"🎯 BEHAVIOR: {behavior_name} (conf={max_prob:.2f})")

            return behaviors

        except Exception as e:
            logger.error(f"TorchScript behavior analysis failed: {e}")
            return behaviors

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

    def _classify_pose_heuristic(self, keypoints: np.ndarray) -> Optional[str]:
        """Simple heuristic fallback for pose classification"""
        try:
            if not isinstance(keypoints, np.ndarray) or len(keypoints) < 12:
                return None

            body_points = keypoints[:12]
            if body_points.ndim == 2 and body_points.shape[1] >= 3:
                conf_mask = body_points[:, 2] > 0.3
                valid = body_points[conf_mask]
                if len(valid) < 4:
                    return None
            else:
                return None

            avg_y = np.mean(valid[:, 1])
            y_ratio = avg_y / 640

            if y_ratio > 0.72:
                return "lie"
            elif y_ratio < 0.35:
                return "stand"
            elif 0.45 <= y_ratio <= 0.68:
                return "sit"
            else:
                return "stand"

        except Exception:
            return None

    def _extract_crop_with_padding(self, frame: np.ndarray, detection: Detection, padding: float = 0.1) -> np.ndarray:
        """Extract crop from frame with padding"""
        h, w = frame.shape[:2]

        # Add padding
        width = detection.width
        height = detection.height
        pad_w = int(width * padding)
        pad_h = int(height * padding)

        # Calculate crop bounds with padding
        x1 = max(0, detection.x1 - pad_w)
        y1 = max(0, detection.y1 - pad_h)
        x2 = min(w, detection.x2 + pad_w)
        y2 = min(h, detection.y2 + pad_h)

        return frame[y1:y2, x1:x2]

    def cleanup(self):
        """Clean up resources"""
        try:
            # Clear model references
            self.active_hef = None
            self.active_network_group = None
            self.active_network_group_params = None
            self.active_input_vstreams_params = None
            self.active_output_vstreams_params = None
            self.active_input_info = None
            self.active_output_infos = None
            self.model_loaded = False

            # Final VDevice cleanup
            if self.vdevice:
                self.vdevice = None

            self.initialized = False
            logger.info("AI Controller cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "initialized": self.initialized,
            "hailo_available": HAILO_AVAILABLE,
            "model": self.config.get("hef_path", "N/A") if self.config else "N/A",
            "model_loaded": self.model_loaded,
            "input_size": self.input_size,
            "behavior_history_length": len(self.behavior_history),
            "vdevice_ready": self.vdevice is not None,
            "active_model_loaded": self.active_hef is not None
        }