#!/usr/bin/env python3
"""
3-Stage AI Controller for TreatSensei DogBot
Stage 1: Dog Detection (dogdetector_14.hef) - 640x640
Stage 2: Pose Estimation (dogpose_14.hef) - 640x640 crops
Stage 3: Behavior Analysis (CPU) - temporal keypoint analysis
"""

import numpy as np
import cv2
import time
import logging
import json
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from collections import deque
from dataclasses import dataclass

try:
    from hailo_platform.pyhailort.pyhailort import InferModel, HailoStreamInterface, FormatType, Device
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

class AI3StageController:
    """3-Stage AI controller: Detection -> Pose -> Behavior"""

    def __init__(self, config_path: str = "ai/models/config.json"):
        """Initialize 3-stage AI controller"""
        self.initialized = False
        self.config_path = config_path
        self.config = None

        # Stage 1: Detection
        self.detection_device = None
        self.detection_model = None
        self.detection_configured_model = None
        self.detection_bindings = None

        # Stage 2: Pose
        self.pose_device = None
        self.pose_model = None
        self.pose_configured_model = None
        self.pose_bindings = None

        # Stage 3: Behavior (CPU)
        self.behavior_history = deque(maxlen=16)  # T=16 frames
        self.behavior_cooldowns = {}

        # Camera settings
        self.camera_rotation = 0  # Fixed at 0 as per your notes
        self.input_size = (640, 640)

        # Load configuration
        self._load_config()

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
        """Initialize all 3 stages"""
        if not HAILO_AVAILABLE:
            logger.error("Hailo platform not available")
            return False

        try:
            # Initialize Stage 1: Detection
            if not self._init_detection():
                return False

            # Initialize Stage 2: Pose
            if not self._init_pose():
                return False

            # Stage 3: Behavior (CPU-based, no init needed)
            logger.info("Stage 3: Behavior analysis ready (CPU)")

            self.initialized = True
            logger.info("3-Stage AI Controller initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize AI controller: {e}")
            return False

    def _init_detection(self) -> bool:
        """Initialize Stage 1: Dog Detection"""
        try:
            model_path = Path("ai/models") / self.config["detect_path"]
            if not model_path.exists():
                logger.error(f"Detection model not found: {model_path}")
                return False

            # Create device
            self.detection_device = Device()
            logger.info("Detection device created")

            # Create InferModel
            self.detection_model = InferModel(self.detection_device, str(model_path))
            logger.info("Detection InferModel created")

            # Configure model
            self.detection_configured_model = self.detection_model.configure()
            logger.info("Detection model configured")

            # Create bindings
            self.detection_bindings = self.detection_configured_model.create_bindings()
            logger.info(f"Detection bindings created: inputs={list(self.detection_bindings.input.keys())}, outputs={list(self.detection_bindings.output.keys())}")

            logger.info("Stage 1: Dog detection model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize detection model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_pose(self) -> bool:
        """Initialize Stage 2: Pose Estimation"""
        try:
            model_path = Path("ai/models") / self.config["hef_path"]
            if not model_path.exists():
                logger.error(f"Pose model not found: {model_path}")
                return False

            # Create separate device for pose model
            self.pose_device = Device()
            logger.info("Pose device created")

            # Create InferModel
            self.pose_model = InferModel(self.pose_device, str(model_path))
            logger.info("Pose InferModel created")

            # Configure model
            self.pose_configured_model = self.pose_model.configure()
            logger.info("Pose model configured")

            # Create bindings
            self.pose_bindings = self.pose_configured_model.create_bindings()
            logger.info(f"Pose bindings created: inputs={list(self.pose_bindings.input.keys())}, outputs={list(self.pose_bindings.output.keys())}")

            logger.info("Stage 2: Pose estimation model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize pose model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def process_frame(self, frame_4k: np.ndarray) -> Tuple[List[Detection], List[PoseKeypoints], List[BehaviorResult]]:
        """
        Process full pipeline on 4K frame
        Returns: (detections, poses, behaviors)
        """
        if not self.initialized:
            return [], [], []

        try:
            # Stage 1: Detect dogs at 640x640
            detections = self._stage1_detect_dogs(frame_4k)

            if not detections:
                return detections, [], []

            # Stage 2: Get poses for each detected dog
            poses = []
            for detection in detections:
                pose = self._stage2_estimate_pose(frame_4k, detection)
                if pose:
                    poses.append(pose)

            # Stage 3: Analyze behaviors from pose history
            behaviors = self._stage3_analyze_behavior(poses)

            return detections, poses, behaviors

        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return [], [], []

    def _stage1_detect_dogs(self, frame_4k: np.ndarray) -> List[Detection]:
        """Stage 1: Dog detection on downsampled frame"""
        try:
            # Downsample 4K to 640x640 for detection
            h_4k, w_4k = frame_4k.shape[:2]
            frame_640 = cv2.resize(frame_4k, self.input_size)

            # Get input name
            input_name = list(self.detection_bindings.input.keys())[0]

            # Prepare input (uint8 format as expected by the model)
            input_data = frame_640.astype(np.uint8)

            # Set input
            self.detection_bindings.input[input_name] = input_data

            # Run inference
            self.detection_configured_model.run(self.detection_bindings)

            # Parse detections and scale back to 4K coordinates
            detections = []
            scale_x = w_4k / 640
            scale_y = h_4k / 640

            # Parse YOLO detection outputs
            for output_name, output_data in self.detection_bindings.output.items():
                # Assuming standard YOLO format: [batch, boxes, 6] or [boxes, 6]
                if len(output_data.shape) >= 2:
                    if len(output_data.shape) == 3:
                        boxes = output_data[0]  # Remove batch dimension
                    else:
                        boxes = output_data

                    for box in boxes:
                        if len(box) >= 5:
                            x1, y1, x2, y2, conf = box[:5]

                            if conf > 0.3:  # Confidence threshold
                                # Scale back to 4K coordinates
                                x1_4k = int(x1 * scale_x)
                                y1_4k = int(y1 * scale_y)
                                x2_4k = int(x2 * scale_x)
                                y2_4k = int(y2 * scale_y)

                                detections.append(Detection(x1_4k, y1_4k, x2_4k, y2_4k, float(conf)))

            logger.debug(f"Stage 1: Found {len(detections)} dogs")
            return detections

        except Exception as e:
            logger.error(f"Stage 1 detection error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _stage2_estimate_pose(self, frame_4k: np.ndarray, detection: Detection) -> Optional[PoseKeypoints]:
        """Stage 2: Pose estimation on cropped region"""
        try:
            # Extract crop from 4K frame with padding
            crop_4k = self._extract_crop_with_padding(frame_4k, detection, padding=0.1)

            # Resize crop to 640x640 for pose model
            crop_640 = cv2.resize(crop_4k, self.input_size)

            # Get input name
            input_name = list(self.pose_bindings.input.keys())[0]

            # Prepare input (uint8 format as expected by the model)
            input_data = crop_640.astype(np.uint8)

            # Set input
            self.pose_bindings.input[input_name] = input_data

            # Run pose inference
            self.pose_configured_model.run(self.pose_bindings)

            # Parse keypoints from pose output
            keypoints = self._parse_pose_output(self.pose_bindings.output)

            if keypoints is not None:
                return PoseKeypoints(keypoints, detection)

            return None

        except Exception as e:
            logger.error(f"Stage 2 pose error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _stage3_analyze_behavior(self, poses: List[PoseKeypoints]) -> List[BehaviorResult]:
        """Stage 3: CPU-based behavior analysis"""
        behaviors = []

        if not poses:
            return behaviors

        try:
            # Add current poses to history
            current_time = time.time()
            self.behavior_history.append({
                'timestamp': current_time,
                'poses': poses
            })

            # Need enough history for temporal analysis
            if len(self.behavior_history) < 8:
                return behaviors

            # Simple behavior analysis (can be enhanced)
            for i, pose in enumerate(poses):
                behavior = self._analyze_single_pose_sequence(pose, i)
                if behavior:
                    behaviors.append(BehaviorResult(behavior, 0.8, current_time))

            return behaviors

        except Exception as e:
            logger.error(f"Stage 3 behavior error: {e}")
            return behaviors

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

    def _parse_pose_output(self, outputs: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Parse pose keypoints from model output"""
        try:
            # Expecting pose output format: [batch, 24, 3] for 24 keypoints with x,y,confidence
            for output_name, output_data in outputs.items():
                if len(output_data.shape) >= 2:
                    if len(output_data.shape) == 3 and output_data.shape[-1] == 3:
                        # Remove batch dimension and return keypoints
                        keypoints = output_data[0]  # Shape: [24, 3]
                        return keypoints
                    elif len(output_data.shape) == 2 and output_data.shape[-1] == 3:
                        # Already flattened, just return
                        return output_data

            logger.warning(f"Could not parse pose output format: {[(name, data.shape) for name, data in outputs.items()]}")
            return None

        except Exception as e:
            logger.error(f"Error parsing pose output: {e}")
            return None

    def _analyze_single_pose_sequence(self, current_pose: PoseKeypoints, pose_index: int) -> Optional[str]:
        """Analyze behavior for a single pose over time"""
        try:
            # Simple placeholder behavior analysis
            # This would be replaced with actual temporal analysis

            keypoints = current_pose.keypoints

            # Example: detect if dog is lying down based on y-coordinates
            if len(keypoints) >= 10:
                body_points = keypoints[:10]  # First 10 keypoints assumed to be body
                avg_y = np.mean(body_points[:, 1])

                # Simple heuristic - if body points are low, might be lying
                frame_height = 640  # Known from our input size
                if avg_y > frame_height * 0.7:
                    return "lie"
                elif avg_y < frame_height * 0.3:
                    return "stand"
                else:
                    return "sit"

            return None

        except Exception as e:
            logger.error(f"Error analyzing pose sequence: {e}")
            return None

    def cleanup(self):
        """Clean up resources"""
        try:
            # No explicit cleanup needed for InferModel - handled automatically
            self.detection_device = None
            self.detection_model = None
            self.detection_configured_model = None
            self.detection_bindings = None

            self.pose_device = None
            self.pose_model = None
            self.pose_configured_model = None
            self.pose_bindings = None

            logger.info("AI Controller cleaned up")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "initialized": self.initialized,
            "hailo_available": HAILO_AVAILABLE,
            "detection_model": self.config.get("detect_path", "N/A") if self.config else "N/A",
            "pose_model": self.config.get("hef_path", "N/A") if self.config else "N/A",
            "input_size": self.input_size,
            "behavior_history_length": len(self.behavior_history)
        }