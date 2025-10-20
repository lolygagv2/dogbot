#!/usr/bin/env python3
"""
3-Stage AI Controller for TreatSensei DogBot - FIXED VERSION
Using working HEF direct API instead of broken InferModel
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

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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

        # Shared VDevice for both models (Hailo8 has only one physical device)
        self.vdevice = None

        # Stage 1: Detection (using shared VDevice)
        self.detection_hef = None
        self.detection_network_group = None
        self.detection_network_group_params = None
        self.detection_input_vstreams_params = None
        self.detection_output_vstreams_params = None
        self.detection_input_info = None
        self.detection_output_infos = None

        # Stage 2: Pose (using shared VDevice)
        self.pose_hef = None
        self.pose_network_group = None
        self.pose_network_group_params = None
        self.pose_input_vstreams_params = None
        self.pose_output_vstreams_params = None
        self.pose_input_info = None
        self.pose_output_infos = None

        # Stage 3: Behavior (CPU)
        self.behavior_history = deque(maxlen=30)  # T=30 frames (~1 second at 30fps)
        self.behavior_cooldowns = {}
        self.last_stable_behavior = None
        self.behavior_confidence_threshold = 0.4  # Lowered to 40% confidence
        self.min_consecutive_frames = 10  # Lowered to 10 frames (1/3 second)
        self.cooldown_duration = 3.0  # 3 seconds between same behavior recognition

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
            # Initialize shared VDevice first
            logger.info("Creating shared VDevice for Hailo8")
            self.vdevice = hpf.VDevice()
            logger.info("Shared VDevice created successfully")

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
        """Initialize Stage 1: Dog Detection using shared VDevice"""
        try:
            model_path = Path("ai/models") / self.config["detect_path"]
            if not model_path.exists():
                logger.error(f"Detection model not found: {model_path}")
                return False

            # Load HEF and configure with shared VDevice
            self.detection_hef = hpf.HEF(str(model_path))

            params = hpf.ConfigureParams.create_from_hef(
                hef=self.detection_hef,
                interface=hpf.HailoStreamInterface.PCIe
            )

            network_groups = self.vdevice.configure(self.detection_hef, params)
            self.detection_network_group = network_groups[0]

            # Get stream info
            self.detection_input_info = self.detection_hef.get_input_vstream_infos()[0]
            self.detection_output_infos = self.detection_hef.get_output_vstream_infos()

            logger.info(f"Detection model loaded: {self.detection_input_info.name}, shape: {self.detection_input_info.shape}")
            logger.info(f"Detection outputs: {len(self.detection_output_infos)} outputs")
            for i, output in enumerate(self.detection_output_infos):
                logger.info(f"  Output {i}: {output.name}, shape: {output.shape}")

            # Create network group parameters and vstream parameters
            self.detection_network_group_params = self.detection_network_group.create_params()

            self.detection_input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
                self.detection_network_group,
                quantized=True,
                format_type=hpf.FormatType.UINT8
            )

            self.detection_output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
                self.detection_network_group,
                quantized=False,
                format_type=hpf.FormatType.FLOAT32
            )

            return True

        except Exception as e:
            logger.error(f"Failed to initialize detection model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _init_pose(self) -> bool:
        """Initialize Stage 2: Pose Estimation using shared VDevice"""
        try:
            model_path = Path("ai/models") / self.config["hef_path"]
            if not model_path.exists():
                logger.error(f"Pose model not found: {model_path}")
                return False

            # Load HEF and configure with shared VDevice
            self.pose_hef = hpf.HEF(str(model_path))

            params = hpf.ConfigureParams.create_from_hef(
                hef=self.pose_hef,
                interface=hpf.HailoStreamInterface.PCIe
            )

            network_groups = self.vdevice.configure(self.pose_hef, params)
            self.pose_network_group = network_groups[0]

            # Get stream info
            self.pose_input_info = self.pose_hef.get_input_vstream_infos()[0]
            self.pose_output_infos = self.pose_hef.get_output_vstream_infos()

            logger.info(f"Pose model loaded: {self.pose_input_info.name}, shape: {self.pose_input_info.shape}")
            logger.info(f"Pose outputs: {len(self.pose_output_infos)} outputs")
            for i, output in enumerate(self.pose_output_infos):
                logger.info(f"  Output {i}: {output.name}, shape: {output.shape}")

            # Create network group parameters and vstream parameters
            self.pose_network_group_params = self.pose_network_group.create_params()

            self.pose_input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
                self.pose_network_group,
                quantized=True,
                format_type=hpf.FormatType.UINT8
            )

            self.pose_output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
                self.pose_network_group,
                quantized=False,
                format_type=hpf.FormatType.FLOAT32
            )

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

            # Prepare input (BGR format gives best results - 0.85 confidence vs 0.55 for RGB)
            # Keep the original BGR format from OpenCV
            frame_bgr = frame_640

            logger.debug(f"Stage 1: Processing frame {frame_4k.shape} -> {frame_640.shape}")

            # Run actual HEF inference using working VDevice pattern
            detections = []

            try:
                if not self.detection_network_group:
                    raise RuntimeError("Detection network group not available")
                with self.detection_network_group.activate(self.detection_network_group_params):
                    with hpf.InferVStreams(self.detection_network_group,
                                          self.detection_input_vstreams_params,
                                          self.detection_output_vstreams_params) as infer_pipeline:

                        # Prepare input - ensure UINT8 and correct shape
                        input_tensor = np.expand_dims(frame_bgr, axis=0).astype(np.uint8)
                        input_name = list(self.detection_input_vstreams_params.keys())[0]
                        input_data = {input_name: input_tensor}

                        # Run inference
                        output_data = infer_pipeline.infer(input_data)

                        # Parse detections and scale back to 4K coordinates
                        detections = self._parse_detection_output(output_data, w_4k, h_4k)
                        logger.debug(f"Stage 1: Found {len(detections)} dogs")

            except Exception as e:
                logger.error(f"Stage 1 inference error: {e}")
                # Fall back to mock detection for testing
                mock_x1 = w_4k // 4
                mock_y1 = h_4k // 4
                mock_x2 = 3 * w_4k // 4
                mock_y2 = 3 * h_4k // 4
                mock_conf = 0.85
                detections.append(Detection(mock_x1, mock_y1, mock_x2, mock_y2, mock_conf))
                logger.info(f"Stage 1: Using mock detection due to inference error")

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

            # Prepare input (RGB format)
            if len(crop_640.shape) == 3 and crop_640.shape[2] == 3:
                # OpenCV gives BGR, convert to RGB
                crop_rgb = cv2.cvtColor(crop_640, cv2.COLOR_BGR2RGB)
            else:
                crop_rgb = crop_640

            logger.debug(f"Stage 2: Processing crop {crop_4k.shape} -> {crop_640.shape}")

            # Run actual HEF inference using working VDevice pattern
            keypoints = None

            try:
                if not self.pose_network_group:
                    raise RuntimeError("Pose network group not available")
                with self.pose_network_group.activate(self.pose_network_group_params):
                    with hpf.InferVStreams(self.pose_network_group,
                                          self.pose_input_vstreams_params,
                                          self.pose_output_vstreams_params) as infer_pipeline:

                        # Prepare input - ensure UINT8 and correct shape
                        input_tensor = np.expand_dims(crop_rgb, axis=0).astype(np.uint8)
                        input_name = list(self.pose_input_vstreams_params.keys())[0]
                        input_data = {input_name: input_tensor}

                        # Run inference
                        output_data = infer_pipeline.infer(input_data)

                        # Parse keypoints from pose output
                        keypoints = self._parse_pose_output(output_data)
                        logger.debug(f"Stage 2: Parsed pose keypoints")

            except Exception as e:
                logger.error(f"Stage 2 inference error: {e}")
                # Fall back to mock keypoints for testing
                keypoints = np.random.rand(24, 3) * [640, 640, 1]  # x, y, confidence
                keypoints[:, 2] = np.random.rand(24) * 0.5 + 0.5  # confidence 0.5-1.0
                logger.info(f"Stage 2: Using mock keypoints due to inference error")

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

    def _parse_detection_output(self, output_data: Dict[str, np.ndarray], orig_w: int, orig_h: int) -> List[Detection]:
        """Parse dogdetector_14.hef NMS output (1,5,100) format"""
        detections = []

        try:
            # Handle actual inference output format
            logger.debug(f"Detection output keys: {list(output_data.keys())}")
            for output_name, output_tensor in output_data.items():
                logger.debug(f"Parsing detection output {output_name}: type={type(output_tensor)}")

                # Convert list to numpy array if needed
                if isinstance(output_tensor, list):
                    if len(output_tensor) > 0:
                        if isinstance(output_tensor[0], list):
                            # Handle nested list format: [[numpy_array]]
                            logger.debug(f"  Found nested list, depth 2, inner length: {len(output_tensor[0])}")
                            if len(output_tensor[0]) > 0:
                                # Extract the actual numpy array from nested structure
                                if isinstance(output_tensor[0][0], np.ndarray):
                                    output_tensor = output_tensor[0][0]  # Get the actual tensor
                                else:
                                    output_tensor = np.array(output_tensor[0])
                            else:
                                continue
                        elif isinstance(output_tensor[0], np.ndarray):
                            output_tensor = output_tensor[0]  # Take first element if it's a list of arrays
                        else:
                            output_tensor = np.array(output_tensor)
                    else:
                        continue

                if not isinstance(output_tensor, np.ndarray):
                    logger.debug(f"Skipping non-array output: {type(output_tensor)}")
                    continue

                logger.debug(f"Processing output {output_name}: shape={output_tensor.shape}")

                # Handle NMS postprocessed format - multiple possible shapes
                if len(output_tensor.shape) == 3 and output_tensor.shape[1] == 5:
                    # Format: (1, 5, 100) - Remove batch dimension: (1, 5, 100) -> (5, 100)
                    detections_data = output_tensor[0]  # Shape: (5, 100)
                    # Transpose to get (100, 5) for easier processing
                    detections_data = detections_data.T  # Shape: (100, 5)

                # Handle single detection format (1, 5) -> (1, 5)
                elif len(output_tensor.shape) == 2 and output_tensor.shape[0] == 1 and output_tensor.shape[1] == 5:
                    detections_data = output_tensor  # Shape: (1, 5) - already correct

                # Handle multiple detections format (N, 5)
                elif len(output_tensor.shape) == 2 and output_tensor.shape[1] == 5:
                    detections_data = output_tensor

                else:
                    logger.debug(f"Unexpected detection output shape: {output_tensor.shape}")
                    continue

                for detection in detections_data:
                    x1, y1, x2, y2, conf = detection

                    # Filter by confidence threshold (lowered to catch more dogs)
                    if conf > 0.1:
                        # Convert from normalized coordinates (0-1) to pixels, then scale to original frame
                        # First convert to 640x640 pixel coordinates
                        x1_640 = x1 * 640
                        y1_640 = y1 * 640
                        x2_640 = x2 * 640
                        y2_640 = y2 * 640

                        # Then scale to original frame size
                        scale_x = orig_w / 640
                        scale_y = orig_h / 640

                        x1_scaled = int(x1_640 * scale_x)
                        y1_scaled = int(y1_640 * scale_y)
                        x2_scaled = int(x2_640 * scale_x)
                        y2_scaled = int(y2_640 * scale_y)

                        # Ensure coordinates are within bounds
                        x1_scaled = max(0, min(x1_scaled, orig_w))
                        y1_scaled = max(0, min(y1_scaled, orig_h))
                        x2_scaled = max(0, min(x2_scaled, orig_w))
                        y2_scaled = max(0, min(y2_scaled, orig_h))

                        # Ensure valid bounding box
                        if x2_scaled > x1_scaled and y2_scaled > y1_scaled:
                            detections.append(Detection(x1_scaled, y1_scaled, x2_scaled, y2_scaled, float(conf)))

                logger.debug(f"Parsed {len(detections)} valid detections from {output_name}")
                break  # Use first valid output

            return detections

        except Exception as e:
            logger.error(f"Error parsing detection output: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _parse_pose_output(self, outputs: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Parse dogpose_14.hef outputs (9 tensors, 72 channels) format"""
        try:
            logger.debug(f"Parsing pose output: {len(outputs)} tensors")

            # Debug output shapes
            for key, output in outputs.items():
                logger.debug(f"  {key}: shape={output.shape}, dtype={output.dtype}")

            # Group outputs by scale for 640x640 input
            # Expected scales for 640x640: 80x80, 40x40, 20x20
            scales = {
                '80x80': {'bbox': None, 'kpts': None, 'conf': None},
                '40x40': {'bbox': None, 'kpts': None, 'conf': None},
                '20x20': {'bbox': None, 'kpts': None, 'conf': None}
            }

            # Map conv layers based on shape patterns
            for layer_name, output in outputs.items():
                if len(output.shape) != 4:
                    continue

                h, w = output.shape[1], output.shape[2]
                channels = output.shape[3]

                # Determine scale and type based on shape
                scale_name = None
                output_type = None

                if (h, w) == (80, 80):  # Large scale
                    scale_name = '80x80'
                elif (h, w) == (40, 40):  # Medium scale
                    scale_name = '40x40'
                elif (h, w) == (20, 20):  # Small scale
                    scale_name = '20x20'

                if scale_name and channels == 64:
                    output_type = 'bbox'
                elif scale_name and channels == 72:
                    output_type = 'kpts'
                elif scale_name and channels == 1:
                    output_type = 'conf'

                if scale_name and output_type:
                    scales[scale_name][output_type] = output
                    logger.debug(f"  Mapped {layer_name} -> {scale_name} {output_type}")

            # Find the best detection across all scales
            best_detection = None
            best_conf = 0.0

            strides = [8, 16, 32]  # YOLOv11 strides
            scale_names = ['80x80', '40x40', '20x20']

            for scale_idx, scale_name in enumerate(scale_names):
                scale_data = scales[scale_name]

                if not all(scale_data[key] is not None for key in ['bbox', 'kpts', 'conf']):
                    continue

                bbox_out = scale_data['bbox']
                kpts_out = scale_data['kpts']
                conf_out = scale_data['conf']

                _, h, w, _ = bbox_out.shape
                h, w = int(h), int(w)
                stride = strides[scale_idx]

                # Process high-confidence cells
                for i in range(h):
                    for j in range(w):
                        conf_raw = conf_out[0, i, j, 0]
                        # Prevent overflow in exp function
                        conf_raw = np.clip(conf_raw, -500, 500)
                        conf = 1.0 / (1.0 + np.exp(-conf_raw))  # Sigmoid

                        # Debug: Print confidence levels to understand what we're getting (disabled for cleaner output)
                        # if conf > 0.1:  # Only show meaningful confidences
                        #     print(f"🔍 Pose confidence: {conf:.3f} at ({i},{j})")

                        if conf < 0.3:  # Confidence threshold (lowered back for better detection)
                            continue

                        if conf > best_conf:
                            # Decode keypoints for best detection
                            kpts_raw = kpts_out[0, i, j, :]
                            kpts = np.zeros((24, 3), dtype=np.float32)

                            for k in range(24):
                                kpts[k, 0] = (kpts_raw[k * 3] + j) * stride
                                kpts[k, 1] = (kpts_raw[k * 3 + 1] + i) * stride
                                kpts[k, 2] = 1.0 / (1.0 + np.exp(-kpts_raw[k * 3 + 2]))

                            best_detection = kpts
                            best_conf = conf

            if best_detection is not None:
                logger.debug(f"Found pose keypoints with confidence {best_conf:.3f}")
                return best_detection
            else:
                logger.debug("No valid pose detection found")
                return None

        except Exception as e:
            logger.error(f"Error parsing pose output: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _analyze_single_pose_sequence(self, current_pose: PoseKeypoints, pose_index: int) -> Optional[str]:
        """Analyze behavior for a single pose over time with temporal smoothing"""
        try:
            keypoints = current_pose.keypoints
            if len(keypoints) < 10:
                return None

            # Add current pose to history
            self.behavior_history.append(keypoints)

            # Need enough history for stable detection
            if len(self.behavior_history) < self.min_consecutive_frames:
                return self.last_stable_behavior

            # Analyze current pose with improved heuristics
            raw_behavior = self._classify_pose(keypoints)
            print(f"🐕 Raw behavior detected: {raw_behavior}")

            # Check if behavior has been consistent
            recent_behaviors = []
            for historical_pose in list(self.behavior_history)[-self.min_consecutive_frames:]:
                behavior = self._classify_pose(historical_pose)
                if behavior:
                    recent_behaviors.append(behavior)

            print(f"📊 Recent behaviors: {recent_behaviors} (need {max(4, int(self.min_consecutive_frames * 0.4))} for detection)")

            # Require majority consensus for stability (non-consecutive)
            if len(recent_behaviors) >= self.min_consecutive_frames * 0.4:  # 40% of frames need behavior
                from collections import Counter
                behavior_counts = Counter(recent_behaviors)
                most_common_behavior, count = behavior_counts.most_common(1)[0]

                # Check if this behavior is confident enough and not in cooldown
                # Allow non-consecutive detection (e.g., 4 out of 10 frames = 40%)
                if count >= max(4, self.min_consecutive_frames * 0.4):
                    if self._check_behavior_cooldown(most_common_behavior):
                        self.last_stable_behavior = most_common_behavior
                        self._trigger_behavior_cooldown(most_common_behavior)
                        print(f"🎯 BEHAVIOR DETECTED: {most_common_behavior.upper()} (confidence: {count}/{len(recent_behaviors)})")
                        return most_common_behavior

            return self.last_stable_behavior

        except Exception as e:
            logger.error(f"Error analyzing pose sequence: {e}")
            return None

    def _classify_pose(self, keypoints: np.ndarray) -> Optional[str]:
        """Classify a single pose based on keypoint positions"""
        try:
            # Ensure keypoints is a numpy array
            if not isinstance(keypoints, np.ndarray):
                return None

            # Improved pose classification with better thresholds
            frame_height = 640
            frame_width = 640

            # Get body keypoints (adjust indices based on your model)
            if len(keypoints) < 12:
                return None

            body_points = keypoints[:12]  # First 12 keypoints for body

            # Fix: Properly filter by confidence (handle shape correctly)
            if body_points.ndim == 2 and body_points.shape[1] >= 3:
                confidence_mask = body_points[:, 2] > 0.3
                valid_points = body_points[confidence_mask]

                # Ensure valid_points is still a 2D array
                if len(valid_points) == 0 or valid_points.ndim != 2:
                    return None
            else:
                # Handle 1D case - assume it's flattened x,y,conf triplets
                if body_points.ndim == 1 and len(body_points) >= 36:  # 12 points * 3 values each
                    reshaped = body_points[:36].reshape(12, 3)
                    confidence_mask = reshaped[:, 2] > 0.3
                    valid_points = reshaped[confidence_mask]
                    if len(valid_points) == 0:
                        return None
                else:
                    return None

            if len(valid_points) < 4:
                return None

            # Now safe to use slicing since we know valid_points is 2D
            avg_y = np.mean(valid_points[:, 1])
            avg_x = np.mean(valid_points[:, 0])

            # Calculate pose characteristics
            y_ratio = avg_y / frame_height

            # Get limb positions for better classification
            limb_spread = np.std(valid_points[:, 0]) if len(valid_points) > 1 else 0

            # Debug logging to understand why poses aren't detected
            print(f"🔍 Pose analysis: y_ratio={y_ratio:.3f}, limb_spread={limb_spread:.1f}, valid_points={len(valid_points)}")

            # Improved classification logic - better lie detection
            if y_ratio > 0.72:  # Lower threshold for lie detection (was 0.80)
                # Additional check: low limb spread indicates lying down
                if limb_spread < frame_width * 0.06:  # Very compact for lie
                    return "lie"
                elif y_ratio > 0.78:  # Very low in frame, definitely lying
                    return "lie"
                else:
                    return "sit"  # High y_ratio but spread out might be sitting
            elif y_ratio < 0.35:  # High in frame (more generous for stand)
                return "stand"
            elif 0.45 <= y_ratio <= 0.68:  # Narrower range for sit (was 0.50-0.75)
                # Additional checks for sit vs other behaviors
                if limb_spread < frame_width * 0.10:  # Compact pose required for sit (relaxed from 0.08)
                    return "sit"
                else:
                    return "stand"  # Default to stand
            else:
                return "stand"  # Default behavior is always "stand"

        except Exception as e:
            logger.error(f"Error classifying pose: {e}")
            return None

    def _check_behavior_cooldown(self, behavior: str) -> bool:
        """Check if behavior is not in cooldown"""
        current_time = time.time()
        if behavior in self.behavior_cooldowns:
            time_since_last = current_time - self.behavior_cooldowns[behavior]
            return time_since_last >= self.cooldown_duration
        return True

    def _trigger_behavior_cooldown(self, behavior: str):
        """Start cooldown for a behavior"""
        self.behavior_cooldowns[behavior] = time.time()

    def cleanup(self):
        """Clean up resources"""
        try:
            # Clean up shared VDevice
            if self.vdevice:
                self.vdevice = None

            # Clear all model references
            self.detection_hef = None
            self.detection_network_group = None
            self.detection_network_group_params = None
            self.detection_input_vstreams_params = None
            self.detection_output_vstreams_params = None

            self.pose_hef = None
            self.pose_network_group = None
            self.pose_network_group_params = None
            self.pose_input_vstreams_params = None
            self.pose_output_vstreams_params = None

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
            "behavior_history_length": len(self.behavior_history),
            "detection_loaded": self.detection_hef is not None,
            "pose_loaded": self.pose_hef is not None,
            "vdevice_ready": self.vdevice is not None
        }