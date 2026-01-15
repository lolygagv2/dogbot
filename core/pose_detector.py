#!/usr/bin/env python3
"""
Pose Detection Module for TreatSensei DogBot
Integrates YOLOv11 pose detection with behavior classification
"""

import os
import json
import time
import collections
import numpy as np
import torch
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import hailo_platform as hpf

logger = logging.getLogger(__name__)

@dataclass
class PoseDetection:
    """Single pose detection result"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    keypoints: np.ndarray  # [24, 3] with x, y, confidence
    confidence: float
    dog_id: Optional[str] = None

class PoseDetector:
    """Pose detection and behavior analysis for dogs using YOLOv11 + LSTM"""

    def __init__(self, config_path: str = "config/config.json"):
        """Initialize pose detector with configuration"""
        self.config_path = config_path
        self.load_config()

        self.initialized = False
        self.hef = None
        self.vdevice = None
        self.network_group = None
        self.infer_pipeline = None
        self.behavior_head = None

        # Per-dog state storage
        self.dog_states = {}
        self.last_dog_ids = []

        # ArUco detector for dog identification
        self.aruco_detector = None
        self.setup_aruco()

        # Statistics
        self.inference_count = 0
        self.last_inference_time = 0

    def load_config(self):
        """Load configuration from JSON file"""
        cfg_path = Path(self.config_path)
        if not cfg_path.exists():
            # Use defaults if config doesn't exist
            logger.warning(f"Config file {cfg_path} not found, using defaults")
            self.config = {
                "imgsz": 640,  # Model was compiled for 640x640, not 896
                "T": 16,
                "behaviors": ["stand", "sit", "lie", "cross", "spin"],
                "prob_th": 0.6,
                "cooldown_s": {"stand": 2, "sit": 5, "lie": 5, "cross": 4, "spin": 8},
                "aruco_dict": "DICT_4X4_1000",
                "dogs": [
                    {"id": "bezik", "marker_id": 315},
                    {"id": "elsa", "marker_id": 832}
                ],
                "hef_path": "ai/models/dogposeV2yolo11.hef",
                "behavior_head_ts": "ai/models/behavior_head.ts"
            }
        else:
            with open(cfg_path, 'r') as f:
                self.config = json.load(f)

        # Extract key parameters
        self.imgsz = int(self.config.get("imgsz", 896))
        self.T = int(self.config.get("T", 16))
        self.behaviors = list(self.config.get("behaviors", ["stand", "sit", "lie", "cross", "spin"]))
        self.prob_threshold = float(self.config.get("prob_th", 0.6))
        self.cooldowns = dict(self.config.get("cooldown_s", {}))
        self.marker_to_dog = {
            int(d["marker_id"]): str(d["id"])
            for d in self.config.get("dogs", [])
        }

        # Model paths
        self.hef_path = self.config.get("hef_path", "ai/models/dogposeV2yolo11.hef")
        self.behavior_head_path = self.config.get("behavior_head_ts", "ai/models/behavior_head.ts")

    def setup_aruco(self):
        """Setup ArUco marker detection"""
        dict_name = str(self.config.get("aruco_dict", "DICT_4X4_1000"))
        dict_const = getattr(cv2.aruco, dict_name)
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_const)

        try:
            self.aruco_detector = cv2.aruco.ArucoDetector(
                aruco_dict,
                cv2.aruco.DetectorParameters()
            )
        except AttributeError:
            # Fallback for older OpenCV
            self.aruco_detector = (aruco_dict, cv2.aruco.DetectorParameters_create())

    def initialize(self) -> bool:
        """Initialize Hailo device and load models"""
        try:
            logger.info("Initializing Pose Detector...")

            # Check model files exist
            if not Path(self.hef_path).exists():
                logger.error(f"HEF model not found: {self.hef_path}")
                return False

            if not Path(self.behavior_head_path).exists():
                logger.error(f"Behavior head model not found: {self.behavior_head_path}")
                return False

            # Load behavior classification head
            logger.info(f"Loading behavior head: {self.behavior_head_path}")
            self.behavior_head = torch.jit.load(self.behavior_head_path, map_location="cpu")
            self.behavior_head.eval()

            # Initialize Hailo device
            logger.info(f"Loading HEF model: {self.hef_path}")
            self.hef = hpf.HEF(self.hef_path)

            # Setup Hailo inference pipeline
            self.vdevice = hpf.VDevice()
            params = hpf.ConfigureParams.create_from_hef(
                hef=self.hef,
                interface=hpf.HailoStreamInterface.PCIe
            )

            network_groups = self.vdevice.configure(self.hef, params)
            self.network_group = network_groups[0]

            # Get input/output info
            input_vstreams_info = self.hef.get_input_vstream_infos()
            output_vstreams_info = self.hef.get_output_vstream_infos()

            logger.info(f"Model inputs: {len(input_vstreams_info)}")
            logger.info(f"Model outputs: {len(output_vstreams_info)}")

            # Create vstream parameters - must be dictionaries not lists
            self.input_vstreams_params = hpf.InputVStreamParams.make(
                self.network_group,
                quantized=True,
                format_type=hpf.FormatType.UINT8
            )

            self.output_vstreams_params = hpf.OutputVStreamParams.make(
                self.network_group,
                quantized=False,
                format_type=hpf.FormatType.FLOAT32
            )

            # Store the input name for later use
            self.input_name = input_vstreams_info[0].name

            self.initialized = True
            logger.info("✅ Pose Detector initialization successful!")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Pose Detector: {e}")
            self.cleanup()
            return False

    def cleanup(self):
        """Clean up Hailo resources"""
        if self.network_group:
            self.network_group = None
        if self.vdevice:
            self.vdevice.release()
            self.vdevice = None

    def letterbox(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float]]:
        """Apply letterbox transformation for model input"""
        h, w = image.shape[:2]

        # Handle both 3 and 4 channel images
        channels = image.shape[2] if len(image.shape) == 3 else 3

        # Convert 4-channel (XBGR/RGBA) to 3-channel BGR if needed
        if channels == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        # Use the configured image size (896 for this model)
        target_size = self.imgsz
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create canvas and center the image
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        dy, dx = (target_size - new_h) // 2, (target_size - new_w) // 2
        canvas[dy:dy+new_h, dx:dx+new_w] = resized

        return canvas, (dx, dy, 1.0/scale)

    def decode_pose_outputs(self, raw_outputs: Dict[str, np.ndarray]) -> List[PoseDetection]:
        """
        Decode YOLOv11 pose HEF outputs from Hailo using the proven working decoder.

        Args:
            raw_outputs: Dictionary of output tensors from Hailo

        Returns:
            List of PoseDetection objects
        """
        detections = []

        # Handle dictionary output from InferVStreams
        if not isinstance(raw_outputs, dict):
            logger.warning("Expected dict output from Hailo")
            return detections

        # Group outputs by scale based on ACTUAL Hailo output layer names
        scales = {
            '80x80': {'bbox': None, 'kpts': None, 'conf': None},
            '40x40': {'bbox': None, 'kpts': None, 'conf': None},
            '20x20': {'bbox': None, 'kpts': None, 'conf': None}
        }

        # Map conv layers to scales and types based on actual debug output
        conv_mapping = {
            # 80x80 scale
            'best_v8/conv63': ('80x80', 'conf'),   # (1, 80, 80, 1)
            'best_v8/conv60': ('80x80', 'kpts'),   # (1, 80, 80, 72)
            'best_v8/conv59': ('80x80', 'bbox'),   # (1, 80, 80, 64)
            # 40x40 scale
            'best_v8/conv78': ('40x40', 'conf'),   # (1, 40, 40, 1)
            'best_v8/conv75': ('40x40', 'kpts'),   # (1, 40, 40, 72)
            'best_v8/conv74': ('40x40', 'bbox'),   # (1, 40, 40, 64)
            # 20x20 scale
            'best_v8/conv97': ('20x20', 'conf'),   # (1, 20, 20, 1)
            'best_v8/conv94': ('20x20', 'kpts'),   # (1, 20, 20, 72)
            'best_v8/conv93': ('20x20', 'bbox'),   # (1, 20, 20, 64)
        }

        # Assign outputs to proper scales
        for layer_name, output in raw_outputs.items():
            if layer_name in conv_mapping:
                scale, output_type = conv_mapping[layer_name]
                scales[scale][output_type] = output

        all_predictions = []
        strides = [8, 16, 32]  # YOLOv11 standard strides
        # For 896x896 input: 112x112, 56x56, 28x28 outputs
        # For 640x640 input: 80x80, 40x40, 20x20 outputs
        # The actual names in the HEF might still say 80x80 etc
        scale_names = ['80x80', '40x40', '20x20']  # These are the layer names, not actual sizes

        for scale_idx, scale_name in enumerate(scale_names):
            scale_data = scales[scale_name]

            # Check if we have all required outputs for this scale
            if not all(scale_data[key] is not None for key in ['bbox', 'kpts', 'conf']):
                continue

            bbox_out = scale_data['bbox']  # Shape: (1, h, w, 64)
            kpts_out = scale_data['kpts']  # Shape: (1, h, w, 72)
            conf_out = scale_data['conf']  # Shape: (1, h, w, 1)

            # Get grid dimensions
            if len(bbox_out.shape) == 4:
                _, h, w, _ = bbox_out.shape
            else:
                continue

            stride = strides[scale_idx]
            scale_detections = 0
            max_detections_per_scale = 20  # Limit detections per scale

            for i in range(min(int(h), conf_out.shape[1])):
                for j in range(min(int(w), conf_out.shape[2])):
                    if scale_detections >= max_detections_per_scale:
                        break

                    # Extract confidence (single value)
                    conf_raw = conf_out[0, i, j, 0]

                    # Apply sigmoid to raw confidence (typical for YOLO models)
                    conf = 1.0 / (1.0 + np.exp(-conf_raw))

                    # Use a lower confidence threshold to detect lying dogs
                    if conf < 0.25:  # Lower threshold for lying poses
                        continue

                    # Extract bbox prediction (64 channels)
                    bbox_raw = bbox_out[0, i, j, :]  # 64 values

                    # YOLOv11 bbox format: first 4 channels are [cx, cy, w, h]
                    cx_raw, cy_raw, w_raw, h_raw = bbox_raw[:4]

                    # Decode bounding box
                    cx = (cx_raw + j) * stride  # Center x
                    cy = (cy_raw + i) * stride  # Center y
                    w = np.exp(w_raw) * stride  # Width
                    h = np.exp(h_raw) * stride  # Height

                    x1 = cx - w / 2
                    y1 = cy - h / 2
                    x2 = cx + w / 2
                    y2 = cy + h / 2

                    # Extract keypoints (72 channels = 24 keypoints * 3 values each)
                    kpts_raw = kpts_out[0, i, j, :]  # 72 values
                    kpts = np.zeros((24, 3), dtype=np.float32)

                    for k in range(24):
                        # Extract x, y, visibility for each keypoint
                        kx_raw = kpts_raw[k * 3]
                        ky_raw = kpts_raw[k * 3 + 1]
                        kv_raw = kpts_raw[k * 3 + 2]

                        # Decode keypoint coordinates - FIXED: multiply raw by 2 (official Hailo YOLOv8 pose)
                        kpts[k, 0] = (kx_raw * 2 + j) * stride  # x coordinate
                        kpts[k, 1] = (ky_raw * 2 + i) * stride  # y coordinate
                        kpts[k, 2] = 1.0 / (1.0 + np.exp(-kv_raw))  # visibility (sigmoid)

                    all_predictions.append({
                        "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                        "kpts": kpts,
                        "conf": conf
                    })
                    scale_detections += 1

        # Apply NMS to remove duplicates
        if len(all_predictions) > 0:
            # Sort by confidence
            all_predictions.sort(key=lambda x: x["conf"], reverse=True)

            # Simple NMS
            keep = []
            for pred in all_predictions:
                should_keep = True

                # Check overlap with kept detections
                for kept in keep:
                    # Calculate IoU
                    x1 = max(pred["xyxy"][0], kept["xyxy"][0])
                    y1 = max(pred["xyxy"][1], kept["xyxy"][1])
                    x2 = min(pred["xyxy"][2], kept["xyxy"][2])
                    y2 = min(pred["xyxy"][3], kept["xyxy"][3])

                    if x2 > x1 and y2 > y1:
                        intersection = (x2 - x1) * (y2 - y1)
                        area1 = (pred["xyxy"][2] - pred["xyxy"][0]) * (pred["xyxy"][3] - pred["xyxy"][1])
                        area2 = (kept["xyxy"][2] - kept["xyxy"][0]) * (kept["xyxy"][3] - kept["xyxy"][1])
                        iou = intersection / (area1 + area2 - intersection + 1e-6)

                        if iou > 0.4:  # NMS threshold
                            should_keep = False
                            break

                if should_keep:
                    keep.append(pred)
                    detections.append(PoseDetection(
                        bbox=pred["xyxy"],
                        keypoints=pred["kpts"],
                        confidence=float(pred["conf"])
                    ))

                    if len(detections) >= 5:  # Reasonable max detections per frame
                        break

        return detections

    def simple_nms(self, boxes: List, scores: List, iou_threshold: float = 0.4) -> List[int]:
        """Simple non-maximum suppression"""
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        scores = np.array(scores)

        # Sort by score
        indices = np.argsort(scores)[::-1]

        keep = []
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            current_box = boxes[current]
            other_boxes = boxes[indices[1:]]

            # Calculate IoU
            ious = self.calculate_iou(current_box, other_boxes)

            # Keep only boxes with IoU less than threshold
            indices = indices[1:][ious < iou_threshold]

        return keep

    def calculate_iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Calculate IoU between one box and multiple boxes"""
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union = box_area + boxes_area - intersection

        return intersection / (union + 1e-6)

    def detect_markers(self, image: np.ndarray) -> List[Tuple[int, float, float]]:
        """Detect ArUco markers in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        try:
            if isinstance(self.aruco_detector, tuple):
                # Old API
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, self.aruco_detector[0],
                    parameters=self.aruco_detector[1]
                )
            else:
                # New API
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        except Exception as e:
            logger.warning(f"ArUco detection failed: {e}")
            return []

        markers = []
        if ids is not None:
            for corner, marker_id in zip(corners, ids.flatten()):
                pts = corner[0]
                cx = float(pts[:, 0].mean())
                cy = float(pts[:, 1].mean())
                markers.append((int(marker_id), cx, cy))

        return markers

    def assign_dog_ids(self, detections: List[PoseDetection], markers: List[Tuple[int, float, float]]):
        """Assign dog IDs to detections based on ArUco markers with smart defaults"""
        # Reset IDs
        for det in detections:
            det.dog_id = None

        used_indices = set()

        # First, assign based on visible markers
        for marker_id, mx, my in markers:
            if marker_id not in self.marker_to_dog:
                continue

            dog_id = self.marker_to_dog[marker_id]

            # Find nearest detection
            min_dist = float('inf')
            min_idx = -1

            for i, det in enumerate(detections):
                if i in used_indices:
                    continue

                # Calculate distance from marker to bbox center
                cx = (det.bbox[0] + det.bbox[2]) / 2
                cy = (det.bbox[1] + det.bbox[3]) / 2

                dist = np.sqrt((cx - mx)**2 + (cy - my)**2)

                if dist < min_dist:
                    min_dist = dist
                    min_idx = i

            if min_idx >= 0:
                detections[min_idx].dog_id = dog_id
                used_indices.add(min_idx)

        # Smart assignment logic for unmarked dogs
        unassigned = [i for i, det in enumerate(detections) if det.dog_id is None]

        if unassigned:
            # Check if Elsa (832) is already detected
            elsa_detected = any(d.dog_id == "elsa" for d in detections)

            if len(unassigned) == 1:
                # Single unassigned dog
                if not elsa_detected:
                    # Default to Elsa if not already detected
                    detections[unassigned[0]].dog_id = "elsa"
                else:
                    # If Elsa is detected, assume the other is Bezik
                    detections[unassigned[0]].dog_id = "bezik"

            elif len(unassigned) == 2:
                # Two unassigned dogs - assign both
                if not elsa_detected:
                    # First is Elsa (default), second is Bezik
                    detections[unassigned[0]].dog_id = "elsa"
                    detections[unassigned[1]].dog_id = "bezik"
                else:
                    # Elsa already assigned, both unassigned are treated as Bezik
                    # (or could implement size-based heuristic here)
                    detections[unassigned[0]].dog_id = "bezik"
                    detections[unassigned[1]].dog_id = "unknown"

            # If only one detection total and no markers, default to Elsa
            elif len(detections) == 1 and len(markers) == 0:
                detections[0].dog_id = "elsa"

        # Keep last IDs if still unassigned (for tracking continuity)
        for i, det in enumerate(detections):
            if det.dog_id is None and i < len(self.last_dog_ids):
                det.dog_id = self.last_dog_ids[i]

        # Update last IDs
        self.last_dog_ids = [d.dog_id for d in detections if d.dog_id is not None]

    def normalize_keypoints(self, bbox: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Normalize keypoints relative to bounding box"""
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)

        normalized = np.zeros((24, 2), dtype=np.float32)
        normalized[:, 0] = np.clip((keypoints[:, 0] - x1) / w, 0, 1)
        normalized[:, 1] = np.clip((keypoints[:, 1] - y1) / h, 0, 1)

        return normalized

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single frame for pose detection and behavior classification

        Args:
            frame: Input image (BGR format)

        Returns:
            Dictionary containing detections and behavior predictions
        """
        if not self.initialized:
            return {"detections": [], "behaviors": {}}

        # Prepare input
        input_image, (dx, dy, scale_inv) = self.letterbox(frame)

        # Run inference
        start_time = time.time()

        # Prepare input for Hailo (NHWC format)
        input_data = {
            self.input_name: np.expand_dims(input_image, axis=0)
        }

        # Run inference with proper context managers
        network_group_params = self.network_group.create_params()
        with self.network_group.activate(network_group_params):
            with hpf.InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params
            ) as infer_pipeline:
                raw_outputs = infer_pipeline.infer(input_data)

        self.last_inference_time = (time.time() - start_time) * 1000
        self.inference_count += 1

        # Decode outputs
        detections = self.decode_pose_outputs(raw_outputs)

        # Scale back to original image coordinates
        for det in detections:
            det.bbox[0] = (det.bbox[0] - dx) * scale_inv
            det.bbox[1] = (det.bbox[1] - dy) * scale_inv
            det.bbox[2] = (det.bbox[2] - dx) * scale_inv
            det.bbox[3] = (det.bbox[3] - dy) * scale_inv

            det.keypoints[:, 0] = (det.keypoints[:, 0] - dx) * scale_inv
            det.keypoints[:, 1] = (det.keypoints[:, 1] - dy) * scale_inv

        # Detect ArUco markers and assign dog IDs
        markers = self.detect_markers(frame)
        self.assign_dog_ids(detections, markers)

        # Process behavior for each detected dog
        behaviors = {}
        current_time = time.time()

        for det in detections:
            if det.dog_id is None:
                continue

            # Get or create dog state
            if det.dog_id not in self.dog_states:
                self.dog_states[det.dog_id] = DogState(self.behavior_head, self.T)

            # Normalize keypoints and predict behavior
            normalized_kpts = self.normalize_keypoints(det.bbox, det.keypoints)

            state = self.dog_states[det.dog_id]
            behavior_idx, confidence, all_probs = state.push_and_predict_with_probs(normalized_kpts)

            if behavior_idx is not None:
                behavior_name = self.behaviors[behavior_idx]

                # Check cooldown
                cooldown = self.cooldowns.get(behavior_name, 3.0)
                can_trigger = (current_time - state.last_emit.get(behavior_name, 0)) >= cooldown

                # Get top 2 behaviors for comparison
                prob_sorted = sorted(enumerate(all_probs), key=lambda x: x[1], reverse=True)
                top_behaviors = []
                for idx, prob in prob_sorted[:2]:
                    if prob > 0.1:  # Only show if meaningful probability
                        top_behaviors.append({
                            "name": self.behaviors[idx],
                            "confidence": float(prob)
                        })

                behaviors[det.dog_id] = {
                    "behavior": behavior_name,
                    "confidence": confidence,
                    "can_trigger": can_trigger and confidence >= self.prob_threshold,
                    "all_behaviors": top_behaviors,  # Show top competing behaviors
                    "ambiguous": len(top_behaviors) > 1 and abs(top_behaviors[0]["confidence"] - top_behaviors[1]["confidence"]) < 0.2
                }

                if can_trigger and confidence >= self.prob_threshold:
                    state.last_emit[behavior_name] = current_time

        return {
            "detections": detections,
            "behaviors": behaviors,
            "inference_time": self.last_inference_time,
            "markers": markers
        }

    def get_status(self) -> Dict[str, Any]:
        """Get current status of pose detector"""
        return {
            "initialized": self.initialized,
            "model_path": self.hef_path,
            "behavior_model": self.behavior_head_path,
            "inference_count": self.inference_count,
            "avg_inference_time": self.last_inference_time,
            "tracked_dogs": list(self.dog_states.keys()),
            "behaviors": self.behaviors
        }


class DogState:
    """Track state for individual dog behavior analysis"""

    def __init__(self, behavior_head: torch.nn.Module, sequence_length: int):
        """Initialize dog state tracker"""
        self.behavior_head = behavior_head
        self.sequence_length = sequence_length
        self.keypoint_buffer = collections.deque(maxlen=sequence_length)
        self.ema_probs = None
        self.last_emit = {}

    def push_and_predict(self, normalized_kpts: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Add keypoints and predict behavior

        Args:
            normalized_kpts: [24, 2] normalized keypoints

        Returns:
            Tuple of (behavior_index, confidence) or (None, 0.0)
        """
        behavior_idx, confidence, _ = self.push_and_predict_with_probs(normalized_kpts)
        return behavior_idx, confidence

    def push_and_predict_with_probs(self, normalized_kpts: np.ndarray) -> Tuple[Optional[int], float, np.ndarray]:
        """
        Add keypoints and predict behavior with full probability distribution

        Args:
            normalized_kpts: [24, 2] normalized keypoints

        Returns:
            Tuple of (behavior_index, confidence, all_probabilities)
        """
        # Flatten keypoints to 48-d vector
        kpts_flat = normalized_kpts.reshape(48)
        self.keypoint_buffer.append(kpts_flat)

        # Need full sequence for prediction
        if len(self.keypoint_buffer) < self.sequence_length:
            return None, 0.0, np.zeros(5)  # Assuming 5 behaviors

        # Prepare input tensor
        sequence = np.array(self.keypoint_buffer, dtype=np.float32)
        input_tensor = torch.tensor(sequence).unsqueeze(0)  # [1, T, 48]

        # Run behavior classification
        with torch.no_grad():
            logits = self.behavior_head(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].numpy()

        # Apply exponential moving average for smoothing
        if self.ema_probs is None:
            self.ema_probs = probs
        else:
            self.ema_probs = 0.8 * self.ema_probs + 0.2 * probs

        # Get prediction
        behavior_idx = int(self.ema_probs.argmax())
        confidence = float(self.ema_probs.max())

        return behavior_idx, confidence, self.ema_probs.copy()