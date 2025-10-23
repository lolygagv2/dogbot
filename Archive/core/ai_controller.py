#!/usr/bin/env python3
"""
AI Controller for TreatSensei DogBot
Uses working HailoRT 4.21 InferVStreams API
"""

import numpy as np
import cv2
import time
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from collections import deque
from dataclasses import dataclass
import hailo_platform as hpf

logger = logging.getLogger(__name__)

@dataclass
class BehaviorFrame:
    """Frame data for behavior analysis"""
    timestamp: float
    bbox: Tuple[int, int, int, int]
    confidence: float
    aspect_ratio: float
    center: Tuple[int, int]

class AIController:
    """AI controller using working HailoRT 4.21 InferVStreams API"""

    def __init__(self):
        """Initialize AI controller"""
        self.initialized = False
        self.hef = None
        self.input_info = None
        self.output_infos = None

        # Model paths - using the working models we tested
        # TEMPORARY: Try NMS model first since raw YOLO parsing is complex
        self.model_path = "ai/models/dogbotyolo8--640x640_quant_hailort_hailo8_1.hef"  # NMS model
        self.backup_model_path = "ai/models/bestdogyolo5.hef"  # Raw YOLO outputs

        # Detection parameters with hysteresis for stability
        self.conf_threshold_low = 0.25   # Lower threshold for maintaining existing detections
        self.conf_threshold_high = 0.35  # Higher threshold for new detections
        self.nms_threshold = 0.4

        # Current confidence threshold (starts high, can go low for continuity)
        self.current_conf_threshold = self.conf_threshold_high

        # Temporal smoothing for stable detections
        self.detection_history = deque(maxlen=5)  # Last 5 frames
        self.last_detection_time = 0
        self.detection_continuity_frames = 0

        # Dog class indices - focusing on most reliable classes
        self.primary_dog_classes = {2}  # Class 2 seems most consistent
        self.secondary_dog_classes = {6, 7, 8, 9}  # Backup classes

        # YOLO class names (assuming COCO classes for now)
        self.class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                           'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog']

        # Statistics
        self.inference_count = 0
        self.last_inference_time = 0
        self.avg_inference_time = 0

        # Behavior detection with improved parameters
        self.behavior_history = deque(maxlen=90)  # ~3 seconds at 30fps
        self.current_behavior = "no_dog"  # Initialize to no_dog instead of unknown
        self.previous_behavior = "no_dog"  # Initialize to no_dog instead of unknown
        self.behavior_change_callback = None
        self.behavior_confidence = 0.0
        self.behavior_start_time = 0

        # Improved behavior thresholds based on real data analysis
        self.behavior_thresholds = {
            'lying_down': {'min_aspect': 0.0, 'max_aspect': 0.8},    # Very wide boxes
            'sitting': {'min_aspect': 0.7, 'max_aspect': 1.4},      # Square-ish boxes
            'standing': {'min_aspect': 1.3, 'max_aspect': 3.0},     # Tall boxes
            'walking': {'movement_threshold': 15}                    # Pixel movement per frame
        }

    def _apply_temporal_smoothing(self, detections: List[Dict]) -> List[Dict]:
        """Apply temporal smoothing to reduce detection flickering"""
        current_time = time.time()

        # Add current detections to history
        self.detection_history.append({
            'detections': detections,
            'timestamp': current_time
        })

        if len(detections) > 0:
            self.detection_continuity_frames += 1
            self.last_detection_time = current_time
            # Lower threshold for continuity after consistent detections
            if self.detection_continuity_frames > 3:
                self.current_conf_threshold = self.conf_threshold_low
        else:
            # No detections - check if we should raise threshold
            if current_time - self.last_detection_time > 2.0:  # 2 seconds without detection
                self.current_conf_threshold = self.conf_threshold_high
                self.detection_continuity_frames = 0

        # If we have recent detection history, use smoothing
        if len(self.detection_history) >= 3:
            # Count detections in recent frames
            recent_frames = list(self.detection_history)[-3:]
            detection_count = sum(1 for frame in recent_frames if len(frame['detections']) > 0)

            # If majority of recent frames have detections, boost weak detections
            if detection_count >= 2 and len(detections) == 0:
                # Look for weak detections that were just below threshold
                # This would require re-running inference with lower threshold
                # For now, just use the existing detections
                pass

        return detections

    def initialize(self):
        """Initialize AI system with working model"""
        try:
            print("🔍 Initializing AI Controller...")

            # Try primary model first
            model_to_use = self.model_path if Path(self.model_path).exists() else self.backup_model_path

            if not Path(model_to_use).exists():
                print(f"❌ No AI models found at {self.model_path} or {self.backup_model_path}")
                return False

            print(f"📁 Loading model: {Path(model_to_use).name}")

            # Load HEF model
            self.hef = hpf.HEF(model_to_use)
            self.input_info = self.hef.get_input_vstream_infos()[0]
            self.output_infos = self.hef.get_output_vstream_infos()

            print(f"✅ Model loaded successfully")
            print(f"   Input: {self.input_info.name}, shape: {self.input_info.shape}")
            print(f"   Outputs: {len(self.output_infos)} outputs")

            # Test inference to validate
            test_frame = np.zeros(self.input_info.shape, dtype=np.uint8)
            test_result = self._run_inference(test_frame)

            if test_result is not None:
                print("✅ AI Controller initialization successful!")
                self.initialized = True
                return True
            else:
                print("❌ AI Controller test inference failed")
                return False

        except Exception as e:
            print(f"❌ AI Controller initialization failed: {e}")
            logger.error(f"AI initialization error: {e}")
            return False

    def is_initialized(self):
        """Check if AI controller is properly initialized"""
        return self.initialized

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame using Hailo AI

        Args:
            frame: Input camera frame (H, W, 3)

        Returns:
            List of detections: [{'bbox': (x, y, w, h), 'confidence': float, 'class': str}, ...]
        """
        if not self.initialized:
            print("🚫 AI not initialized - returning empty detections")
            return []

        try:
            start_time = time.time()
            # Reduce logging noise - only log every 30th frame
            log_this_frame = (self.inference_count % 30 == 0)

            if log_this_frame:
                print(f"🔍 Starting detection on frame {frame.shape}")

            # Preprocess frame
            input_frame = self._preprocess_frame(frame)

            # Run inference
            raw_results = self._run_inference(input_frame)

            if raw_results is None:
                if log_this_frame:
                    print("❌ No raw results from inference")
                return []

            # Post-process results
            detections = self._postprocess_results(raw_results, frame.shape)

            # Always log when we find detections, or periodically for debugging
            if len(detections) > 0 or log_this_frame:
                print(f"🎯 Found {len(detections)} detections (frame #{self.inference_count})")

            # Update statistics
            inference_time = time.time() - start_time
            self._update_stats(inference_time)

            # Filter for dogs and high-confidence detections
            dog_detections = []
            for i, det in enumerate(detections):
                print(f"   Detection {i}: class={det['class']}, conf={det['confidence']:.3f}, bbox={det['bbox']}")
                if det['class'] == 'dog' and det['confidence'] > self.current_conf_threshold:
                    # Add behavior analysis to each detection
                    det['behavior'] = self._analyze_behavior(det)
                    dog_detections.append(det)
                    print(f"   ✅ Dog detection accepted: {det['behavior']}")
                elif det['class'] == 'dog':
                    print(f"   ❌ Dog detection rejected: confidence {det['confidence']:.3f} < {self.current_conf_threshold}")

            print(f"🐕 Final dog detections: {len(dog_detections)}")

            # Handle behavior changes based on detection results
            if dog_detections:
                # Check for behavior changes when dogs are present
                current_behavior = dog_detections[0].get('behavior', 'unknown')
                if current_behavior != self.previous_behavior:
                    self._handle_behavior_change(current_behavior)
            else:
                # Clear behavior when no dogs detected
                if self.current_behavior != "no_dog":
                    self._handle_behavior_change("no_dog")
                    self.behavior_history.clear()  # Clear history when no dogs

            return dog_detections

        except Exception as e:
            logger.error(f"Detection error: {e}")
            print(f"❌ Detection error: {e}")
            return []

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO input"""
        # Resize to model input size
        target_shape = self.input_info.shape
        resized = cv2.resize(frame, (target_shape[1], target_shape[0]))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Ensure correct data type
        return rgb.astype(np.uint8)

    def _run_inference(self, input_frame: np.ndarray) -> Optional[Dict]:
        """Run inference using working InferVStreams API"""
        try:
            with hpf.VDevice() as target:
                configure_params = hpf.ConfigureParams.create_from_hef(
                    self.hef, interface=hpf.HailoStreamInterface.PCIe)
                network_group = target.configure(self.hef, configure_params)[0]
                network_group_params = network_group.create_params()

                input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
                    network_group, quantized=True, format_type=hpf.FormatType.UINT8)
                output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
                    network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

                with network_group.activate(network_group_params):
                    with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                        # Prepare input
                        input_data = {self.input_info.name: np.expand_dims(input_frame, axis=0)}

                        # Run inference
                        results = infer_pipeline.infer(input_data)

                        return results

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None

    def _postprocess_results(self, raw_results: Dict, frame_shape: Tuple) -> List[Dict]:
        """Post-process inference results to extract detections"""
        detections = []

        try:
            # Handle different model output formats
            if "dogbotyolo8" in self.model_path or "nms" in self.model_path.lower():
                # NMS postprocessed outputs
                print("📦 Using NMS postprocessed model")
                detections = self._process_nms_outputs(raw_results, frame_shape)
            elif "bestdogyolo5" in self.model_path:
                # Raw YOLO outputs - need custom NMS
                print("🔧 Using raw YOLO model")
                detections = self._process_raw_yolo_outputs(raw_results, frame_shape)
            else:
                # Auto-detect based on output structure
                print("🤔 Auto-detecting model format...")
                if len(raw_results) == 1:
                    detections = self._process_nms_outputs(raw_results, frame_shape)
                else:
                    detections = self._process_raw_yolo_outputs(raw_results, frame_shape)

        except Exception as e:
            logger.error(f"Postprocessing error: {e}")
            print(f"❌ Postprocessing error: {e}")

        return detections

    def _process_raw_yolo_outputs(self, raw_results: Dict, frame_shape: Tuple) -> List[Dict]:
        """Process raw YOLO outputs (bestdogyolo5.hef with 6 outputs)"""
        detections = []

        try:
            # Raw YOLO model outputs multiple layers at different scales
            # Process each output layer and collect detections
            all_boxes = []
            all_confidences = []
            all_class_ids = []

            for output_name, output_data in raw_results.items():
                if isinstance(output_data, np.ndarray):
                    # Parse YOLO output format
                    boxes, confidences, class_ids = self._parse_yolo_layer(output_data, frame_shape)
                    all_boxes.extend(boxes)
                    all_confidences.extend(confidences)
                    all_class_ids.extend(class_ids)

            # Apply Non-Maximum Suppression to remove duplicate detections
            if len(all_boxes) > 0:
                boxes = np.array(all_boxes)
                confidences = np.array(all_confidences)
                class_ids = np.array(all_class_ids)

                # Apply NMS
                indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(),
                                         self.current_conf_threshold, self.nms_threshold)

                if len(indices) > 0:
                    for i in indices.flatten():
                        x, y, w, h = boxes[i]
                        confidence = confidences[i]
                        class_id = int(class_ids[i])

                        # Only include dogs (class_id 16 in COCO)
                        if class_id == 16 and confidence > self.current_conf_threshold:
                            detections.append({
                                'bbox': (int(x), int(y), int(w), int(h)),
                                'confidence': float(confidence),
                                'class': 'dog'
                            })

        except Exception as e:
            logger.error(f"Raw YOLO processing error: {e}")

        return detections

    def _parse_yolo_layer(self, output_data: np.ndarray, frame_shape: Tuple) -> Tuple[List, List, List]:
        """Parse individual YOLO output layer"""
        boxes = []
        confidences = []
        class_ids = []

        try:
            print(f"🔬 Parsing YOLO layer: input shape {output_data.shape}")

            # Handle different output shapes
            if len(output_data.shape) == 4:  # Batch, Height, Width, Channels
                print(f"   Removing batch dimension: {output_data.shape} -> {output_data[0].shape}")
                output_data = output_data[0]  # Remove batch dimension

            if len(output_data.shape) == 3:  # Height, Width, Channels
                height, width, channels = output_data.shape
                print(f"   3D output: H={height}, W={width}, C={channels}")

                # YOLO v5 output format: [x, y, w, h, obj_conf, class_probs...]
                # Each detection has 5 + num_classes values
                num_classes = 80  # COCO dataset
                num_attrs = 5 + num_classes
                print(f"   Expected attributes per detection: {num_attrs}")

                if channels >= num_attrs:
                    # Reshape to [num_anchors, num_attrs]
                    detections = output_data.reshape(-1, num_attrs)
                    print(f"   Reshaped to detections: {detections.shape}")

                    h, w = frame_shape[:2]
                    valid_detections = 0

                    for i, detection in enumerate(detections):
                        # Extract box coordinates (normalized)
                        center_x, center_y, box_w, box_h = detection[:4]
                        objectness = detection[4]

                        # Get class probabilities
                        class_probs = detection[5:]
                        class_id = np.argmax(class_probs)
                        class_confidence = class_probs[class_id]

                        # Calculate final confidence
                        confidence = objectness * class_confidence

                        # Debug first few detections
                        if i < 5:
                            print(f"     Det {i}: obj={objectness:.3f}, cls_conf={class_confidence:.3f}, final={confidence:.3f}, class={class_id}")

                        if confidence > self.current_conf_threshold:
                            valid_detections += 1
                            # Convert to pixel coordinates
                            x = int((center_x - box_w/2) * w)
                            y = int((center_y - box_h/2) * h)
                            width = int(box_w * w)
                            height = int(box_h * h)

                            # Ensure box is within frame bounds
                            x = max(0, min(x, w - 1))
                            y = max(0, min(y, h - 1))
                            width = max(1, min(width, w - x))
                            height = max(1, min(height, h - y))

                            boxes.append([x, y, width, height])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                    print(f"   Valid detections above threshold: {valid_detections}")
                else:
                    print(f"   ❌ Channel count {channels} < required {num_attrs}")
            else:
                print(f"   ❌ Unexpected output shape: {output_data.shape}")

        except Exception as e:
            logger.error(f"YOLO layer parsing error: {e}")
            print(f"❌ YOLO parsing error: {e}")

        print(f"   Final output: {len(boxes)} boxes, {len(confidences)} confidences, {len(class_ids)} classes")
        return boxes, confidences, class_ids

    def _process_nms_outputs(self, raw_results: Dict, frame_shape: Tuple) -> List[Dict]:
        """Process NMS postprocessed outputs"""
        detections = []

        try:
            h, w = frame_shape[:2]
            log_details = (self.inference_count % 30 == 0)  # Match main logging frequency

            if log_details:
                print(f"🔍 Processing NMS outputs...")

            # NMS outputs are usually numpy arrays with detection results
            for output_name, output_data in raw_results.items():
                if log_details:
                    print(f"   Output '{output_name}': {type(output_data)}")

                if isinstance(output_data, np.ndarray):
                    print(f"     Shape: {output_data.shape}, dtype: {output_data.dtype}")

                    # Common NMS format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
                    if len(output_data.shape) == 3 and output_data.shape[2] >= 6:
                        batch_detections = output_data[0]  # Remove batch dimension

                        for i, detection in enumerate(batch_detections):
                            x1, y1, x2, y2, conf, class_id = detection[:6]

                            if i < 3:  # Debug first few
                                print(f"     Det {i}: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] conf={conf:.3f} class={class_id}")

                            if conf > self.current_conf_threshold and class_id == 16:  # Class 16 = dog in COCO
                                # Convert coordinates to x,y,w,h format
                                x = int(x1 * w)
                                y = int(y1 * h)
                                box_w = int((x2 - x1) * w)
                                box_h = int((y2 - y1) * h)

                                # Ensure within bounds
                                x = max(0, min(x, w - 1))
                                y = max(0, min(y, h - 1))
                                box_w = max(1, min(box_w, w - x))
                                box_h = max(1, min(box_h, h - y))

                                detections.append({
                                    'bbox': (x, y, box_w, box_h),
                                    'confidence': float(conf),
                                    'class': 'dog'
                                })
                                print(f"     ✅ Added dog detection: conf={conf:.3f}, bbox=({x},{y},{box_w},{box_h})")

                    # Alternative format: [num_detections, 6]
                    elif len(output_data.shape) == 2 and output_data.shape[1] >= 6:
                        for i, detection in enumerate(output_data):
                            x1, y1, x2, y2, conf, class_id = detection[:6]

                            if i < 3:  # Debug first few
                                print(f"     Det {i}: [{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}] conf={conf:.3f} class={class_id}")

                            if conf > self.current_conf_threshold and class_id == 16:  # Class 16 = dog in COCO
                                # Convert coordinates
                                x = int(x1 * w) if x1 <= 1.0 else int(x1)
                                y = int(y1 * h) if y1 <= 1.0 else int(y1)
                                box_w = int((x2 - x1) * w) if x2 <= 1.0 else int(x2 - x1)
                                box_h = int((y2 - y1) * h) if y2 <= 1.0 else int(y2 - y1)

                                detections.append({
                                    'bbox': (x, y, box_w, box_h),
                                    'confidence': float(conf),
                                    'class': 'dog'
                                })
                                print(f"     ✅ Added dog detection: conf={conf:.3f}, bbox=({x},{y},{box_w},{box_h})")

                elif isinstance(output_data, (list, tuple)) and len(output_data) > 0:
                    print(f"     List format with {len(output_data)} items")

                    # Deep inspect the first item to understand structure
                    first_item = output_data[0]
                    print(f"     First item type: {type(first_item)}")

                    # CRITICAL FIX: Check if first_item is itself a list containing the class arrays
                    if isinstance(first_item, (list, tuple)) and len(first_item) > 0:
                        print(f"     Nested list detected with {len(first_item)} class arrays!")

                        # This is the actual class-specific array format
                        class_arrays = first_item

                        for class_idx, class_array in enumerate(class_arrays):
                            if isinstance(class_array, np.ndarray) and class_array.size > 0:
                                if log_details or class_array.shape[0] > 0:
                                    print(f"       Class {class_idx}: {class_array.shape} - {class_array.flatten()[:5] if class_array.size > 0 else 'empty'}")

                                # Check if this array has detections
                                if len(class_array.shape) == 2 and class_array.shape[1] >= 5:
                                    for det_idx, detection in enumerate(class_array):
                                        x1, y1, x2, y2, conf = detection[:5]

                                        if log_details or conf > self.current_conf_threshold:
                                            print(f"         Det {det_idx}: class={class_idx}, [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}] conf={conf:.3f}")

                                        # Prioritize primary classes, allow secondary classes
                                        is_primary_class = class_idx in self.primary_dog_classes
                                        is_dog_class = is_primary_class or class_idx in self.secondary_dog_classes

                                        # Use adaptive threshold - lower for primary classes
                                        threshold = self.current_conf_threshold
                                        if is_primary_class:
                                            threshold *= 0.9  # 10% lower threshold for primary classes

                                        # Accept detections from dog classes that meet threshold
                                        if conf > threshold and is_dog_class:
                                            # Convert normalized coordinates to pixels
                                            x = int(x1 * w) if x1 <= 1.0 else int(x1)
                                            y = int(y1 * h) if y1 <= 1.0 else int(y1)
                                            box_w = int((x2 - x1) * w) if x2 <= 1.0 else int(x2 - x1)
                                            box_h = int((y2 - y1) * h) if y2 <= 1.0 else int(y2 - y1)

                                            # Ensure valid bounding box
                                            x = max(0, min(x, w - 1))
                                            y = max(0, min(y, h - 1))
                                            box_w = max(1, min(box_w, w - x))
                                            box_h = max(1, min(box_h, h - y))

                                            detections.append({
                                                'bbox': (x, y, box_w, box_h),
                                                'confidence': float(conf),
                                                'class': 'dog',  # Assume all detections are dogs for now
                                                'class_id': class_idx  # Track which class this came from
                                            })
                                            print(f"         ✅ Added detection: class_idx={class_idx}, conf={conf:.3f}, bbox=({x},{y},{box_w},{box_h})")

                                elif len(class_array.shape) == 1 and class_array.shape[0] >= 5:
                                    # Single detection in this class
                                    x1, y1, x2, y2, conf = class_array[:5]

                                    if log_details or conf > self.current_conf_threshold:
                                        print(f"         Single det: class={class_idx}, [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}] conf={conf:.3f}")

                                    if conf > self.current_conf_threshold:
                                        x = int(x1 * w) if x1 <= 1.0 else int(x1)
                                        y = int(y1 * h) if y1 <= 1.0 else int(y1)
                                        box_w = int((x2 - x1) * w) if x2 <= 1.0 else int(x2 - x1)
                                        box_h = int((y2 - y1) * h) if y2 <= 1.0 else int(y2 - y1)

                                        x = max(0, min(x, w - 1))
                                        y = max(0, min(y, h - 1))
                                        box_w = max(1, min(box_w, w - x))
                                        box_h = max(1, min(box_h, h - y))

                                        detections.append({
                                            'bbox': (x, y, box_w, box_h),
                                            'confidence': float(conf),
                                            'class': 'dog',
                                            'class_id': class_idx
                                        })
                                        print(f"         ✅ Added single detection: class_idx={class_idx}, conf={conf:.3f}")

                    elif isinstance(first_item, np.ndarray):
                        print(f"     First item shape: {first_item.shape}, dtype: {first_item.dtype}")
                        print(f"     First item content sample: {first_item.flatten()[:10] if first_item.size > 0 else 'empty'}")

                        # Handle numpy array within list
                        if len(first_item.shape) == 2 and first_item.shape[1] >= 6:
                            # Standard format: [num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
                            for i, detection in enumerate(first_item):
                                x1, y1, x2, y2, conf, class_id = detection[:6]

                                if i < 3:  # Debug first few
                                    print(f"       Det {i}: [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}] conf={conf:.3f} class={class_id}")

                                if conf > self.current_conf_threshold and class_id == 16:  # Class 16 = dog
                                    # Convert coordinates
                                    x = int(x1 * w) if x1 <= 1.0 else int(x1)
                                    y = int(y1 * h) if y1 <= 1.0 else int(y1)
                                    box_w = int((x2 - x1) * w) if x2 <= 1.0 else int(x2 - x1)
                                    box_h = int((y2 - y1) * h) if y2 <= 1.0 else int(y2 - y1)

                                    detections.append({
                                        'bbox': (x, y, box_w, box_h),
                                        'confidence': float(conf),
                                        'class': 'dog'
                                    })
                                    print(f"       ✅ Added dog: conf={conf:.3f}, bbox=({x},{y},{box_w},{box_h})")

                        elif len(first_item.shape) == 1 and first_item.shape[0] >= 6:
                            # Single detection format
                            x1, y1, x2, y2, conf, class_id = first_item[:6]
                            print(f"     Single detection: [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}] conf={conf:.3f} class={class_id}")

                            if conf > self.current_conf_threshold and class_id == 16:
                                x = int(x1 * w) if x1 <= 1.0 else int(x1)
                                y = int(y1 * h) if y1 <= 1.0 else int(y1)
                                box_w = int((x2 - x1) * w) if x2 <= 1.0 else int(x2 - x1)
                                box_h = int((y2 - y1) * h) if y2 <= 1.0 else int(y2 - y1)

                                detections.append({
                                    'bbox': (x, y, box_w, box_h),
                                    'confidence': float(conf),
                                    'class': 'dog'
                                })
                                print(f"     ✅ Added single dog: conf={conf:.3f}")

                    else:
                        # Handle class-specific array format (YOLO8 NMS postprocessed)
                        # The list contains arrays, one per class - we need to find the dog class
                        print(f"     Processing {len(output_data)} class-specific arrays...")

                        for class_idx, class_array in enumerate(output_data):
                            if isinstance(class_array, np.ndarray) and class_array.size > 0:
                                if log_details or class_array.shape[0] > 0:
                                    print(f"       Class {class_idx}: {class_array.shape} - {class_array.flatten()[:5] if class_array.size > 0 else 'empty'}")

                                # Check if this array has detections
                                if len(class_array.shape) == 2 and class_array.shape[1] >= 5:
                                    for det_idx, detection in enumerate(class_array):
                                        x1, y1, x2, y2, conf = detection[:5]

                                        if log_details or conf > self.current_conf_threshold:
                                            print(f"         Det {det_idx}: class={class_idx}, [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}] conf={conf:.3f}")

                                        # For now, accept any high-confidence detection regardless of class
                                        # We'll determine which class is "dog" based on the model
                                        if conf > self.current_conf_threshold:
                                            # Convert normalized coordinates to pixels
                                            x = int(x1 * w) if x1 <= 1.0 else int(x1)
                                            y = int(y1 * h) if y1 <= 1.0 else int(y1)
                                            box_w = int((x2 - x1) * w) if x2 <= 1.0 else int(x2 - x1)
                                            box_h = int((y2 - y1) * h) if y2 <= 1.0 else int(y2 - y1)

                                            # Ensure valid bounding box
                                            x = max(0, min(x, w - 1))
                                            y = max(0, min(y, h - 1))
                                            box_w = max(1, min(box_w, w - x))
                                            box_h = max(1, min(box_h, h - y))

                                            detections.append({
                                                'bbox': (x, y, box_w, box_h),
                                                'confidence': float(conf),
                                                'class': 'dog',  # Assume all detections are dogs for now
                                                'class_id': class_idx  # Track which class this came from
                                            })
                                            print(f"         ✅ Added detection: class_idx={class_idx}, conf={conf:.3f}, bbox=({x},{y},{box_w},{box_h})")

                                elif len(class_array.shape) == 1 and class_array.shape[0] >= 5:
                                    # Single detection in this class
                                    x1, y1, x2, y2, conf = class_array[:5]

                                    if log_details or conf > self.current_conf_threshold:
                                        print(f"         Single det: class={class_idx}, [{x1:.3f},{y1:.3f},{x2:.3f},{y2:.3f}] conf={conf:.3f}")

                                    if conf > self.current_conf_threshold:
                                        x = int(x1 * w) if x1 <= 1.0 else int(x1)
                                        y = int(y1 * h) if y1 <= 1.0 else int(y1)
                                        box_w = int((x2 - x1) * w) if x2 <= 1.0 else int(x2 - x1)
                                        box_h = int((y2 - y1) * h) if y2 <= 1.0 else int(y2 - y1)

                                        x = max(0, min(x, w - 1))
                                        y = max(0, min(y, h - 1))
                                        box_w = max(1, min(box_w, w - x))
                                        box_h = max(1, min(box_h, h - y))

                                        detections.append({
                                            'bbox': (x, y, box_w, box_h),
                                            'confidence': float(conf),
                                            'class': 'dog',
                                            'class_id': class_idx
                                        })
                                        print(f"         ✅ Added single detection: class_idx={class_idx}, conf={conf:.3f}")

            print(f"   📊 Total detections found: {len(detections)}")

            # Apply temporal smoothing to reduce detection flickering
            detections = self._apply_temporal_smoothing(detections)

        except Exception as e:
            logger.error(f"NMS processing error: {e}")
            print(f"❌ NMS processing error: {e}")

        return detections

    def _update_stats(self, inference_time: float):
        """Update inference statistics"""
        self.inference_count += 1
        self.last_inference_time = inference_time

        # Rolling average
        alpha = 0.1
        if self.avg_inference_time == 0:
            self.avg_inference_time = inference_time
        else:
            self.avg_inference_time = alpha * inference_time + (1 - alpha) * self.avg_inference_time

    def _analyze_behavior(self, detection: Dict) -> str:
        """Analyze dog behavior from detection data"""
        bbox = detection.get('bbox', (0, 0, 1, 1))
        x, y, w, h = bbox

        # Calculate aspect ratio and center
        aspect_ratio = h / w if w > 0 else 1.0
        center = (x + w//2, y + h//2)

        # Create behavior frame
        behavior_frame = BehaviorFrame(
            timestamp=time.time(),
            bbox=bbox,
            confidence=detection['confidence'],
            aspect_ratio=aspect_ratio,
            center=center
        )

        # Add to history
        self.behavior_history.append(behavior_frame)

        # Analyze pose based on aspect ratio
        behavior = self._detect_pose_from_aspect_ratio(aspect_ratio)

        # Check for movement-based behaviors if we have history
        if len(self.behavior_history) >= 30:
            movement_behavior = self._detect_movement_behavior()
            if movement_behavior:
                behavior = movement_behavior

        self.current_behavior = behavior
        return behavior

    def _detect_pose_from_aspect_ratio(self, aspect_ratio: float) -> str:
        """
        Detect pose based on bounding box aspect ratio

        Aspect ratios (height/width):
        - Lying down: < 0.6 (dog is horizontal, wider than tall)
        - Sitting: 0.8 - 1.2 (roughly square)
        - Standing: > 1.3 (taller than wide)
        """
        if aspect_ratio < 0.6:
            return "lying_down"
        elif 0.8 <= aspect_ratio <= 1.2:
            return "sitting"
        elif aspect_ratio > 1.3:
            return "standing"
        else:
            # Transitional poses
            if aspect_ratio < 0.8:
                return "lying_to_sit"  # Transitioning from lying to sitting
            else:
                return "sit_to_stand"  # Transitioning from sitting to standing

    def _detect_movement_behavior(self) -> Optional[str]:
        """Detect movement-based behaviors from history"""
        if len(self.behavior_history) < 30:
            return None

        # Get recent positions
        recent_frames = list(self.behavior_history)[-30:]
        positions = [frame.center for frame in recent_frames]

        # Calculate movement variance
        x_positions = [p[0] for p in positions]
        y_positions = [p[1] for p in positions]

        x_variance = np.var(x_positions) if x_positions else 0
        y_variance = np.var(y_positions) if y_positions else 0

        total_movement = x_variance + y_variance

        # Check for staying still (low movement)
        if total_movement < 100:  # pixels^2
            # Check the pose while staying
            avg_aspect_ratio = np.mean([f.aspect_ratio for f in recent_frames[-10:]])
            if avg_aspect_ratio < 0.6:
                return "lying_down"  # Confirmed lying down
            elif 0.8 <= avg_aspect_ratio <= 1.2:
                return "staying_sit"  # Staying in sit position
            else:
                return "staying_stand"  # Staying in stand position

        # Check for walking/running (high movement)
        elif total_movement > 1000:
            # Check speed of movement
            time_diff = recent_frames[-1].timestamp - recent_frames[0].timestamp
            if time_diff > 0:
                distance = np.sqrt((positions[-1][0] - positions[0][0])**2 +
                                 (positions[-1][1] - positions[0][1])**2)
                speed = distance / time_diff

                if speed > 100:  # pixels/second
                    return "running"
                else:
                    return "walking"

        return None

    def _handle_behavior_change(self, new_behavior: str):
        """Handle behavior change event"""
        if new_behavior != self.previous_behavior:
            change_msg = f"🐕 Behavior changed: {self.previous_behavior} → {new_behavior}"
            print(change_msg)

            # Update behavior tracking
            self.previous_behavior = self.current_behavior
            self.behavior_start_time = time.time()

            # Call callback if registered
            if self.behavior_change_callback:
                self.behavior_change_callback({
                    'previous': self.previous_behavior,
                    'current': new_behavior,
                    'timestamp': time.time()
                })

    def set_behavior_change_callback(self, callback):
        """Set callback for behavior change notifications"""
        self.behavior_change_callback = callback

    def get_current_behavior(self) -> Dict:
        """Get current behavior state"""
        return {
            'behavior': self.current_behavior,
            'confidence': self.behavior_confidence,
            'duration': time.time() - self.behavior_start_time if self.behavior_start_time else 0
        }

    def get_status(self) -> Dict:
        """Get AI controller status"""
        return {
            'initialized': self.initialized,
            'model_loaded': self.hef is not None,
            'inference_count': self.inference_count,
            'last_inference_time': f"{self.last_inference_time:.3f}s",
            'avg_inference_time': f"{self.avg_inference_time:.3f}s",
            'current_behavior': self.current_behavior,
            'behavior_confidence': self.behavior_confidence
        }

    def cleanup(self):
        """Clean up AI resources"""
        try:
            # HailoRT resources are cleaned up automatically with context managers
            self.initialized = False
            print("AI Controller cleanup complete")

        except Exception as e:
            logger.error(f"AI cleanup error: {e}")

# Test function
def test_ai_controller():
    """Test AI controller with camera or test frame"""
    ai = AIController()

    if ai.initialize():
        print("Testing AI detection...")

        # Create test frame
        test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_frame, (100, 100), (300, 300), (255, 255, 255), -1)

        # Run detection
        detections = ai.detect_objects(test_frame)
        print(f"Detections: {detections}")

        # Get status
        status = ai.get_status()
        print(f"Status: {status}")

        ai.cleanup()
        print("AI test complete!")
    else:
        print("AI initialization failed")

if __name__ == "__main__":
    test_ai_controller()