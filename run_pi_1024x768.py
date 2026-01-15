#!/usr/bin/env python3
"""
Enhanced Pose Detection with 1024x768 Resolution Support
Includes ArUco visualization, behavior cooldowns, and servo tracking
"""

import os
import json
import time
import collections
import numpy as np
import torch
import cv2
import threading
from datetime import datetime, timedelta
from pathlib import Path

# Hailo imports
try:
    import hailo_platform as hpf
    HAILO_AVAILABLE = True
except ImportError:
    print("[WARNING] HailoRT not available - will use CPU fallback")
    HAILO_AVAILABLE = False

# ------------------------
# Config
# ------------------------
CFG = json.load(open("config/config.json"))

# Handle both single value and [height, width] format
imgsz_cfg = CFG.get("imgsz", [768, 1024])
if isinstance(imgsz_cfg, list):
    IMGSZ_H, IMGSZ_W = imgsz_cfg  # [768, 1024] -> height=768, width=1024
else:
    IMGSZ_H = IMGSZ_W = imgsz_cfg  # backward compatibility for square models

T = int(CFG.get("T", 16))
BEHAVIORS = list(CFG.get("behaviors", ["stand", "sit", "lie", "cross", "spin"]))
PROB_TH = float(CFG.get("prob_th", 0.7))
COOLDOWN_S = dict(CFG.get("cooldown_s", {"stand": 2, "sit": 5, "lie": 5, "cross": 4, "spin": 8}))
ASSUME_OTHER = bool(CFG.get("assume_other_if_two_boxes_one_marker", True))
CAM_ROT_DEG = int(CFG.get("camera_rotation_deg", 90))
HEF_PATH = CFG.get("hef_path", "ai/models/yolo_pose_1024x768.hef")
HEAD_TS = CFG.get("behavior_head_ts", "ai/models/behavior_head.ts")

MARKER_TO_DOG = {int(d["marker_id"]): str(d["id"]) for d in CFG.get("dogs", [])}

print(f"[CONFIG] Model resolution: {IMGSZ_W}x{IMGSZ_H} (WxH)")
print(f"[CONFIG] HEF model: {HEF_PATH}")
print(f"[CONFIG] Behavior head: {HEAD_TS}")
print(f"[CONFIG] Camera rotation: {CAM_ROT_DEG}°")

# ------------------------
# Behavior Tracking
# ------------------------
class BehaviorTracker:
    """Track behaviors with cooldowns and smoothing"""

    def __init__(self, dog_id):
        self.dog_id = dog_id
        self.last_behaviors = {}  # behavior -> last_time
        self.behavior_history = collections.deque(maxlen=5)  # Last 5 behaviors
        self.cooldown_active = {}  # behavior -> cooldown_end_time

    def can_trigger(self, behavior, current_time):
        """Check if behavior can be triggered (cooldown expired)"""
        if behavior in self.cooldown_active:
            if current_time < self.cooldown_active[behavior]:
                return False
        return True

    def record_behavior(self, behavior, confidence, current_time):
        """Record a behavior and manage cooldowns"""
        if not self.can_trigger(behavior, current_time):
            return False

        # Record the behavior
        self.last_behaviors[behavior] = current_time
        self.behavior_history.append((behavior, confidence, current_time))

        # Set cooldown
        cooldown_duration = COOLDOWN_S.get(behavior, 5)
        self.cooldown_active[behavior] = current_time + timedelta(seconds=cooldown_duration)

        return True

    def get_active_cooldowns(self, current_time):
        """Get list of behaviors currently in cooldown"""
        active = []
        for behavior, end_time in list(self.cooldown_active.items()):
            if current_time < end_time:
                remaining = (end_time - current_time).total_seconds()
                active.append((behavior, remaining))
            else:
                # Remove expired cooldown
                del self.cooldown_active[behavior]
        return active

# ------------------------
# ArUco Detection
# ------------------------
def setup_aruco():
    dict_name = str(CFG.get("aruco_dict", "DICT_4X4_1000"))
    dconst = getattr(cv2.aruco, dict_name)
    dic = cv2.aruco.getPredefinedDictionary(dconst)
    try:
        det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    except AttributeError:  # older OpenCV
        det = (dic, cv2.aruco.DetectorParameters_create())
    return det

ARUCO_DET = setup_aruco()

def detect_markers_visual(bgr):
    """Detect ArUco markers and return with visualization info"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    try:
        corners, ids, _ = ARUCO_DET.detectMarkers(gray)
    except Exception:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DET[0], parameters=ARUCO_DET[1])

    markers = []
    vis_data = []  # For visualization

    if ids is not None:
        for c, id_ in zip(corners, ids.flatten()):
            pts = c[0]
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            markers.append((int(id_), cx, cy))
            vis_data.append({
                'id': int(id_),
                'corners': pts,
                'center': (int(cx), int(cy)),
                'dog_name': MARKER_TO_DOG.get(int(id_), f"Unknown_{id_}")
            })

    return markers, vis_data

def draw_aruco_markers(img, vis_data):
    """Draw ArUco markers on image"""
    for marker in vis_data:
        # Draw marker outline
        pts = marker['corners'].astype(int)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

        # Draw center point
        cv2.circle(img, marker['center'], 5, (0, 255, 0), -1)

        # Draw ID and dog name
        text = f"ID:{marker['id']} ({marker['dog_name']})"
        cv2.putText(img, text,
                   (marker['center'][0] - 30, marker['center'][1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# ------------------------
# Image Preprocessing
# ------------------------
def letterbox(img, target_w=IMGSZ_W, target_h=IMGSZ_H):
    """Letterbox image to target size preserving aspect ratio"""
    h, w = img.shape[:2]

    # Handle 4-channel images (RGBA/XBGR)
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Calculate scale to fit image in target size
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create canvas and place resized image
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    dy, dx = (target_h - new_h) // 2, (target_w - new_w) // 2
    canvas[dy:dy+new_h, dx:dx+new_w] = resized

    return canvas, (dx, dy, 1.0/scale)

def rotate_image(img, angle):
    """Rotate image by angle (0, 90, 180, 270)"""
    if angle == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

# ------------------------
# Hailo Inference
# ------------------------
class HailoPoseInference:
    def __init__(self, hef_path):
        self.hef_path = hef_path
        self.initialized = False
        self.hef = None
        self.vdevice = None
        self.network_group = None
        self.infer_pipeline = None

    def initialize(self):
        """Initialize Hailo device"""
        if not HAILO_AVAILABLE:
            print("[ERROR] HailoRT not available")
            return False

        try:
            print(f"[HAILO] Loading HEF: {self.hef_path}")
            self.hef = hpf.HEF(self.hef_path)

            # Setup device
            self.vdevice = hpf.VDevice()
            params = hpf.ConfigureParams.create_from_hef(
                hef=self.hef,
                interface=hpf.HailoStreamInterface.PCIe
            )

            network_groups = self.vdevice.configure(self.hef, params)
            self.network_group = network_groups[0]

            # Get stream info
            input_vstreams_info = self.hef.get_input_vstream_infos()
            output_vstreams_info = self.hef.get_output_vstream_infos()

            print(f"[HAILO] Model inputs: {len(input_vstreams_info)}")
            for info in input_vstreams_info:
                print(f"  Input: {info.name}, shape: {info.shape}")

            print(f"[HAILO] Model outputs: {len(output_vstreams_info)}")
            for info in output_vstreams_info:
                print(f"  Output: {info.name}, shape: {info.shape}")

            # Create network group parameters and vstream parameters
            self.network_group_params = self.network_group.create_params()

            self.input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
                self.network_group,
                quantized=True,
                format_type=hpf.FormatType.UINT8
            )

            self.output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
                self.network_group,
                quantized=False,
                format_type=hpf.FormatType.FLOAT32
            )

            self.initialized = True
            return True

        except Exception as e:
            print(f"[ERROR] Failed to initialize Hailo: {e}")
            return False

    def infer(self, img_preprocessed):
        """Run inference on preprocessed image"""
        if not self.initialized:
            return None

        try:
            # Use the working pattern from run_pi.py
            with self.network_group.activate(self.network_group_params):
                with hpf.InferVStreams(self.network_group,
                                      self.input_vstreams_params,
                                      self.output_vstreams_params) as infer_pipeline:

                    # Prepare input - ensure UINT8 and correct shape
                    input_tensor = np.expand_dims(img_preprocessed, axis=0).astype(np.uint8)
                    input_data = {list(self.input_vstreams_params.keys())[0]: input_tensor}

                    # Run inference
                    output_data = infer_pipeline.infer(input_data)
                    return output_data

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return None

    def cleanup(self):
        """Clean up Hailo resources"""
        if self.vdevice:
            self.vdevice = None
        self.initialized = False

# ------------------------
# Pose Decoding
# ------------------------
def decode_hailo_pose_outputs(raw_outputs):
    """Decode YOLOv11 pose outputs from Hailo - 1024x768 version"""
    if not isinstance(raw_outputs, dict):
        return []

    print(f"\n[DECODE] Processing {len(raw_outputs)} outputs")

    # Debug output shapes
    for key, output in raw_outputs.items():
        print(f"  {key}: shape={output.shape}, dtype={output.dtype}")

    detections = []

    # Group outputs by scale - adjust grid sizes for 1024x768
    # Expected scales for 1024x768: 128x96, 64x48, 32x24
    scales = {
        '128x96': {'bbox': None, 'kpts': None, 'conf': None},
        '64x48': {'bbox': None, 'kpts': None, 'conf': None},
        '32x24': {'bbox': None, 'kpts': None, 'conf': None}
    }

    # Map conv layers based on shape patterns
    for layer_name, output in raw_outputs.items():
        h, w = output.shape[1], output.shape[2] if len(output.shape) == 4 else (0, 0)
        channels = output.shape[3] if len(output.shape) == 4 else 0

        # Determine scale and type based on shape
        scale_name = None
        output_type = None

        if (h, w) == (128, 96) or (h, w) == (96, 128):  # Large scale
            scale_name = '128x96'
        elif (h, w) == (64, 48) or (h, w) == (48, 64):  # Medium scale
            scale_name = '64x48'
        elif (h, w) == (32, 24) or (h, w) == (24, 32):  # Small scale
            scale_name = '32x24'

        if scale_name and channels == 64:
            output_type = 'bbox'
        elif scale_name and channels == 72:
            output_type = 'kpts'
        elif scale_name and channels == 1:
            output_type = 'conf'

        if scale_name and output_type:
            scales[scale_name][output_type] = output
            print(f"  Mapped {layer_name} -> {scale_name} {output_type}")

    # Process each scale
    all_predictions = []
    strides = [8, 16, 32]  # YOLOv11 strides
    scale_names = ['128x96', '64x48', '32x24']

    for scale_idx, scale_name in enumerate(scale_names):
        scale_data = scales[scale_name]

        if not all(scale_data[key] is not None for key in ['bbox', 'kpts', 'conf']):
            continue

        bbox_out = scale_data['bbox']
        kpts_out = scale_data['kpts']
        conf_out = scale_data['conf']

        _, h, w, _ = bbox_out.shape
        h, w = int(h), int(w)  # Ensure integers
        stride = strides[scale_idx]

        # Process high-confidence cells
        for i in range(h):
            for j in range(w):
                conf_raw = conf_out[0, i, j, 0]
                conf = 1.0 / (1.0 + np.exp(-conf_raw))  # Sigmoid

                if conf < 0.25:  # Confidence threshold
                    continue

                # Decode bounding box
                bbox_raw = bbox_out[0, i, j, :4]
                cx = (bbox_raw[0] + j) * stride
                cy = (bbox_raw[1] + i) * stride
                bbox_w = np.exp(bbox_raw[2]) * stride
                bbox_h = np.exp(bbox_raw[3]) * stride

                x1 = cx - bbox_w / 2
                y1 = cy - bbox_h / 2
                x2 = cx + bbox_w / 2
                y2 = cy + bbox_h / 2

                # Decode keypoints
                kpts_raw = kpts_out[0, i, j, :]
                kpts = np.zeros((24, 3), dtype=np.float32)

                for k in range(24):
                    # FIXED: raw values must be multiplied by 2 (per official Hailo YOLOv8 pose decoding)
                    kpts[k, 0] = (kpts_raw[k * 3] * 2 + j) * stride
                    kpts[k, 1] = (kpts_raw[k * 3 + 1] * 2 + i) * stride
                    kpts[k, 2] = 1.0 / (1.0 + np.exp(-kpts_raw[k * 3 + 2]))

                all_predictions.append({
                    'bbox': np.array([x1, y1, x2, y2]),
                    'keypoints': kpts,
                    'confidence': conf
                })

    # Apply NMS
    if all_predictions:
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        keep = []
        for pred in all_predictions:
            should_keep = True

            for kept in keep:
                # Calculate IoU
                x1 = max(pred['bbox'][0], kept['bbox'][0])
                y1 = max(pred['bbox'][1], kept['bbox'][1])
                x2 = min(pred['bbox'][2], kept['bbox'][2])
                y2 = min(pred['bbox'][3], kept['bbox'][3])

                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (pred['bbox'][2] - pred['bbox'][0]) * (pred['bbox'][3] - pred['bbox'][1])
                    area2 = (kept['bbox'][2] - kept['bbox'][0]) * (kept['bbox'][3] - kept['bbox'][1])
                    iou = intersection / (area1 + area2 - intersection + 1e-6)

                    if iou > 0.4:
                        should_keep = False
                        break

            if should_keep:
                keep.append(pred)
                detections.append(pred)

                if len(detections) >= 10:
                    break

    print(f"[DECODE] Found {len(detections)} detections after NMS")
    return detections

# ------------------------
# Behavior Classification
# ------------------------
class BehaviorClassifier:
    def __init__(self, model_path):
        self.model = None
        self.model_path = model_path
        self.trackers = {}  # dog_id -> BehaviorTracker

    def load(self):
        """Load TorchScript behavior model"""
        try:
            self.model = torch.jit.load(self.model_path, map_location="cpu")
            self.model.eval()
            print(f"[BEHAVIOR] Loaded model: {self.model_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load behavior model: {e}")
            return False

    def get_tracker(self, dog_id):
        """Get or create behavior tracker for dog"""
        if dog_id not in self.trackers:
            self.trackers[dog_id] = BehaviorTracker(dog_id)
        return self.trackers[dog_id]

    def classify(self, keypoint_sequence, dog_id="unknown"):
        """Classify behavior from keypoint sequence"""
        if self.model is None:
            return None, 0.0, False

        try:
            with torch.no_grad():
                # Prepare input - keypoint_sequence is (T, 24, 2) for temporal sequence
                if len(keypoint_sequence.shape) == 3 and keypoint_sequence.shape[-1] == 3:
                    # Remove confidence scores, keep only x,y: (T, 24, 3) -> (T, 24, 2)
                    keypoint_sequence = keypoint_sequence[:, :, :2]

                # Expected shape: (T, 24, 2) -> flatten to (T, 48)
                if len(keypoint_sequence.shape) == 3:
                    # Flatten each frame: (T, 24, 2) -> (T, 48)
                    flattened_sequence = keypoint_sequence.reshape(keypoint_sequence.shape[0], -1)
                else:
                    # Single frame: (24, 2) -> (1, 48)
                    flattened_sequence = keypoint_sequence.reshape(1, -1)

                # Ensure correct dimensions
                expected_features = 48  # 24 keypoints × 2 coordinates
                if flattened_sequence.shape[1] != expected_features:
                    print(f"[WARNING] Expected {expected_features}D features, got {flattened_sequence.shape[1]}D")
                    return None, 0.0, False

                input_tensor = torch.from_numpy(flattened_sequence).float()
                # Input should be (batch_size, sequence_length, features) or (batch_size, features)
                if input_tensor.dim() == 2:
                    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

                # Run inference
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=-1)

                # Get top behavior
                confidence, idx = probs.max(dim=-1)
                behavior = BEHAVIORS[idx.item()]
                conf_value = confidence.item()

                # Check if confidence meets threshold
                if conf_value >= PROB_TH:
                    tracker = self.get_tracker(dog_id)
                    current_time = datetime.now()

                    # Try to record behavior (respecting cooldowns)
                    if tracker.record_behavior(behavior, conf_value, current_time):
                        return behavior, conf_value, True  # Behavior triggered
                    else:
                        return behavior, conf_value, False  # In cooldown

                return None, conf_value, False

        except Exception as e:
            print(f"[ERROR] Behavior classification failed: {e}")
            return None, 0.0, False

# ------------------------
# Servo Tracking Controller
# ------------------------
class ServoTracker:
    """Control servo tracking of detected dogs"""

    def __init__(self):
        self.tracking_enabled = False
        self.tracking_dog_id = None
        self.tracking_start_time = None
        self.last_position = None
        self.servo_controller = None

        # Tracking parameters
        self.MIN_TRACKING_TIME = 5.0  # Minimum 5 seconds before tracking
        self.DEADZONE = 0.1  # 10% deadzone in center
        self.SERVO_SPEED = 0.05  # Servo movement speed

    def set_servo_controller(self, controller):
        """Set the servo controller instance"""
        self.servo_controller = controller

    def update(self, detections, img_width, img_height):
        """Update servo tracking based on detections"""
        if not self.servo_controller:
            return

        current_time = time.time()

        # Find if we have a dog to track
        if detections:
            # Get the first detection (or implement selection logic)
            det = detections[0]
            dog_id = det.get('dog_id', 'unknown')
            bbox = det.get('bbox', None)

            if bbox is not None:
                # Calculate center of detection
                cx = (bbox[0] + bbox[2]) / 2 / img_width
                cy = (bbox[1] + bbox[3]) / 2 / img_height

                # Check if we should start tracking
                if not self.tracking_enabled:
                    if self.tracking_dog_id == dog_id:
                        # Same dog, check time
                        if current_time - self.tracking_start_time > self.MIN_TRACKING_TIME:
                            self.tracking_enabled = True
                            print(f"[SERVO] Started tracking {dog_id}")
                    else:
                        # New dog, reset timer
                        self.tracking_dog_id = dog_id
                        self.tracking_start_time = current_time

                # Perform tracking if enabled
                if self.tracking_enabled:
                    self.track_position(cx, cy)

        else:
            # No detection, reset to neutral if tracking
            if self.tracking_enabled:
                print("[SERVO] Lost tracking, returning to neutral")
                self.return_to_neutral()
                self.tracking_enabled = False
                self.tracking_dog_id = None

    def track_position(self, norm_x, norm_y):
        """Track normalized position (0-1) with servos"""
        # Calculate error from center
        error_x = norm_x - 0.5
        error_y = norm_y - 0.5

        # Apply deadzone
        if abs(error_x) < self.DEADZONE:
            error_x = 0
        if abs(error_y) < self.DEADZONE:
            error_y = 0

        # Calculate servo adjustments
        if error_x != 0:
            # Pan servo (horizontal)
            current_pan = self.servo_controller.get_pan_angle()
            pan_adjust = -error_x * self.SERVO_SPEED * 180  # Invert for correct direction
            new_pan = max(-90, min(90, current_pan + pan_adjust))
            self.servo_controller.set_pan_angle(new_pan)

        if error_y != 0:
            # Tilt servo (vertical)
            current_tilt = self.servo_controller.get_tilt_angle()
            tilt_adjust = error_y * self.SERVO_SPEED * 180
            new_tilt = max(-90, min(90, current_tilt + tilt_adjust))
            self.servo_controller.set_tilt_angle(new_tilt)

    def return_to_neutral(self):
        """Return ONLY camera servos to neutral position (NEVER touch treat servos)"""
        if self.servo_controller:
            # Only center pan and tilt servos (channels 0,1) - NEVER touch treat servos (channels 2,3)
            self.servo_controller.set_pan_angle(0)
            self.servo_controller.set_tilt_angle(0)

# ------------------------
# Main Application
# ------------------------
class PoseDetectionApp:
    def __init__(self):
        self.hailo = HailoPoseInference(HEF_PATH)
        self.behavior_classifier = BehaviorClassifier(HEAD_TS)
        self.servo_tracker = ServoTracker()
        self.running = False

        # Keypoint buffers for each dog
        self.keypoint_buffers = {}

        # Statistics
        self.fps_tracker = collections.deque(maxlen=30)

    def initialize(self):
        """Initialize all components"""
        if not self.hailo.initialize():
            print("[ERROR] Failed to initialize Hailo")
            return False

        if not self.behavior_classifier.load():
            print("[ERROR] Failed to load behavior classifier")
            return False

        # DISABLE servo controller initialization to prevent treat servo activation
        # Only enable when specifically needed for camera tracking in production
        print("[SERVO] Controller disabled to prevent treat servo activation during testing")

        return True

    def process_frame(self, frame):
        """Process a single frame"""
        start_time = time.time()

        # Rotate if configured
        if CAM_ROT_DEG:
            frame = rotate_image(frame, CAM_ROT_DEG)

        # Preprocess
        preprocessed, (dx, dy, scale_inv) = letterbox(frame, IMGSZ_W, IMGSZ_H)

        # Detect ArUco markers
        markers, aruco_vis = detect_markers_visual(frame)

        # Run pose detection
        outputs = self.hailo.infer(preprocessed)
        detections = decode_hailo_pose_outputs(outputs) if outputs else []

        # Process detections
        results = {
            'detections': [],
            'behaviors': {},
            'aruco_markers': aruco_vis,
            'fps': 0
        }

        for det in detections:
            # Map back to original coordinates
            bbox = det['bbox']
            bbox[0] = (bbox[0] - dx) * scale_inv
            bbox[1] = (bbox[1] - dy) * scale_inv
            bbox[2] = (bbox[2] - dx) * scale_inv
            bbox[3] = (bbox[3] - dy) * scale_inv

            # Assign dog ID based on ArUco markers
            dog_id = self.assign_dog_id(bbox, markers)
            det['dog_id'] = dog_id

            # Update keypoint buffer
            if dog_id not in self.keypoint_buffers:
                self.keypoint_buffers[dog_id] = collections.deque(maxlen=T)

            # Normalize and buffer keypoints
            if 'keypoints' in det:
                normalized_kpts = self.normalize_keypoints(bbox, det['keypoints'])
                self.keypoint_buffers[dog_id].append(normalized_kpts)

                # Classify behavior if we have enough frames
                if len(self.keypoint_buffers[dog_id]) == T:
                    kpt_sequence = np.array(self.keypoint_buffers[dog_id])
                    behavior, conf, triggered = self.behavior_classifier.classify(kpt_sequence, dog_id)

                    if behavior:
                        results['behaviors'][dog_id] = {
                            'behavior': behavior,
                            'confidence': conf,
                            'triggered': triggered
                        }

                        if triggered:
                            print(f"[BEHAVIOR] {dog_id}: {behavior} (conf: {conf:.2f})")
                            self.trigger_action(dog_id, behavior)

            results['detections'].append(det)

        # Update servo tracking
        self.servo_tracker.update(results['detections'], frame.shape[1], frame.shape[0])

        # Calculate FPS
        process_time = time.time() - start_time
        self.fps_tracker.append(1.0 / max(process_time, 0.001))
        results['fps'] = np.mean(self.fps_tracker) if self.fps_tracker else 0

        return results

    def assign_dog_id(self, bbox, markers):
        """Assign dog ID based on ArUco markers"""
        if not markers:
            return "unknown"

        # Find nearest marker
        bbox_cx = (bbox[0] + bbox[2]) / 2
        bbox_cy = (bbox[1] + bbox[3]) / 2

        min_dist = float('inf')
        nearest_marker_id = None

        for marker_id, mx, my in markers:
            dist = ((bbox_cx - mx)**2 + (bbox_cy - my)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                nearest_marker_id = marker_id

        return MARKER_TO_DOG.get(nearest_marker_id, f"dog_{nearest_marker_id}")

    def normalize_keypoints(self, bbox, keypoints):
        """Normalize keypoints relative to bounding box"""
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)

        normalized = np.zeros((24, 2))
        for i in range(min(24, len(keypoints))):
            normalized[i, 0] = np.clip((keypoints[i, 0] - x1) / w, 0, 1)
            normalized[i, 1] = np.clip((keypoints[i, 1] - y1) / h, 0, 1)

        return normalized

    def trigger_action(self, dog_id, behavior):
        """Trigger action for detected behavior"""
        # This is where you'd integrate with the treat dispenser
        print(f"[ACTION] Would dispense treat for {dog_id} - {behavior}")

        # Try to trigger treat dispenser if available
        try:
            from core.hardware.treat_dispenser import dispense_treat
            dispense_treat()
        except:
            pass

    def draw_results(self, frame, results):
        """Draw detection results on frame"""
        # Draw ArUco markers
        if results['aruco_markers']:
            draw_aruco_markers(frame, results['aruco_markers'])

        # Draw detections
        for det in results['detections']:
            bbox = det['bbox'].astype(int)
            dog_id = det.get('dog_id', 'unknown')

            # Draw bounding box
            color = (0, 255, 0) if dog_id != 'unknown' else (255, 0, 0)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Draw label
            label = f"{dog_id}"
            if dog_id in results['behaviors']:
                behavior_info = results['behaviors'][dog_id]
                label += f" - {behavior_info['behavior']} ({behavior_info['confidence']:.2f})"
                if not behavior_info['triggered']:
                    label += " [COOLDOWN]"

            cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw keypoints if available
            if 'keypoints' in det:
                kpts = det['keypoints']
                for i, (x, y, conf) in enumerate(kpts):
                    if conf > 0.5:
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        # Draw FPS
        fps_text = f"FPS: {results['fps']:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Draw cooldown status
        y_offset = 60
        for dog_id, tracker in self.behavior_classifier.trackers.items():
            cooldowns = tracker.get_active_cooldowns(datetime.now())
            if cooldowns:
                text = f"{dog_id} cooldowns: "
                for behavior, remaining in cooldowns:
                    text += f"{behavior}({remaining:.1f}s) "
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 25

        return frame

    def run_camera(self):
        """Run with camera feed"""
        # Try Picamera2 first
        try:
            from picamera2 import Picamera2
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={"size": (1024, 768), "format": "XBGR8888"}
            )
            camera.configure(config)
            camera.start()
            print("[CAMERA] Using Picamera2")
            use_picamera = True
        except Exception as e:
            print(f"[CAMERA] Picamera2 not available: {e}")
            print("[CAMERA] Using OpenCV")
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
            use_picamera = False

        cv2.namedWindow("Pose Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pose Detection", 1024, 768)

        self.running = True

        try:
            while self.running:
                # Capture frame
                if use_picamera:
                    frame = camera.capture_array()
                else:
                    ret, frame = camera.read()
                    if not ret:
                        continue

                # Process frame
                results = self.process_frame(frame)

                # Draw results
                display_frame = self.draw_results(frame.copy(), results)

                # Display
                cv2.imshow("Pose Detection", display_frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"pose_detection_{timestamp}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"[SAVE] Screenshot saved: {filename}")
                elif key == ord('t'):
                    # Toggle tracking
                    self.servo_tracker.tracking_enabled = not self.servo_tracker.tracking_enabled
                    state = "ON" if self.servo_tracker.tracking_enabled else "OFF"
                    print(f"[SERVO] Tracking toggled: {state}")
                elif key == ord('r'):
                    # Reset servos to neutral
                    self.servo_tracker.return_to_neutral()
                    print("[SERVO] Reset to neutral")

        finally:
            self.running = False
            cv2.destroyAllWindows()

            if use_picamera:
                camera.stop()
            else:
                camera.release()

            self.hailo.cleanup()
            print("[APP] Shutdown complete")

# ------------------------
# Main Entry Point
# ------------------------
def main():
    print("=" * 60)
    print("POSE DETECTION WITH 1024x768 RESOLUTION")
    print("=" * 60)
    print("Controls:")
    print("  Q - Quit")
    print("  S - Save screenshot")
    print("  T - Toggle servo tracking")
    print("  R - Reset servos to neutral")
    print("=" * 60)

    app = PoseDetectionApp()

    if not app.initialize():
        print("[ERROR] Failed to initialize application")
        return 1

    try:
        app.run_camera()
    except KeyboardInterrupt:
        print("\n[APP] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Application error: {e}")
        import traceback
        traceback.print_exc()

    return 0

if __name__ == "__main__":
    exit(main())