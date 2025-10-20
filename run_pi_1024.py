#!/usr/bin/env python3
"""
Pose Detection with New 1024x1024 Model
Includes ArUco visualization, behavior cooldowns, and servo tracking
Updated for retrained model with better false positive handling
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

# Check if GUI is available
CV2_GUI_AVAILABLE = False
try:
    # Try to create a test window
    test_win = "test"
    cv2.namedWindow(test_win)
    cv2.destroyWindow(test_win)
    CV2_GUI_AVAILABLE = True
    print("[INFO] OpenCV GUI support detected")
except cv2.error:
    print("[WARNING] OpenCV GUI not available - running in headless mode")
    print("[WARNING] Results will be saved to disk but not displayed")

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
imgsz_cfg = CFG.get("imgsz", [1024, 1024])
if isinstance(imgsz_cfg, list):
    IMGSZ_H, IMGSZ_W = imgsz_cfg  # [1024, 1024] -> height=1024, width=1024
else:
    IMGSZ_H = IMGSZ_W = imgsz_cfg  # backward compatibility for square models

T = int(CFG.get("T", 14))
BEHAVIORS = list(CFG.get("behaviors", ["stand", "sit", "lie", "cross", "spin"]))
PROB_TH = float(CFG.get("prob_th", 0.6))
COOLDOWN_S = dict(CFG.get("cooldown_s", {"stand": 2, "sit": 5, "lie": 5, "cross": 4, "spin": 8}))
ASSUME_OTHER = bool(CFG.get("assume_other_if_two_boxes_one_marker", True))
CAM_ROT_DEG = int(CFG.get("camera_rotation_deg", 90))
HEF_PATH = CFG.get("hef_path", "ai/models/yolo_pose.hef")
HEAD_TS = CFG.get("behavior_head_ts", "ai/models/behavior_head.ts")

MARKER_TO_DOG = {int(d["marker_id"]): str(d["id"]) for d in CFG.get("dogs", [])}

print(f"[CONFIG] Model resolution: {IMGSZ_W}x{IMGSZ_H} (WxH)")
print(f"[CONFIG] HEF model: {HEF_PATH}")
print(f"[CONFIG] Behavior head: {HEAD_TS}")
print(f"[CONFIG] Camera rotation: {CAM_ROT_DEG}°")
print(f"[CONFIG] Detection threshold: {PROB_TH}")
print(f"[CONFIG] Behaviors: {BEHAVIORS}")

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
# YOLO Decoder (multi-scale)
# ------------------------
def parse_multi_scale_outputs(outputs, img_h, img_w, pad_t, pad_l, scale, conf_thresh=0.3):
    """Parse multi-scale YOLO outputs"""

    # Expected output shapes for pose detection
    EXPECTED_SHAPES = [
        (1, 144, 128, 128),  # stride 8
        (1, 144, 64, 64),    # stride 16
        (1, 144, 32, 32),    # stride 32
    ]

    strides = [8, 16, 32]
    all_boxes = []
    all_scores = []
    all_keypoints = []

    for out, stride, expected_shape in zip(outputs, strides, EXPECTED_SHAPES):
        # Verify shape
        if out.shape != expected_shape:
            print(f"[WARNING] Unexpected output shape: {out.shape}, expected {expected_shape}")
            continue

        bs, ch, gy, gx = out.shape

        # Generate grid
        yv, xv = np.meshgrid(np.arange(gy), np.arange(gx), indexing='ij')
        grid = np.stack([xv, yv], axis=2).reshape(1, gy, gx, 2).astype(np.float32)

        # Process predictions
        out_reshaped = out.reshape(bs, ch, gy * gx).transpose(0, 2, 1)  # [1, gy*gx, 144]

        for b in range(bs):
            for anchor_idx in range(gy * gx):
                pred = out_reshaped[b, anchor_idx]

                # Extract components (adjusted for pose model)
                # Format: [x, y, w, h, obj_conf, class_scores..., keypoints...]
                # With 24 keypoints * 3 values = 72 keypoint values
                cx = (pred[0] * 2 - 0.5 + grid.reshape(-1, 2)[anchor_idx, 0]) * stride
                cy = (pred[1] * 2 - 0.5 + grid.reshape(-1, 2)[anchor_idx, 1]) * stride
                w = (pred[2] * 2) ** 2 * stride
                h = (pred[3] * 2) ** 2 * stride
                obj_conf = pred[4]

                # Skip low confidence
                if obj_conf < conf_thresh:
                    continue

                # Get class probabilities (if any)
                num_classes = 1  # For dog-only model
                if ch > 77:  # 5 + 72 for pose
                    class_probs = pred[5:5+num_classes]
                    keypoint_data = pred[5+num_classes:5+num_classes+72]
                else:
                    class_probs = np.array([1.0])  # Assume dog if no class scores
                    keypoint_data = pred[5:77]

                # Compute final score
                cls_conf = np.max(class_probs)
                final_score = obj_conf * cls_conf

                if final_score < conf_thresh:
                    continue

                # Convert box to image coordinates
                x1 = (cx - w/2 - pad_l) / scale
                y1 = (cy - h/2 - pad_t) / scale
                x2 = (cx + w/2 - pad_l) / scale
                y2 = (cy + h/2 - pad_t) / scale

                # Clip to image bounds
                x1 = max(0, min(x1, img_w))
                y1 = max(0, min(y1, img_h))
                x2 = max(0, min(x2, img_w))
                y2 = max(0, min(y2, img_h))

                # Parse keypoints (24 points x 3 values)
                keypoints = []
                for kp_idx in range(24):
                    kp_x = keypoint_data[kp_idx * 3] * stride
                    kp_y = keypoint_data[kp_idx * 3 + 1] * stride
                    kp_conf = keypoint_data[kp_idx * 3 + 2]

                    # Convert to image coordinates
                    kp_x = (kp_x - pad_l) / scale
                    kp_y = (kp_y - pad_t) / scale

                    keypoints.extend([kp_x, kp_y, kp_conf])

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(final_score)
                all_keypoints.append(keypoints)

    return all_boxes, all_scores, all_keypoints

# ------------------------
# NMS Decoder
# ------------------------
def parse_nms_output(output, img_h, img_w, pad_t, pad_l, scale):
    """Parse NMS output format [n, 82] where each row is a detection"""
    boxes = []
    scores = []
    keypoints = []

    for det in output:
        # Format: [x1, y1, x2, y2, score, class_id, ...keypoints...]
        x1, y1, x2, y2, score, class_id = det[:6]

        # Skip invalid detections
        if score < 0.01 or x1 >= x2 or y1 >= y2:
            continue

        # Convert to image coordinates
        x1 = (x1 - pad_l) / scale
        y1 = (y1 - pad_t) / scale
        x2 = (x2 - pad_l) / scale
        y2 = (y2 - pad_t) / scale

        # Clip to image bounds
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        # Extract keypoints if present
        kpts = []
        if len(det) > 6:  # Has keypoint data
            kp_data = det[6:]
            num_kpts = len(kp_data) // 3
            for i in range(min(num_kpts, 24)):  # Up to 24 keypoints
                kp_x = (kp_data[i*3] - pad_l) / scale
                kp_y = (kp_data[i*3 + 1] - pad_t) / scale
                kp_conf = kp_data[i*3 + 2]
                kpts.extend([kp_x, kp_y, kp_conf])

        # Pad to 24 keypoints if needed
        while len(kpts) < 72:
            kpts.extend([0, 0, 0])

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        keypoints.append(kpts)

    return boxes, scores, keypoints

# ------------------------
# Hailo Inference
# ------------------------
def infer_hailo(hef_path, img):
    """Run inference using HailoRT"""

    if not HAILO_AVAILABLE:
        return [], [], []

    try:
        # Preprocess image
        orig_h, orig_w = img.shape[:2]

        # Letterbox to square 1024x1024
        scale = min(IMGSZ_W / orig_w, IMGSZ_H / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # Resize image
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        pad_w = IMGSZ_W - new_w
        pad_h = IMGSZ_H - new_h
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l

        padded = cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Convert BGR to RGB and keep as uint8 (model expects uint8)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.uint8)

        # Add batch dimension and convert to CHW
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC -> CHW
        input_data = np.expand_dims(input_data, 0)  # Add batch

        # Run inference
        with hpf.VDevice() as target:
            hef = hpf.HEF(hef_path)
            configure_params = hpf.ConfigureParams.create_from_hef(
                hef, interface=hpf.HailoStreamInterface.PCIe)

            network_group = target.configure(hef, configure_params)[0]
            network_group_params = network_group.create_params()

            input_vstreams_params = hpf.InputVStreamParams.make(network_group)
            output_vstreams_params = hpf.OutputVStreamParams.make(network_group)

            with network_group.activate(network_group_params):
                with hpf.InferVStreams(network_group, input_vstreams_params,
                                      output_vstreams_params) as infer_pipeline:

                    # Prepare input dict - get input names from network group
                    input_vstream_infos = network_group.get_input_vstream_infos()
                    input_dict = {
                        info.name: input_data
                        for info in input_vstream_infos
                    }

                    outputs = infer_pipeline.infer(input_dict)
                    output_data = {
                        name: data.copy()
                        for name, data in outputs.items()
                    }

        # Process outputs
        output_arrays = list(output_data.values())

        # Check output format
        if len(output_arrays) == 1 and output_arrays[0].ndim == 2:
            # NMS output format
            print(f"[DEBUG] NMS output shape: {output_arrays[0].shape}")
            boxes, scores, keypoints = parse_nms_output(
                output_arrays[0], orig_h, orig_w, pad_t, pad_l, scale)
        else:
            # Multi-scale outputs
            print(f"[DEBUG] Multi-scale outputs: {[out.shape for out in output_arrays]}")
            boxes, scores, keypoints = parse_multi_scale_outputs(
                output_arrays, orig_h, orig_w, pad_t, pad_l, scale, conf_thresh=PROB_TH)

        return boxes, scores, keypoints

    except Exception as e:
        print(f"[ERROR] Hailo inference failed: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

# ------------------------
# Behavior Head
# ------------------------
def load_behavior_head():
    """Load the TorchScript behavior classification head"""
    if not Path(HEAD_TS).exists():
        print(f"[WARNING] Behavior head not found: {HEAD_TS}")
        return None

    try:
        model = torch.jit.load(HEAD_TS)
        model.eval()
        print(f"[INFO] Loaded behavior head from {HEAD_TS}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load behavior head: {e}")
        return None

def predict_behavior(model, keypoints_seq):
    """Predict behavior from keypoint sequence"""
    if model is None or len(keypoints_seq) < T:
        return None, 0.0

    try:
        # Take last T frames
        seq = keypoints_seq[-T:]

        # Convert to tensor [1, T, 48] (24 keypoints * 2 coords)
        kpts_tensor = []
        for kpts in seq:
            # Take only x,y coords (ignore confidence)
            coords = []
            for i in range(24):
                coords.extend([kpts[i*3], kpts[i*3+1]])
            kpts_tensor.append(coords)

        input_tensor = torch.tensor(kpts_tensor).unsqueeze(0).float()  # [1, T, 48]

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze().numpy()

        # Get top prediction
        best_idx = np.argmax(probs)
        best_prob = probs[best_idx]

        if best_prob >= PROB_TH:
            return BEHAVIORS[best_idx], best_prob

        return None, best_prob

    except Exception as e:
        print(f"[ERROR] Behavior prediction failed: {e}")
        return None, 0.0

# ------------------------
# Camera Interface
# ------------------------
def get_camera():
    """Initialize camera (Picamera2 or OpenCV)"""
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        print("[INFO] Using Picamera2")
        return cam, "picamera2"
    except:
        print("[INFO] Falling back to OpenCV camera")
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        return cam, "opencv"

def capture_frame(cam, cam_type):
    """Capture a frame from the camera"""
    if cam_type == "picamera2":
        frame = cam.capture_array()
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()
        if not ret:
            return None

    # Apply rotation if configured
    if CAM_ROT_DEG == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif CAM_ROT_DEG == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif CAM_ROT_DEG == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return frame

# ------------------------
# ArUco Detection
# ------------------------
def detect_aruco(img):
    """Detect ArUco markers"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    corners, ids, _ = detector.detectMarkers(gray)

    markers = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            corners_2d = corners[i][0]
            center = np.mean(corners_2d, axis=0)
            markers[int(marker_id)] = {
                'corners': corners_2d,
                'center': center
            }

    return markers

# ------------------------
# Visualization
# ------------------------
def draw_detections(img, boxes, scores, keypoints, dog_ids, behaviors, trackers):
    """Draw bounding boxes, keypoints, and behavior info"""

    # Define keypoint connections for dog skeleton
    SKELETON = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Head to tail
        (5, 6), (6, 7), (7, 8),  # Front left leg
        (9, 10), (10, 11), (11, 12),  # Front right leg
        (13, 14), (14, 15), (15, 16),  # Back left leg
        (17, 18), (18, 19), (19, 20),  # Back right leg
    ]

    for i, (box, score, kpts, dog_id) in enumerate(zip(boxes, scores, keypoints, dog_ids)):
        x1, y1, x2, y2 = box

        # Choose color based on dog
        if dog_id == "bezik":
            color = (255, 0, 0)  # Blue for Bezik
        elif dog_id == "elsa":
            color = (0, 255, 0)  # Green for Elsa
        else:
            color = (128, 128, 128)  # Gray for unknown

        # Draw box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # Draw label
        behavior_info = behaviors.get(i, ("", 0.0))
        behavior, behavior_conf = behavior_info

        label = f"{dog_id}: {score:.2f}"
        if behavior:
            label += f" | {behavior} ({behavior_conf:.2f})"

        # Add cooldown info
        if dog_id in trackers:
            current_time = datetime.now()
            cooldowns = trackers[dog_id].get_active_cooldowns(current_time)
            if cooldowns:
                cooldown_str = ", ".join([f"{b}:{t:.1f}s" for b, t in cooldowns])
                label += f" | CD: {cooldown_str}"

        cv2.putText(img, label, (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw keypoints and skeleton
        kp_threshold = 0.3

        # Draw keypoints
        for j in range(24):
            x = kpts[j*3]
            y = kpts[j*3 + 1]
            conf = kpts[j*3 + 2]

            if conf > kp_threshold:
                cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)

        # Draw skeleton
        for connection in SKELETON:
            kp1, kp2 = connection
            if kp1 < 24 and kp2 < 24:
                x1, y1, c1 = kpts[kp1*3:kp1*3+3]
                x2, y2, c2 = kpts[kp2*3:kp2*3+3]

                if c1 > kp_threshold and c2 > kp_threshold:
                    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                           (0, 255, 0), 2)

    return img

# ------------------------
# Main Pipeline
# ------------------------
def main():
    print(f"\n{'='*60}")
    print("🐕 Dog Pose Detection & Behavior Analysis")
    print(f"{'='*60}")

    # Initialize components
    print("\n[1] Initializing components...")

    # Load behavior head
    behavior_model = load_behavior_head()

    # Initialize camera
    cam, cam_type = get_camera()
    time.sleep(2)  # Let camera warm up

    # Initialize trackers
    dog_trackers = {}
    keypoint_sequences = {}

    # Detection stats
    frame_count = 0
    detection_count = 0
    last_fps_time = time.time()
    fps = 0
    start_time = time.time()
    max_runtime_seconds = 60 if not CV2_GUI_AVAILABLE else float('inf')  # 60 seconds in headless mode

    # Create output directory
    output_dir = Path("detection_results")
    output_dir.mkdir(exist_ok=True)

    print("\n[2] Starting detection loop...")
    if CV2_GUI_AVAILABLE:
        print("Press 'q' to quit, 's' to save current frame")
    else:
        print("Running in headless mode - press Ctrl+C to quit")
        print(f"Will run for maximum {max_runtime_seconds} seconds")
        print("Frames with detections will be saved automatically")
    print(f"{'='*60}\n")

    try:
        while True:
            # Check runtime limit in headless mode
            if not CV2_GUI_AVAILABLE and (time.time() - start_time) > max_runtime_seconds:
                print(f"\n[INFO] Reached {max_runtime_seconds} second runtime limit")
                break
            # Capture frame
            frame = capture_frame(cam, cam_type)
            if frame is None:
                continue

            frame_count += 1

            # Run detection
            t_start = time.time()
            boxes, scores, keypoints = infer_hailo(HEF_PATH, frame)
            t_infer = time.time() - t_start

            # Detect ArUco markers
            markers = detect_aruco(frame)

            # Assign dog IDs
            dog_ids = []
            for box in boxes:
                dog_id = "unknown"
                box_center = [(box[0] + box[2])/2, (box[1] + box[3])/2]

                # Check proximity to markers
                min_dist = float('inf')
                closest_marker = None

                for marker_id, marker_info in markers.items():
                    dist = np.linalg.norm(np.array(box_center) - marker_info['center'])
                    if dist < min_dist and dist < 200:  # Within 200 pixels
                        min_dist = dist
                        closest_marker = marker_id

                if closest_marker and closest_marker in MARKER_TO_DOG:
                    dog_id = MARKER_TO_DOG[closest_marker]
                elif ASSUME_OTHER and len(boxes) == 2 and len(markers) == 1:
                    # If 2 dogs but only 1 marker, assume the other
                    marked_dog = MARKER_TO_DOG.get(list(markers.keys())[0], None)
                    if marked_dog == "bezik":
                        dog_id = "elsa"
                    elif marked_dog == "elsa":
                        dog_id = "bezik"

                dog_ids.append(dog_id)

            # Track behaviors
            behaviors = {}
            current_time = datetime.now()

            for i, (dog_id, kpts) in enumerate(zip(dog_ids, keypoints)):
                if dog_id != "unknown":
                    # Initialize tracker if needed
                    if dog_id not in dog_trackers:
                        dog_trackers[dog_id] = BehaviorTracker(dog_id)
                        keypoint_sequences[dog_id] = collections.deque(maxlen=T)

                    # Add keypoints to sequence
                    keypoint_sequences[dog_id].append(kpts)

                    # Predict behavior
                    if len(keypoint_sequences[dog_id]) >= T:
                        behavior, conf = predict_behavior(behavior_model,
                                                         keypoint_sequences[dog_id])

                        if behavior:
                            behaviors[i] = (behavior, conf)

                            # Try to trigger behavior
                            if dog_trackers[dog_id].record_behavior(behavior, conf, current_time):
                                print(f"[BEHAVIOR] {dog_id}: {behavior} (conf: {conf:.2f})")

                                # Here you could trigger treat dispensing
                                if behavior in ["sit", "lie", "cross"]:
                                    print(f"  → 🍖 Dispensing treat for {dog_id}!")

            # Update stats
            if len(boxes) > 0:
                detection_count += 1

            # Calculate FPS
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - last_fps_time)
                last_fps_time = time.time()

            # Draw visualizations
            vis_frame = frame.copy()
            vis_frame = draw_detections(vis_frame, boxes, scores, keypoints,
                                       dog_ids, behaviors, dog_trackers)

            # Draw ArUco markers
            for marker_id, marker_info in markers.items():
                cv2.aruco.drawDetectedMarkers(vis_frame,
                                             [marker_info['corners'].reshape(1, 4, 2)],
                                             np.array([[marker_id]]))

            # Add stats overlay
            stats_text = [
                f"FPS: {fps:.1f}",
                f"Inference: {t_infer*1000:.1f}ms",
                f"Detections: {len(boxes)}",
                f"Frame: {frame_count}"
            ]

            for i, text in enumerate(stats_text):
                cv2.putText(vis_frame, text, (10, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Show frame if GUI available
            if CV2_GUI_AVAILABLE:
                cv2.imshow("Dog Pose Detection", vis_frame)
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    save_frame = True
                else:
                    save_frame = False
            else:
                # In headless mode, save frames with detections automatically
                save_frame = len(boxes) > 0
                # Small delay to not overwhelm the system
                time.sleep(0.1)

            if save_frame:
                # Save frame
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = output_dir / f"detection_{timestamp}.jpg"
                cv2.imwrite(str(filename), vis_frame)
                print(f"[SAVED] {filename}")

                # Save detection data
                data_file = output_dir / f"detection_{timestamp}.json"
                detection_data = {
                    "timestamp": timestamp,
                    "boxes": boxes,
                    "scores": [float(s) for s in scores],
                    "dog_ids": dog_ids,
                    "behaviors": behaviors,
                    "markers": {str(k): {"center": v["center"].tolist()}
                               for k, v in markers.items()}
                }
                with open(data_file, 'w') as f:
                    json.dump(detection_data, f, indent=2)
                print(f"[SAVED] {data_file}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        if CV2_GUI_AVAILABLE:
            cv2.destroyAllWindows()
        if cam_type == "picamera2":
            cam.stop()
        else:
            cam.release()

        # Print final stats
        print(f"\n{'='*60}")
        print("📊 Session Statistics:")
        print(f"  Total frames: {frame_count}")
        print(f"  Frames with detections: {detection_count}")
        print(f"  Detection rate: {detection_count/max(frame_count,1)*100:.1f}%")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()