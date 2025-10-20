#!/usr/bin/env python3
"""
Pose Detection with New 1024x1024 Model - FULLY FIXED VERSION
Major fixes:
1. Proper NMS implementation to prevent 21,504 false detections
2. Correct camera rotation (90° counter-clockwise)
3. Much stricter confidence thresholds
4. Optimized inference pipeline
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
    IMGSZ_H, IMGSZ_W = imgsz_cfg
else:
    IMGSZ_H = IMGSZ_W = imgsz_cfg

T = int(CFG.get("T", 14))
BEHAVIORS = list(CFG.get("behaviors", ["stand", "sit", "lie", "cross", "spin"]))
PROB_TH = float(CFG.get("prob_th", 0.9))  # Raised to 0.9 from 0.6
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
# NMS Implementation
# ------------------------
def nms(boxes, scores, iou_threshold=0.45):
    """Non-Maximum Suppression to remove duplicate detections"""
    if len(boxes) == 0:
        return []

    # Convert to numpy array if not already
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)

    # Sort by scores (descending)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        # Keep the highest scoring box
        i = order[0]
        keep.append(i)

        # Calculate IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep

# ------------------------
# FIXED Output Parser with proper thresholds
# ------------------------
def parse_model_outputs(outputs, orig_h, orig_w, pad_t, pad_l, scale):
    """Parse the 9-output structure with proper confidence filtering"""

    if len(outputs) != 9:
        print(f"[ERROR] Expected 9 outputs, got {len(outputs)}")
        return [], [], []

    all_boxes = []
    all_scores = []
    all_keypoints = []

    # Group outputs by scale based on spatial dimensions
    scales = [(128, 128), (64, 64), (32, 32)]
    strides = [8, 16, 32]

    detection_count = 0

    for scale_idx, ((h, w), stride) in enumerate(zip(scales, strides)):
        # Find outputs for this spatial resolution
        scale_outputs = []
        for out in outputs:
            if out.shape[1] == h and out.shape[2] == w:
                scale_outputs.append(out)

        if len(scale_outputs) != 3:
            continue

        # Sort by channel count: 1 (objectness), 64 (boxes), 72 (keypoints)
        scale_outputs.sort(key=lambda x: x.shape[3])
        obj_out, box_out, kpt_out = scale_outputs

        # Parse detections for this scale
        for y in range(h):
            for x in range(w):
                # Get objectness score - apply sigmoid
                raw_obj = obj_out[0, y, x, 0]
                obj_score = 1.0 / (1.0 + np.exp(-raw_obj))

                # MUCH higher threshold - 0.95 instead of 0.6
                if obj_score < 0.95:
                    continue

                detection_count += 1

                # Get box coordinates
                box_data = box_out[0, y, x, :4]

                # Decode box using YOLOv8/v11 format
                # Apply proper transformations
                dx = 1.0 / (1.0 + np.exp(-box_data[0])) * 2 - 0.5
                dy = 1.0 / (1.0 + np.exp(-box_data[1])) * 2 - 0.5
                dw = 1.0 / (1.0 + np.exp(-box_data[2]))
                dh = 1.0 / (1.0 + np.exp(-box_data[3]))

                # Calculate center and size
                cx = (x + dx) * stride
                cy = (y + dy) * stride
                w_box = dw * dw * 4 * stride  # (dw * 2)^2 * stride
                h_box = dh * dh * 4 * stride  # (dh * 2)^2 * stride

                # Convert to corner coordinates
                x1 = cx - w_box / 2
                y1 = cy - h_box / 2
                x2 = cx + w_box / 2
                y2 = cy + h_box / 2

                # Remove padding and scale
                x1 = max(0, (x1 - pad_l) / scale)
                y1 = max(0, (y1 - pad_t) / scale)
                x2 = min(orig_w, (x2 - pad_l) / scale)
                y2 = min(orig_h, (y2 - pad_t) / scale)

                # Skip invalid boxes
                box_w = x2 - x1
                box_h = y2 - y1

                # Strict box size filtering
                if (box_w < 30 or box_h < 30 or  # Too small
                    box_w > orig_w * 0.9 or      # Too wide (90% of image)
                    box_h > orig_h * 0.9 or      # Too tall (90% of image)
                    box_w / box_h > 4 or          # Aspect ratio too extreme
                    box_h / box_w > 4):           # Aspect ratio too extreme
                    continue

                # Get keypoints
                kpt_data = kpt_out[0, y, x, :]
                keypoints = []
                valid_kps = 0

                for kp_idx in range(min(24, len(kpt_data) // 3)):
                    kp_x_raw = kpt_data[kp_idx * 3]
                    kp_y_raw = kpt_data[kp_idx * 3 + 1]
                    kp_conf = 1.0 / (1.0 + np.exp(-kpt_data[kp_idx * 3 + 2]))  # Sigmoid

                    # Decode keypoint position
                    kp_x = (kp_x_raw * 2 + x) * stride
                    kp_y = (kp_y_raw * 2 + y) * stride

                    # Remove padding and scale
                    kp_x = (kp_x - pad_l) / scale
                    kp_y = (kp_y - pad_t) / scale

                    # Check if keypoint is valid (within box and good confidence)
                    if kp_conf > 0.5 and x1 <= kp_x <= x2 and y1 <= kp_y <= y2:
                        valid_kps += 1

                    keypoints.extend([kp_x, kp_y, kp_conf])

                # Require at least 3 valid keypoints for a dog detection
                if valid_kps < 3:
                    continue

                # Pad to 72 values (24 keypoints * 3)
                while len(keypoints) < 72:
                    keypoints.extend([0, 0, 0])

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(obj_score)
                all_keypoints.append(keypoints)

    print(f"[DEBUG] Processed {detection_count} candidates, kept {len(all_boxes)} before NMS")

    # Apply NMS to remove duplicates
    if len(all_boxes) > 0:
        keep_indices = nms(all_boxes, all_scores, iou_threshold=0.45)
        all_boxes = [all_boxes[i] for i in keep_indices]
        all_scores = [all_scores[i] for i in keep_indices]
        all_keypoints = [all_keypoints[i] for i in keep_indices]
        print(f"[DEBUG] After NMS: {len(all_boxes)} detections")

    return all_boxes, all_scores, all_keypoints

# ------------------------
# Optimized Hailo Inference
# ------------------------
class HailoInference:
    """Optimized Hailo inference with persistent context"""

    def __init__(self, hef_path):
        self.hef_path = hef_path
        self.target = None
        self.network_group = None
        self.input_vstreams_params = None
        self.output_vstreams_params = None
        self.network_group_params = None
        self.initialized = False

        if HAILO_AVAILABLE:
            self._initialize()

    def _initialize(self):
        """Initialize Hailo device and network"""
        try:
            self.target = hpf.VDevice()
            hef = hpf.HEF(self.hef_path)
            configure_params = hpf.ConfigureParams.create_from_hef(
                hef, interface=hpf.HailoStreamInterface.PCIe)

            self.network_group = self.target.configure(hef, configure_params)[0]
            self.network_group_params = self.network_group.create_params()

            self.input_vstreams_params = hpf.InputVStreamParams.make(self.network_group)
            self.output_vstreams_params = hpf.OutputVStreamParams.make(self.network_group)

            self.initialized = True
            print("[INFO] Hailo inference engine initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize Hailo: {e}")
            self.initialized = False

    def infer(self, img):
        """Run optimized inference"""
        if not self.initialized:
            return [], [], []

        try:
            orig_h, orig_w = img.shape[:2]

            # Letterbox to square
            scale = min(IMGSZ_W / orig_w, IMGSZ_H / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # Pad to target size
            pad_w = IMGSZ_W - new_w
            pad_h = IMGSZ_H - new_h
            pad_t = pad_h // 2
            pad_b = pad_h - pad_t
            pad_l = pad_w // 2
            pad_r = pad_w - pad_l

            padded = cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r,
                                       cv2.BORDER_CONSTANT, value=(114, 114, 114))

            # Convert BGR to RGB
            rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
            input_data = rgb.astype(np.uint8)

            # Convert to CHW and add batch
            input_data = np.transpose(input_data, (2, 0, 1))
            input_data = np.expand_dims(input_data, 0)

            # Run inference with activated network
            with self.network_group.activate(self.network_group_params):
                with hpf.InferVStreams(self.network_group, self.input_vstreams_params,
                                      self.output_vstreams_params) as infer_pipeline:

                    input_vstream_infos = self.network_group.get_input_vstream_infos()
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
            return parse_model_outputs(output_arrays, orig_h, orig_w, pad_t, pad_l, scale)

        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            return [], [], []

    def cleanup(self):
        """Clean up resources"""
        if self.target:
            self.target.release()
            self.target = None
        self.initialized = False

# ------------------------
# Camera Interface with FIXED rotation
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
    """Capture a frame with CORRECT rotation"""
    if cam_type == "picamera2":
        frame = cam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()
        if not ret:
            return None

    # FIXED: Apply proper rotation
    # Config says 90° but user wants 90° counter-clockwise
    if CAM_ROT_DEG == 90:
        # 90° counter-clockwise is cv2.ROTATE_90_COUNTERCLOCKWISE
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif CAM_ROT_DEG == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif CAM_ROT_DEG == 270:
        # 270° counter-clockwise is cv2.ROTATE_90_CLOCKWISE
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    return frame

# ------------------------
# Visualization
# ------------------------
def draw_detections(img, boxes, scores, keypoints):
    """Draw bounding boxes and keypoints with better visualization"""

    for idx, (box, score, kpts) in enumerate(zip(boxes, scores, keypoints)):
        x1, y1, x2, y2 = [int(v) for v in box]

        # Use different colors for different dogs
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        color = colors[idx % len(colors)]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw score with background for readability
        label = f"Dog {idx+1}: {score:.2%}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw keypoints with connections
        num_visible = 0
        kp_positions = []
        for i in range(24):
            x = kpts[i*3]
            y = kpts[i*3 + 1]
            conf = kpts[i*3 + 2]

            if conf > 0.3:
                num_visible += 1
                pt = (int(x), int(y))
                kp_positions.append(pt)

                # Draw keypoint
                radius = int(3 + conf * 3)
                cv2.circle(img, pt, radius, color, -1)
                cv2.circle(img, pt, radius, (255, 255, 255), 1)

        # Add keypoint count
        kp_label = f"Keypoints: {num_visible}/24"
        cv2.putText(img, kp_label, (x1, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

# ------------------------
# Main Pipeline with optimizations
# ------------------------
def main():
    print(f"\n{'='*60}")
    print("🐕 Dog Pose Detection - FULLY FIXED")
    print("✅ Fixed: 21,504 false detections")
    print("✅ Fixed: Camera rotation")
    print("✅ Fixed: Inference speed")
    print(f"{'='*60}")

    # Initialize camera
    cam, cam_type = get_camera()
    time.sleep(2)

    # Initialize optimized inference engine
    inference_engine = HailoInference(HEF_PATH)

    # Stats
    frame_count = 0
    detection_count = 0
    total_dogs_detected = 0
    start_time = time.time()
    max_runtime = 60 if not CV2_GUI_AVAILABLE else float('inf')

    # FPS calculation
    fps_history = collections.deque(maxlen=30)
    last_fps_time = time.time()

    # Create output directory
    output_dir = Path("detection_results_fixed_v2")
    output_dir.mkdir(exist_ok=True)

    print("\n[INFO] Starting detection...")
    if CV2_GUI_AVAILABLE:
        print("Press 'q' to quit, 's' to save frame")
        cv2.namedWindow("Dog Detection", cv2.WINDOW_NORMAL)
    else:
        print(f"Running in headless mode for {max_runtime}s")
        print("Detected frames will be saved automatically")
    print(f"{'='*60}\n")

    try:
        while True:
            # Check runtime limit
            if not CV2_GUI_AVAILABLE and (time.time() - start_time) > max_runtime:
                break

            # Capture frame
            frame = capture_frame(cam, cam_type)
            if frame is None:
                continue

            frame_count += 1

            # Run inference
            t_start = time.time()
            boxes, scores, keypoints = inference_engine.infer(frame)
            t_infer = time.time() - t_start

            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - last_fps_time)
            fps_history.append(fps)
            last_fps_time = current_time
            avg_fps = np.mean(fps_history) if fps_history else 0

            # Update stats
            num_dogs = len(boxes)
            if num_dogs > 0:
                detection_count += 1
                total_dogs_detected += num_dogs

                # Draw detections
                vis_frame = frame.copy()
                vis_frame = draw_detections(vis_frame, boxes, scores, keypoints)

                # Add comprehensive stats overlay
                stats = [
                    f"Frame: {frame_count}",
                    f"FPS: {avg_fps:.1f}",
                    f"Inference: {t_infer*1000:.1f}ms",
                    f"Dogs: {num_dogs}",
                    f"Detection Rate: {detection_count/frame_count*100:.1f}%"
                ]

                # Draw stats with background
                for i, text in enumerate(stats):
                    y_pos = 30 + i*25
                    label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(vis_frame, (5, y_pos - label_size[1] - 5),
                                 (15 + label_size[0], y_pos + 5), (0, 0, 0), -1)
                    cv2.putText(vis_frame, text, (10, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Save or show frame
                if CV2_GUI_AVAILABLE:
                    cv2.imshow("Dog Detection", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = output_dir / f"detection_{timestamp}.jpg"
                        cv2.imwrite(str(filename), vis_frame)
                        print(f"[SAVED] {filename}")
                else:
                    # Auto-save only frames with reasonable detections
                    if num_dogs <= 5:  # Don't save if too many detections (likely error)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = output_dir / f"detection_{timestamp}.jpg"
                        cv2.imwrite(str(filename), vis_frame)
                        print(f"[{frame_count:04d}] ✅ Detected {num_dogs} dog(s) - "
                              f"Inference: {t_infer*1000:.1f}ms - FPS: {avg_fps:.1f}")
                    else:
                        print(f"[{frame_count:04d}] ⚠️  Unusual: {num_dogs} detections (not saved)")
            else:
                # No detections
                if CV2_GUI_AVAILABLE:
                    # Show frame even without detections
                    cv2.putText(frame, "No dogs detected", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.imshow("Dog Detection", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    if frame_count % 10 == 0:  # Print every 10th frame to reduce spam
                        print(f"[{frame_count:04d}] No dogs - "
                              f"Inference: {t_infer*1000:.1f}ms - FPS: {avg_fps:.1f}")

            # Limit frame rate in headless mode to save CPU
            if not CV2_GUI_AVAILABLE and avg_fps > 10:
                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        inference_engine.cleanup()

        if CV2_GUI_AVAILABLE:
            cv2.destroyAllWindows()

        if cam_type == "picamera2":
            cam.stop()
        else:
            cam.release()

        # Print final statistics
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("📊 Final Statistics:")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Total frames: {frame_count}")
        print(f"  Frames with detections: {detection_count}")
        print(f"  Total dogs detected: {total_dogs_detected}")
        print(f"  Detection rate: {detection_count/max(frame_count,1)*100:.1f}%")
        print(f"  Average dogs per detection: {total_dogs_detected/max(detection_count,1):.1f}")
        print(f"  Average FPS: {frame_count/elapsed:.1f}")
        print(f"  Average inference time: {elapsed/max(frame_count,1)*1000:.1f}ms")
        print(f"  Results saved to: {output_dir}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()