#!/usr/bin/env python3
"""
Pose Detection with New 1024x1024 Model - FIXED VERSION
Handles the actual 9-output structure of the retrained model
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
# NEW Output Parser for 9-output model
# ------------------------
def parse_model_outputs(outputs, orig_h, orig_w, pad_t, pad_l, scale):
    """Parse the 9-output structure from the retrained model"""

    if len(outputs) != 9:
        print(f"[ERROR] Expected 9 outputs, got {len(outputs)}")
        return [], [], []

    print(f"[DEBUG] Output shapes: {[out.shape for out in outputs]}")

    all_boxes = []
    all_scores = []
    all_keypoints = []

    # Group outputs by scale based on spatial dimensions
    scales = [(128, 128), (64, 64), (32, 32)]
    strides = [8, 16, 32]

    for scale_idx, ((h, w), stride) in enumerate(zip(scales, strides)):
        # Find outputs for this spatial resolution
        scale_outputs = []
        for out in outputs:
            if out.shape[1] == h and out.shape[2] == w:
                scale_outputs.append(out)

        if len(scale_outputs) != 3:
            print(f"[WARNING] Expected 3 outputs for scale {h}x{w}, got {len(scale_outputs)}")
            continue

        # Sort by channel count: 1 (objectness), 64 (boxes), 72 (keypoints)
        scale_outputs.sort(key=lambda x: x.shape[3])
        obj_out, box_out, kpt_out = scale_outputs

        print(f"[DEBUG] Scale {h}x{w}: obj={obj_out.shape}, box={box_out.shape}, kpt={kpt_out.shape}")

        # Parse detections for this scale
        for y in range(h):
            for x in range(w):
                # Get objectness score - apply sigmoid since it's raw output
                obj_score = 1.0 / (1.0 + np.exp(-obj_out[0, y, x, 0]))  # sigmoid

                # Much higher threshold to prevent false positives
                if obj_score < 0.8:  # Raised from 0.6 to 0.8
                    continue

                # Get box coordinates (assuming first 4 channels are x,y,w,h)
                box_data = box_out[0, y, x, :4]

                # Apply sigmoid to box coordinates too
                box_x = 1.0 / (1.0 + np.exp(-box_data[0]))
                box_y = 1.0 / (1.0 + np.exp(-box_data[1]))
                box_w = np.exp(box_data[2])  # exp for width/height
                box_h = np.exp(box_data[3])

                # Decode box (YOLOv11 format)
                cx = (box_x * 2 - 0.5 + x) * stride
                cy = (box_y * 2 - 0.5 + y) * stride
                w_box = (box_w * 2) ** 2 * stride
                h_box = (box_h * 2) ** 2 * stride

                # Convert to image coordinates
                x1 = max(0, (cx - w_box/2 - pad_l) / scale)
                y1 = max(0, (cy - h_box/2 - pad_t) / scale)
                x2 = min(orig_w, (cx + w_box/2 - pad_l) / scale)
                y2 = min(orig_h, (cy + h_box/2 - pad_t) / scale)

                # Skip tiny or huge boxes - be much more restrictive
                box_w_img = x2 - x1
                box_h_img = y2 - y1
                box_area = box_w_img * box_h_img
                img_area = orig_w * orig_h

                # Box must be reasonable size (1% to 60% of image)
                if (box_w_img < 50 or box_h_img < 50 or  # Too small
                    box_area < img_area * 0.01 or        # Less than 1% of image
                    box_area > img_area * 0.6 or         # More than 60% of image
                    box_w_img > orig_w * 0.8 or          # Too wide
                    box_h_img > orig_h * 0.8):           # Too tall
                    continue

                # Get keypoints (72 channels = 24 keypoints * 3 values)
                kpt_data = kpt_out[0, y, x, :]
                keypoints = []

                for kp_idx in range(24):
                    if kp_idx * 3 + 2 < len(kpt_data):
                        kp_x = kpt_data[kp_idx * 3] * stride
                        kp_y = kpt_data[kp_idx * 3 + 1] * stride
                        kp_conf = kpt_data[kp_idx * 3 + 2]

                        # Convert to image coordinates
                        kp_x = max(0, (kp_x - pad_l) / scale)
                        kp_y = max(0, (kp_y - pad_t) / scale)

                        keypoints.extend([kp_x, kp_y, kp_conf])
                    else:
                        keypoints.extend([0, 0, 0])

                # Pad to 24 keypoints
                while len(keypoints) < 72:
                    keypoints.extend([0, 0, 0])

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(obj_score)
                all_keypoints.append(keypoints)

    print(f"[DEBUG] Found {len(all_boxes)} detections")
    return all_boxes, all_scores, all_keypoints

# ------------------------
# Hailo Inference
# ------------------------
def infer_hailo(hef_path, img):
    """Run inference using HailoRT"""

    if not HAILO_AVAILABLE:
        return [], [], []

    try:
        orig_h, orig_w = img.shape[:2]

        # Letterbox to square 1024x1024
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

        # Convert BGR to RGB and keep as uint8 (model expects uint8)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.uint8)

        # Convert to CHW and add batch
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, 0)

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

                    # Get input names from network group
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

        # Process outputs using new parser
        output_arrays = list(output_data.values())
        return parse_model_outputs(output_arrays, orig_h, orig_w, pad_t, pad_l, scale)

    except Exception as e:
        print(f"[ERROR] Hailo inference failed: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

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
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()
        if not ret:
            return None

    # Apply rotation if configured (90° in config means 270° actual rotation needed)
    if CAM_ROT_DEG == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Fixed rotation
    elif CAM_ROT_DEG == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif CAM_ROT_DEG == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    return frame

# ------------------------
# Visualization
# ------------------------
def draw_detections(img, boxes, scores, keypoints):
    """Draw bounding boxes and keypoints"""

    for box, score, kpts in zip(boxes, scores, keypoints):
        x1, y1, x2, y2 = [int(v) for v in box]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw score
        label = f"Dog: {score:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw keypoints
        num_visible = 0
        for i in range(24):
            x = kpts[i*3]
            y = kpts[i*3 + 1]
            conf = kpts[i*3 + 2]

            if conf > 0.3:
                num_visible += 1
                color = (0, int(255 * conf), int(255 * (1 - conf)))
                cv2.circle(img, (int(x), int(y)), 3, color, -1)

        # Add keypoint count
        kp_label = f"KPs: {num_visible}/24"
        cv2.putText(img, kp_label, (x1, y2 + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    return img

# ------------------------
# Main Pipeline
# ------------------------
def main():
    print(f"\n{'='*60}")
    print("🐕 Dog Pose Detection (Fixed for 9-output model)")
    print(f"{'='*60}")

    # Initialize camera
    cam, cam_type = get_camera()
    time.sleep(2)

    # Stats
    frame_count = 0
    detection_count = 0
    start_time = time.time()
    max_runtime = 60 if not CV2_GUI_AVAILABLE else float('inf')

    # Create output directory
    output_dir = Path("detection_results_fixed")
    output_dir.mkdir(exist_ok=True)

    print("\n[INFO] Starting detection...")
    if CV2_GUI_AVAILABLE:
        print("Press 'q' to quit, 's' to save frame")
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
            boxes, scores, keypoints = infer_hailo(HEF_PATH, frame)
            t_infer = time.time() - t_start

            # Update stats
            if len(boxes) > 0:
                detection_count += 1

                # Draw detections
                vis_frame = frame.copy()
                vis_frame = draw_detections(vis_frame, boxes, scores, keypoints)

                # Add stats
                stats = [
                    f"Frame: {frame_count}",
                    f"Inference: {t_infer*1000:.1f}ms",
                    f"Detections: {len(boxes)}"
                ]
                for i, text in enumerate(stats):
                    cv2.putText(vis_frame, text, (10, 30 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Save or show frame
                if CV2_GUI_AVAILABLE:
                    cv2.imshow("Dog Pose Detection", vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = output_dir / f"detection_{timestamp}.jpg"
                        cv2.imwrite(str(filename), vis_frame)
                        print(f"[SAVED] {filename}")
                else:
                    # Auto-save in headless mode
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = output_dir / f"detection_{timestamp}.jpg"
                    cv2.imwrite(str(filename), vis_frame)
                    print(f"[{frame_count:04d}] Detected {len(boxes)} dog(s) - "
                          f"Inference: {t_infer*1000:.1f}ms - Saved: {filename.name}")
            else:
                if not CV2_GUI_AVAILABLE:
                    print(f"[{frame_count:04d}] No detections - Inference: {t_infer*1000:.1f}ms")

            # Small delay if headless
            if not CV2_GUI_AVAILABLE:
                time.sleep(0.1)

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

        # Print stats
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("📊 Final Statistics:")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Total frames: {frame_count}")
        print(f"  Frames with detections: {detection_count}")
        print(f"  Detection rate: {detection_count/max(frame_count,1)*100:.1f}%")
        print(f"  Avg FPS: {frame_count/elapsed:.1f}")
        print(f"  Results saved to: {output_dir}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()