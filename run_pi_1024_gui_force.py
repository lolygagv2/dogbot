#!/usr/bin/env python3
"""
Pose Detection with FORCED GUI MODE
Forces GUI display so you can see what's happening
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

# FORCE GUI MODE - skip detection
CV2_GUI_AVAILABLE = True
print("[INFO] FORCING GUI MODE - you should see a window")

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
PROB_TH = float(CFG.get("prob_th", 0.6))  # Lower threshold for testing
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
# Simplified Output Parser for testing
# ------------------------
def parse_model_outputs(outputs, orig_h, orig_w, pad_t, pad_l, scale):
    """Parse the 9-output structure from the retrained model"""

    if len(outputs) != 9:
        print(f"[ERROR] Expected 9 outputs, got {len(outputs)}")
        return [], [], []

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
            continue

        # Sort by channel count: 1 (objectness), 64 (boxes), 72 (keypoints)
        scale_outputs.sort(key=lambda x: x.shape[3])
        obj_out, box_out, kpt_out = scale_outputs

        # Only process a few points for speed - sample every 4 pixels
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                # Get objectness score - try with and without sigmoid
                raw_score = obj_out[0, y, x, 0]
                sig_score = 1.0 / (1.0 + np.exp(-np.clip(raw_score, -10, 10)))

                # Use whichever makes more sense (raw or sigmoid)
                obj_score = max(raw_score, sig_score)

                # Much lower threshold for testing
                if obj_score < PROB_TH:
                    continue

                # Get box coordinates
                box_data = box_out[0, y, x, :4]

                # Try simple decoding first
                cx = (box_data[0] + x) * stride
                cy = (box_data[1] + y) * stride
                w_box = box_data[2] * stride * 4  # Scale up
                h_box = box_data[3] * stride * 4

                # Convert to image coordinates
                x1 = max(0, (cx - w_box/2 - pad_l) / scale)
                y1 = max(0, (cy - h_box/2 - pad_t) / scale)
                x2 = min(orig_w, (cx + w_box/2 - pad_l) / scale)
                y2 = min(orig_h, (cy + h_box/2 - pad_t) / scale)

                # Basic size filter
                box_w_img = x2 - x1
                box_h_img = y2 - y1

                if (box_w_img < 30 or box_h_img < 30 or
                    box_w_img > orig_w * 0.9 or box_h_img > orig_h * 0.9):
                    continue

                # Simple keypoints (just copy some values)
                keypoints = []
                kpt_data = kpt_out[0, y, x, :]
                for i in range(24):
                    if i * 3 + 2 < len(kpt_data):
                        keypoints.extend([kpt_data[i*3], kpt_data[i*3+1], kpt_data[i*3+2]])
                    else:
                        keypoints.extend([0, 0, 0])

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(obj_score)
                all_keypoints.append(keypoints)

                # Limit detections for testing
                if len(all_boxes) >= 10:
                    return all_boxes, all_scores, all_keypoints

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

    # Apply rotation - FIXED
    if CAM_ROT_DEG == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
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
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw score
        label = f"Dog: {score:.3f}"
        cv2.putText(img, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw some keypoints for visualization
        for i in range(min(8, 24)):  # Just first 8 keypoints
            x = kpts[i*3]
            y = kpts[i*3 + 1]
            conf = kpts[i*3 + 2]

            if conf > 0.1:
                cv2.circle(img, (int(x), int(y)), 4, (0, 0, 255), -1)

    return img

# ------------------------
# Main Pipeline
# ------------------------
def main():
    print(f"\n{'='*60}")
    print("🐕 Dog Pose Detection (FORCED GUI MODE)")
    print(f"{'='*60}")

    # Initialize camera
    cam, cam_type = get_camera()
    time.sleep(2)

    # Stats
    frame_count = 0
    detection_count = 0

    # Create output directory
    output_dir = Path("detection_results_gui")
    output_dir.mkdir(exist_ok=True)

    print("\n[INFO] Starting detection with GUI...")
    print("Press 'q' to quit, 's' to save frame (any frame), 'd' for debug info")
    print("You should see a live video window!")
    print(f"{'='*60}\n")

    try:
        while True:
            # Capture frame
            frame = capture_frame(cam, cam_type)
            if frame is None:
                continue

            frame_count += 1

            # Run inference every 3rd frame to speed up display
            if frame_count % 3 == 0:
                t_start = time.time()
                boxes, scores, keypoints = infer_hailo(HEF_PATH, frame)
                t_infer = time.time() - t_start

                if len(boxes) > 0:
                    detection_count += 1
                    print(f"Frame {frame_count}: Found {len(boxes)} dogs!")
            else:
                boxes, scores, keypoints = [], [], []
                t_infer = 0

            # Draw detections
            vis_frame = frame.copy()
            if len(boxes) > 0:
                vis_frame = draw_detections(vis_frame, boxes, scores, keypoints)

            # Add status text
            status_text = f"Frame: {frame_count} | Detections: {len(boxes)} | FPS: {1/(t_infer+0.001):.1f}"
            cv2.putText(vis_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show frame - FORCE IT
            try:
                cv2.imshow("Dog Detection - LIVE VIEW", vis_frame)
            except Exception as e:
                print(f"[ERROR] Failed to show window: {e}")
                print("GUI really not available - install libgtk2.0-dev")
                break

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Save ANY frame when 's' is pressed
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = output_dir / f"frame_{timestamp}.jpg"
                cv2.imwrite(str(filename), vis_frame)
                print(f"[SAVED] {filename}")
            elif key == ord('d'):
                # Debug info
                print(f"DEBUG: Frame {frame_count}, {len(boxes)} detections")
                if len(scores) > 0:
                    print(f"  Scores: {[f'{s:.3f}' for s in scores[:3]]}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        cv2.destroyAllWindows()
        if cam_type == "picamera2":
            cam.stop()
        else:
            cam.release()

        print(f"\n📊 Total frames: {frame_count}, Detections: {detection_count}")

if __name__ == "__main__":
    main()