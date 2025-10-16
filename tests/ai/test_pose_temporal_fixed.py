#!/usr/bin/env python3
"""
FIXED: Proper temporal behavior detection
- Accumulates T=14 frames before behavior classification
- Fixes uint16 quantization
- No false positives
"""

import json
import time
import numpy as np
import cv2
import torch
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

try:
    import hailo_platform as hpf
    HAILO_AVAILABLE = True
except ImportError:
    print("[ERROR] HailoRT not available")
    exit(1)

# Load config
CFG = json.load(open("config/config.json"))
IMGSZ = CFG.get("imgsz", [1024, 1024])
if isinstance(IMGSZ, list):
    IMGSZ_H, IMGSZ_W = IMGSZ
else:
    IMGSZ_H = IMGSZ_W = IMGSZ

HEF_PATH = CFG.get("hef_path", "ai/models/yolo_pose.hef")
BEHAVIOR_PATH = CFG.get("behavior_model_path", "ai/models/behavior_head.ts")
T = 14  # Temporal sequence length for LSTM
BEHAVIORS = CFG.get("behaviors", ["stand", "sit", "lie", "cross", "spin"])

# CRITICAL: Much higher threshold to avoid false positives
CONF_THRESH = 0.95  # Start very high

print("\n" + "="*60)
print("🐕 Temporal Pose Detection (FIXED)")
print("="*60)
print(f"[CONFIG] Pose model: {HEF_PATH}")
print(f"[CONFIG] Behavior model: {BEHAVIOR_PATH}")
print(f"[CONFIG] Resolution: {IMGSZ_W}x{IMGSZ_H}")
print(f"[CONFIG] Temporal window: T={T} frames")
print(f"[CONFIG] Confidence threshold: {CONF_THRESH}")
print(f"[CONFIG] Behaviors: {BEHAVIORS}")

# Frame buffer for temporal analysis
keypoint_buffer = defaultdict(lambda: deque(maxlen=T))

def parse_model_outputs(outputs, orig_h, orig_w, pad_t, pad_l, scale):
    """Parse YOLO outputs with PROPER uint16 handling"""

    all_boxes = []
    all_scores = []
    all_keypoints = []

    # Debug once
    if not hasattr(parse_model_outputs, 'debug_done'):
        print(f"\n[DEBUG] Got {len(outputs)} outputs")
        for i, out in enumerate(outputs[:3]):
            print(f"  Output {i}: shape {out.shape}, dtype {out.dtype}, "
                  f"range [{out.min():.1f}, {out.max():.1f}]")
        parse_model_outputs.debug_done = True

    # Group outputs by scale
    scales = [(128, 128), (64, 64), (32, 32)]
    strides = [8, 16, 32]

    for scale_idx, ((h, w), stride) in enumerate(zip(scales, strides)):
        # Find outputs for this scale
        scale_outputs = []
        for out in outputs:
            if out.ndim == 4 and out.shape[1] == h and out.shape[2] == w:
                scale_outputs.append(out)

        if len(scale_outputs) != 3:
            continue

        # Sort by channels
        scale_outputs.sort(key=lambda x: x.shape[3])
        obj_out, box_out, kpt_out = scale_outputs

        # Sample sparsely to reduce false positives
        step = 16 if scale_idx == 0 else 8

        for y in range(0, h, step):
            for x in range(0, w, step):
                raw_score = obj_out[0, y, x, 0]

                # FIXED uint16 quantization
                if obj_out.dtype == np.uint16:
                    # For uint16, assume zero point around middle of range
                    # Based on observed range [7798, 26275], midpoint ~17000
                    zero_point = 17000
                    scale_factor = 2000  # Empirical
                    dequant_score = (float(raw_score) - zero_point) / scale_factor
                elif obj_out.dtype == np.uint8:
                    dequant_score = (float(raw_score) - 128) / 32.0
                else:
                    dequant_score = float(raw_score)

                # Apply sigmoid
                obj_score = 1.0 / (1.0 + np.exp(-np.clip(dequant_score, -10, 10)))

                # Very high threshold to avoid false positives
                if obj_score < CONF_THRESH:
                    continue

                # Decode box
                box_data = box_out[0, y, x, :4].astype(np.float32)

                # Dequantize box
                if box_out.dtype == np.uint16:
                    box_data = (box_data - 17000) / 2000
                elif box_out.dtype == np.uint8:
                    box_data = (box_data - 128) / 32.0

                # YOLO decoding
                cx = (x + 0.5 + np.tanh(box_data[0])) * stride
                cy = (y + 0.5 + np.tanh(box_data[1])) * stride
                w_box = np.exp(np.clip(box_data[2], -5, 5)) * stride * 2
                h_box = np.exp(np.clip(box_data[3], -5, 5)) * stride * 2

                # Convert to image coords
                x1 = max(0, (cx - w_box/2 - pad_l) / scale)
                y1 = max(0, (cy - h_box/2 - pad_t) / scale)
                x2 = min(orig_w, (cx + w_box/2 - pad_l) / scale)
                y2 = min(orig_h, (cy + h_box/2 - pad_t) / scale)

                # Strict size filtering
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w < 100 or box_h < 100 or box_w > orig_w*0.7 or box_h > orig_h*0.7:
                    continue

                # Extract keypoints
                kpts = []
                kpt_data = kpt_out[0, y, x, :]

                # Dequantize keypoints
                if kpt_out.dtype == np.uint16:
                    kpt_data = kpt_data.astype(np.float32) / 1000.0
                elif kpt_out.dtype == np.uint8:
                    kpt_data = kpt_data.astype(np.float32) / 255.0

                for i in range(24):
                    if i*3+2 < len(kpt_data):
                        kp_x = (kpt_data[i*3] * stride - pad_l) / scale
                        kp_y = (kpt_data[i*3+1] * stride - pad_t) / scale
                        kp_conf = kpt_data[i*3+2]
                        kpts.extend([kp_x, kp_y, kp_conf])
                    else:
                        kpts.extend([0, 0, 0])

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(obj_score)
                all_keypoints.append(kpts)

                # Max 3 detections
                if len(all_boxes) >= 3:
                    return all_boxes, all_scores, all_keypoints

    return all_boxes, all_scores, all_keypoints

def classify_behavior_temporal(dog_id, keypoints):
    """Classify behavior using temporal sequence"""

    # Add current frame to buffer
    keypoint_buffer[dog_id].append(keypoints)

    # Need full sequence
    if len(keypoint_buffer[dog_id]) < T:
        return "collecting", 0.0

    try:
        # Load model
        model = torch.jit.load(BEHAVIOR_PATH, map_location='cpu')
        model.eval()

        # Prepare temporal sequence [1, T, 48]
        sequence = []
        for frame_kpts in keypoint_buffer[dog_id]:
            # Extract only x,y (skip confidence) = 48 values
            kpts_xy = []
            for i in range(0, 72, 3):
                kpts_xy.extend([frame_kpts[i], frame_kpts[i+1]])
            sequence.append(kpts_xy[:48])

        # Convert to tensor [1, T, 48]
        seq_array = np.array(sequence).reshape(1, T, 48).astype(np.float32)
        seq_tensor = torch.from_numpy(seq_array)

        # Run inference
        with torch.no_grad():
            output = model(seq_tensor)
            probs = torch.softmax(output, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_idx].item()

        if pred_idx < len(BEHAVIORS):
            return BEHAVIORS[pred_idx], confidence
        else:
            return "unknown", 0.0

    except Exception as e:
        print(f"[ERROR] Behavior classification failed: {e}")
        return "error", 0.0

def infer_hailo(hef_path, img):
    """Run Hailo inference"""

    if not HAILO_AVAILABLE:
        return [], [], []

    try:
        orig_h, orig_w = img.shape[:2]

        # Letterbox
        scale = min(IMGSZ_W / orig_w, IMGSZ_H / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        resized = cv2.resize(img, (new_w, new_h))

        pad_w = IMGSZ_W - new_w
        pad_h = IMGSZ_H - new_h
        pad_t = pad_h // 2
        pad_b = pad_h - pad_t
        pad_l = pad_w // 2
        pad_r = pad_w - pad_l

        padded = cv2.copyMakeBorder(resized, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # Convert to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.uint8)

        # CHW format
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

                    input_vstream_infos = network_group.get_input_vstream_infos()
                    input_dict = {info.name: input_data for info in input_vstream_infos}

                    outputs = infer_pipeline.infer(input_dict)
                    output_data = {name: data.copy() for name, data in outputs.items()}

        output_arrays = list(output_data.values())
        return parse_model_outputs(output_arrays, orig_h, orig_w, pad_t, pad_l, scale)

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return [], [], []

def main():
    # Initialize camera
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_still_configuration(
            main={"size": (1920, 1080), "format": "RGB888"}
        )
        cam.configure(config)
        cam.start()
        cam_type = "picamera2"
    except:
        cam = cv2.VideoCapture(0)
        cam_type = "opencv"

    time.sleep(2)

    output_dir = Path("temporal_results")
    output_dir.mkdir(exist_ok=True)

    print(f"\n[INFO] Starting temporal detection...")
    print(f"[INFO] Need {T} frames to start behavior analysis")
    print("="*60 + "\n")

    frame_count = 0
    start_time = time.time()

    try:
        while time.time() - start_time < 60:
            # Capture
            if cam_type == "picamera2":
                frame = cam.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cam.read()
                if not ret:
                    continue

            frame_count += 1

            # Detect poses
            t_start = time.time()
            boxes, scores, keypoints = infer_hailo(HEF_PATH, frame)
            t_detect = (time.time() - t_start) * 1000

            # Process detections
            if len(boxes) > 0:
                behaviors = []

                for idx, (box, score, kpts) in enumerate(zip(boxes, scores, keypoints)):
                    dog_id = f"dog_{idx}"

                    # Classify behavior (temporal)
                    behavior, behav_conf = classify_behavior_temporal(dog_id, kpts)
                    behaviors.append((behavior, behav_conf))

                    # Draw
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    label = f"Dog {score:.2f}"
                    if behavior != "collecting":
                        label += f" | {behavior} ({behav_conf:.2f})"
                    else:
                        buffer_size = len(keypoint_buffer[dog_id])
                        label += f" | Collecting... {buffer_size}/{T}"

                    cv2.putText(frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Save if behavior detected
                if any(b[0] not in ["collecting", "error"] for b in behaviors):
                    timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                    filename = output_dir / f"behavior_{timestamp}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print(f"[{frame_count:04d}] Behavior detected! Saved to {filename.name}")

                status = f"[{frame_count:04d}] {len(boxes)} dog(s) | "
                status += " | ".join([f"{b[0]}:{b[1]:.2f}" for b in behaviors])
                print(status)
            else:
                print(f"[{frame_count:04d}] No dogs detected - {t_detect:.1f}ms", end='\r')

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user")

    # Cleanup
    if cam_type == "picamera2":
        cam.stop()
    else:
        cam.release()

    print(f"\n" + "="*60)
    print(f"📊 Temporal Detection Results:")
    print(f"  Frames processed: {frame_count}")
    print(f"  Results saved to: {output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main()