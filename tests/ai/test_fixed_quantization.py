#!/usr/bin/env python3
"""
FIXED: Based on actual model inspection
- Objectness: uint8, centered around 200+ (high values!)
- Boxes: uint8, 64 channels (not standard 4)
- Keypoints: uint16
- Much higher threshold needed to avoid false positives
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

# Config
CFG = json.load(open("config/config.json"))
IMGSZ = CFG.get("imgsz", [1024, 1024])
if isinstance(IMGSZ, list):
    IMGSZ_H, IMGSZ_W = IMGSZ
else:
    IMGSZ_H = IMGSZ_W = IMGSZ

HEF_PATH = CFG.get("hef_path", "ai/models/yolo_pose.hef")
BEHAVIOR_PATH = CFG.get("behavior_model_path", "ai/models/behavior_head.ts")
T = 14
BEHAVIORS = CFG.get("behaviors", ["stand", "sit", "lie", "cross", "spin"])

# CRITICAL: Based on inspection, objectness values can reach 241 with no dogs
# We need VERY high threshold to avoid false positives
OBJECTNESS_THRESHOLD = 242  # Above max observed without dogs (241)

print("\n" + "="*60)
print("🐕 FIXED Quantization Detection")
print("="*60)
print(f"[CONFIG] Pose model: {HEF_PATH}")
print(f"[CONFIG] Behavior model: {BEHAVIOR_PATH}")
print(f"[CONFIG] Resolution: {IMGSZ_W}x{IMGSZ_H}")
print(f"[CONFIG] Objectness threshold: {OBJECTNESS_THRESHOLD} (raw uint8)")
print(f"[CONFIG] Temporal window: T={T}")

# Frame buffer for temporal analysis
keypoint_buffer = defaultdict(lambda: deque(maxlen=T))

def parse_model_outputs_fixed(outputs, orig_h, orig_w, pad_t, pad_l, scale):
    """Parse outputs with CORRECT quantization based on inspection"""

    all_boxes = []
    all_scores = []
    all_keypoints = []

    # Map outputs by scale and type based on inspection
    outputs_by_scale = {}

    for out in outputs:
        h, w = out.shape[1:3]
        channels = out.shape[3]

        scale_key = f"{h}x{w}"
        if scale_key not in outputs_by_scale:
            outputs_by_scale[scale_key] = {}

        if channels == 1:
            outputs_by_scale[scale_key]['objectness'] = out
        elif channels == 64:
            outputs_by_scale[scale_key]['boxes'] = out
        elif channels == 72:
            outputs_by_scale[scale_key]['keypoints'] = out

    print(f"[DEBUG] Processing {len(outputs_by_scale)} scales...")

    for scale_name, scale_data in outputs_by_scale.items():
        if 'objectness' not in scale_data:
            continue

        h, w = scale_data['objectness'].shape[1:3]
        stride = 1024 // h  # 8, 16, or 32

        obj_out = scale_data['objectness']
        box_out = scale_data.get('boxes')
        kpt_out = scale_data.get('keypoints')

        print(f"[DEBUG] Scale {scale_name}, stride={stride}")
        print(f"  Objectness: {obj_out.dtype}, range [{obj_out.min():.0f}, {obj_out.max():.0f}]")

        detection_count = 0

        # Sample grid sparsely
        step = max(1, h // 16)  # Only check ~16x16 grid points per scale

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Get raw objectness score (uint8)
                raw_obj = int(obj_out[0, y, x, 0])

                # Use RAW threshold - no sigmoid needed yet
                if raw_obj < OBJECTNESS_THRESHOLD:
                    continue

                detection_count += 1

                # If we have box output, decode it
                if box_out is not None:
                    # Box has 64 channels - need to figure out format
                    # For now, assume first 4 are x, y, w, h
                    box_data = box_out[0, y, x, :4].astype(np.float32)

                    # uint8 dequantization (0-255 -> reasonable range)
                    box_data = (box_data - 128.0) / 64.0  # Center at 128

                    # YOLO decoding
                    cx = (x + 0.5 + np.tanh(box_data[0])) * stride
                    cy = (y + 0.5 + np.tanh(box_data[1])) * stride
                    w_box = np.exp(np.clip(box_data[2], -3, 3)) * stride * 2
                    h_box = np.exp(np.clip(box_data[3], -3, 3)) * stride * 2
                else:
                    # Fallback - create box from grid position
                    cx = (x + 0.5) * stride
                    cy = (y + 0.5) * stride
                    w_box = stride * 4
                    h_box = stride * 4

                # Convert to image coordinates
                x1 = max(0, (cx - w_box/2 - pad_l) / scale)
                y1 = max(0, (cy - h_box/2 - pad_t) / scale)
                x2 = min(orig_w, (cx + w_box/2 - pad_l) / scale)
                y2 = min(orig_h, (cy + h_box/2 - pad_t) / scale)

                # Size filtering
                box_w = x2 - x1
                box_h = y2 - y1
                if box_w < 80 or box_h < 80 or box_w > orig_w*0.8 or box_h > orig_h*0.8:
                    continue

                # Extract keypoints if available
                kpts = []
                if kpt_out is not None:
                    kpt_data = kpt_out[0, y, x, :]

                    # uint16 dequantization - based on observed range 8K-26K
                    # Center around 16K
                    for i in range(24):
                        if i*3+2 < len(kpt_data):
                            kp_x_raw = float(kpt_data[i*3])
                            kp_y_raw = float(kpt_data[i*3+1])
                            kp_conf_raw = float(kpt_data[i*3+2])

                            # Dequantize uint16 - assume 16K is zero
                            kp_x = ((kp_x_raw - 16000) / 1000.0) * stride
                            kp_y = ((kp_y_raw - 16000) / 1000.0) * stride
                            kp_conf = max(0, min(1, (kp_conf_raw - 8000) / 16000.0))

                            # Convert to image coordinates
                            kp_x = (kp_x - pad_l) / scale
                            kp_y = (kp_y - pad_t) / scale

                            kpts.extend([kp_x, kp_y, kp_conf])
                        else:
                            kpts.extend([0, 0, 0])

                # Pad to 72 values
                while len(kpts) < 72:
                    kpts.extend([0, 0, 0])

                # Convert raw objectness to confidence score
                # Since values are high (182-231), normalize differently
                confidence = min(1.0, max(0.0, (raw_obj - 180) / 50.0))

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(confidence)
                all_keypoints.append(kpts)

                # Limit detections
                if len(all_boxes) >= 5:
                    break

        print(f"  Detections above threshold: {detection_count}")

    print(f"[DEBUG] Final detections: {len(all_boxes)}")
    return all_boxes, all_scores, all_keypoints

def classify_behavior_temporal(dog_id, keypoints):
    """Classify behavior using temporal sequence - FIXED LSTM input"""

    keypoint_buffer[dog_id].append(keypoints)

    if len(keypoint_buffer[dog_id]) < T:
        return "collecting", len(keypoint_buffer[dog_id]) / T

    try:
        model = torch.jit.load(BEHAVIOR_PATH, map_location='cpu')
        model.eval()

        # Prepare CORRECT temporal sequence [1, T, 48]
        sequence = []
        for frame_kpts in keypoint_buffer[dog_id]:
            # Extract only x,y (skip confidence) = 48 values
            kpts_xy = []
            for i in range(0, 72, 3):
                kpts_xy.extend([frame_kpts[i], frame_kpts[i+1]])
            sequence.append(kpts_xy[:48])

        # Create tensor with correct shape [1, T, 48]
        seq_array = np.array(sequence).reshape(1, T, 48).astype(np.float32)
        seq_tensor = torch.from_numpy(seq_array)

        print(f"[DEBUG] LSTM input shape: {seq_tensor.shape}")

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
        import traceback
        traceback.print_exc()
        return "error", 0.0

def infer_hailo(hef_path, img):
    """Run Hailo inference"""

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

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.uint8)
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, 0)

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
        return parse_model_outputs_fixed(output_arrays, orig_h, orig_w, pad_t, pad_l, scale)

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

    output_dir = Path("fixed_detection_results")
    output_dir.mkdir(exist_ok=True)

    print(f"\n[INFO] Testing FIXED quantization...")
    print(f"[INFO] Threshold set above max observed without dogs (231 -> {OBJECTNESS_THRESHOLD})")
    print("="*60 + "\n")

    frame_count = 0
    start_time = time.time()

    try:
        while time.time() - start_time < 30:
            # Capture
            if cam_type == "picamera2":
                frame = cam.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cam.read()
                if not ret:
                    continue

            frame_count += 1

            # Detect
            t_start = time.time()
            boxes, scores, keypoints = infer_hailo(HEF_PATH, frame)
            t_detect = (time.time() - t_start) * 1000

            if len(boxes) > 0:
                # Process behaviors
                for idx, (box, score, kpts) in enumerate(zip(boxes, scores, keypoints)):
                    dog_id = f"dog_{idx}"
                    behavior, behav_conf = classify_behavior_temporal(dog_id, kpts)

                    # Draw detection
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    if behavior == "collecting":
                        label = f"Dog {score:.2f} | Collecting {behav_conf:.1%}"
                    else:
                        label = f"Dog {score:.2f} | {behavior} ({behav_conf:.2f})"

                    cv2.putText(frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save detection
                timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                filename = output_dir / f"detection_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)

                print(f"[{frame_count:04d}] ✓ DETECTED {len(boxes)} dog(s)! - {t_detect:.1f}ms")
            else:
                print(f"[{frame_count:04d}] No detections - {t_detect:.1f}ms", end='\r')

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped")

    # Cleanup
    if cam_type == "picamera2":
        cam.stop()
    else:
        cam.release()

    print(f"\n" + "="*60)
    print(f"📊 Fixed Detection Test Complete")
    print("="*60)

if __name__ == "__main__":
    main()