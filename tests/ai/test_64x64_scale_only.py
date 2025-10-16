#!/usr/bin/env python3
"""
Use ONLY 64x64 scale for detection (has good quantization)
Ignore 32x32 and 128x128 scales (still have bad ranges)
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

# Use 64x64 scale threshold (99th percentile was 168)
OBJECTNESS_THRESHOLD_RAW = 175  # Above 99th percentile for 64x64

print("\n" + "="*60)
print("🎯 Using ONLY 64x64 Scale (Good Quantization)")
print("="*60)
print(f"[CONFIG] Pose model: {HEF_PATH}")
print(f"[CONFIG] Behavior model: {BEHAVIOR_PATH}")
print(f"[CONFIG] Using ONLY 64x64 scale (stride 16)")
print(f"[CONFIG] 64x64 objectness threshold: {OBJECTNESS_THRESHOLD_RAW} (raw uint8)")
print(f"[CONFIG] Temporal window: T={T}")

# Frame buffer
keypoint_buffer = defaultdict(lambda: deque(maxlen=T))

def parse_64x64_only(outputs, orig_h, orig_w, pad_t, pad_l, scale):
    """Use ONLY the 64x64 scale outputs (good quantization)"""

    all_boxes = []
    all_scores = []
    all_keypoints = []

    # Find 64x64 outputs only
    obj_64x64 = None
    box_64x64 = None
    kpt_64x64 = None

    for out in outputs:
        if out.ndim == 4 and out.shape[1] == 64 and out.shape[2] == 64:
            channels = out.shape[3]
            if channels == 1:
                obj_64x64 = out
            elif channels == 64:
                box_64x64 = out
            elif channels == 72:
                kpt_64x64 = out

    if obj_64x64 is None:
        print("[ERROR] No 64x64 objectness output found")
        return [], [], []

    stride = 16  # 1024 / 64 = 16
    h, w = 64, 64

    print(f"[DEBUG] Using 64x64 scale only, stride={stride}")
    print(f"  Objectness range: [{obj_64x64.min():.0f}, {obj_64x64.max():.0f}]")

    detection_count = 0

    # Sample every 4 pixels to reduce computation
    for y in range(0, h, 4):
        for x in range(0, w, 4):
            raw_obj = int(obj_64x64[0, y, x, 0])

            # Use raw threshold for 64x64 scale
            if raw_obj < OBJECTNESS_THRESHOLD_RAW:
                continue

            detection_count += 1

            # Decode box if available
            if box_64x64 is not None:
                box_data = box_64x64[0, y, x, :4].astype(np.float32)

                # Standard uint8 dequantization for 64x64 (centered ~127)
                box_data = (box_data - 128.0) / 64.0

                # YOLO decoding
                cx = (x + 0.5 + np.tanh(box_data[0])) * stride
                cy = (y + 0.5 + np.tanh(box_data[1])) * stride
                w_box = np.exp(np.clip(box_data[2], -3, 3)) * stride * 2
                h_box = np.exp(np.clip(box_data[3], -3, 3)) * stride * 2
            else:
                # Fallback
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                w_box = stride * 4
                h_box = stride * 4

            # Convert to image coords
            x1 = max(0, (cx - w_box/2 - pad_l) / scale)
            y1 = max(0, (cy - h_box/2 - pad_t) / scale)
            x2 = min(orig_w, (cx + w_box/2 - pad_l) / scale)
            y2 = min(orig_h, (cy + h_box/2 - pad_t) / scale)

            # Size filter
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w < 60 or box_h < 60 or box_w > orig_w*0.8 or box_h > orig_h*0.8:
                continue

            # Extract keypoints from 64x64 scale
            kpts = []
            if kpt_64x64 is not None:
                kpt_data = kpt_64x64[0, y, x, :]

                # uint16 dequantization for keypoints (based on inspection)
                for i in range(24):
                    if i*3+2 < len(kpt_data):
                        kp_x_raw = float(kpt_data[i*3])
                        kp_y_raw = float(kpt_data[i*3+1])
                        kp_conf_raw = float(kpt_data[i*3+2])

                        # Dequantize uint16 keypoints (centered around 19751 for 64x64)
                        kp_x = ((kp_x_raw - 19000) / 2000.0) * stride
                        kp_y = ((kp_y_raw - 19000) / 2000.0) * stride
                        kp_conf = max(0, min(1, (kp_conf_raw - 12000) / 10000.0))

                        # Convert to image coordinates
                        kp_x = (kp_x - pad_l) / scale
                        kp_y = (kp_y - pad_t) / scale

                        kpts.extend([kp_x, kp_y, kp_conf])
                    else:
                        kpts.extend([0, 0, 0])

            while len(kpts) < 72:
                kpts.extend([0, 0, 0])

            # Convert raw objectness to confidence (for 64x64 scale)
            # Since it's centered around 127, use standard quantization
            dequant = (raw_obj - 128.0) / 32.0
            confidence = 1.0 / (1.0 + np.exp(-np.clip(dequant, -10, 10)))

            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(confidence)
            all_keypoints.append(kpts)

            # Limit to 5 detections
            if len(all_boxes) >= 5:
                break

    print(f"  Raw detections above {OBJECTNESS_THRESHOLD_RAW}: {detection_count}")
    print(f"  Final detections after filtering: {len(all_boxes)}")

    return all_boxes, all_scores, all_keypoints

def classify_behavior_temporal(dog_id, keypoints):
    """Temporal behavior classification"""

    keypoint_buffer[dog_id].append(keypoints)

    if len(keypoint_buffer[dog_id]) < T:
        return "collecting", len(keypoint_buffer[dog_id]) / T

    try:
        model = torch.jit.load(BEHAVIOR_PATH, map_location='cpu')
        model.eval()

        # Create sequence [1, T, 48]
        sequence = []
        for frame_kpts in keypoint_buffer[dog_id]:
            kpts_xy = []
            for i in range(0, 72, 3):
                kpts_xy.extend([frame_kpts[i], frame_kpts[i+1]])
            sequence.append(kpts_xy[:48])

        seq_array = np.array(sequence).reshape(1, T, 48).astype(np.float32)
        seq_tensor = torch.from_numpy(seq_array)

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
        print(f"[ERROR] Behavior classification: {e}")
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
        return parse_64x64_only(output_arrays, orig_h, orig_w, pad_t, pad_l, scale)

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

    output_dir = Path("64x64_test_results")
    output_dir.mkdir(exist_ok=True)

    print(f"\n[INFO] Testing 64x64 scale only...")
    print(f"[INFO] Should have 0 false positives and detect real dogs")
    print("="*60 + "\n")

    frame_count = 0
    start_time = time.time()

    try:
        while time.time() - start_time < 45:
            # Capture
            if cam_type == "picamera2":
                frame = cam.capture_array()
                frame = cv2.cvtColor(frame, frame, cv2.COLOR_RGB2BGR)
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

                    # Draw
                    x1, y1, x2, y2 = [int(v) for v in box]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                    if behavior == "collecting":
                        label = f"Dog {score:.3f} | Collecting {behav_conf:.1%}"
                    else:
                        label = f"Dog {score:.3f} | {behavior} ({behav_conf:.2f})"

                    cv2.putText(frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Save
                timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
                filename = output_dir / f"detection_{timestamp}.jpg"
                cv2.imwrite(str(filename), frame)

                print(f"[{frame_count:04d}] ✓ DETECTED {len(boxes)} dog(s)! - {t_detect:.1f}ms")
            else:
                print(f"[{frame_count:04d}] No detections - {t_detect:.1f}ms", end='\r')

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Stopped")

    # Cleanup
    if cam_type == "picamera2":
        cam.stop()
    else:
        cam.release()

    print(f"\n" + "="*60)
    print(f"📊 64x64 Scale Test Results")
    print("="*60)

if __name__ == "__main__":
    main()