#!/usr/bin/env python3
"""
Fixed headless pose detection test with quantization handling
- Fixes camera rotation (90° counter-clockwise)
- Handles quantized uint8 outputs properly
- Adds debug output for better troubleshooting
"""

import os
import json
import time
import numpy as np
import cv2
from datetime import datetime
from pathlib import Path

# Hailo imports
try:
    import hailo_platform as hpf
    HAILO_AVAILABLE = True
except ImportError:
    print("[WARNING] HailoRT not available")
    HAILO_AVAILABLE = False

# Load config
CFG = json.load(open("config/config.json"))
IMGSZ = CFG.get("imgsz", [1024, 1024])
if isinstance(IMGSZ, list):
    IMGSZ_H, IMGSZ_W = IMGSZ
else:
    IMGSZ_H = IMGSZ_W = IMGSZ

HEF_PATH = CFG.get("hef_path", "ai/models/yolo_pose.hef")
BEHAVIOR_PATH = CFG.get("behavior_model_path", "ai/models/behavior_head.ts")
CAM_ROT_DEG = CFG.get("camera_rotation_deg", 90)
CONF_THRESH = CFG.get("detection_threshold", 0.6)
BEHAVIORS = CFG.get("behaviors", ["stand", "sit", "lie", "cross", "spin"])

print("[WARNING] OpenCV GUI not available - running in headless mode")
print("[WARNING] Results will be saved to disk but not displayed")
print(f"[CONFIG] Model resolution: {IMGSZ_W}x{IMGSZ_H} (WxH)")
print(f"[CONFIG] HEF model: {HEF_PATH}")
print(f"[CONFIG] Behavior head: {BEHAVIOR_PATH}")
print(f"[CONFIG] Camera rotation: {CAM_ROT_DEG}°")
print(f"[CONFIG] Detection threshold: {CONF_THRESH}")
print(f"[CONFIG] Behaviors: {BEHAVIORS}")

def parse_model_outputs(outputs, orig_h, orig_w, pad_t, pad_l, scale):
    """Parse quantized YOLO outputs with proper dequantization"""

    all_boxes = []
    all_scores = []
    all_keypoints = []

    # Sort outputs by size to identify scales
    outputs_by_size = {}
    for out in outputs:
        if out.ndim == 4:
            h, w = out.shape[1:3]
            key = (h, w)
            if key not in outputs_by_size:
                outputs_by_size[key] = []
            outputs_by_size[key].append(out)

    # Debug output shapes
    print(f"[DEBUG] Output shapes: {[out.shape for out in outputs]}")

    # Process each scale
    for (h, w), scale_outputs in outputs_by_size.items():
        stride = IMGSZ_H // h  # Calculate stride from output size

        # Find objectness, box, and keypoint outputs
        obj_out = None
        box_out = None
        kpt_out = None

        for out in scale_outputs:
            channels = out.shape[3]
            if channels == 1:
                obj_out = out
            elif channels == 64:
                box_out = out
            elif channels == 72:
                kpt_out = out

        if obj_out is None or box_out is None or kpt_out is None:
            continue

        print(f"[DEBUG] Scale {h}x{w}: obj={obj_out.shape}, box={box_out.shape}, kpt={kpt_out.shape}")

        # Check if outputs are quantized (uint8)
        is_quantized = obj_out.dtype == np.uint8
        if is_quantized:
            print(f"[DEBUG] Detected quantized outputs (uint8)")

        # Parse detections for this scale
        for y in range(h):
            for x in range(w):
                # Get objectness score with quantization handling
                raw_score = obj_out[0, y, x, 0]

                if is_quantized:
                    # Dequantize: uint8 is quantized with 128 as zero point
                    dequant_score = (float(raw_score) - 128.0) / 32.0
                    obj_score = 1.0 / (1.0 + np.exp(-np.clip(dequant_score, -10, 10)))
                else:
                    # Regular sigmoid for float outputs
                    obj_score = 1.0 / (1.0 + np.exp(-raw_score))

                if obj_score < CONF_THRESH:
                    continue

                # Get box coordinates
                box_data = box_out[0, y, x, :4]

                if is_quantized:
                    # Dequantize box coordinates
                    box_data = (box_data.astype(np.float32) - 128.0) / 32.0

                # Apply proper transformations
                box_x = 1.0 / (1.0 + np.exp(-box_data[0]))
                box_y = 1.0 / (1.0 + np.exp(-box_data[1]))
                box_w = np.exp(np.clip(box_data[2], -10, 10))
                box_h = np.exp(np.clip(box_data[3], -10, 10))

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

                # Filter unreasonable boxes
                box_w_img = x2 - x1
                box_h_img = y2 - y1
                box_area = box_w_img * box_h_img
                img_area = orig_w * orig_h

                if (box_w_img < 30 or box_h_img < 30 or  # Too small
                    box_area < img_area * 0.002 or        # Less than 0.2% of image
                    box_area > img_area * 0.8 or          # More than 80% of image
                    box_w_img > orig_w * 0.95 or          # Too wide
                    box_h_img > orig_h * 0.95):           # Too tall
                    continue

                # Get keypoints
                kpt_data = kpt_out[0, y, x, :]
                keypoints = []

                if is_quantized:
                    # Dequantize keypoints
                    kpt_data = (kpt_data.astype(np.float32) - 128.0) / 32.0

                for kp_idx in range(24):
                    if kp_idx * 3 + 2 < len(kpt_data):
                        kp_x = kpt_data[kp_idx * 3] * stride
                        kp_y = kpt_data[kp_idx * 3 + 1] * stride
                        kp_conf = 1.0 / (1.0 + np.exp(-kpt_data[kp_idx * 3 + 2]))

                        # Convert to image coordinates
                        kp_x = max(0, (kp_x - pad_l) / scale)
                        kp_y = max(0, (kp_y - pad_t) / scale)

                        keypoints.extend([kp_x, kp_y, kp_conf])
                    else:
                        keypoints.extend([0, 0, 0])

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(obj_score)
                all_keypoints.append(keypoints)

    print(f"[DEBUG] Found {len(all_boxes)} detections")
    return all_boxes, all_scores, all_keypoints

def infer_hailo(hef_path, img):
    """Run inference using Hailo"""

    if not HAILO_AVAILABLE:
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

        # Convert to RGB and keep as uint8 (model expects uint8)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        input_data = rgb.astype(np.uint8)

        # Convert to CHW and add batch
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, 0)

        # Add input name handling
        print("Converting bestz_v8/input_layer1 numpy array to be C_CONTIGUOUS")

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

        # Parse outputs with quantization handling
        output_arrays = list(output_data.values())
        return parse_model_outputs(output_arrays, orig_h, orig_w, pad_t, pad_l, scale)

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

def get_camera():
    """Initialize camera"""
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
        print("[INFO] Using OpenCV camera")
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        return cam, "opencv"

def capture_frame(cam, cam_type):
    """Capture a frame with fixed rotation"""
    if cam_type == "picamera2":
        frame = cam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()
        if not ret:
            return None

    # Apply rotation - FIXED TO COUNTER-CLOCKWISE
    if CAM_ROT_DEG == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)  # FIXED!
    elif CAM_ROT_DEG == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif CAM_ROT_DEG == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    return frame

def draw_detections(img, boxes, scores, keypoints):
    """Draw detections on image"""

    for box, score, kpts in zip(boxes, scores, keypoints):
        x1, y1, x2, y2 = [int(v) for v in box]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw score
        label = f"Dog: {score:.2f}"
        cv2.putText(img, label, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw keypoints with confidence
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

def main():
    print("\n" + "="*60)
    print("🐕 Dog Pose Detection (Fixed for 9-output model)")
    print("="*60)

    # Create output directory
    output_dir = Path("detection_results_fixed")
    output_dir.mkdir(exist_ok=True)

    # Initialize camera
    cam, cam_type = get_camera()
    time.sleep(2)  # Warm up

    duration = 60  # Run for 60 seconds
    print(f"[INFO] Starting detection...")
    print(f"Running in headless mode for {duration}s")
    print(f"Detected frames will be saved automatically")
    print("="*60 + "\n")

    # Stats
    frame_count = 0
    detection_count = 0
    total_detections = 0
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            # Capture frame
            frame = capture_frame(cam, cam_type)
            if frame is None:
                continue

            frame_count += 1

            # Run inference
            t_start = time.time()
            boxes, scores, keypoints = infer_hailo(HEF_PATH, frame)
            t_infer = (time.time() - t_start) * 1000  # Convert to ms

            # Update stats
            num_dogs = len(boxes)
            if num_dogs > 0:
                detection_count += 1
                total_detections += num_dogs

                # Save detection frame
                vis_frame = frame.copy()
                vis_frame = draw_detections(vis_frame, boxes, scores, keypoints)

                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = output_dir / f"detection_{timestamp}.jpg"
                cv2.imwrite(str(filename), vis_frame)

                print(f"[{frame_count:04d}] Detected {num_dogs} dog(s) - Inference: {t_infer:.1f}ms - Saved: {filename.name}")
            else:
                print(f"[{frame_count:04d}] No dogs detected - Inference: {t_infer:.1f}ms", end='\r')

            # Small delay to avoid overwhelming the system
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    # Cleanup
    if cam_type == "picamera2":
        cam.stop()
    else:
        cam.release()

    # Print statistics
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    print("\n" + "="*60)
    print("📊 Final Statistics:")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Total frames: {frame_count}")
    print(f"  Frames with detections: {detection_count}")
    print(f"  Detection rate: {detection_count/frame_count*100:.1f}%" if frame_count > 0 else "N/A")
    print(f"  Avg FPS: {fps:.1f}")
    print(f"  Results saved to: {output_dir}")

if __name__ == "__main__":
    main()