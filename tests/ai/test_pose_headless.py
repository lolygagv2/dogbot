#!/usr/bin/env python3
"""
Headless pose detection test - saves results without GUI
Perfect for testing the new retrained 1024x1024 model
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
CAM_ROT_DEG = CFG.get("camera_rotation_deg", 90)
CONF_THRESH = 0.3  # Start with lower threshold for testing

print(f"[CONFIG] Model: {HEF_PATH}")
print(f"[CONFIG] Resolution: {IMGSZ_W}x{IMGSZ_H}")
print(f"[CONFIG] Confidence threshold: {CONF_THRESH}")

def parse_outputs(outputs, orig_h, orig_w, pad_t, pad_l, scale):
    """Parse YOLO outputs - handles both multi-scale and NMS formats"""

    # Check if NMS output (single 2D array)
    if len(outputs) == 1 and outputs[0].ndim == 2:
        return parse_nms_output(outputs[0], orig_h, orig_w, pad_t, pad_l, scale)
    else:
        return parse_multi_scale_outputs(outputs, orig_h, orig_w, pad_t, pad_l, scale)

def parse_nms_output(output, orig_h, orig_w, pad_t, pad_l, scale):
    """Parse NMS output format"""
    boxes = []
    scores = []
    keypoints = []

    for det in output:
        if len(det) < 6:
            continue

        x1, y1, x2, y2, score, class_id = det[:6]

        if score < CONF_THRESH:
            continue

        # Convert to image coordinates
        x1 = max(0, (x1 - pad_l) / scale)
        y1 = max(0, (y1 - pad_t) / scale)
        x2 = min(orig_w, (x2 - pad_l) / scale)
        y2 = min(orig_h, (y2 - pad_t) / scale)

        # Extract keypoints if present
        kpts = []
        if len(det) > 6:
            kp_data = det[6:]
            num_kpts = len(kp_data) // 3
            for i in range(min(num_kpts, 24)):
                kp_x = (kp_data[i*3] - pad_l) / scale
                kp_y = (kp_data[i*3 + 1] - pad_t) / scale
                kp_conf = kp_data[i*3 + 2]
                kpts.extend([kp_x, kp_y, kp_conf])

        # Pad to 24 keypoints
        while len(kpts) < 72:
            kpts.extend([0, 0, 0])

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        keypoints.append(kpts)

    return boxes, scores, keypoints

def parse_multi_scale_outputs(outputs, orig_h, orig_w, pad_t, pad_l, scale):
    """Parse multi-scale YOLO outputs"""
    boxes = []
    scores = []
    keypoints = []

    # Process each scale
    strides = [8, 16, 32]

    for out, stride in zip(outputs, strides):
        if out.ndim != 4:
            continue

        bs, ch, gy, gx = out.shape

        # Generate grid
        yv, xv = np.meshgrid(np.arange(gy), np.arange(gx), indexing='ij')
        grid = np.stack([xv, yv], axis=2).reshape(1, gy, gx, 2).astype(np.float32)

        # Process predictions
        out_reshaped = out.reshape(bs, ch, gy * gx).transpose(0, 2, 1)

        for b in range(bs):
            for anchor_idx in range(gy * gx):
                pred = out_reshaped[b, anchor_idx]

                # Decode box
                cx = (pred[0] * 2 - 0.5 + grid.reshape(-1, 2)[anchor_idx, 0]) * stride
                cy = (pred[1] * 2 - 0.5 + grid.reshape(-1, 2)[anchor_idx, 1]) * stride
                w = (pred[2] * 2) ** 2 * stride
                h = (pred[3] * 2) ** 2 * stride
                obj_conf = pred[4]

                if obj_conf < CONF_THRESH:
                    continue

                # Convert to image coords
                x1 = max(0, (cx - w/2 - pad_l) / scale)
                y1 = max(0, (cy - h/2 - pad_t) / scale)
                x2 = min(orig_w, (cx + w/2 - pad_l) / scale)
                y2 = min(orig_h, (cy + h/2 - pad_t) / scale)

                # Extract keypoints
                kpts = []
                if ch > 77:  # Has keypoints
                    kp_data = pred[5:77] if ch > 82 else pred[5:]
                    for i in range(24):
                        if i * 3 + 2 < len(kp_data):
                            kp_x = (kp_data[i*3] * stride - pad_l) / scale
                            kp_y = (kp_data[i*3+1] * stride - pad_t) / scale
                            kp_conf = kp_data[i*3+2]
                            kpts.extend([kp_x, kp_y, kp_conf])

                # Pad keypoints
                while len(kpts) < 72:
                    kpts.extend([0, 0, 0])

                boxes.append([x1, y1, x2, y2])
                scores.append(obj_conf)
                keypoints.append(kpts)

    return boxes, scores, keypoints

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

        # Parse outputs
        output_arrays = list(output_data.values())
        return parse_outputs(output_arrays, orig_h, orig_w, pad_t, pad_l, scale)

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
    """Capture a frame"""
    if cam_type == "picamera2":
        frame = cam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()
        if not ret:
            return None

    # Apply rotation
    if CAM_ROT_DEG == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif CAM_ROT_DEG == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif CAM_ROT_DEG == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

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
    print("🐕 Headless Pose Detection Test")
    print("="*60)

    # Create output directory
    output_dir = Path("pose_test_results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Initialize camera
    cam, cam_type = get_camera()
    time.sleep(2)  # Warm up

    print(f"\n[INFO] Saving results to {output_dir}")
    print("[INFO] Running for 30 seconds or until Ctrl+C")
    print("="*60 + "\n")

    # Stats
    frame_count = 0
    detection_count = 0
    total_detections = 0
    start_time = time.time()

    try:
        while time.time() - start_time < 30:  # Run for 30 seconds
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
                total_detections += len(boxes)

                # Save detection frame
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

                # Save image
                filename = output_dir / f"detection_{timestamp}_{frame_count:04d}.jpg"
                cv2.imwrite(str(filename), vis_frame)

                # Save data
                data = {
                    "frame": frame_count,
                    "timestamp": datetime.now().isoformat(),
                    "inference_ms": t_infer * 1000,
                    "detections": len(boxes),
                    "boxes": [[float(v) for v in box] for box in boxes],
                    "scores": [float(s) for s in scores],
                    "keypoints_visible": [
                        sum(1 for i in range(24) if kpts[i*3+2] > 0.3)
                        for kpts in keypoints
                    ]
                }

                # Append to JSON log
                log_file = output_dir / f"log_{timestamp}.jsonl"
                with open(log_file, 'a') as f:
                    f.write(json.dumps(data) + '\n')

                print(f"[{frame_count:04d}] Detected {len(boxes)} dog(s) - "
                      f"Inference: {t_infer*1000:.1f}ms - "
                      f"Saved: {filename.name}")
            else:
                print(f"[{frame_count:04d}] No detections - "
                      f"Inference: {t_infer*1000:.1f}ms")

            # Small delay
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        if cam_type == "picamera2":
            cam.stop()
        else:
            cam.release()

        # Print summary
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print("📊 Test Summary:")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Total frames: {frame_count}")
        print(f"  Frames with detections: {detection_count}")
        print(f"  Total dogs detected: {total_detections}")
        print(f"  Detection rate: {detection_count/max(frame_count,1)*100:.1f}%")
        print(f"  Avg FPS: {frame_count/elapsed:.1f}")
        print(f"  Results saved to: {output_dir}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()