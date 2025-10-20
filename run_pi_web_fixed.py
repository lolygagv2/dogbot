#!/usr/bin/env python3
"""
Dog Pose Detection with Web GUI - FIXED VERSION
Fixed camera format and rotation issues
"""

import os
import json
import time
import threading
import base64
import io
import collections
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import cv2
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler

# Hailo imports
try:
    import hailo_platform as hpf
    HAILO_AVAILABLE = True
except ImportError:
    print("[WARNING] HailoRT not available")
    HAILO_AVAILABLE = False

# ------------------------
# Config
# ------------------------
CFG = json.load(open("config/config.json"))

imgsz_cfg = CFG.get("imgsz", [1024, 1024])
if isinstance(imgsz_cfg, list):
    IMGSZ_H, IMGSZ_W = imgsz_cfg
else:
    IMGSZ_H = IMGSZ_W = imgsz_cfg

T = int(CFG.get("T", 14))
BEHAVIORS = list(CFG.get("behaviors", ["stand", "sit", "lie", "cross", "spin"]))
PROB_TH = float(CFG.get("prob_th", 0.6))
COOLDOWN_S = dict(CFG.get("cooldown_s", {"stand": 2, "sit": 5, "lie": 5, "cross": 4, "spin": 8}))
CAM_ROT_DEG = int(CFG.get("camera_rotation_deg", 90))
HEF_PATH = CFG.get("hef_path", "ai/models/yolo_pose.hef")
HEAD_TS = CFG.get("behavior_head_ts", "ai/models/behavior_head.ts")

MARKER_TO_DOG = {int(d["marker_id"]): str(d["id"]) for d in CFG.get("dogs", [])}

# Behavior tracking
behavior_history = collections.defaultdict(lambda: collections.deque(maxlen=T))
last_behavior_time = collections.defaultdict(lambda: collections.defaultdict(lambda: datetime.min))

print(f"[CONFIG] Model resolution: {IMGSZ_W}x{IMGSZ_H}")
print(f"[CONFIG] HEF model: {HEF_PATH}")
print(f"[CONFIG] Behavior head: {HEAD_TS}")
print(f"[CONFIG] Detection threshold: {PROB_TH}")
print(f"[CONFIG] Camera rotation: {CAM_ROT_DEG}°")
print(f"[CONFIG] Behaviors: {BEHAVIORS}")
print(f"[CONFIG] Dogs: {MARKER_TO_DOG}")

# Global variables for web interface
current_frame = None
current_stats = {"frame": 0, "detections": 0, "fps": 0}
frame_lock = threading.Lock()

# ------------------------
# Simplified Model Parser
# ------------------------
def parse_nms_output(output, orig_h, orig_w, pad_t, pad_l, scale):
    """Parse NMS-processed output"""
    boxes = []
    scores = []
    keypoints = []

    for detection in output:
        if len(detection) < 6:
            continue

        # Standard YOLO NMS format: x1, y1, x2, y2, score, class_id, [keypoints...]
        x1, y1, x2, y2, score, class_id = detection[:6]

        # Only process if score is high enough
        if score < PROB_TH:
            continue

        # Convert to original image coordinates
        x1 = max(0, (x1 - pad_l) / scale)
        y1 = max(0, (y1 - pad_t) / scale)
        x2 = min(orig_w, (x2 - pad_l) / scale)
        y2 = min(orig_h, (y2 - pad_t) / scale)

        # Extract keypoints if present
        kpts = []
        if len(detection) > 6:
            kp_data = detection[6:]
            for i in range(0, min(len(kp_data), 72), 3):
                if i+2 < len(kp_data):
                    kp_x = (kp_data[i] - pad_l) / scale
                    kp_y = (kp_data[i+1] - pad_t) / scale
                    kp_conf = kp_data[i+2]
                    kpts.extend([kp_x, kp_y, kp_conf])

        # Pad keypoints to 72 values
        while len(kpts) < 72:
            kpts.extend([0, 0, 0])

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        keypoints.append(kpts)

    return boxes, scores, keypoints

# Global flag to only debug once
debug_done = False

def parse_model_outputs(outputs, orig_h, orig_w, pad_t, pad_l, scale):
    """Parser that handles both raw and NMS outputs"""
    global debug_done

    # Debug output shapes (only once)
    if not debug_done:
        print(f"[DEBUG] Got {len(outputs)} outputs")
        for i, out in enumerate(outputs[:3]):  # Only show first 3
            print(f"  Output {i}: shape {out.shape}, dtype {out.dtype}, range [{out.min():.2f}, {out.max():.2f}]")
        debug_done = True

    # Check if this is NMS output (single 2D array) or raw outputs (9 4D arrays)
    if len(outputs) == 1 and outputs[0].ndim == 2:
        print("[INFO] Detected NMS format output")
        return parse_nms_output(outputs[0], orig_h, orig_w, pad_t, pad_l, scale)

    if len(outputs) != 9:
        print(f"[ERROR] Expected 9 outputs for raw format, got {len(outputs)}")
        return [], [], []

    all_boxes = []
    all_scores = []
    all_keypoints = []

    # Debug: track max scores
    max_raw = -1000
    max_sig = 0

    scales = [(128, 128), (64, 64), (32, 32)]
    strides = [8, 16, 32]

    for scale_idx, ((h, w), stride) in enumerate(zip(scales, strides)):
        # Find outputs for this scale
        scale_outputs = []
        for out in outputs:
            if out.shape[1] == h and out.shape[2] == w:
                scale_outputs.append(out)

        if len(scale_outputs) != 3:
            continue

        # Sort by channels: 1, 64, 72
        scale_outputs.sort(key=lambda x: x.shape[3])
        obj_out, box_out, kpt_out = scale_outputs

        # Sample every 8 pixels to avoid too many detections
        step = 8 if scale_idx == 0 else 4

        for y in range(0, h, step):
            for x in range(0, w, step):
                # Get objectness score - HANDLE QUANTIZED UINT8
                raw_score = obj_out[0, y, x, 0]

                # Dequantize: uint8 is quantized with 128 as zero point
                # Values > 128 are positive, < 128 are negative
                dequant_score = (raw_score - 128) / 32.0  # Scale factor guess

                # Apply sigmoid to dequantized score
                obj_score = 1.0 / (1.0 + np.exp(-np.clip(dequant_score, -10, 10)))

                # Track max scores for debugging
                if raw_score > max_raw:
                    max_raw = raw_score
                if obj_score > max_sig:
                    max_sig = obj_score

                # Use global threshold
                if obj_score < PROB_TH:
                    continue

                # Box decoding - HANDLE QUANTIZED DATA
                box_data = box_out[0, y, x, :4].astype(np.float32)

                # Dequantize box values (centered at 128)
                box_data = (box_data - 128) / 32.0

                # YOLO box decoding
                cx = (x + 0.5 + box_data[0]) * stride
                cy = (y + 0.5 + box_data[1]) * stride
                w_box = np.exp(box_data[2]) * stride * 4
                h_box = np.exp(box_data[3]) * stride * 4

                # Convert to image coords
                x1 = max(0, (cx - w_box/2 - pad_l) / scale)
                y1 = max(0, (cy - h_box/2 - pad_t) / scale)
                x2 = min(orig_w, (cx + w_box/2 - pad_l) / scale)
                y2 = min(orig_h, (cy + h_box/2 - pad_t) / scale)

                # Size filter
                if (x2-x1) < 40 or (y2-y1) < 40 or (x2-x1) > orig_w*0.8 or (y2-y1) > orig_h*0.8:
                    continue

                # Simple keypoints
                keypoints = []
                kpt_data = kpt_out[0, y, x, :]
                for i in range(24):
                    if i * 3 + 2 < len(kpt_data):
                        kp_x = (kpt_data[i*3] * stride - pad_l) / scale
                        kp_y = (kpt_data[i*3+1] * stride - pad_t) / scale
                        kp_conf = abs(kpt_data[i*3+2])
                        keypoints.extend([kp_x, kp_y, kp_conf])
                    else:
                        keypoints.extend([0, 0, 0])

                all_boxes.append([x1, y1, x2, y2])
                all_scores.append(obj_score)
                all_keypoints.append(keypoints)

                # Limit detections
                if len(all_boxes) >= 5:
                    break

    # Debug output every few frames
    import random
    if random.random() < 0.05:  # 5% of frames
        print(f"[DEBUG] Max raw score: {max_raw:.3f}, Max sigmoid: {max_sig:.3f}, Detections: {len(all_boxes)}")

    return all_boxes, all_scores, all_keypoints

# ------------------------
# Behavior Classification
# ------------------------
def classify_behavior(keypoints_list):
    """Classify behavior using PyTorch model"""
    if not keypoints_list:
        return []

    try:
        # Load model
        model = torch.jit.load(HEAD_TS, map_location='cpu')
        model.eval()

        behaviors = []
        for kpts in keypoints_list:
            # Extract only x,y coordinates (no confidence) = 24 keypoints * 2 = 48 values
            kpts_xy = []
            for i in range(0, min(72, len(kpts)), 3):
                if i+1 < len(kpts):
                    kpts_xy.extend([kpts[i], kpts[i+1]])  # Only x, y (skip confidence)

            # Pad to 48 values if needed
            while len(kpts_xy) < 48:
                kpts_xy.extend([0.0, 0.0])

            # Reshape for model input
            kpts_array = np.array(kpts_xy[:48]).reshape(1, -1).astype(np.float32)

            # Convert to tensor
            kpts_tensor = torch.from_numpy(kpts_array)

            with torch.no_grad():
                output = model(kpts_tensor)
                probs = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(probs, dim=1).item()
                confidence = probs[0, pred_idx].item()

                if pred_idx < len(BEHAVIORS):
                    behaviors.append((BEHAVIORS[pred_idx], confidence))
                else:
                    behaviors.append(("unknown", 0.0))

        return behaviors

    except Exception as e:
        print(f"[ERROR] Behavior classification failed: {e}")
        return [("unknown", 0.0)] * len(keypoints_list)

def detect_aruco_markers(img):
    """Detect ArUco markers for dog identification"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ArUco detection
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        corners, ids, _ = detector.detectMarkers(gray)

        markers = []
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in MARKER_TO_DOG:
                    # Get marker center
                    corner = corners[i][0]
                    center = np.mean(corner, axis=0)
                    markers.append({
                        'id': int(marker_id),
                        'dog': MARKER_TO_DOG[marker_id],
                        'center': center,
                        'corners': corner
                    })

        return markers

    except Exception as e:
        print(f"[ERROR] ArUco detection failed: {e}")
        return []

def match_detections_to_markers(boxes, markers):
    """Match pose detections to ArUco markers"""
    matches = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        box_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        best_marker = None
        min_dist = float('inf')

        for marker in markers:
            dist = np.linalg.norm(box_center - marker['center'])
            if dist < min_dist and dist < 200:  # 200 pixel threshold
                min_dist = dist
                best_marker = marker

        matches.append({
            'detection_idx': i,
            'marker': best_marker,
            'dog_id': best_marker['dog'] if best_marker else f"unknown_{i}"
        })

    return matches

# ------------------------
# Hailo Inference
# ------------------------
def infer_hailo(hef_path, img):
    """Simplified Hailo inference"""
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

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.uint8)
        input_data = np.transpose(rgb, (2, 0, 1))
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

# ------------------------
# Camera - BACK TO WORKING VERSION
# ------------------------
def get_camera():
    """Initialize camera - SIMPLE, NO MANUAL CONTROLS"""
    from picamera2 import Picamera2

    cam = Picamera2()
    config = cam.create_still_configuration(main={"size": (1920, 1080), "format": "RGB888"})
    cam.configure(config)
    cam.start()

    # NO MANUAL CONTROLS - let camera auto-expose
    print("[INFO] Picamera2 - RGB888, auto-exposure")
    return cam, "picamera2"

def capture_frame(cam, cam_type):
    """Capture frame - NO COLOR CONVERSION AT ALL"""
    frame = cam.capture_array()
    # NO conversion - just return the raw RGB888 frame
    return frame

# ------------------------
# Visualization
# ------------------------
def draw_detections(img, boxes, scores, keypoints):
    """Draw simple detections (fallback)"""
    for box, score, kpts in zip(boxes, scores, keypoints):
        x1, y1, x2, y2 = [int(v) for v in box]

        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"Dog: {score:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw some keypoints
        for i in range(0, min(72, len(kpts)), 3):
            x, y, conf = kpts[i:i+3]
            if conf > 0.3:
                cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)

    return img

def draw_full_detections(img, boxes, scores, keypoints, markers, matches, behaviors):
    """Draw full detection pipeline results"""

    # Draw ArUco markers first
    for marker in markers:
        corners = marker['corners'].astype(int)
        cv2.polylines(img, [corners], True, (0, 255, 255), 2)

        # Draw marker ID and dog name
        center = marker['center'].astype(int)
        cv2.putText(img, f"{marker['dog']} ({marker['id']})",
                   (center[0]-50, center[1]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Draw detections with behavior info
    for i, (box, score, kpts) in enumerate(zip(boxes, scores, keypoints)):
        x1, y1, x2, y2 = [int(v) for v in box]

        # Get match and behavior info
        match = matches[i] if i < len(matches) else None
        behavior = behaviors[i] if i < len(behaviors) else ("unknown", 0.0)
        behavior_name, behavior_conf = behavior

        dog_id = match['dog_id'] if match else f"unknown_{i}"

        # Choose color based on dog ID
        if match and match['marker']:
            color = (0, 255, 0)  # Green for identified dogs
        else:
            color = (0, 165, 255)  # Orange for unidentified

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw labels
        labels = [
            f"{dog_id}: {score:.2f}",
            f"{behavior_name}: {behavior_conf:.2f}"
        ]

        for j, label in enumerate(labels):
            cv2.putText(img, label, (x1, y1-30+j*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw keypoints
        for k in range(0, min(72, len(kpts)), 3):
            x, y, conf = kpts[k:k+3]
            if conf > 0.3:
                cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)

    return img

# ------------------------
# Web Server
# ------------------------
class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global current_frame, current_stats

        if self.path == '/':
            # HTML page
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dog Detection Live Feed - FIXED</title>
                <style>
                    body { font-family: Arial; text-align: center; background: #222; color: white; }
                    img { max-width: 90%; height: auto; border: 2px solid #00ff00; }
                    .stats { margin: 20px; font-size: 18px; color: #00ff00; }
                    .button { margin: 10px; padding: 10px; font-size: 16px; background: #00ff00; color: black; border: none; cursor: pointer; }
                    .status { color: #ffff00; }
                </style>
            </head>
            <body>
                <h1>🐕 Dog Pose Detection - COLOR FIX ATTEMPT</h1>
                <div class="status">SIMPLIFIED: Picamera2 only, no OpenCV fallback | RGB888→BGR | 0° rotation</div>
                <div class="stats">
                    <span id="stats">Frame: 0, Detections: 0, FPS: 0</span>
                </div>
                <img id="video" src="/stream" alt="Loading camera feed...">
                <br>
                <button class="button" onclick="saveFrame()">💾 Save Current Frame</button>
                <button class="button" onclick="location.reload()">🔄 Refresh Page</button>
                <script>
                    function updateStats() {
                        fetch('/stats').then(r => r.json()).then(data => {
                            document.getElementById('stats').textContent =
                                `Frame: ${data.frame}, Detections: ${data.detections}, FPS: ${data.fps}`;
                        }).catch(e => console.log('Stats error:', e));
                    }
                    function saveFrame() {
                        fetch('/save').then(r => r.text()).then(msg => alert(msg))
                            .catch(e => alert('Save failed: ' + e));
                    }
                    setInterval(updateStats, 1000);
                    setInterval(() => {
                        document.getElementById('video').src = '/stream?' + Date.now();
                    }, 500);
                    console.log('Web interface loaded successfully');
                </script>
            </body>
            </html>
            """
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())

        elif self.path.startswith('/stream'):
            # Video stream
            try:
                with frame_lock:
                    if current_frame is not None:
                        _, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        img_data = buffer.tobytes()

                        self.send_response(200)
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Cache-Control', 'no-cache')
                        self.end_headers()
                        self.wfile.write(img_data)
                    else:
                        self.send_error(404, "No frame available")
            except Exception as e:
                print(f"[ERROR] Stream error: {e}")
                self.send_error(500, "Stream error")

        elif self.path == '/stats':
            # Stats JSON
            try:
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(current_stats).encode())
            except Exception as e:
                print(f"[ERROR] Stats error: {e}")
                self.send_error(500, "Stats error")

        elif self.path == '/save':
            # Save frame
            try:
                with frame_lock:
                    if current_frame is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"detection_web_fixed_{timestamp}.jpg"
                        cv2.imwrite(filename, current_frame)
                        msg = f"✅ Saved: {filename}"
                    else:
                        msg = "❌ No frame available"

                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(msg.encode())
            except Exception as e:
                print(f"[ERROR] Save error: {e}")
                self.send_error(500, f"Save error: {e}")

    def log_message(self, format, *args):
        pass  # Suppress HTTP log messages

# ------------------------
# Main Detection Loop
# ------------------------
def detection_loop():
    """Main detection loop running in background"""
    global current_frame, current_stats

    try:
        print("[INFO] Initializing camera...")
        cam, cam_type = get_camera()
        time.sleep(3)  # Longer warmup time

        frame_count = 0
        detection_count = 0
        start_time = time.time()

        print("[INFO] Starting detection loop...")

        while True:
            try:
                frame = capture_frame(cam, cam_type)
                if frame is None:
                    print("[WARNING] No frame captured")
                    time.sleep(0.5)
                    continue

                frame_count += 1

                # Run full detection pipeline every few frames
                if frame_count % 5 == 0:  # Every 5th frame
                    # 1. Pose detection
                    boxes, scores, keypoints = infer_hailo(HEF_PATH, frame)

                    # 2. ArUco marker detection
                    markers = detect_aruco_markers(frame)

                    # 3. Match detections to markers
                    matches = match_detections_to_markers(boxes, markers)

                    # 4. Behavior classification
                    behaviors = classify_behavior(keypoints) if keypoints else []

                    if len(boxes) > 0:
                        detection_count += 1

                        # Update behavior history
                        current_time = datetime.now()
                        for i, (match, behavior) in enumerate(zip(matches, behaviors)):
                            dog_id = match['dog_id']
                            behavior_name, confidence = behavior

                            # Add to history
                            behavior_history[dog_id].append(behavior_name)

                            # Check for behavior triggers
                            if len(behavior_history[dog_id]) >= T:
                                recent_behaviors = list(behavior_history[dog_id])[-T:]
                                most_common = max(set(recent_behaviors), key=recent_behaviors.count)

                                # Check cooldown
                                last_time = last_behavior_time[dog_id][most_common]
                                cooldown = timedelta(seconds=COOLDOWN_S.get(most_common, 2))

                                if current_time - last_time > cooldown:
                                    print(f"[BEHAVIOR] {dog_id}: {most_common} (confidence: {confidence:.2f})")
                                    last_behavior_time[dog_id][most_common] = current_time

                else:
                    boxes, scores, keypoints, markers, matches, behaviors = [], [], [], [], [], []

                # Draw everything
                vis_frame = frame.copy()
                if boxes:
                    vis_frame = draw_full_detections(vis_frame, boxes, scores, keypoints, markers, matches, behaviors)

                # Add status overlay
                elapsed = time.time() - start_time
                fps = frame_count / max(elapsed, 0.001)

                status_lines = [
                    f"Frame: {frame_count}",
                    f"Dogs: {len(boxes)}",
                    f"FPS: {fps:.1f}",
                    f"Total detections: {detection_count}"
                ]

                for i, line in enumerate(status_lines):
                    cv2.putText(vis_frame, line, (10, 30 + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Update globals thread-safely
                with frame_lock:
                    current_frame = vis_frame.copy()
                    current_stats = {
                        "frame": frame_count,
                        "detections": len(boxes),
                        "fps": round(fps, 1)
                    }

                # Print status occasionally
                if frame_count % 50 == 0:
                    print(f"[STATUS] Frame {frame_count}, FPS: {fps:.1f}, Detections: {detection_count}")

                time.sleep(0.05)  # Small delay to prevent overload

            except Exception as e:
                print(f"[ERROR] Detection loop error: {e}")
                time.sleep(1)

    except Exception as e:
        print(f"[FATAL] Detection loop crashed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print(f"\n{'='*60}")
    print("🐕 Dog Pose Detection - FIXED Web Interface")
    print(f"{'='*60}")
    print("FIXES: Camera rotation CLOCKWISE + Normal colors")

    # Start detection in background
    print("[INFO] Starting detection thread...")
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()
    time.sleep(2)  # Let it initialize

    # Start web server
    port = 8081
    print(f"\n[INFO] Starting FIXED web server on port {port}")
    print(f"[INFO] Remote access: http://192.168.50.69:{port}")
    print(f"[INFO] Local access: http://localhost:{port}")
    print(f"[INFO] Press Ctrl+C to quit")
    print(f"{'='*60}\n")

    try:
        server = HTTPServer(('0.0.0.0', port), WebHandler)
        print("[SUCCESS] Web server started successfully!")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"[ERROR] Server failed: {e}")
    finally:
        try:
            server.shutdown()
        except:
            pass

if __name__ == "__main__":
    main()