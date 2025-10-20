#!/usr/bin/env python3
"""
Dog Pose Detection with Web GUI
Since OpenCV GUI isn't available, use web interface on port 8080
"""

import os
import json
import time
import threading
import base64
import io
from pathlib import Path
from datetime import datetime
import numpy as np
import cv2
from http.server import HTTPServer, BaseHTTPRequestHandler
import socketserver

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

PROB_TH = 0.6
CAM_ROT_DEG = int(CFG.get("camera_rotation_deg", 90))
HEF_PATH = CFG.get("hef_path", "ai/models/yolo_pose.hef")

print(f"[CONFIG] Model resolution: {IMGSZ_W}x{IMGSZ_H}")
print(f"[CONFIG] Detection threshold: {PROB_TH}")
print(f"[CONFIG] Camera rotation: {CAM_ROT_DEG}°")

# Global variables for web interface
current_frame = None
current_stats = {"frame": 0, "detections": 0, "fps": 0}
frame_lock = threading.Lock()

# ------------------------
# Simplified Model Parser
# ------------------------
def parse_model_outputs(outputs, orig_h, orig_w, pad_t, pad_l, scale):
    """Simple parser for testing"""
    if len(outputs) != 9:
        return [], [], []

    all_boxes = []
    all_scores = []
    all_keypoints = []

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
                # Get objectness with simple threshold
                raw_score = obj_out[0, y, x, 0]

                # Try both raw and sigmoid
                if raw_score > 0.5:  # Raw threshold
                    obj_score = raw_score
                else:
                    obj_score = 1.0 / (1.0 + np.exp(-np.clip(raw_score, -10, 10)))
                    if obj_score < 0.7:  # Sigmoid threshold
                        continue

                # Simple box decoding
                box_data = box_out[0, y, x, :4]

                # Basic box calculation
                cx = (x + 0.5 + box_data[0]) * stride
                cy = (y + 0.5 + box_data[1]) * stride
                w_box = abs(box_data[2]) * stride * 8
                h_box = abs(box_data[3]) * stride * 8

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

    return all_boxes, all_scores, all_keypoints

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
# Camera
# ------------------------
def get_camera():
    """Initialize camera"""
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        # Use BGR888 format directly for normal colors, higher resolution
        config = cam.create_still_configuration(main={"size": (1920, 1080), "format": "BGR888"})
        cam.configure(config)
        cam.start()
        return cam, "picamera2"
    except:
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        return cam, "opencv"

def capture_frame(cam, cam_type):
    """Capture frame"""
    if cam_type == "picamera2":
        frame = cam.capture_array()
        # Now using BGR888 format directly - no conversion needed for normal colors
    else:
        ret, frame = cam.read()
        if not ret:
            return None

    # Rotate - FIXED: now rotate CLOCKWISE from current position
    if CAM_ROT_DEG == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    return frame

# ------------------------
# Visualization
# ------------------------
def draw_detections(img, boxes, scores, keypoints):
    """Draw detections"""
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
                <title>Dog Detection Live Feed</title>
                <style>
                    body { font-family: Arial; text-align: center; background: #222; color: white; }
                    img { max-width: 90%; height: auto; border: 2px solid #00ff00; }
                    .stats { margin: 20px; font-size: 18px; }
                    .button { margin: 10px; padding: 10px; font-size: 16px; }
                </style>
            </head>
            <body>
                <h1>🐕 Dog Pose Detection - Live Feed</h1>
                <div class="stats">
                    <span id="stats">Frame: 0, Detections: 0, FPS: 0</span>
                </div>
                <img id="video" src="/stream" alt="Loading...">
                <br>
                <button class="button" onclick="saveFrame()">Save Current Frame</button>
                <script>
                    function updateStats() {
                        fetch('/stats').then(r => r.json()).then(data => {
                            document.getElementById('stats').textContent =
                                `Frame: ${data.frame}, Detections: ${data.detections}, FPS: ${data.fps}`;
                        });
                    }
                    function saveFrame() {
                        fetch('/save').then(r => r.text()).then(msg => alert(msg));
                    }
                    setInterval(updateStats, 1000);
                    setInterval(() => {
                        document.getElementById('video').src = '/stream?' + Date.now();
                    }, 500);
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
            with frame_lock:
                if current_frame is not None:
                    _, buffer = cv2.imencode('.jpg', current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    img_data = buffer.tobytes()

                    self.send_response(200)
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    self.wfile.write(img_data)
                else:
                    self.send_error(404)

        elif self.path == '/stats':
            # Stats JSON
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(current_stats).encode())

        elif self.path == '/save':
            # Save frame
            with frame_lock:
                if current_frame is not None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_web_{timestamp}.jpg"
                    cv2.imwrite(filename, current_frame)
                    msg = f"Saved: {filename}"
                else:
                    msg = "No frame available"

            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(msg.encode())

    def log_message(self, format, *args):
        pass  # Suppress log messages

# ------------------------
# Main Loop
# ------------------------
def detection_loop():
    """Main detection loop running in background"""
    global current_frame, current_stats

    print("[INFO] Starting camera...")
    cam, cam_type = get_camera()
    time.sleep(2)

    frame_count = 0
    last_time = time.time()

    while True:
        try:
            frame = capture_frame(cam, cam_type)
            if frame is None:
                continue

            frame_count += 1

            # Run detection every few frames
            if frame_count % 3 == 0:
                boxes, scores, keypoints = infer_hailo(HEF_PATH, frame)
            else:
                boxes, scores, keypoints = [], [], []

            # Draw detections
            vis_frame = frame.copy()
            if boxes:
                vis_frame = draw_detections(vis_frame, boxes, scores, keypoints)

            # Add status text
            fps = frame_count / (time.time() - last_time + 0.001)
            status = f"Frame: {frame_count} | Dogs: {len(boxes)} | FPS: {fps:.1f}"
            cv2.putText(vis_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Update globals
            with frame_lock:
                current_frame = vis_frame.copy()
                current_stats = {
                    "frame": frame_count,
                    "detections": len(boxes),
                    "fps": round(fps, 1)
                }

            time.sleep(0.1)

        except Exception as e:
            print(f"[ERROR] Detection loop: {e}")
            time.sleep(1)

def main():
    print(f"\n{'='*60}")
    print("🐕 Dog Pose Detection - Web Interface")
    print(f"{'='*60}")

    # Start detection in background
    detection_thread = threading.Thread(target=detection_loop, daemon=True)
    detection_thread.start()

    # Start web server
    port = 8080
    print(f"\n[INFO] Starting web server on port {port}")
    print(f"[INFO] Open your browser and go to: http://localhost:{port}")
    print(f"[INFO] Or from another device: http://<pi-ip-address>:{port}")
    print(f"[INFO] Press Ctrl+C to quit")
    print(f"{'='*60}\n")

    try:
        server = HTTPServer(('0.0.0.0', port), WebHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
        server.shutdown()

if __name__ == "__main__":
    main()