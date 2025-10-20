# run_pi.py  — unified runtime: Hailo pose + TorchScript behavior + ArUco IDs
import os, json, time, collections
import numpy as np
import torch, cv2

# ------------------------
# Config
# ------------------------
CFG = json.load(open("config/config.json"))
IMGSZ        = int(CFG.get("imgsz", 896))
T            = int(CFG.get("T", 16))
BEHAVIORS    = list(CFG.get("behaviors", ["stand","sit","lie","cross","spin"]))
PROB_TH      = float(CFG.get("prob_th", 0.6))
VOTE_LEN     = int(CFG.get("vote_len", 10))  # we use EMA; VOTE_LEN kept for completeness
COOLDOWN_S   = dict(CFG.get("cooldown_s", {"stand":2,"sit":5,"lie":5,"cross":4,"spin":8}))
ASSUME_OTHER = bool(CFG.get("assume_other_if_two_boxes_one_marker", True))
CAM_ROT_DEG  = int(CFG.get("camera_rotation_deg", 0))  # 0/90/180/270
DEBUG_CPU_PT = CFG.get("debug_cpu_pose_pt", None)      # optional path to .pt to bypass Hailo for testing
HEF_PATH     = CFG.get("hef_path", "ai/models/dogposeV2yolo11.hef")    # Hailo HEF path
HEAD_TS      = CFG.get("behavior_head_ts", "behavior_head.ts")

MARKER_TO_DOG = { int(d["marker_id"]): str(d["id"]) for d in CFG.get("dogs", []) }

# ------------------------
# ArUco setup
# ------------------------
def setup_aruco():
    dict_name = str(CFG.get("aruco_dict", "DICT_4X4_1000"))
    dconst = getattr(cv2.aruco, dict_name)
    dic    = cv2.aruco.getPredefinedDictionary(dconst)
    try:
        det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    except AttributeError:  # older OpenCV fallback
        det = (dic, cv2.aruco.DetectorParameters_create())
    return det

ARUCO_DET = setup_aruco()

def detect_markers(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    try:
        corners, ids, _ = ARUCO_DET.detectMarkers(gray)
    except Exception:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DET[0], parameters=ARUCO_DET[1])
    out=[]
    if ids is not None:
        for c, id_ in zip(corners, ids.flatten()):
            pts = c[0]
            cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())
            out.append((int(id_), cx, cy))
    return out  # [(marker_id, cx, cy)]

def assign_ids(boxes, markers, last_ids):
    # boxes: list[np.array([x1,y1,x2,y2])]
    dog_ids = [None]*len(boxes)
    used = set()
    # nearest marker → box
    for mid, cx, cy in markers:
        if not boxes: break
        j = int(min(range(len(boxes)), key=lambda i: (( (boxes[i][0]+boxes[i][2])/2 - cx )**2
                                                    + ((boxes[i][1]+boxes[i][3])/2 - cy )**2 )))
        if j not in used:
            dog_ids[j] = MARKER_TO_DOG.get(mid)
            used.add(j)
    # if 2 boxes and 1 marker, optionally infer the other ID
    if len(boxes)==2 and dog_ids.count(None)==1 and ASSUME_OTHER:
        other = 1 - dog_ids.index(None)
        remaining = [d for d in {v for v in MARKER_TO_DOG.values()} if d != dog_ids[other]]
        dog_ids[1-other] = remaining[0] if remaining else None
    # keep last IDs if still None
    for i in range(len(boxes)):
        if dog_ids[i] is None and i < len(last_ids):
            dog_ids[i] = last_ids[i]
    return dog_ids

def assign_markers_to_boxes(markers, boxes, dx, dy, scale_inv):
    """Assign markers to boxes and return (marker_id, box_index) pairs.

    Args:
        markers: List of (marker_id, cx, cy) tuples
        boxes: List of detection boxes [x1, y1, x2, y2]
        dx, dy, scale_inv: Transform parameters from letterboxing

    Returns:
        List of (marker_id, box_index) pairs
    """
    assigned = []
    used_boxes = set()

    # Transform marker coordinates back to detection space
    for marker_id, cx, cy in markers:
        # Undo letterbox transform
        orig_cx = (cx - dx) * scale_inv
        orig_cy = (cy - dy) * scale_inv

        if not boxes:
            continue

        # Find closest box to this marker
        min_dist = float('inf')
        best_box_idx = None

        for i, box in enumerate(boxes):
            if i in used_boxes:
                continue

            # Box center
            box_cx = (box[0] + box[2]) / 2
            box_cy = (box[1] + box[3]) / 2

            # Distance to marker
            dist = ((box_cx - orig_cx)**2 + (box_cy - orig_cy)**2)**0.5

            if dist < min_dist:
                min_dist = dist
                best_box_idx = i

        if best_box_idx is not None:
            assigned.append((marker_id, best_box_idx))
            used_boxes.add(best_box_idx)

    return assigned

def decode_pose_multi_hailo(raw_outputs):
    """Decode Hailo YOLO pose model outputs"""
    print("HAILO DECODE: Function called!")
    print(f"HAILO DECODE: raw_outputs type = {type(raw_outputs)}")

    # Force print to stdout immediately
    import sys
    sys.stdout.flush()

    return []

def decode_pose_multi_cpu_ultra(inp):
    """Placeholder for CPU debug mode"""
    return []  # No CPU model available

# ------------------------
# Pose preprocessing
# ------------------------
def letterbox(img, size=IMGSZ):
    # simple square letterbox, like training
    # Handle both 3 and 4 channel images
    h, w = img.shape[:2]
    channels = img.shape[2] if len(img.shape) == 3 else 3

    # Convert 4-channel (XBGR/RGBA) to 3-channel BGR if needed
    if channels == 4:
        # For XBGR8888 format from picamera2, convert to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    s = min(size/w, size/h)
    nw, nh = int(w*s), int(h*s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    dy, dx = (size - nh)//2, (size - nw)//2
    canvas[dy:dy+nh, dx:dx+nw] = r
    return canvas, (dx, dy, 1.0/s)

def norm_kpts(box, kpts):  # kpts [24,3] with absolute xy and conf ∈ [0,1]
    x1,y1,x2,y2 = box
    w=max(x2-x1,1e-6); h=max(y2-y1,1e-6)
    nk = np.stack([
        np.clip((kpts[:,0]-x1)/w, 0, 1),
        np.clip((kpts[:,1]-y1)/h, 0, 1)
    ], axis=1)  # [24,2] for the head
    return nk

# ------------------------
# Pose decode
# ------------------------
ULTRA_MODEL = None
def decode_pose_multi_cpu_ultra(bgr):
    """Debug path: run Ultralytics .pt on CPU to get the same outputs format."""
    global ULTRA_MODEL
    if ULTRA_MODEL is None:
        from ultralytics import YOLO
        ULTRA_MODEL = YOLO(DEBUG_CPU_PT)
        assert ULTRA_MODEL.task == "pose"
    dets=[]
    for r in ULTRA_MODEL.predict(source=bgr[...,::-1], imgsz=IMGSZ, conf=0.005, verbose=False):
        if len(r)==0 or r.boxes is None or r.keypoints is None: 
            continue
        # take all detections
        for i in range(len(r.boxes)):
            xyxy = r.boxes.xyxy[i].detach().cpu().numpy()
            kxy  = r.keypoints.xy[i].detach().cpu().numpy()
            kcf  = r.keypoints.conf[i].detach().cpu().numpy()
            K = int(kxy.shape[0])
            if K < 24:
                pad_xy = np.zeros((24,2), np.float32); pad_cf = np.zeros(24, np.float32)
                pad_xy[:K], pad_cf[:K] = kxy, kcf
                kxy, kcf = pad_xy, pad_cf
            elif K > 24:
                continue
            kpts = np.stack([kxy[:,0], kxy[:,1], kcf], axis=1)  # [24,3]
            dets.append({"xyxy": xyxy, "kpts": kpts})
    return dets

def decode_pose_multi_hailo(raw):
    """
    Decode YOLOv11 pose HEF outputs from Hailo.

    Expected structure based on debug output:
    - 3 detection scales (80x80, 40x40, 20x20)
    - Each scale has 3 outputs: bbox coords (64 channels), keypoints (72 channels), confidence (1 channel)
    - Total: 9 conv layers

    YOLOv11 pose format: [cx, cy, w, h] + [x1,y1,v1, x2,y2,v2, ..., x24,y24,v24]
    where v is visibility/confidence for each keypoint

    Returns: list of {"xyxy": np.array([x1,y1,x2,y2]), "kpts": np.array([24,3])}
    """
    print(f"\n=== DECODING HAILO YOLOv11 POSE OUTPUTS ===")
    print(f"Raw type: {type(raw)}")

    if isinstance(raw, dict):
        print(f"Dict keys: {list(raw.keys())}")
        for k, v in raw.items():
            print(f"  {k}: shape={getattr(v, 'shape', 'no shape')}, type={type(v)}")

    dets = []

    # Handle dictionary output from InferVStreams
    if not isinstance(raw, dict):
        print("Expected dict output from Hailo, got:", type(raw))
        return dets

    # Group outputs by scale based on conv layer names
    # Pattern from debug: best_v8/conv63, best_v8/conv60, best_v8/conv59 (80x80)
    #                    best_v8/conv49, best_v8/conv46, best_v8/conv45 (40x40)
    #                    best_v8/conv35, best_v8/conv32, best_v8/conv31 (20x20)

    scales = {
        '80x80': {'bbox': None, 'kpts': None, 'conf': None},
        '40x40': {'bbox': None, 'kpts': None, 'conf': None},
        '20x20': {'bbox': None, 'kpts': None, 'conf': None}
    }

    # Map conv layers to scales and types based on ACTUAL debug output
    # From the real output we have:
    # best_v8/conv97: (1, 20, 20, 1) - likely 20x20 conf
    # best_v8/conv60: (1, 80, 80, 72) - 80x80 kpts
    # best_v8/conv75: (1, 40, 40, 72) - 40x40 kpts
    # best_v8/conv63: (1, 80, 80, 1) - 80x80 conf
    # best_v8/conv74: (1, 40, 40, 64) - 40x40 bbox
    # best_v8/conv93: (1, 20, 20, 64) - 20x20 bbox
    # best_v8/conv94: (1, 20, 20, 72) - 20x20 kpts
    # best_v8/conv78: (1, 40, 40, 1) - 40x40 conf
    # best_v8/conv59: (1, 80, 80, 64) - 80x80 bbox

    conv_mapping = {
        # 80x80 scale
        'best_v8/conv63': ('80x80', 'conf'),   # (1, 80, 80, 1)
        'best_v8/conv60': ('80x80', 'kpts'),   # (1, 80, 80, 72)
        'best_v8/conv59': ('80x80', 'bbox'),   # (1, 80, 80, 64)
        # 40x40 scale
        'best_v8/conv78': ('40x40', 'conf'),   # (1, 40, 40, 1)
        'best_v8/conv75': ('40x40', 'kpts'),   # (1, 40, 40, 72)
        'best_v8/conv74': ('40x40', 'bbox'),   # (1, 40, 40, 64)
        # 20x20 scale
        'best_v8/conv97': ('20x20', 'conf'),   # (1, 20, 20, 1)
        'best_v8/conv94': ('20x20', 'kpts'),   # (1, 20, 20, 72)
        'best_v8/conv93': ('20x20', 'bbox'),   # (1, 20, 20, 64)
    }

    # Assign outputs to proper scales
    for layer_name, output in raw.items():
        if layer_name in conv_mapping:
            scale, output_type = conv_mapping[layer_name]
            scales[scale][output_type] = output
            print(f"Mapped {layer_name} -> {scale} {output_type}: {output.shape}")

            # Debug: show data range for each output type
            print(f"  Data range: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")
            if output_type == 'conf':
                # Show some actual confidence values
                sample_conf = output[0, :3, :3, 0].flatten()
                print(f"  Sample conf values: {[f'{x:.4f}' for x in sample_conf]}")
        else:
            print(f"Unknown layer: {layer_name}")

    all_predictions = []
    strides = [8, 16, 32]  # YOLOv11 strides for 640x640 input
    scale_names = ['80x80', '40x40', '20x20']

    for scale_idx, scale_name in enumerate(scale_names):
        scale_data = scales[scale_name]

        # Check if we have all required outputs for this scale
        if not all(scale_data[key] is not None for key in ['bbox', 'kpts', 'conf']):
            print(f"Missing outputs for scale {scale_name}, skipping")
            continue

        bbox_out = scale_data['bbox']  # Shape: (1, h, w, 64)
        kpts_out = scale_data['kpts']  # Shape: (1, h, w, 72)
        conf_out = scale_data['conf']  # Shape: (1, h, w, 1)

        # Get grid dimensions
        if len(bbox_out.shape) == 4:
            _, h, w, _ = bbox_out.shape
        else:
            print(f"Unexpected bbox shape for {scale_name}: {bbox_out.shape}")
            continue

        stride = strides[scale_idx]
        print(f"Processing scale {scale_name}: {h}x{w}, stride={stride}")

        # Process each grid cell
        processed_cells = 0
        high_conf_cells = 0
        scale_detections = 0
        max_detections_per_scale = 20  # Limit detections per scale

        for i in range(min(int(h), conf_out.shape[1])):
            for j in range(min(int(w), conf_out.shape[2])):
                if scale_detections >= max_detections_per_scale:
                    break
                # Extract confidence (single value)
                conf_raw = conf_out[0, i, j, 0]

                # Try different confidence interpretations
                conf_sigmoid = 1.0 / (1.0 + np.exp(-conf_raw))  # Sigmoid
                conf_normalized = (conf_raw + 20) / 20  # Normalize from [-20, 0] to [0, 1]
                conf_softmax = np.exp(conf_raw) / (1 + np.exp(conf_raw))  # Alternative

                processed_cells += 1
                if processed_cells <= 5:  # Debug first few cells
                    print(f"  Cell [{i},{j}]: raw={conf_raw:.3f}, sigmoid={conf_sigmoid:.6f}, norm={conf_normalized:.3f}")

                # Try the normalized version since all values are around -15
                conf = max(0, conf_normalized)

                if conf > 0.01:  # Much lower threshold for debugging
                    high_conf_cells += 1
                    if high_conf_cells <= 3:  # Debug first few high conf cells
                        print(f"  HIGH CONF Cell [{i},{j}]: conf={conf:.4f} (raw={conf_raw:.3f})")

                # Use a higher threshold to get fewer but better detections
                if conf < 0.5:  # Higher threshold - only process very confident cells
                    continue

                # Extract bbox prediction (64 channels)
                bbox_raw = bbox_out[0, i, j, :]  # 64 values

                # YOLOv11 bbox format: first 4 channels are [cx, cy, w, h]
                cx_raw, cy_raw, w_raw, h_raw = bbox_raw[:4]

                # Decode bounding box - YOLOv11 uses DFL (Distribution Focal Loss)
                # For simplicity, use the first 4 channels as direct bbox prediction
                cx = (cx_raw + j) * stride  # Center x
                cy = (cy_raw + i) * stride  # Center y
                w = np.exp(w_raw) * stride  # Width
                h = np.exp(h_raw) * stride  # Height

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                # Extract keypoints (72 channels = 24 keypoints * 3 values each)
                kpts_raw = kpts_out[0, i, j, :]  # 72 values
                kpts = np.zeros((24, 3), dtype=np.float32)

                for k in range(24):
                    # Extract x, y, visibility for each keypoint
                    kx_raw = kpts_raw[k * 3]
                    ky_raw = kpts_raw[k * 3 + 1]
                    kv_raw = kpts_raw[k * 3 + 2]

                    # Decode keypoint coordinates
                    kpts[k, 0] = (kx_raw + j) * stride  # x coordinate
                    kpts[k, 1] = (ky_raw + i) * stride  # y coordinate
                    kpts[k, 2] = 1.0 / (1.0 + np.exp(-kv_raw))  # visibility (sigmoid)

                all_predictions.append({
                    "xyxy": np.array([x1, y1, x2, y2], dtype=np.float32),
                    "kpts": kpts,
                    "conf": conf
                })
                scale_detections += 1

        print(f"  Scale {scale_name}: processed {processed_cells} cells, {high_conf_cells} with conf>0.01, {len([p for p in all_predictions if p['conf'] >= 0.25])} above threshold")

    print(f"Found {len(all_predictions)} raw detections before NMS")

    # Apply NMS to remove duplicates
    if len(all_predictions) > 0:
        # Sort by confidence
        all_predictions.sort(key=lambda x: x["conf"], reverse=True)

        # Simple NMS
        keep = []
        for pred in all_predictions:
            should_keep = True

            # Check overlap with kept detections
            for kept in keep:
                # Calculate IoU
                x1 = max(pred["xyxy"][0], kept["xyxy"][0])
                y1 = max(pred["xyxy"][1], kept["xyxy"][1])
                x2 = min(pred["xyxy"][2], kept["xyxy"][2])
                y2 = min(pred["xyxy"][3], kept["xyxy"][3])

                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (pred["xyxy"][2] - pred["xyxy"][0]) * (pred["xyxy"][3] - pred["xyxy"][1])
                    area2 = (kept["xyxy"][2] - kept["xyxy"][0]) * (kept["xyxy"][3] - kept["xyxy"][1])
                    iou = intersection / (area1 + area2 - intersection + 1e-6)

                    if iou > 0.4:  # NMS threshold
                        should_keep = False
                        break

            if should_keep:
                keep.append(pred)
                dets.append({"xyxy": pred["xyxy"], "kpts": pred["kpts"]})

                if len(dets) >= 10:  # Max detections
                    break

    print(f"After NMS: {len(dets)} final detections")
    return dets

# ------------------------
# Behavior head and per-dog state
# ------------------------
class DogState:
    def __init__(self, head):
        self.buf = collections.deque(maxlen=T)  # stores 48-d vectors
        self.ema = None
        self.last_emit = {b: 0.0 for b in BEHAVIORS}
        self.head = head

    def push_and_predict(self, kpts_xy):
        # kpts_xy: [24,2] normalized
        self.buf.append(kpts_xy.reshape(48))
        if len(self.buf) < T:
            return None, 0.0
        x = torch.tensor(np.array(self.buf, dtype=np.float32)[None, ...])
        with torch.no_grad():
            p = torch.softmax(self.head(x), 1)[0].numpy()
        # EMA smoothing
        self.ema = (0.8*self.ema + 0.2*p) if self.ema is not None else p
        return int(self.ema.argmax()), float(self.ema.max())

def dispense_treat(dog_id, behavior):
    # replace with GPIO/serial/i2c call
    print(f"[DISPENSE] to {dog_id} for {behavior}")

# ------------------------
# Camera
# ------------------------
def get_camera():
    # Try OpenCV first - but actually test frame capture
    print("Initializing camera with OpenCV...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Test if we can actually capture a frame
        test_ok, test_frame = cap.read()
        if test_ok and test_frame is not None:
            print("OpenCV camera initialized successfully")

            def grab():
                ok, frame = cap.read()
                if not ok:
                    raise SystemExit("Camera read failed")
                if CAM_ROT_DEG:
                    if CAM_ROT_DEG == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif CAM_ROT_DEG == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif CAM_ROT_DEG == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return frame
            return grab
        else:
            print("OpenCV camera opens but can't capture frames")
            cap.release()

    # Use picamera2 since OpenCV doesn't work
    try:
        print("OpenCV failed, trying picamera2...")
        from picamera2 import Picamera2
        import time

        cam = Picamera2()
        config = cam.create_video_configuration(main={"size": (1280, 720)})
        cam.configure(config)
        cam.start()
        time.sleep(2)  # Let camera initialize

        print("Picamera2 initialized successfully")

        def grab():
            try:
                frame = cam.capture_array()
                # Convert XBGR to BGR if needed
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                return frame
            except Exception as e:
                print(f"Camera capture error: {e}")
                raise SystemExit("Camera read failed")
        return grab

    except Exception as e:
        print(f"Both camera methods failed: {e}")
        raise SystemExit("No camera available")

# ------------------------
# Hailo
# ------------------------
def run_hailo_main_loop():
    import hailo_platform as hpf

    # Load HEF
    hef = hpf.HEF(HEF_PATH)

    # Get stream info
    input_vstream_info = hef.get_input_vstream_infos()[0]
    output_vstream_infos = hef.get_output_vstream_infos()

    # Debug: Print expected input shape
    print(f"Model expects input: {input_vstream_info.name}, shape: {input_vstream_info.shape}")
    # Shape is (H, W, C) for Hailo models
    expected_h, expected_w, expected_c = input_vstream_info.shape
    print(f"Expected size: {expected_w}x{expected_h}x{expected_c}")

    # Use model's expected size for letterboxing
    model_expected_size = expected_w  # Assuming square input

    # load behavior head
    head = torch.jit.load(HEAD_TS, map_location="cpu").eval()

    # camera - initialize once
    grab = get_camera()

    # per-dog state storage
    states = {}

    # Setup device and run inference in context managers
    with hpf.VDevice() as target:
        configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
        network_group = target.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            network_group, quantized=True, format_type=hpf.FormatType.UINT8)
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        with network_group.activate(network_group_params):
            with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:

                try:
                    while True:
                        frame = grab()
                        inp, (dx,dy,scale_inv) = letterbox(frame, model_expected_size)

                        # pose inference
                        # Hailo expects NHWC uint8
                        # Prepare input data
                        input_tensor = np.expand_dims(inp, axis=0)

                        # Debug info removed - inference working correctly

                        input_data = {
                            input_vstream_info.name: input_tensor
                        }

                        # Run inference
                        raw = infer_pipeline.infer(input_data)
                        print("DECODE TEST: About to call decode function")
                        dets = decode_pose_multi_hailo(raw)
                        print(f"DECODE TEST: Got {len(dets)} detections back")

                        # boxes + kpts for assignment and behavior
                        boxes = [d["xyxy"].astype(np.float32) for d in dets]
                        kptss = [d["kpts"].astype(np.float32) for d in dets]

                        # detect markers and assign
                        markers = detect_markers(frame)
                        assigned = assign_markers_to_boxes(markers, boxes, dx, dy, scale_inv)

                        # Debug output every 30 frames to show what's being detected
                        if hasattr(run_hailo_main_loop, 'frame_count'):
                            run_hailo_main_loop.frame_count += 1
                        else:
                            run_hailo_main_loop.frame_count = 1

                        if run_hailo_main_loop.frame_count % 30 == 0:
                            print(f"\n=== Frame {run_hailo_main_loop.frame_count} ===")
                            print(f"Raw detections from Hailo: {len(dets)}")

                            # Show raw detection details
                            for i, det in enumerate(dets):
                                if 'conf' in det:
                                    print(f"  Detection {i}: conf={det.get('conf', 'N/A'):.3f}")
                                else:
                                    print(f"  Detection {i}: {det.keys()}")

                            print(f"Filtered boxes: {len(boxes)} dogs, {len(markers)} markers")
                            if markers:
                                print(f"  Markers: {[(m[0], f'({m[1]:.0f},{m[2]:.0f})') for m in markers]}")
                            if assigned:
                                print(f"  Assigned: {assigned}")
                            else:
                                print("  No assignments made")
                            print("=" * 40)

                        for marker_id, box_i in assigned:
                            if marker_id not in MARKER_TO_DOG: continue
                            dog_id = MARKER_TO_DOG[marker_id]
                            if dog_id not in states:
                                states[dog_id] = DogState(head)

                            kpts_xy = norm_kpts(boxes[box_i], kptss[box_i])
                            beh_idx, conf = states[dog_id].push_and_predict(kpts_xy)

                            if beh_idx is not None and conf > PROB_TH:
                                beh = BEHAVIORS[beh_idx]
                                now = time.time()
                                cooldown = COOLDOWN_S.get(beh, 2.0)

                                if (now - states[dog_id].last_emit.get(beh, 0)) > cooldown:
                                    print(f"{dog_id}: {beh} ({conf:.2f})")
                                    dispense_treat(dog_id, beh)
                                    states[dog_id].last_emit[beh] = now

                except KeyboardInterrupt:
                    print("Stopping...")
                    return

# ------------------------
# Main
# ------------------------
def main():
    # choose pose backend
    use_cpu_debug = bool(DEBUG_CPU_PT)
    if not use_cpu_debug:
        # Run Hailo mode using context managers
        run_hailo_main_loop()
        return

    # CPU debug mode only - initialize camera here
    # load behavior head
    head = torch.jit.load(HEAD_TS, map_location="cpu").eval()

    # camera
    grab = get_camera()

    # per-dog state storage
    states = {}

    try:
        while True:
            frame = grab()
            inp, (dx,dy,scale_inv) = letterbox(frame, IMGSZ)

            # pose inference (CPU debug mode only here)
            dets = decode_pose_multi_cpu_ultra(inp)

            # boxes + kpts for assignment and behavior
            boxes = [d["xyxy"].astype(np.float32) for d in dets]
            kptss = [d["kpts"].astype(np.float32) for d in dets]

            # detect markers and assign
            markers = detect_markers(frame)
            assigned = assign_markers_to_boxes(markers, boxes, dx, dy, scale_inv)

            for marker_id, box_i in assigned:
                if marker_id not in MARKER_TO_DOG: continue
                dog_id = MARKER_TO_DOG[marker_id]
                if dog_id not in states:
                    states[dog_id] = DogState(head)

                kpts_xy = norm_kpts(boxes[box_i], kptss[box_i])
                beh_idx, conf = states[dog_id].push_and_predict(kpts_xy)

                if beh_idx is not None and conf > PROB_TH:
                    beh = BEHAVIORS[beh_idx]
                    now = time.time()
                    cooldown = COOLDOWN_S.get(beh, 2.0)

                    if (now - states[dog_id].last_emit.get(beh, 0)) > cooldown:
                        print(f"{dog_id}: {beh} ({conf:.2f})")
                        dispense_treat(dog_id, beh)
                        states[dog_id].last_emit[beh] = now

    except KeyboardInterrupt:
        print("Stopping...")

if __name__=="__main__":
    main()
