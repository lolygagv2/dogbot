#!/usr/bin/env python3
"""
Check what the model is ACTUALLY outputting
No detection logic, just raw output inspection
"""

import json
import numpy as np
import cv2
import time

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

print("\n" + "="*60)
print("🔍 Raw Model Output Inspection")
print("="*60)
print(f"[CONFIG] Model: {HEF_PATH}")
print(f"[CONFIG] Input size: {IMGSZ_W}x{IMGSZ_H}")

def inspect_outputs(outputs):
    """Detailed output inspection"""

    print(f"\n[INSPECTION] Total outputs: {len(outputs)}")
    print("-" * 60)

    # Check if it's NMS post-processed (single 2D output)
    if len(outputs) == 1 and outputs[0].ndim == 2:
        out = outputs[0]
        print(f"[NMS OUTPUT DETECTED]")
        print(f"  Shape: {out.shape}")
        print(f"  Dtype: {out.dtype}")
        print(f"  Range: [{out.min():.2f}, {out.max():.2f}]")
        print(f"  Number of detections: {out.shape[0]}")
        if out.shape[0] > 0:
            print(f"  Detection format (first row): {out[0][:10]}")  # First 10 values
            print(f"  Expected format: [x1, y1, x2, y2, score, class, ...keypoints...]")
        return

    # Raw multi-scale outputs
    print("[RAW OUTPUTS - Multi-scale]")

    # Group by shape
    by_shape = {}
    for i, out in enumerate(outputs):
        key = str(out.shape)
        if key not in by_shape:
            by_shape[key] = []
        by_shape[key].append((i, out))

    for shape_str, outputs_list in by_shape.items():
        print(f"\n  Shape group: {shape_str}")
        for idx, out in outputs_list:
            print(f"    Output {idx}:")
            print(f"      Dtype: {out.dtype}")
            print(f"      Range: [{out.min():.2f}, {out.max():.2f}]")
            print(f"      Mean: {out.mean():.2f}, Std: {out.std():.2f}")

            # Check data distribution
            flat = out.flatten()
            print(f"      Percentiles: 1%={np.percentile(flat, 1):.1f}, "
                  f"50%={np.percentile(flat, 50):.1f}, "
                  f"99%={np.percentile(flat, 99):.1f}")

            # Identify output type by channels
            if out.ndim == 4:
                channels = out.shape[3]
                if channels == 1:
                    print(f"      Type: Objectness (1 channel)")
                elif channels == 4:
                    print(f"      Type: Boxes (4 channels)")
                elif channels == 64:
                    print(f"      Type: Boxes? (64 channels - unexpected)")
                elif channels == 72:
                    print(f"      Type: Keypoints (72 = 24 points * 3)")
                else:
                    print(f"      Type: Unknown ({channels} channels)")

def run_inference(img):
    """Run single inference"""

    # Prepare image
    scale = min(IMGSZ_W / img.shape[1], IMGSZ_H / img.shape[0])
    new_w = int(img.shape[1] * scale)
    new_h = int(img.shape[0] * scale)

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

    print(f"\n[INPUT] Shape: {input_data.shape}, Dtype: {input_data.dtype}")

    # Run inference
    with hpf.VDevice() as target:
        hef = hpf.HEF(HEF_PATH)

        # Get network info
        print(f"[HEF INFO] Number of networks: {len(hef.get_network_group_names())}")
        for name in hef.get_network_group_names():
            print(f"  Network: {name}")

        configure_params = hpf.ConfigureParams.create_from_hef(
            hef, interface=hpf.HailoStreamInterface.PCIe)

        network_group = target.configure(hef, configure_params)[0]

        # Get stream info
        input_vstream_infos = network_group.get_input_vstream_infos()
        output_vstream_infos = network_group.get_output_vstream_infos()

        print(f"\n[STREAMS]")
        print(f"  Inputs: {len(input_vstream_infos)}")
        for info in input_vstream_infos:
            print(f"    {info.name}: {info.shape}, {info.format}")

        print(f"  Outputs: {len(output_vstream_infos)}")
        for info in output_vstream_infos:
            print(f"    {info.name}: {info.shape}, {info.format}")

        network_group_params = network_group.create_params()

        input_vstreams_params = hpf.InputVStreamParams.make(network_group)
        output_vstreams_params = hpf.OutputVStreamParams.make(network_group)

        with network_group.activate(network_group_params):
            with hpf.InferVStreams(network_group, input_vstreams_params,
                                  output_vstreams_params) as infer_pipeline:

                input_dict = {info.name: input_data for info in input_vstream_infos}

                outputs = infer_pipeline.infer(input_dict)
                output_data = {name: data.copy() for name, data in outputs.items()}

    return list(output_data.values())

def main():
    # Get camera
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

    print("\n[INFO] Capturing frame...")

    # Capture frame
    if cam_type == "picamera2":
        frame = cam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()

    # Save for reference
    cv2.imwrite("inspect_frame.jpg", frame)
    print("[INFO] Saved frame to inspect_frame.jpg")

    # Run inference
    print("\n[INFO] Running inference...")
    outputs = run_inference(frame)

    # Inspect outputs
    inspect_outputs(outputs)

    # Cleanup
    if cam_type == "picamera2":
        cam.stop()
    else:
        cam.release()

    print("\n" + "="*60)
    print("INSPECTION COMPLETE")
    print("="*60)
    print("\nKey findings to check:")
    print("1. Is it NMS post-processed (1 output) or raw (9 outputs)?")
    print("2. What's the actual dtype? (uint8, uint16, float32?)")
    print("3. What's the value range? (0-255 for uint8, different for uint16)")
    print("4. Are the channel counts correct? (1, 64, 72)")

if __name__ == "__main__":
    main()