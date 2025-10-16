#!/usr/bin/env python3
"""
Test uncalibrated uint8 model (no calibration dataset)
Should have different/better quantization ranges
"""

import json
import time
import numpy as np
import cv2
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

print("\n" + "="*60)
print("🔍 Testing UNCALIBRATED Model (No calibration dataset)")
print("="*60)
print(f"[CONFIG] Model: {HEF_PATH}")
print(f"[CONFIG] Should have better uint8 quantization ranges")

def inspect_uncalibrated_outputs(outputs):
    """Check what the uncalibrated quantization looks like"""

    print(f"\n[UNCALIBRATED INSPECTION]")
    print(f"Total outputs: {len(outputs)}")
    print("-" * 60)

    for i, out in enumerate(outputs[:9]):  # Show first 9
        print(f"Output {i}:")
        print(f"  Shape: {out.shape}")
        print(f"  Dtype: {out.dtype}")
        print(f"  Range: [{out.min():.1f}, {out.max():.1f}]")

        if out.ndim == 4:
            channels = out.shape[3]
            h, w = out.shape[1:3]

            # Get distribution
            flat = out.flatten()
            p1 = np.percentile(flat, 1)
            p50 = np.percentile(flat, 50)
            p99 = np.percentile(flat, 99)

            print(f"  Percentiles: 1%={p1:.1f}, 50%={p50:.1f}, 99%={p99:.1f}")

            if channels == 1:
                print(f"  Type: Objectness ({h}x{w})")

                # Check if this looks like standard quantization
                if abs(p50 - 128) < 20:  # Standard uint8 center
                    print(f"  ✓ Standard quantization (centered at 128)")
                    suggested_thresh = p99 + 5
                    print(f"  Suggested threshold: {suggested_thresh:.0f}")
                else:
                    print(f"  ⚠ Non-standard quantization")

            elif channels == 64:
                print(f"  Type: Boxes ({h}x{w})")
            elif channels == 72:
                print(f"  Type: Keypoints ({h}x{w})")

    print("-" * 60)

def run_inference_inspect(img):
    """Run inference just for inspection"""

    # Standard preprocessing
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

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    input_data = rgb.astype(np.uint8)
    input_data = np.transpose(input_data, (2, 0, 1))
    input_data = np.expand_dims(input_data, 0)

    # Run inference
    with hpf.VDevice() as target:
        hef = hpf.HEF(HEF_PATH)
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

    print("\n[INFO] Capturing test frame (NO DOGS for baseline)...")

    # Capture baseline frame
    if cam_type == "picamera2":
        frame = cam.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        ret, frame = cam.read()

    cv2.imwrite("uncalibrated_test_frame.jpg", frame)
    print("[INFO] Saved to uncalibrated_test_frame.jpg")

    # Run inference
    print("\n[INFO] Running inference on uncalibrated model...")
    outputs = run_inference_inspect(frame)

    # Inspect outputs
    inspect_uncalibrated_outputs(outputs)

    # Cleanup
    if cam_type == "picamera2":
        cam.stop()
    else:
        cam.release()

    print("\n" + "="*60)
    print("UNCALIBRATED MODEL ANALYSIS COMPLETE")
    print("="*60)
    print("\nKey comparisons:")
    print("OLD (calibrated):   Objectness range 182-241, centered ~200+")
    print("NEW (uncalibrated): Check above ranges")
    print("\nIf new ranges are centered around 128, use standard uint8 quantization!")

if __name__ == "__main__":
    main()