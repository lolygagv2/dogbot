#!/usr/bin/env python3
"""Check raw Hailo keypoint output values to determine correct decoding"""

import sys
sys.path.insert(0, '/home/morgan/dogbot')

import numpy as np
import cv2
import time

try:
    from picamera2 import Picamera2
    import hailo_platform as hpf
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)

# Load model
HEF_PATH = "/home/morgan/dogbot/ai/models/dogpose_14.hef"
hef = hpf.HEF(HEF_PATH)

vdevice = hpf.VDevice()
params = hpf.ConfigureParams.create_from_hef(hef=hef, interface=hpf.HailoStreamInterface.PCIe)
network_groups = vdevice.configure(hef, params)
ng = network_groups[0]

input_info = hef.get_input_vstream_infos()[0]
output_infos = hef.get_output_vstream_infos()

ng_params = ng.create_params()
input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(ng, quantized=True, format_type=hpf.FormatType.UINT8)
output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(ng, quantized=False, format_type=hpf.FormatType.FLOAT32)

# Camera
camera = Picamera2()
config = camera.create_preview_configuration(main={"size": (640, 640), "format": "RGB888"})
camera.configure(config)
camera.start()
time.sleep(1)

print("=" * 70)
print("RAW KEYPOINT VALUE ANALYSIS")
print("=" * 70)
print("Testing keypoint decoding approaches:")
print("  Approach A (my code): kpt = (raw + cell_pos) * stride")
print("  Approach B (run_pi):  kpt = raw * stride")
print("-" * 70)

for frame_num in range(60):
    frame = camera.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    with ng.activate(ng_params):
        with hpf.InferVStreams(ng, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            input_tensor = np.expand_dims(frame, axis=0).astype(np.uint8)
            input_name = list(input_vstreams_params.keys())[0]
            outputs = infer_pipeline.infer({input_name: input_tensor})

            # Find outputs
            conf_layer = None
            kpts_layer = None

            for layer_name, output in outputs.items():
                if len(output.shape) == 4:
                    if output.shape[3] == 1:
                        conf_layer = (layer_name, output)
                    elif output.shape[3] == 72:
                        kpts_layer = (layer_name, output)

            if conf_layer and kpts_layer:
                conf_name, conf_out = conf_layer
                kpts_name, kpts_out = kpts_layer

                h, w = kpts_out.shape[1], kpts_out.shape[2]

                # Determine stride based on output size
                if (h, w) == (80, 80):
                    stride = 8
                elif (h, w) == (40, 40):
                    stride = 16
                elif (h, w) == (20, 20):
                    stride = 32
                else:
                    continue

                # Apply sigmoid to confidence
                conf_map = conf_out[0, :, :, 0]
                conf_map = 1.0 / (1.0 + np.exp(-np.clip(conf_map, -500, 500)))

                # Find best cell
                best_idx = np.unravel_index(np.argmax(conf_map), conf_map.shape)
                best_conf = conf_map[best_idx]

                if best_conf > 0.25:
                    i, j = best_idx
                    kpts_raw = kpts_out[0, i, j, :]

                    print(f"\nFrame {frame_num} - Detection at cell ({i}, {j}), stride={stride}")
                    print(f"  Confidence: {best_conf:.2f}")
                    print(f"  Grid size: {h}x{w}")
                    print()

                    # Show raw values
                    print(f"  RAW keypoint values (first 5 of 24):")
                    for k in range(5):
                        x_raw = kpts_raw[k*3]
                        y_raw = kpts_raw[k*3+1]
                        c_raw = kpts_raw[k*3+2]
                        print(f"    KP{k}: x_raw={x_raw:8.3f}, y_raw={y_raw:8.3f}, conf_raw={c_raw:6.3f}")

                    # Show all keypoint x/y ranges
                    x_raws = kpts_raw[0::3]
                    y_raws = kpts_raw[1::3]
                    print(f"\n  All 24 keypoints RAW range:")
                    print(f"    X: min={x_raws.min():.3f}, max={x_raws.max():.3f}, mean={x_raws.mean():.3f}")
                    print(f"    Y: min={y_raws.min():.3f}, max={y_raws.max():.3f}, mean={y_raws.mean():.3f}")

                    # Compare decoding approaches
                    print(f"\n  APPROACH A (my code): kpt = (raw + cell_pos) * stride")
                    approach_a_x = (x_raws + j) * stride
                    approach_a_y = (y_raws + i) * stride
                    print(f"    X range: [{approach_a_x.min():.1f}, {approach_a_x.max():.1f}]")
                    print(f"    Y range: [{approach_a_y.min():.1f}, {approach_a_y.max():.1f}]")

                    print(f"\n  APPROACH B (run_pi_1024): kpt = raw * stride")
                    approach_b_x = x_raws * stride
                    approach_b_y = y_raws * stride
                    print(f"    X range: [{approach_b_x.min():.1f}, {approach_b_x.max():.1f}]")
                    print(f"    Y range: [{approach_b_y.min():.1f}, {approach_b_y.max():.1f}]")

                    # Check which makes sense for 640x640 frame
                    print(f"\n  ANALYSIS (frame is 640x640):")
                    a_in_frame = (approach_a_x.min() >= 0 and approach_a_x.max() <= 640 and
                                  approach_a_y.min() >= 0 and approach_a_y.max() <= 640)
                    b_in_frame = (approach_b_x.min() >= 0 and approach_b_x.max() <= 640 and
                                  approach_b_y.min() >= 0 and approach_b_y.max() <= 640)

                    print(f"    Approach A in frame bounds: {a_in_frame}")
                    print(f"    Approach B in frame bounds: {b_in_frame}")

                    # Show where keypoints land relative to cell center
                    cell_center_x = (j + 0.5) * stride
                    cell_center_y = (i + 0.5) * stride
                    print(f"\n  Cell center: ({cell_center_x:.1f}, {cell_center_y:.1f})")

                    # Exit after first good detection
                    break

    time.sleep(0.2)

camera.stop()
camera.close()
print("\nDone!")
