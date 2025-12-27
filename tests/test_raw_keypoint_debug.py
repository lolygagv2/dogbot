#!/usr/bin/env python3
"""Debug Hailo outputs to understand keypoint format"""

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
print("HAILO OUTPUT DEBUG")
print("=" * 70)

for frame_num in range(30):
    frame = camera.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    with ng.activate(ng_params):
        with hpf.InferVStreams(ng, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            input_tensor = np.expand_dims(frame, axis=0).astype(np.uint8)
            input_name = list(input_vstreams_params.keys())[0]
            outputs = infer_pipeline.infer({input_name: input_tensor})

            if frame_num == 0:
                print(f"\nAll output layers:")
                for layer_name, output in outputs.items():
                    print(f"  {layer_name}: shape={output.shape}, dtype={output.dtype}")
                print()

            # Group by scale
            scales = {}
            for layer_name, output in outputs.items():
                if len(output.shape) == 4:
                    h, w = output.shape[1], output.shape[2]
                    ch = output.shape[3]
                    key = f"{h}x{w}"
                    if key not in scales:
                        scales[key] = {}
                    scales[key][ch] = (layer_name, output)

            # Find any confidence > 0.2
            found = False
            for scale_name, layers in scales.items():
                if 1 in layers:  # conf layer
                    conf_name, conf_out = layers[1]
                    conf_map_raw = conf_out[0, :, :, 0]
                    conf_map = 1.0 / (1.0 + np.exp(-np.clip(conf_map_raw, -500, 500)))

                    max_conf = conf_map.max()
                    if max_conf > 0.2:
                        found = True
                        best_idx = np.unravel_index(np.argmax(conf_map), conf_map.shape)
                        print(f"Frame {frame_num} - Scale {scale_name}: max_conf={max_conf:.3f} at {best_idx}")

                        # If we have keypoints layer
                        if 72 in layers:
                            kpts_name, kpts_out = layers[72]
                            i, j = best_idx
                            kpts_raw = kpts_out[0, i, j, :]

                            # Determine stride
                            h, w = kpts_out.shape[1], kpts_out.shape[2]
                            if (h, w) == (80, 80):
                                stride = 8
                            elif (h, w) == (40, 40):
                                stride = 16
                            elif (h, w) == (20, 20):
                                stride = 32
                            else:
                                stride = 640 // h

                            x_raws = kpts_raw[0::3]
                            y_raws = kpts_raw[1::3]

                            print(f"  Raw keypoints range: x=[{x_raws.min():.2f}, {x_raws.max():.2f}], y=[{y_raws.min():.2f}, {y_raws.max():.2f}]")

                            # Test both approaches
                            approach_a_x = (x_raws + j) * stride
                            approach_a_y = (y_raws + i) * stride
                            approach_b_x = x_raws * stride
                            approach_b_y = y_raws * stride

                            print(f"  Approach A (raw+cell)*stride: x=[{approach_a_x.min():.0f}, {approach_a_x.max():.0f}], y=[{approach_a_y.min():.0f}, {approach_a_y.max():.0f}]")
                            print(f"  Approach B raw*stride: x=[{approach_b_x.min():.0f}, {approach_b_x.max():.0f}], y=[{approach_b_y.min():.0f}, {approach_b_y.max():.0f}]")
                            print(f"  Cell (i,j)=({i},{j}), stride={stride}, cell_center=({(j+0.5)*stride:.0f}, {(i+0.5)*stride:.0f})")
                            print()

            if not found:
                # Print max conf for each scale
                if frame_num % 10 == 0:
                    print(f"Frame {frame_num} - No detection (conf < 0.2)")
                    for scale_name, layers in scales.items():
                        if 1 in layers:
                            conf_name, conf_out = layers[1]
                            conf_map_raw = conf_out[0, :, :, 0]
                            conf_map = 1.0 / (1.0 + np.exp(-np.clip(conf_map_raw, -500, 500)))
                            print(f"  {scale_name}: max_conf={conf_map.max():.4f}")

    time.sleep(0.1)

camera.stop()
camera.close()
print("\nDone!")
