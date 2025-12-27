#!/usr/bin/env python3
"""Check raw Hailo keypoint output values"""

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

print("Analyzing Hailo keypoint output...")
print("-" * 60)

for frame_num in range(10):
    frame = camera.capture_array()
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    with ng.activate(ng_params):
        with hpf.InferVStreams(ng, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            input_tensor = np.expand_dims(frame, axis=0).astype(np.uint8)
            input_name = list(input_vstreams_params.keys())[0]
            outputs = infer_pipeline.infer({input_name: input_tensor})

            # Find keypoints output (72 channels)
            for layer_name, output in outputs.items():
                if output.shape[3] == 72:  # Keypoints layer
                    # Find cell with highest confidence
                    # Find the confidence layer first
                    for conf_name, conf_out in outputs.items():
                        if conf_out.shape[3] == 1 and conf_out.shape[1] == output.shape[1]:
                            h, w = output.shape[1], output.shape[2]
                            conf_map = conf_out[0, :, :, 0]

                            # Apply sigmoid
                            conf_map = 1.0 / (1.0 + np.exp(-np.clip(conf_map, -500, 500)))

                            # Find best cell
                            best_idx = np.unravel_index(np.argmax(conf_map), conf_map.shape)
                            best_conf = conf_map[best_idx]

                            if best_conf > 0.3:
                                i, j = best_idx
                                kpts_raw = output[0, i, j, :]

                                print(f"\nFrame {frame_num} - Detection at cell ({i}, {j}) conf={best_conf:.2f}")
                                print(f"Layer: {layer_name}, shape: {output.shape}")
                                print(f"Raw keypoint values (first 5 keypoints):")
                                for k in range(5):
                                    x_raw = kpts_raw[k*3]
                                    y_raw = kpts_raw[k*3+1]
                                    c_raw = kpts_raw[k*3+2]
                                    c_sig = 1.0 / (1.0 + np.exp(-c_raw))
                                    print(f"  KP{k}: x_raw={x_raw:.3f}, y_raw={y_raw:.3f}, conf_raw={c_raw:.3f} -> conf={c_sig:.2f}")

                                # Show stats
                                x_vals = kpts_raw[0::3]
                                y_vals = kpts_raw[1::3]
                                print(f"\nAll keypoint stats:")
                                print(f"  X: min={x_vals.min():.3f}, max={x_vals.max():.3f}, mean={x_vals.mean():.3f}")
                                print(f"  Y: min={y_vals.min():.3f}, max={y_vals.max():.3f}, mean={y_vals.mean():.3f}")

                            break
                    break

    time.sleep(0.3)

camera.stop()
camera.close()
print("\nDone!")
