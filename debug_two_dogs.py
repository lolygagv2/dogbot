#!/usr/bin/env python3
"""
Debug the two-dog frame specifically
"""

import sys
import os
import numpy as np
import cv2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import hailo_platform as hpf

def debug_two_dog_frame():
    """Debug exactly what's in the 2-dog frame"""

    frame_path = "simple_test_results_20251016_174201/frame_00480_174306.jpg"
    model_path = "ai/models/dogdetector_14.hef"

    frame = cv2.imread(frame_path)
    print(f"📷 Testing 2-dog frame: {frame.shape}")

    # Initialize model directly (like our working debug script)
    vdevice = hpf.VDevice()
    hef_model = hpf.HEF(model_path)

    configure_params = hpf.ConfigureParams.create_from_hef(hef_model, interface=hpf.HailoStreamInterface.PCIe)
    network_groups = vdevice.configure(hef_model, configure_params)
    network_group = network_groups[0]
    network_group_params = network_group.create_params()

    # Resize and process
    frame_640 = cv2.resize(frame, (640, 640))

    print(f"📐 Original: {frame.shape} -> Resized: {frame_640.shape}")

    # Test BGR (best performing)
    with network_group.activate(network_group_params):
        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.UINT8)
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

        with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
            input_tensor = np.expand_dims(frame_640, axis=0).astype(np.uint8)
            input_name = list(input_vstreams_params.keys())[0]
            input_data = {input_name: input_tensor}

            output_data = infer_pipeline.infer(input_data)

            for output_name, output_tensor in output_data.items():
                print(f"\n🔍 Raw output: {output_name}")

                # Navigate the nested structure
                if isinstance(output_tensor, list) and len(output_tensor) > 0:
                    if isinstance(output_tensor[0], list) and len(output_tensor[0]) > 0:
                        actual_data = output_tensor[0][0]

                        print(f"📊 Detection data shape: {actual_data.shape}")
                        print(f"📊 Data type: {actual_data.dtype}")

                        if len(actual_data.shape) == 2 and actual_data.shape[1] == 5:
                            num_dets = actual_data.shape[0]
                            print(f"🎯 Found {num_dets} raw detection(s)")

                            for i in range(num_dets):
                                x1, y1, x2, y2, conf = actual_data[i]
                                print(f"  Det {i}: x1={x1:.3f}, y1={y1:.3f}, x2={x2:.3f}, y2={y2:.3f}, conf={conf:.3f}")

                                # Check if these are normalized coordinates (0-1) or pixel coordinates
                                if x2 <= 1.0 and y2 <= 1.0:
                                    print(f"    -> Normalized coordinates (0-1 range)")
                                    # Convert to pixel coordinates
                                    x1_pix = int(x1 * 640)
                                    y1_pix = int(y1 * 640)
                                    x2_pix = int(x2 * 640)
                                    y2_pix = int(y2 * 640)
                                    print(f"    -> Pixel coords: ({x1_pix},{y1_pix},{x2_pix},{y2_pix})")
                                else:
                                    print(f"    -> Already pixel coordinates")

                                # Filter by confidence
                                if conf > 0.1:
                                    print(f"    ✅ Above 0.1 threshold")
                                else:
                                    print(f"    ❌ Below 0.1 threshold")

if __name__ == "__main__":
    debug_two_dog_frame()