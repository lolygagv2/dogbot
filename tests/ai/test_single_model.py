#!/usr/bin/env python3
"""
Test single YOLOv8 model with NMS (single output)
"""

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hailo_platform.pyhailort.pyhailort import VDevice

def test_yolov8_model():
    """Test YOLOv8 model with single output"""
    model_path = "ai/models/dogbotyolo8--640x640_quant_hailort_hailo8_1.hef"

    print(f"Testing: {os.path.basename(model_path)}")
    print("-" * 50)

    try:
        # Create VDevice and InferModel
        vdevice = VDevice()
        infer_model = vdevice.create_infer_model(model_path)

        print(f"Inputs: {infer_model.input_names}")
        print(f"Outputs: {infer_model.output_names}")
        print(f"Output count: {len(infer_model.output_names)}")

        # Configure model
        configured_model = infer_model.configure()
        print("✅ Model configured")

        # Create bindings
        bindings = configured_model.create_bindings()
        print("✅ Bindings created")

        # Get input info
        input_obj = infer_model.input()
        input_name = infer_model.input_names[0]
        input_shape = input_obj[input_name].shape()
        print(f"Input shape: {input_shape}")

        # Create test input
        dummy_input = np.zeros(input_shape, dtype=np.uint8)
        print(f"Created dummy input: {dummy_input.shape}")

        # Set input buffer
        bindings.input().set_buffer(dummy_input)
        print("✅ Input buffer set")

        # Run inference
        print("Running inference...")
        configured_model.run(bindings)
        print("✅ INFERENCE SUCCESSFUL!")

        # Get output
        output_buffer = bindings.output().get_buffer()
        print(f"Output shape: {output_buffer.shape}")
        print(f"Output dtype: {output_buffer.dtype}")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        if "buffer as view" in str(e):
            print("⚠️  Buffer/view issue")
        return False
    finally:
        # Force cleanup to avoid segfault
        os._exit(0 if locals().get('output_buffer') is not None else 1)

if __name__ == "__main__":
    print("Testing YOLOv8 model with fixed NumPy")
    print("="*50)

    test_yolov8_model()  # Will exit directly from function