#!/usr/bin/env python3
"""
Test HEF model with segfault workaround
"""

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hailo_platform.pyhailort.pyhailort import VDevice

def test_hef_models():
    """Test all HEF models with proper cleanup"""
    models = [
        "ai/models/dogbotyolo8--640x640_quant_hailort_hailo8_1.hef",
        "yolov8n.hef",
        "yolov8s.hef"
    ]

    success_count = 0

    for model_path in models:
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            continue

        print(f"\n🔍 Testing: {os.path.basename(model_path)}")
        print("-" * 50)

        try:
            # Create VDevice and load model
            vdevice = VDevice()
            infer_model = vdevice.create_infer_model(model_path)
            print(f"✅ Model loaded")
            print(f"   Inputs: {infer_model.input_names}")
            print(f"   Outputs: {infer_model.output_names}")

            # Configure model
            configured_model = infer_model.configure()
            print("✅ Model configured")

            # Create bindings
            bindings = configured_model.create_bindings()
            print("✅ Bindings created")

            # Create dummy input (640x640x3 for YOLO models)
            dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
            bindings.input().set_buffer(dummy_input)
            print("✅ Input buffer set")

            # Run inference
            configured_model.run(bindings)
            print("✅ Inference completed")

            # Get output
            output_buffer = bindings.output().get_buffer()
            print(f"✅ Output retrieved: shape={output_buffer.shape}, dtype={output_buffer.dtype}")

            success_count += 1
            print(f"🎉 SUCCESS for {os.path.basename(model_path)}!")

            # Clean up this iteration
            del output_buffer, bindings, configured_model, infer_model, vdevice

        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n📊 Results: {success_count}/{len(models)} models working")

    # Force exit to avoid cleanup segfault
    os._exit(0 if success_count > 0 else 1)

if __name__ == "__main__":
    print("Testing HEF Models with HailoRT 4.20")
    print("=" * 50)
    test_hef_models()
