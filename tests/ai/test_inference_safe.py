#!/usr/bin/env python3
"""
Safe inference test with proper error handling
"""

import sys
import os
import numpy as np

def test_inference_safe(model_path):
    try:
        print(f"🔍 Testing inference: {os.path.basename(model_path)}")

        from hailo_platform.pyhailort.pyhailort import VDevice

        # Create VDevice and load model
        vdevice = VDevice()
        infer_model = vdevice.create_infer_model(model_path)
        print(f"✅ Model loaded: {infer_model.input_names} -> {infer_model.output_names}")

        # Configure model
        configured_model = infer_model.configure()
        print("✅ Model configured")

        # Create bindings
        bindings = configured_model.create_bindings()
        print("✅ Bindings created")

        # Use correct input shape for shortcut_net (224x224x3)
        dummy_input = np.zeros((224, 224, 3), dtype=np.uint8)
        print(f"✅ Created input: {dummy_input.shape}")

        # Set input buffer
        bindings.input().set_buffer(dummy_input)
        print("✅ Input buffer set")

        # Run inference with timeout
        print("🔍 Running inference...")
        # HailoRT 4.21 API expects list of bindings
        configured_model.run([bindings], 5000)  # 5 second timeout
        print("✅ Inference completed")

        # Get output
        output_buffer = bindings.output().get_buffer()
        print(f"✅ Output: shape={output_buffer.shape}, dtype={output_buffer.dtype}")

        # Show some output stats
        print(f"   Output range: [{output_buffer.min():.6f}, {output_buffer.max():.6f}]")

        print(f"🎉 SUCCESS! Model {os.path.basename(model_path)} working with HailoRT 4.21")

        # Cleanup
        del output_buffer, bindings, configured_model, infer_model, vdevice
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    models = [
        "env_new/lib/python3.11/site-packages/hailo_tutorials/hefs/shortcut_net.hef"
    ]

    success_count = 0
    for model in models:
        if os.path.exists(model):
            if test_inference_safe(model):
                success_count += 1
            print("-" * 60)

    print(f"\n🎯 FINAL RESULT: {success_count}/{len(models)} models working!")

    if success_count > 0:
        print("🚀 HailoRT 4.21 is working - ready to integrate!")
    else:
        print("❌ Still having issues - need more debugging")