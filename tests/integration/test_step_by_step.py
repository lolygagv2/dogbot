#!/usr/bin/env python3
"""
Step-by-step HEF model test
"""

import sys
import os

def test_model_step_by_step(model_path):
    try:
        print(f"🔍 Testing model: {os.path.basename(model_path)}")

        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return False

        print("🔍 Importing HailoRT...")
        from hailo_platform.pyhailort.pyhailort import VDevice
        print("✅ Import successful")

        print("🔍 Creating VDevice...")
        vdevice = VDevice()
        print("✅ VDevice created")

        print("🔍 Loading model...")
        infer_model = vdevice.create_infer_model(model_path)
        print("✅ Model loaded successfully")
        print(f"   Inputs: {infer_model.input_names}")
        print(f"   Outputs: {infer_model.output_names}")

        # Clean up
        del infer_model, vdevice
        print("✅ Cleanup complete")
        return True

    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    models = [
        "ai/models/bestdogyolo5-9_raw.hef",
        "env_new/lib/python3.11/site-packages/hailo_tutorials/hefs/shortcut_net.hef"
    ]

    for model in models:
        success = test_model_step_by_step(model)
        if success:
            print(f"🎉 {model} - SUCCESS!")
        else:
            print(f"💥 {model} - FAILED!")
        print("-" * 50)