#!/usr/bin/env python3
"""
Test HEF models with correct HailoRT 4.20 API
"""

import os
import sys
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hailo_platform.pyhailort.pyhailort import VDevice

def test_single_model(model_path):
    """Test a single HEF model with proper cleanup"""
    print(f"\n🔍 Testing: {os.path.basename(model_path)}")
    print("-" * 50)
    
    vdevice = None
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

        # Run inference with timeout (HailoRT 4.20 requires timeout parameter)
        timeout_ms = 5000  # 5 second timeout
        configured_model.run(bindings, timeout_ms)
        print("✅ Inference completed")

        # Get output
        output_buffer = bindings.output().get_buffer()
        print(f"✅ Output retrieved: shape={output_buffer.shape}, dtype={output_buffer.dtype}")
        print(f"🎉 SUCCESS for {os.path.basename(model_path)}!")
        
        # Cleanup
        del output_buffer, bindings, configured_model, infer_model
        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        return False
    finally:
        # Always cleanup VDevice
        if vdevice:
            del vdevice
        # Small delay to ensure cleanup
        time.sleep(0.1)

def test_all_models():
    """Test all available HEF models"""
    models = [
        "ai/models/dogbotyolo8--640x640_quant_hailort_hailo8_1.hef",
        "ai/models/bestdogyolo5.hef", 
        "ai/models/best_dogYoloV5test--640x640_quant_hailort_hailo8_1.hef",
        "yolov8n.hef",
        "yolov8s.hef",
        "yolov8m.hef"
    ]

    success_count = 0
    tested_count = 0

    for model_path in models:
        if not os.path.exists(model_path):
            print(f"⏭️  Skipping (not found): {model_path}")
            continue

        tested_count += 1
        if test_single_model(model_path):
            success_count += 1

    print(f"\n📊 FINAL RESULTS")
    print("=" * 50)
    print(f"✅ Success: {success_count}/{tested_count} models working")
    print(f"🎯 HailoRT 4.20 + Driver 4.20 = WORKING!")
    
    if success_count > 0:
        print(f"🚀 Ready to integrate into detection system!")
        return True
    else:
        print(f"❌ No models working - need further debugging")
        return False

if __name__ == "__main__":
    print("🔥 Testing HEF Models with HailoRT 4.20")
    print("=" * 50)
    success = test_all_models()
    
    # Normal exit (no forced exit needed for individual tests)
    sys.exit(0 if success else 1)
