#!/usr/bin/env python3
"""
Test single HEF model safely
"""

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_safe(model_path):
    """Test model with forced cleanup"""
    try:
        from hailo_platform.pyhailort.pyhailort import VDevice
        
        print(f"Testing: {os.path.basename(model_path)}")
        
        # Create and test
        vdevice = VDevice()
        infer_model = vdevice.create_infer_model(model_path)
        print(f"✅ Model loaded: {infer_model.input_names} -> {infer_model.output_names}")
        
        configured_model = infer_model.configure()
        print("✅ Model configured")
        
        bindings = configured_model.create_bindings()
        print("✅ Bindings created")
        
        # Simplified input
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        bindings.input().set_buffer(dummy_input)
        print("✅ Input set")
        
        # Run with timeout
        configured_model.run(bindings, 5000)
        print("✅ Inference complete")
        
        output = bindings.output().get_buffer()
        print(f"✅ Output: {output.shape}, {output.dtype}")
        print("🎉 SUCCESS!")
        
        # Force exit to avoid cleanup issues
        os._exit(0)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        os._exit(1)

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "ai/models/dogbotyolo8--640x640_quant_hailort_hailo8_1.hef"
    test_model_safe(model)
