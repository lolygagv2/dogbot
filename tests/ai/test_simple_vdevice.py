#!/usr/bin/env python3
"""
Minimal test to identify the issue
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hailo_platform.pyhailort.pyhailort import VDevice

print("Testing VDevice creation...")

try:
    # Method 1: Direct instantiation
    print("Method 1: Direct VDevice()...")
    vdevice = VDevice()
    print(f"  Success! Type: {type(vdevice)}")
except Exception as e:
    print(f"  Failed: {e}")

try:
    # Method 2: With params
    print("\nMethod 2: VDevice with params...")
    params = VDevice.create_params()
    print(f"  Params created: {type(params)}")
    vdevice = VDevice(params)
    print(f"  Success! Type: {type(vdevice)}")

    # Try to create infer model
    print("\nTrying to create InferModel...")
    model_path = "ai/models/bestdogyolo5.hef"
    if os.path.exists(model_path):
        infer_model = vdevice.create_infer_model(model_path)
        print(f"  InferModel created: {type(infer_model)}")

        print("\nTrying to configure...")
        configured = infer_model.configure()
        print(f"  Configured: {type(configured)}")
    else:
        print(f"  Model not found: {model_path}")

except Exception as e:
    print(f"  Failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone")