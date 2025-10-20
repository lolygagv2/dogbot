#!/usr/bin/env python3
"""
Debug InferModel creation
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hailo_platform.pyhailort.pyhailort import VDevice, HEF

model_path = "ai/models/bestdogyolo5.hef"

print("Step 1: Create VDevice...")
vdevice = VDevice()
print(f"  Success: {type(vdevice)}")

print("\nStep 2: Load HEF separately...")
hef = HEF(model_path)
print(f"  Success: {hef.get_network_group_names()}")

print("\nStep 3: Check VDevice methods...")
print(f"  Has create_infer_model: {hasattr(vdevice, 'create_infer_model')}")

print("\nStep 4: Check create_infer_model signature...")
import inspect
if hasattr(vdevice, 'create_infer_model'):
    sig = inspect.signature(vdevice.create_infer_model)
    print(f"  Signature: {sig}")

print("\nStep 5: Try creating InferModel with just path...")
try:
    # Maybe it just needs the path
    print(f"  Calling create_infer_model('{model_path}')...")
    infer_model = vdevice.create_infer_model(model_path)
    print(f"  Success! Type: {type(infer_model)}")
except Exception as e:
    print(f"  Failed: {e}")
    import traceback
    traceback.print_exc()

print("\nStep 6: Try other arguments...")
try:
    # Maybe it needs the HEF object
    print(f"  Calling create_infer_model(hef)...")
    infer_model = vdevice.create_infer_model(hef)
    print(f"  Success! Type: {type(infer_model)}")
except Exception as e:
    print(f"  Failed: {e}")

print("\nDone")