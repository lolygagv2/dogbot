#!/usr/bin/env python3
"""
Test if we can configure and run inference
"""

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hailo_platform.pyhailort.pyhailort import VDevice

model_path = "ai/models/bestdogyolo5.hef"

print("Creating VDevice...")
vdevice = VDevice()

print("Creating InferModel...")
infer_model = vdevice.create_infer_model(model_path)
print(f"  Success: {type(infer_model)}")

print("\nChecking InferModel properties...")
print(f"  input_names: {infer_model.input_names}")
print(f"  output_names: {infer_model.output_names}")

print("\nConfiguring...")
try:
    configured_model = infer_model.configure()
    print(f"  Success! Type: {type(configured_model)}")

    print("\nCreating bindings...")
    bindings = configured_model.create_bindings()
    print(f"  Inputs: {list(bindings.input.keys())}")
    print(f"  Outputs: {list(bindings.output.keys())}")

    # Get input shape
    input_name = list(bindings.input.keys())[0]
    input_binding = bindings.input[input_name]
    shape = input_binding.shape()
    print(f"  Input shape: {shape}")

    # Create test input
    dummy_input = np.zeros(shape, dtype=np.uint8)
    print(f"  Test input created: {dummy_input.shape}")

    # Set input
    bindings.input[input_name] = dummy_input

    print("\nRunning inference...")
    configured_model.run(bindings)

    print("✅ INFERENCE SUCCESSFUL!")

    # Check outputs
    for name, output in bindings.output.items():
        print(f"  Output '{name}': {output.shape}")

except Exception as e:
    print(f"  Failed: {e}")
    if "buffer as view" in str(e):
        print("\n⚠️  This is the buffer/view issue")
    import traceback
    traceback.print_exc()