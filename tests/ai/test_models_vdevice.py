#!/usr/bin/env python3
"""
Test HEF models using VDevice.create_infer_model
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hailo_platform.pyhailort.pyhailort import VDevice, FormatType

def test_model(model_path):
    """Test model with VDevice.create_infer_model"""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(model_path).name}")
    print('-'*60)

    try:
        # Create VDevice (without params to avoid device conflicts)
        print("Creating VDevice...")
        vdevice = VDevice()  # Simple instantiation works best
        print("✅ VDevice created")

        # Create InferModel from VDevice
        print("Creating InferModel...")
        infer_model = vdevice.create_infer_model(str(model_path))
        print("✅ InferModel created")

        # Configure the model
        print("Configuring model...")
        configured_model = infer_model.configure()
        print("✅ Model configured")

        # Create bindings
        print("Creating bindings...")
        bindings = configured_model.create_bindings()

        print(f"  Inputs: {list(bindings.input.keys())}")
        print(f"  Outputs: {list(bindings.output.keys())}")

        # Get input info and create test data
        input_name = list(bindings.input.keys())[0]
        input_binding = bindings.input[input_name]
        input_shape = input_binding.shape()
        print(f"  Input shape: {input_shape}")

        # Create dummy input
        dummy_input = np.zeros(input_shape, dtype=np.uint8)
        print(f"  Test input shape: {dummy_input.shape}")

        # Set input
        bindings.input[input_name] = dummy_input

        # Run inference
        print("\nRunning inference...")
        configured_model.run(bindings)

        print("✅ INFERENCE SUCCESSFUL!")

        # Get output info
        for name, output in bindings.output.items():
            print(f"  Output '{name}': shape {output.shape}")

        return True

    except Exception as e:
        error_msg = str(e)
        print(f"❌ Failed: {error_msg}")

        if "buffer as view" in error_msg:
            print("\n⚠️  Buffer/view configuration issue detected")
            print("This is the known HailoRT issue with this model")

        return False

def main():
    models = [
        "ai/models/bestdogyolo5.hef",
        "ai/models/best_dogYoloV5test--640x640_quant_hailort_hailo8_1.hef",
        "ai/models/dogbotyolo8--640x640_quant_hailort_hailo8_1.hef"
    ]

    print("="*60)
    print("HAILO MODEL TEST - VDEVICE APPROACH")
    print("="*60)

    results = {}
    for model in models:
        if Path(model).exists():
            results[Path(model).name] = test_model(model)
        else:
            print(f"Model not found: {model}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    working = [k for k,v in results.items() if v]
    buffer_issue = [k for k,v in results.items() if not v]

    if working:
        print(f"\n✅ WORKING MODELS ({len(working)}):")
        for m in working:
            print(f"  - {m}")

        print(f"\n🎯 RECOMMENDED: Use {working[0]}")
        print(f"   Full path: ai/models/{working[0]}")

        # Save result
        with open("working_hailo_model.txt", "w") as f:
            f.write(f"ai/models/{working[0]}\n")
        print("\n✅ Working model saved to working_hailo_model.txt")

    else:
        print("\n❌ No models passed inference test")

        if buffer_issue:
            print(f"\n⚠️  All {len(buffer_issue)} models have the buffer/view issue:")
            for m in buffer_issue:
                print(f"  - {m}")

            print("\nThis appears to be a consistent issue across all models.")
            print("The models load and configure correctly, but fail at inference.")

    print(f"\nFinal result: {len(working)}/{len(results)} models working")

if __name__ == "__main__":
    main()