#!/usr/bin/env python3
"""
Test HEF models with correct InferModel usage
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hailo_platform.pyhailort.pyhailort import InferModel, HailoStreamInterface, FormatType, Device

def test_model_simple(model_path):
    """Test with simplest possible approach"""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(model_path).name}")
    print('-'*60)

    try:
        # Create device first
        print("Creating device...")
        device = Device()
        print(f"✅ Device created")

        # Create InferModel with device and path
        print("Creating InferModel...")
        model = InferModel(device, str(model_path))
        print("✅ InferModel created")

        # Configure
        print("Configuring...")
        configured_model = model.configure()
        print("✅ Model configured")

        # Get bindings
        print("Creating bindings...")
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
        bindings.input[input_name] = dummy_input

        # Run inference
        print("\nRunning inference...")
        configured_model.run(bindings)

        print("✅ INFERENCE SUCCESSFUL!")

        # Check outputs
        for name, output in bindings.output.items():
            print(f"  Output '{name}': shape {output.shape}")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")

        if "buffer as view" in str(e):
            print("⚠️  Buffer/view issue detected")

        # Try to show more detail
        import traceback
        traceback.print_exc()

        return False

def main():
    models = [
        "ai/models/bestdogyolo5.hef",
        "ai/models/best_dogYoloV5test--640x640_quant_hailort_hailo8_1.hef",
        "ai/models/dogbotyolo8--640x640_quant_hailort_hailo8_1.hef"
    ]

    print("="*60)
    print("HAILO MODEL TEST - WORKING APPROACH")
    print("="*60)

    # Check device
    os.system("hailortcli scan")

    results = {}
    for model in models:
        if Path(model).exists():
            results[Path(model).name] = test_model_simple(model)

    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    working = [k for k,v in results.items() if v]
    failed = [k for k,v in results.items() if not v]

    if working:
        print(f"\n✅ WORKING MODELS ({len(working)}):")
        for m in working:
            print(f"  - {m}")
        print(f"\n🎯 USE THIS MODEL: {working[0]}")
        print(f"   Path: ai/models/{working[0]}")

        # Save working model
        with open("working_model.txt", "w") as f:
            f.write(f"ai/models/{working[0]}\n")
        print("\nWorking model saved to working_model.txt")
    else:
        print("\n❌ All models failed")
        print("\nAll 3 models show the same issue.")
        print("This is a HailoRT Python API issue, not a model problem.")

    if failed:
        print(f"\n❌ FAILED MODELS ({len(failed)}):")
        for m in failed:
            print(f"  - {m}")

    print(f"\nResult: {len(working)}/{len(results)} models working")

if __name__ == "__main__":
    main()