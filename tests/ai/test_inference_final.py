#!/usr/bin/env python3
"""
Final test - get inference working
"""

import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hailo_platform.pyhailort.pyhailort import VDevice

def test_model(model_path):
    print(f"\nTesting: {model_path}")
    print("-" * 50)

    try:
        # Create VDevice and InferModel
        vdevice = VDevice()
        infer_model = vdevice.create_infer_model(model_path)

        print(f"Model loaded:")
        print(f"  Inputs: {infer_model.input_names}")
        print(f"  Outputs: {infer_model.output_names}")

        # Configure
        configured_model = infer_model.configure()
        print("✅ Model configured")

        # Create bindings
        bindings = configured_model.create_bindings()
        print(f"Bindings type: {type(bindings)}")

        # Check what bindings actually has
        print(f"Bindings attributes: {[x for x in dir(bindings) if not x.startswith('_')][:10]}")

        # Get input/output properly
        input_dict = bindings.input()
        output_dict = bindings.output()

        print(f"Input dict: {list(input_dict.keys())}")
        print(f"Output dict: {list(output_dict.keys())}")

        # Get input shape
        input_name = infer_model.input_names[0]
        input_binding = input_dict[input_name]
        shape = input_binding.shape()
        print(f"Input shape: {shape}")

        # Create test input
        dummy_input = np.zeros(shape, dtype=np.uint8)

        # Set input
        input_dict[input_name] = dummy_input
        print(f"Set input: {dummy_input.shape}")

        # Run inference
        print("Running inference...")
        configured_model.run(bindings)

        print("✅ INFERENCE SUCCESSFUL!")

        # Get outputs
        for name in infer_model.output_names:
            output = output_dict[name]
            print(f"  Output '{name}': {output.shape}")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        if "buffer as view" in str(e):
            print("⚠️  Buffer/view issue")
        return False

def main():
    models = [
        "ai/models/bestdogyolo5.hef",
        "ai/models/best_dogYoloV5test--640x640_quant_hailort_hailo8_1.hef",
        "ai/models/dogbotyolo8--640x640_quant_hailort_hailo8_1.hef"
    ]

    print("="*60)
    print("HAILO INFERENCE TEST")
    print("="*60)

    results = {}
    for model in models:
        if os.path.exists(model):
            results[os.path.basename(model)] = test_model(model)

    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    working = [k for k,v in results.items() if v]
    if working:
        print(f"\n✅ WORKING: {working}")
    else:
        print("\n❌ No models working")

    print(f"\nSuccess: {len(working)}/{len(results)}")

if __name__ == "__main__":
    main()