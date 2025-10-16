#!/usr/bin/env python3
"""
Test HEF models with correct constructor usage
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hailo_platform.pyhailort.pyhailort import InferModel, HailoStreamInterface, FormatType, HEF

def test_model(model_path):
    """Test with proper InferModel constructor"""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(model_path).name}")
    print('-'*60)

    try:
        # Load HEF first
        print("Loading HEF...")
        hef = HEF(str(model_path))
        print(f"✅ HEF loaded, networks: {hef.get_network_group_names()}")

        # Create InferModel with proper constructor
        print("Creating InferModel...")
        # InferModel constructor takes (hef, interface)
        model = InferModel(hef, HailoStreamInterface.PCIe)
        print("✅ InferModel created")

        # Configure the model
        print("Configuring model...")
        model.configure()
        print("✅ Model configured")

        # Get input/output info
        print(f"Inputs: {model.input_names}")
        print(f"Outputs: {model.output_names}")

        # Get input stream
        input_stream = model.input()
        first_input_name = model.input_names[0]
        input_obj = input_stream[first_input_name]

        # Get shape
        shape = input_obj.shape()
        print(f"Input shape: {shape}")

        # Create dummy input
        dummy_input = np.zeros(shape, dtype=np.uint8)
        print(f"Test input shape: {dummy_input.shape}")

        # Set format type for outputs
        output_stream = model.output()
        for name in model.output_names:
            output_stream[name].set_format_type(FormatType.FLOAT32)
            print(f"Set output '{name}' to FLOAT32")

        # Create InferStream for running inference
        print("\nRunning inference...")
        infer_stream = model.InferStream()

        # Set input
        input_stream[first_input_name].set_buffer(dummy_input)

        # Run inference
        infer_stream.run()

        # Get outputs
        print("✅ INFERENCE SUCCESSFUL!")

        for name in model.output_names:
            output_buffer = output_stream[name].get_buffer()
            print(f"  Output '{name}': shape {output_buffer.shape}")

        return True

    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

        if "buffer as view" in str(e):
            print("\n⚠️  This is the buffer/view issue - trying workaround...")
            try:
                # Try different format type
                print("Retrying with UINT8 format...")
                output_stream = model.output()
                for name in model.output_names:
                    output_stream[name].set_format_type(FormatType.UINT8)

                # Try again
                infer_stream = model.InferStream()
                input_stream[first_input_name].set_buffer(dummy_input)
                infer_stream.run()

                print("✅ Workaround successful!")
                return True

            except Exception as e2:
                print(f"  Workaround also failed: {e2}")

        return False

def main():
    models = [
        "ai/models/bestdogyolo5.hef",
        "ai/models/best_dogYoloV5test--640x640_quant_hailort_hailo8_1.hef",
        "ai/models/dogbotyolo8--640x640_quant_hailort_hailo8_1.hef"
    ]

    print("="*60)
    print("HAILO MODEL TEST - CORRECT API")
    print("="*60)

    results = {}
    for model in models:
        if Path(model).exists():
            results[Path(model).name] = test_model(model)

    # Summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    working = [k for k,v in results.items() if v]
    failed = [k for k,v in results.items() if not v]

    if working:
        print(f"\n✅ WORKING MODELS ({len(working)}):")
        for m in working:
            print(f"  - {m}")
        print(f"\n🎯 RECOMMENDED MODEL: {working[0]}")

    if failed:
        print(f"\n❌ FAILED MODELS ({len(failed)}):")
        for m in failed:
            print(f"  - {m}")

    print(f"\nSuccess rate: {len(working)}/{len(results)}")

if __name__ == "__main__":
    main()