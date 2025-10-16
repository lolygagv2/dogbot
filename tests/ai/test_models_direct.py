#!/usr/bin/env python3
"""
Direct test of each HEF model
"""

import os
import sys
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hailo_platform.pyhailort import pyhailort

def test_model(model_path):
    """Test a single model directly"""
    print(f"\nTesting: {model_path}")
    print("-" * 50)

    try:
        # Load HEF
        print("Loading HEF...")
        with open(model_path, 'rb') as f:
            hef_buffer = f.read()

        hef = pyhailort.Hef(hef_buffer)
        print(f"✅ Loaded, networks: {hef.get_network_group_names()}")

        # Create device
        print("Creating device...")
        target = pyhailort.create_vdevice()
        print("✅ Device created")

        # Configure
        print("Configuring...")
        configure_params = hef.create_configure_params(interface=pyhailort.HailoStreamInterface.PCIE)
        network_groups = target.configure(hef, configure_params)

        if len(network_groups) == 0:
            raise RuntimeError("Failed to configure network")

        network_group = network_groups[0]
        print("✅ Configured")

        # Get network info
        print("Network info:")
        input_vstream_params = network_group.make_input_vstream_params(
            format_type=pyhailort.FormatType.UINT8,
            quantized=True
        )

        output_vstream_params = network_group.make_output_vstream_params(
            format_type=pyhailort.FormatType.UINT8,
            quantized=True
        )

        input_vstreams = network_group.create_input_vstreams(input_vstream_params)
        output_vstreams = network_group.create_output_vstreams(output_vstream_params)

        print(f"  Inputs: {len(input_vstreams)}")
        for i, vs in enumerate(input_vstreams):
            info = vs.get_info()
            print(f"    {i}: {info.name} - shape {info.shape}")

        print(f"  Outputs: {len(output_vstreams)}")
        for i, vs in enumerate(output_vstreams):
            info = vs.get_info()
            print(f"    {i}: {info.name} - shape {info.shape}")

        # Test inference
        print("Testing inference...")
        input_info = input_vstreams[0].get_info()
        dummy_input = np.zeros(input_info.shape, dtype=np.uint8)

        with network_group.activate():
            input_vstreams[0].send(dummy_input)

            outputs = []
            for output_vstream in output_vstreams:
                output = output_vstream.recv()
                outputs.append(output)

        print(f"✅ SUCCESS! Got {len(outputs)} outputs")
        for i, out in enumerate(outputs):
            print(f"  Output {i}: shape {out.shape}")

        # Cleanup
        input_vstreams.clear()
        output_vstreams.clear()

        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        if "buffer as view" in str(e):
            print("  (This is the buffer/view configuration issue)")
        return False

def main():
    models = [
        "ai/models/bestdogyolo5.hef",
        "ai/models/best_dogYoloV5test--640x640_quant_hailort_hailo8_1.hef",
        "ai/models/dogbotyolo8--640x640_quant_hailort_hailo8_1.hef"
    ]

    print("="*60)
    print("Testing all HEF models")
    print("="*60)

    results = {}
    for model in models:
        if Path(model).exists():
            success = test_model(model)
            results[Path(model).name] = success
        else:
            print(f"Model not found: {model}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    working = [k for k, v in results.items() if v]
    failed = [k for k, v in results.items() if not v]

    if working:
        print(f"\n✅ WORKING ({len(working)}):")
        for m in working:
            print(f"  - {m}")

    if failed:
        print(f"\n❌ FAILED ({len(failed)}):")
        for m in failed:
            print(f"  - {m}")

    print(f"\nResult: {len(working)}/{len(results)} models working")

if __name__ == "__main__":
    main()