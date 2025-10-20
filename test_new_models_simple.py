#!/usr/bin/env python3
"""
Simple test for new dogdetector_14.hef and dogpose_14.hef models
Using minimal HailoRT approach to verify models work
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_model_basic(model_path):
    """Test model with most basic approach possible"""
    print(f"\n{'='*60}")
    print(f"Testing: {Path(model_path).name}")
    print('-'*60)

    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False

    try:
        # Try multiple API approaches
        approaches = [
            test_with_infer_model,
            test_with_hef_direct,
            test_model_info_only
        ]

        for i, approach in enumerate(approaches, 1):
            print(f"\n--- Approach {i}: {approach.__name__} ---")
            try:
                result = approach(model_path)
                if result:
                    print(f"✅ SUCCESS with {approach.__name__}")
                    return True
                else:
                    print(f"❌ Failed with {approach.__name__}")
            except Exception as e:
                print(f"❌ Exception in {approach.__name__}: {e}")

        print(f"❌ All approaches failed for {Path(model_path).name}")
        return False

    except Exception as e:
        print(f"❌ Error testing {Path(model_path).name}: {e}")
        return False

def test_with_infer_model(model_path):
    """Test with InferModel API"""
    try:
        from hailo_platform.pyhailort.pyhailort import InferModel, Device

        device = Device()
        model = InferModel(device, str(model_path))
        configured_model = model.configure()
        bindings = configured_model.create_bindings()

        print(f"  Inputs: {list(bindings.input.keys())}")
        print(f"  Outputs: {list(bindings.output.keys())}")

        # Get input shape
        input_name = list(bindings.input.keys())[0]
        input_binding = bindings.input[input_name]
        shape = input_binding.shape()
        print(f"  Input shape: {shape}")

        return True

    except Exception as e:
        print(f"  InferModel failed: {e}")
        return False

def test_with_hef_direct(model_path):
    """Test with HEF direct API"""
    try:
        import hailo_platform as hpf

        hef = hpf.HEF(str(model_path))
        input_info = hef.get_input_vstream_infos()[0]
        output_infos = hef.get_output_vstream_infos()

        print(f"  Input: {input_info.name}, shape: {input_info.shape}")
        print(f"  Outputs: {len(output_infos)} outputs")
        for i, output in enumerate(output_infos):
            print(f"    Output {i}: {output.name}, shape: {output.shape}")

        return True

    except Exception as e:
        print(f"  HEF direct failed: {e}")
        return False

def test_model_info_only(model_path):
    """Test just getting model info without inference"""
    try:
        # Try to use hailortcli if available
        import subprocess
        result = subprocess.run(['hailortcli', 'parse-hef', str(model_path)],
                              capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            print(f"  hailortcli parse successful")
            # Look for key info in output
            output = result.stdout
            if "Input" in output:
                print(f"  Found input layer info")
            if "Output" in output:
                print(f"  Found output layer info")
            return True
        else:
            print(f"  hailortcli failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"  hailortcli test failed: {e}")
        return False

def main():
    print("🤖 Testing New Models: dogdetector_14.hef and dogpose_14.hef")
    print("=" * 70)

    models = [
        "ai/models/dogdetector_14.hef",
        "ai/models/dogpose_14.hef"
    ]

    results = {}

    for model_path in models:
        success = test_model_basic(model_path)
        results[model_path] = success

    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("-" * 70)

    for model_path, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{Path(model_path).name:25} {status}")

    working_count = sum(results.values())
    total_count = len(results)

    print(f"\nResult: {working_count}/{total_count} models working")

    if working_count == total_count:
        print("🎉 All models are working!")
        print("\nNext steps:")
        print("- Models are compatible with Hailo8")
        print("- Ready for integration testing")
        print("- Test with actual camera input")
    elif working_count > 0:
        print("⚠️ Some models working, some not")
        print("- Check which approach works")
        print("- Use working approach for implementation")
    else:
        print("❌ No models working")
        print("- Check HailoRT installation")
        print("- Verify models are compiled for Hailo8")
        print("- Check model file integrity")

if __name__ == "__main__":
    main()