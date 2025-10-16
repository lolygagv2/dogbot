#!/usr/bin/env python3
"""
Test all HEF models with Hailo-8L hardware
Tests different models and approaches to find working configuration
"""

import os
import sys
import numpy as np
from pathlib import Path
import traceback
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hailo_platform import (
        HEF, VDevice, HailoStreamInterface, InferVStreams,
        ConfigureParams, FormatType, HailoSchedulingAlgorithm
    )
    print("✅ Hailo platform imports successful")
except ImportError as e:
    print(f"❌ Failed to import Hailo platform: {e}")
    sys.exit(1)

class HEFModelTester:
    def __init__(self):
        self.models_dir = Path("ai/models")
        self.test_results = []

    def get_all_hef_files(self):
        """Get all HEF files in the models directory"""
        return sorted(self.models_dir.glob("*.hef"))

    def test_model_loading(self, model_path):
        """Test if a model can be loaded"""
        print(f"\n{'='*60}")
        print(f"Testing: {model_path.name}")
        print(f"Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        print('-'*60)

        result = {
            'model': model_path.name,
            'load_success': False,
            'device_success': False,
            'inference_success': False,
            'error': None,
            'details': {}
        }

        try:
            # Step 1: Load HEF file
            print("1. Loading HEF file...")
            hef = HEF(str(model_path))
            result['load_success'] = True
            print("   ✅ HEF loaded successfully")

            # Get network info
            network_groups = hef.get_networks_names()
            print(f"   Networks: {network_groups}")
            result['details']['networks'] = network_groups

            # Step 2: Create VDevice
            print("2. Creating VDevice...")
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

            with VDevice(params) as vdevice:
                result['device_success'] = True
                print("   ✅ VDevice created successfully")

                # Step 3: Get network group
                network_group = hef.get_network_group_names()[0]
                print(f"3. Using network group: {network_group}")

                # Step 4: Configure network
                print("4. Configuring network...")
                configure_params = ConfigureParams.create_from_hef(
                    hef=hef,
                    interface=HailoStreamInterface.PCIe
                )

                network_group_handle = vdevice.configure(hef, configure_params)[0]
                print("   ✅ Network configured")

                # Step 5: Get input/output info
                input_vstreams_params = network_group_handle.make_input_vstream_params(
                    format_type=FormatType.UINT8,
                    quantized=True
                )
                output_vstreams_params = network_group_handle.make_output_vstream_params(
                    format_type=FormatType.UINT8,
                    quantized=True
                )

                input_vstreams = network_group_handle.create_input_vstreams(input_vstreams_params)
                output_vstreams = network_group_handle.create_output_vstreams(output_vstreams_params)

                print(f"5. Stream info:")
                for i, input_stream in enumerate(input_vstreams):
                    info = input_stream.get_info()
                    print(f"   Input {i}: {info.name}")
                    print(f"     Shape: {info.shape}")
                    print(f"     Format: {info.format}")
                    result['details'][f'input_{i}'] = {
                        'name': info.name,
                        'shape': str(info.shape),
                        'format': str(info.format)
                    }

                for i, output_stream in enumerate(output_vstreams):
                    info = output_stream.get_info()
                    print(f"   Output {i}: {info.name}")
                    print(f"     Shape: {info.shape}")
                    print(f"     Format: {info.format}")
                    result['details'][f'output_{i}'] = {
                        'name': info.name,
                        'shape': str(info.shape),
                        'format': str(info.format)
                    }

                # Step 6: Test inference with dummy data
                print("6. Testing inference with dummy data...")

                # Create dummy input matching the expected shape
                input_info = input_vstreams[0].get_info()
                input_shape = input_info.shape

                # Create dummy image (black image)
                if len(input_shape) == 4:  # NHWC format
                    dummy_input = np.zeros(input_shape, dtype=np.uint8)
                else:
                    # Try to infer batch size of 1
                    dummy_input = np.zeros((1, *input_shape), dtype=np.uint8)

                print(f"   Input data shape: {dummy_input.shape}")

                # Prepare input dict
                input_dict = {input_vstreams[0].get_info().name: dummy_input}

                # Run inference
                with InferVStreams(network_group_handle, input_vstreams, output_vstreams) as infer_pipeline:
                    with network_group_handle.activate():
                        outputs = infer_pipeline.infer(input_dict)

                        print("   ✅ Inference completed!")
                        result['inference_success'] = True

                        # Check output shapes
                        for name, output in outputs.items():
                            print(f"   Output '{name}' shape: {output.shape}")
                            result['details'][f'output_{name}_shape'] = str(output.shape)

                # Clean up
                input_vstreams.clear()
                output_vstreams.clear()

        except Exception as e:
            result['error'] = str(e)
            print(f"   ❌ Error: {e}")
            traceback.print_exc()

        self.test_results.append(result)
        return result

    def run_all_tests(self):
        """Test all HEF models"""
        hef_files = self.get_all_hef_files()

        if not hef_files:
            print("No HEF files found in ai/models/")
            return

        print(f"Found {len(hef_files)} HEF models to test:")
        for f in hef_files:
            print(f"  - {f.name}")

        for model_path in hef_files:
            self.test_model_loading(model_path)

        # Print summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        for result in self.test_results:
            status = "✅" if result['inference_success'] else "❌"
            print(f"\n{status} {result['model']}:")
            print(f"  - Load HEF: {'✅' if result['load_success'] else '❌'}")
            print(f"  - Create Device: {'✅' if result['device_success'] else '❌'}")
            print(f"  - Run Inference: {'✅' if result['inference_success'] else '❌'}")

            if result['error']:
                print(f"  - Error: {result['error']}")

            if result['inference_success'] and result['details']:
                print("  - Model Details:")
                for key, value in result['details'].items():
                    if not key.startswith('output_') or '_shape' in key:
                        print(f"    {key}: {value}")

        # Count successes
        successful = [r for r in self.test_results if r['inference_success']]
        print(f"\n{'='*60}")
        print(f"Results: {len(successful)}/{len(self.test_results)} models working")

        if successful:
            print("\n✅ Working models:")
            for r in successful:
                print(f"  - {r['model']}")
        else:
            print("\n❌ No models successfully completed inference")

def main():
    print("Hailo HEF Model Compatibility Tester")
    print("="*60)

    # Check Hailo device first
    print("Checking Hailo device status...")
    os.system("hailortcli fw-control identify")
    print()

    tester = HEFModelTester()
    tester.run_all_tests()

if __name__ == "__main__":
    main()