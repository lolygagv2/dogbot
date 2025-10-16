#!/usr/bin/env python3
"""
Test all HEF models with proper Hailo API
Based on working test_hailo_detection.py structure
"""

import os
import sys
import numpy as np
from pathlib import Path
import traceback
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Hailo
from hailo_platform.pyhailort import HailoRt

class ModelTester:
    def __init__(self):
        self.models_dir = Path("ai/models")
        self.test_results = []

    def test_model(self, model_path):
        """Test a single model"""
        print(f"\n{'='*60}")
        print(f"Testing: {model_path.name}")
        print(f"Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        print('-'*60)

        result = {
            'model': model_path.name,
            'success': False,
            'error': None,
            'details': {}
        }

        try:
            # Step 1: Load HEF
            print("1. Loading HEF...")
            hef = HailoRt.read_hef_file(str(model_path))

            if not hef:
                raise ValueError("Failed to load HEF file")

            print("   ✅ HEF loaded")

            # Step 2: Create device
            print("2. Creating device...")
            devices = HailoRt.scan_devices()

            if not devices:
                raise RuntimeError("No Hailo devices found")

            print(f"   Found {len(devices)} device(s)")

            # Get target from HEF
            target = HailoRt.get_hef_target(hef)
            print(f"   HEF target: {target}")

            # Create device
            device = HailoRt.create_device()
            if not device:
                raise RuntimeError("Failed to create device")

            print("   ✅ Device created")

            # Step 3: Get model info
            print("3. Getting model info...")

            # Get network groups info
            network_groups_infos = HailoRt.get_hef_network_groups_infos(hef)

            if not network_groups_infos:
                raise ValueError("No network groups found in HEF")

            network_group_info = network_groups_infos[0]
            network_name = network_group_info.name

            print(f"   Network: {network_name}")
            result['details']['network'] = network_name

            # Step 4: Configure network
            print("4. Configuring network...")

            # Configure the device with HEF
            configure_params = HailoRt.create_configure_params(hef)
            network_groups = device.configure(hef, configure_params)

            if not network_groups:
                raise RuntimeError("Failed to configure device")

            network_group = network_groups[0]
            print("   ✅ Network configured")

            # Step 5: Create input/output vstreams
            print("5. Creating vstreams...")

            # Get default vstream params
            input_vstreams_params = network_group.make_input_vstreams_params()
            output_vstreams_params = network_group.make_output_vstreams_params()

            # Create vstreams
            input_vstreams = HailoRt.create_input_vstreams(network_group, input_vstreams_params)
            output_vstreams = HailoRt.create_output_vstreams(network_group, output_vstreams_params)

            if not input_vstreams or not output_vstreams:
                raise RuntimeError("Failed to create vstreams")

            print(f"   ✅ Created {len(input_vstreams)} input, {len(output_vstreams)} output vstreams")

            # Get vstream info
            for vs in input_vstreams:
                info = vs.get_info()
                print(f"   Input: {info.name}, shape: {info.shape}")
                result['details'][f'input_{info.name}'] = str(info.shape)

            for vs in output_vstreams:
                info = vs.get_info()
                print(f"   Output: {info.name}, shape: {info.shape}")
                result['details'][f'output_{info.name}'] = str(info.shape)

            # Step 6: Prepare dummy input
            print("6. Preparing test input...")

            # Get input shape from first vstream
            input_info = input_vstreams[0].get_info()
            input_shape = input_info.shape

            # Create dummy input (black image)
            dummy_input = np.zeros(input_shape, dtype=np.uint8)
            print(f"   Input shape: {dummy_input.shape}")

            # Step 7: Run inference
            print("7. Running inference...")

            # Activate network group
            with network_group.activate():
                # Send input
                input_vstreams[0].send(dummy_input)

                # Get output
                outputs = []
                for output_vstream in output_vstreams:
                    output = output_vstream.recv()
                    outputs.append(output)
                    print(f"   Output shape: {output.shape}")

            print("   ✅ Inference successful!")
            result['success'] = True

            # Cleanup
            for vs in input_vstreams:
                vs.clear()
            for vs in output_vstreams:
                vs.clear()

        except Exception as e:
            result['error'] = str(e)
            print(f"   ❌ Error: {e}")

            # Check for specific error
            if "buffer as view" in str(e):
                print("   Note: Buffer/view configuration issue detected")
                result['details']['error_type'] = 'buffer_view_issue'

        self.test_results.append(result)
        return result

    def run_all_tests(self):
        """Test all models"""
        hef_files = sorted(self.models_dir.glob("*.hef"))

        if not hef_files:
            print("No HEF files found!")
            return []

        print(f"Found {len(hef_files)} models:")
        for f in hef_files:
            print(f"  - {f.name}")

        # Test each
        for model_path in hef_files:
            try:
                self.test_model(model_path)
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                print(f"Unexpected error testing {model_path.name}: {e}")

        # Summary
        self.print_summary()

        return [r for r in self.test_results if r['success']]

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)

        working = []
        buffer_issue = []
        other_failed = []

        for r in self.test_results:
            if r['success']:
                working.append(r)
            elif r.get('details', {}).get('error_type') == 'buffer_view_issue':
                buffer_issue.append(r)
            else:
                other_failed.append(r)

        # Working models
        if working:
            print(f"\n✅ WORKING MODELS ({len(working)}):")
            for r in working:
                print(f"  {r['model']}")
                for k, v in r['details'].items():
                    if k != 'error_type':
                        print(f"    {k}: {v}")

        # Buffer issue models
        if buffer_issue:
            print(f"\n⚠️  BUFFER/VIEW ISSUE ({len(buffer_issue)}):")
            for r in buffer_issue:
                print(f"  {r['model']}")

        # Other failures
        if other_failed:
            print(f"\n❌ OTHER FAILURES ({len(other_failed)}):")
            for r in other_failed:
                print(f"  {r['model']}: {r['error']}")

        print(f"\n{'='*60}")
        print(f"Results: {len(working)}/{len(self.test_results)} models working")

        if working:
            print(f"\n🎯 RECOMMENDED: Use {working[0]['model']}")

def main():
    print("Hailo Model Compatibility Test")
    print("="*60)

    # Check device status
    print("Checking device...")
    os.system("hailortcli fw-control identify")
    print()

    # Run tests
    tester = ModelTester()
    working_models = tester.run_all_tests()

    # Save results
    if working_models:
        with open("working_models.txt", "w") as f:
            for model in working_models:
                f.write(f"{model['model']}\n")
        print(f"\nWorking models saved to working_models.txt")

if __name__ == "__main__":
    main()