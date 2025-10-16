#!/usr/bin/env python3
"""
Test HEF models using simplified InferModel API
"""

import os
import sys
import numpy as np
from pathlib import Path
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from hailo_platform import HEF, FormatType, InferModel
    print("✅ Hailo platform imports successful")
except ImportError as e:
    print(f"❌ Failed to import Hailo platform: {e}")
    sys.exit(1)

class SimpleHEFTester:
    def __init__(self):
        self.models_dir = Path("ai/models")
        self.test_results = []

    def test_model(self, model_path):
        """Test a single model with InferModel API"""
        print(f"\n{'='*60}")
        print(f"Testing: {model_path.name}")
        print(f"Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        print('-'*60)

        result = {
            'model': model_path.name,
            'load_success': False,
            'infer_model_success': False,
            'inference_success': False,
            'error': None,
            'input_shape': None,
            'output_shapes': []
        }

        try:
            # Step 1: Load HEF
            print("1. Loading HEF file...")
            hef = HEF(str(model_path))
            result['load_success'] = True
            print("   ✅ HEF loaded")

            # Get network info
            networks = hef.get_networks_names()
            print(f"   Networks: {networks}")

            # Step 2: Create InferModel
            print("2. Creating InferModel...")
            infer_model = InferModel(str(model_path), FormatType.UINT8)
            result['infer_model_success'] = True
            print("   ✅ InferModel created")

            # Step 3: Get input shape
            print("3. Getting model info...")
            input_shape = infer_model.get_input_shape()
            print(f"   Input shape: {input_shape}")
            result['input_shape'] = str(input_shape)

            # Step 4: Create dummy input
            print("4. Creating test input...")
            # YOLOv5/v8 expects (batch, height, width, channels)
            if len(input_shape) == 4:
                dummy_input = np.zeros(input_shape, dtype=np.uint8)
            else:
                # Assume YOLO with 640x640x3 input
                dummy_input = np.zeros((1, 640, 640, 3), dtype=np.uint8)

            print(f"   Input data shape: {dummy_input.shape}")

            # Step 5: Run inference
            print("5. Running inference...")
            outputs = infer_model.run([dummy_input])
            result['inference_success'] = True
            print("   ✅ Inference successful!")

            # Step 6: Check outputs
            print("6. Output info:")
            for i, output in enumerate(outputs):
                print(f"   Output {i}: shape {output.shape}, dtype {output.dtype}")
                result['output_shapes'].append(str(output.shape))

            # Clean up
            del infer_model

        except Exception as e:
            result['error'] = str(e)
            print(f"   ❌ Error: {e}")
            if "buffer as view" in str(e):
                print("   Note: This is the known buffer/view configuration issue")
            traceback.print_exc()

        self.test_results.append(result)
        return result

    def run_all_tests(self):
        """Test all HEF models"""
        hef_files = sorted(self.models_dir.glob("*.hef"))

        if not hef_files:
            print("No HEF files found in ai/models/")
            return

        print(f"Found {len(hef_files)} HEF models to test:")
        for f in hef_files:
            print(f"  - {f.name}")

        # Test each model
        for model_path in hef_files:
            self.test_model(model_path)

        # Print summary
        print("\n" + "="*60)
        print("FINAL TEST SUMMARY")
        print("="*60)

        working_models = []
        buffer_issue_models = []
        other_failure_models = []

        for result in self.test_results:
            if result['inference_success']:
                working_models.append(result)
            elif result['error'] and "buffer as view" in result['error']:
                buffer_issue_models.append(result)
            else:
                other_failure_models.append(result)

        # Print working models
        if working_models:
            print(f"\n✅ WORKING MODELS ({len(working_models)}):")
            for r in working_models:
                print(f"  {r['model']}")
                print(f"    - Input: {r['input_shape']}")
                print(f"    - Outputs: {r['output_shapes']}")
        else:
            print("\n❌ No models successfully completed inference")

        # Print buffer issue models
        if buffer_issue_models:
            print(f"\n⚠️  MODELS WITH BUFFER/VIEW ISSUE ({len(buffer_issue_models)}):")
            for r in buffer_issue_models:
                print(f"  {r['model']}")

        # Print other failures
        if other_failure_models:
            print(f"\n❌ OTHER FAILURES ({len(other_failure_models)}):")
            for r in other_failure_models:
                print(f"  {r['model']}: {r['error']}")

        print(f"\n{'='*60}")
        print(f"Summary: {len(working_models)}/{len(self.test_results)} models working")

        return working_models

def main():
    print("Hailo HEF Model Tester (Simplified)")
    print("="*60)

    # Check device
    print("Checking Hailo device...")
    os.system("hailortcli fw-control identify")
    print()

    tester = SimpleHEFTester()
    working_models = tester.run_all_tests()

    # If we have working models, save the info
    if working_models:
        print("\n" + "="*60)
        print("RECOMMENDED MODEL:")
        print("="*60)
        best_model = working_models[0]
        print(f"Use: {best_model['model']}")
        print(f"Path: ai/models/{best_model['model']}")

if __name__ == "__main__":
    main()