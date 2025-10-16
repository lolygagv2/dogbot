#!/usr/bin/env python3
"""
Working HailoRT 4.21 inference using InferVStreams API
"""

import numpy as np
import hailo_platform as hpf
import os

def test_model_infervstreams(hef_path):
    """Test model using InferVStreams API (correct for HailoRT 4.21)"""
    print(f"🔍 Testing: {os.path.basename(hef_path)}")

    try:
        hef = hpf.HEF(hef_path)

        with hpf.VDevice() as target:
            configure_params = hpf.ConfigureParams.create_from_hef(hef, interface=hpf.HailoStreamInterface.PCIe)
            network_group = target.configure(hef, configure_params)[0]
            network_group_params = network_group.create_params()

            input_vstream_info = hef.get_input_vstream_infos()[0]
            output_vstream_infos = hef.get_output_vstream_infos()

            print(f"✅ Input: {input_vstream_info.name}, shape: {input_vstream_info.shape}")
            print(f"✅ Outputs: {len(output_vstream_infos)} outputs")
            for i, out_info in enumerate(output_vstream_infos):
                print(f"   Output {i}: {out_info.name}, shape: {out_info.shape}")

            input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
                network_group, quantized=True, format_type=hpf.FormatType.UINT8)
            output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
                network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

            with network_group.activate(network_group_params):
                with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                    # Create dummy input
                    dummy_input = np.zeros(input_vstream_info.shape, dtype=np.uint8)
                    input_data = {input_vstream_info.name: np.expand_dims(dummy_input, axis=0)}

                    print("🔍 Running inference...")
                    results = infer_pipeline.infer(input_data)
                    print("✅ Inference completed!")

                    # Display results
                    for output_name, output_data in results.items():
                        print(f"✅ {output_name}: shape={output_data.shape}, dtype={output_data.dtype}")
                        if hasattr(output_data, 'min'):
                            print(f"   Range: [{output_data.min():.6f}, {output_data.max():.6f}]")

                    print(f"🎉 SUCCESS! {os.path.basename(hef_path)} working with InferVStreams API!")
                    return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_all_models():
    """Test all available models with InferVStreams API"""
    models = [
        "ai/models/dogbotyolo8--640x640_quant_hailort_hailo8_1.hef",
        "ai/models/bestdogyolo5.hef",
        "ai/models/best_dogYoloV5test--640x640_quant_hailort_hailo8_1.hef",
        "env_new/lib/python3.11/site-packages/hailo_tutorials/hefs/resnet_v1_18.hef"
    ]

    success_count = 0
    tested_count = 0

    for model_path in models:
        if not os.path.exists(model_path):
            print(f"⏭️  Skipping (not found): {model_path}")
            continue

        tested_count += 1
        if test_model_infervstreams(model_path):
            success_count += 1
        print("-" * 60)

    print(f"\n📊 FINAL RESULTS")
    print("=" * 60)
    print(f"✅ Success: {success_count}/{tested_count} models working")
    print(f"🎯 HailoRT 4.21 + InferVStreams API = WORKING!")

    if success_count > 0:
        print(f"🚀 Ready to integrate into detection system!")
        return True
    else:
        print(f"❌ No models working - need further debugging")
        return False

if __name__ == "__main__":
    print("🔥 Testing HEF Models with HailoRT 4.21 InferVStreams API")
    print("=" * 60)
    success = test_all_models()
    print(f"\n🎯 Result: {'SUCCESS' if success else 'FAILED'}")