#!/usr/bin/env python3
"""
Debug detection model with real dog frames
Use saved frames from the working video to debug why models aren't detecting dogs
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import hailo_platform as hpf

def debug_detection_model(image_path, model_path):
    """Debug detection model with a specific image"""
    print(f"🔍 Debugging Detection Model")
    print(f"Image: {image_path}")
    print(f"Model: {model_path}")
    print("=" * 50)

    # Load image
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Failed to load image")
        return

    print(f"📷 Loaded image: {frame.shape}")

    # Initialize Hailo
    try:
        vdevice = hpf.VDevice()
        hef_model = hpf.HEF(model_path)

        configure_params = hpf.ConfigureParams.create_from_hef(hef_model, interface=hpf.HailoStreamInterface.PCIe)
        network_groups = vdevice.configure(hef_model, configure_params)
        network_group = network_groups[0]
        network_group_params = network_group.create_params()

        print(f"✅ Model loaded")

        # Get model info
        input_info = hef_model.get_input_vstream_infos()[0]
        output_infos = hef_model.get_output_vstream_infos()

        print(f"📊 Model expects: {input_info.shape}")
        print(f"📊 Model outputs: {len(output_infos)} tensors")
        for i, info in enumerate(output_infos):
            print(f"   Output {i}: {info.name} -> {info.shape}")

    except Exception as e:
        print(f"❌ Model init failed: {e}")
        return

    # Try different preprocessing approaches
    preprocessing_methods = [
        ("Original BGR", lambda img: img),
        ("RGB conversion", lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        ("Normalized 0-1", lambda img: (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)),
        ("Normalized mean/std", lambda img: normalize_imagenet(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))),
    ]

    for method_name, preprocess_func in preprocessing_methods:
        print(f"\n🧪 Testing: {method_name}")

        try:
            # Resize to model input size
            frame_resized = cv2.resize(frame, (640, 640))

            # Apply preprocessing
            processed = preprocess_func(frame_resized)

            # Ensure correct format for model
            if processed.dtype != np.uint8:
                if processed.max() <= 1.0:
                    processed = (processed * 255).astype(np.uint8)
                else:
                    processed = processed.astype(np.uint8)

            print(f"   Input shape: {processed.shape}, dtype: {processed.dtype}")
            print(f"   Value range: {processed.min()} to {processed.max()}")

            # Run inference
            with network_group.activate(network_group_params):
                input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
                    network_group, quantized=False, format_type=hpf.FormatType.UINT8)
                output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
                    network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

                with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                    input_tensor = np.expand_dims(processed, axis=0)
                    input_name = list(input_vstreams_params.keys())[0]
                    input_data = {input_name: input_tensor}

                    start_time = time.time()
                    output_data = infer_pipeline.infer(input_data)
                    inference_time = (time.time() - start_time) * 1000

                    print(f"   Inference: {inference_time:.1f}ms")

                    # Analyze outputs
                    for output_name, output_tensor in output_data.items():
                        if isinstance(output_tensor, list) and len(output_tensor) > 0:
                            if isinstance(output_tensor[0], list) and len(output_tensor[0]) > 0:
                                actual_tensor = output_tensor[0][0]
                                if hasattr(actual_tensor, 'shape'):
                                    print(f"   Output {output_name}: {actual_tensor.shape}")
                                    print(f"   Detection count: {actual_tensor.shape[0] if len(actual_tensor.shape) > 0 else 0}")

                                    if actual_tensor.shape[0] > 0:
                                        print(f"   🎉 FOUND DETECTIONS!")
                                        confidences = actual_tensor[:, 4] if actual_tensor.shape[1] > 4 else actual_tensor[:, -1]
                                        print(f"   Top confidences: {np.sort(confidences)[-5:]}")
                                    else:
                                        print(f"   ❌ No detections")

        except Exception as e:
            print(f"   ❌ Failed: {e}")

def normalize_imagenet(img):
    """ImageNet normalization"""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img_norm = img.astype(np.float32) / 255.0
    img_norm = (img_norm - mean) / std
    return img_norm

def main():
    # Find the most recent test results
    test_dirs = list(Path(".").glob("simple_test_results_*"))
    if not test_dirs:
        print("❌ No test results found. Run test_live_dogs_simple.py first.")
        return

    latest_dir = sorted(test_dirs)[-1]
    print(f"Using results from: {latest_dir}")

    # Find a frame image
    frame_files = list(latest_dir.glob("frame_*.jpg"))
    if not frame_files:
        print("❌ No frame images found")
        return

    # Use the latest frame
    latest_frame = sorted(frame_files)[-1]
    print(f"Using frame: {latest_frame}")

    # Test both models
    models = [
        ("dogdetector_14.hef", "ai/models/dogdetector_14.hef"),
        ("YOLOv8m baseline", "/home/morgan/dogbot/hailoRTsuite/tappas/detection/resources/yolov8m.hef")
    ]

    for model_name, model_path in models:
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")

        if os.path.exists(model_path):
            debug_detection_model(str(latest_frame), model_path)
        else:
            print(f"❌ Model not found: {model_path}")

if __name__ == "__main__":
    main()