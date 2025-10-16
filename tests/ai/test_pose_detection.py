#!/usr/bin/env python3
"""
Test script for YOLOv11 pose detection with Hailo
"""

import sys
import time
import numpy as np
import cv2
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_hailo_inference():
    """Test basic Hailo inference with pose model"""
    try:
        import hailo_platform as hpf

        # Check for HEF model
        hef_path = "ai/models/dogposeV2yolo11.hef"
        if not Path(hef_path).exists():
            logger.error(f"HEF model not found at {hef_path}")
            return False

        logger.info(f"Loading HEF model: {hef_path}")
        hef = hpf.HEF(hef_path)

        # Get model info
        input_info = hef.get_input_vstream_infos()
        output_info = hef.get_output_vstream_infos()

        logger.info(f"Model loaded successfully")
        logger.info(f"Inputs: {len(input_info)}")
        for i, info in enumerate(input_info):
            logger.info(f"  Input {i}: {info.name}, shape: {info.shape}, format: {info.format}")

        logger.info(f"Outputs: {len(output_info)}")
        for i, info in enumerate(output_info):
            logger.info(f"  Output {i}: {info.name}, shape: {info.shape}, format: {info.format}")

        # Setup device with proper context managers
        logger.info("Setting up Hailo device...")

        with hpf.VDevice() as vdevice:
            # Configure
            params = hpf.ConfigureParams.create_from_hef(
                hef=hef,
                interface=hpf.HailoStreamInterface.PCIe
            )
            network_groups = vdevice.configure(hef, params)
            network_group = network_groups[0]

            # Create vstream params - must be dictionaries not lists
            input_vstreams_params = hpf.InputVStreamParams.make(
                network_group,
                quantized=True,
                format_type=hpf.FormatType.UINT8
            )

            output_vstreams_params = hpf.OutputVStreamParams.make(
                network_group,
                quantized=False,
                format_type=hpf.FormatType.FLOAT32
            )

            # Activate network with context manager
            network_group_params = network_group.create_params()
            with network_group.activate(network_group_params):
                # Create inference pipeline with context manager
                with hpf.InferVStreams(
                    network_group,
                    input_vstreams_params,
                    output_vstreams_params
                ) as infer_pipeline:
                    # Create test input (896x896x3 image)
                    test_image = np.zeros((896, 896, 3), dtype=np.uint8)
                    test_image[:, :] = [128, 128, 128]  # Gray image

                    # Resize to expected input size (640x640)
                    if input_info[0].shape[0] == 640:
                        test_image = cv2.resize(test_image, (640, 640))

                    # Prepare input
                    input_data = {
                        input_info[0].name: np.expand_dims(test_image, axis=0)
                    }

                    # Run inference
                    logger.info("Running test inference...")
                    start_time = time.time()
                    outputs = infer_pipeline.infer(input_data)
                    inference_time = (time.time() - start_time) * 1000

                    logger.info(f"✅ Inference successful! Time: {inference_time:.2f}ms")

                    # Check outputs
                    logger.info("Output details:")
                    for name, output in outputs.items():
                        logger.info(f"  {name}: shape={output.shape}, dtype={output.dtype}")
                        logger.info(f"    min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

        return True

    except Exception as e:
        logger.error(f"Hailo test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pose_detector():
    """Test the integrated PoseDetector class"""
    try:
        # Add parent directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        from core.pose_detector import PoseDetector

        logger.info("Testing PoseDetector class...")
        detector = PoseDetector()

        # Initialize
        if not detector.initialize():
            logger.error("Failed to initialize PoseDetector")
            return False

        logger.info("✅ PoseDetector initialized successfully")

        # Get status
        status = detector.get_status()
        logger.info(f"Status: {status}")

        # Test with blank frame
        test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        test_frame[:, :] = [128, 128, 128]  # Gray

        logger.info("Processing test frame...")
        result = detector.process_frame(test_frame)

        logger.info(f"Results:")
        logger.info(f"  Detections: {len(result['detections'])}")
        logger.info(f"  Behaviors: {result['behaviors']}")
        logger.info(f"  Inference time: {result.get('inference_time', 0):.2f}ms")

        # Cleanup
        detector.cleanup()

        return True

    except Exception as e:
        logger.error(f"PoseDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_standalone_script():
    """Test the standalone run_pi.py script"""
    try:
        logger.info("Testing standalone run_pi.py...")

        # Check if script exists
        if not Path("run_pi.py").exists():
            logger.error("run_pi.py not found")
            return False

        # Import and check functions
        sys.path.insert(0, ".")
        import run_pi

        # Check critical functions exist
        functions_to_check = [
            "decode_pose_multi_hailo",
            "detect_markers",
            "assign_ids",
            "letterbox",
            "norm_kpts"
        ]

        for func_name in functions_to_check:
            if hasattr(run_pi, func_name):
                logger.info(f"  ✅ Function {func_name} found")
            else:
                logger.error(f"  ❌ Function {func_name} missing")
                return False

        # Test decode function with mock data
        logger.info("Testing decode_pose_multi_hailo with mock data...")

        # Mock single detection output
        mock_output = np.zeros((1, 77), dtype=np.float32)
        mock_output[0, :4] = [100, 100, 200, 200]  # bbox
        mock_output[0, 4] = 0.8  # confidence
        # Add some keypoints
        for i in range(24):
            mock_output[0, 5 + i*3] = 150  # x
            mock_output[0, 6 + i*3] = 150  # y
            mock_output[0, 7 + i*3] = 0.9  # confidence

        result = run_pi.decode_pose_multi_hailo(mock_output)
        logger.info(f"  Decoded {len(result)} detections")

        if len(result) > 0:
            det = result[0]
            logger.info(f"  First detection:")
            logger.info(f"    bbox: {det['xyxy']}")
            logger.info(f"    keypoints shape: {det['kpts'].shape}")

        return True

    except Exception as e:
        logger.error(f"Standalone script test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("=" * 60)
    logger.info("Starting Pose Detection Tests")
    logger.info("=" * 60)

    tests = [
        ("Basic Hailo Inference", test_hailo_inference),
        ("PoseDetector Class", test_pose_detector),
        ("Standalone Script", test_standalone_script)
    ]

    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing: {test_name} ---")
        success = test_func()
        results.append((test_name, success))
        logger.info(f"Result: {'✅ PASSED' if success else '❌ FAILED'}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("Test Summary:")
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"  {test_name}: {status}")

    all_passed = all(success for _, success in results)
    logger.info(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())