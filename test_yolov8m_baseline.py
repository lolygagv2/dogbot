#!/usr/bin/env python3
"""
Test baseline detection with known working YOLOv8m model
This will help isolate if the issue is with our pipeline or the dogdetector_14.hef model
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

import hailo_platform as hpf

class YOLOv8BaselineTest:
    """Test with YOLOv8m baseline model"""

    def __init__(self):
        self.model_path = "/home/morgan/dogbot/hailoRTsuite/tappas/detection/resources/yolov8m.hef"
        self.vdevice = None
        self.network_group = None
        self.camera = None

        self.output_dir = Path(f"yolov8m_test_{time.strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(exist_ok=True)

    def initialize(self):
        """Initialize Hailo and camera"""
        print("🧪 YOLOv8m Baseline Test")
        print("=" * 30)

        # Check model exists
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            return False
        print(f"✅ Model found: {self.model_path}")

        # Initialize Hailo
        try:
            self.vdevice = hpf.VDevice()
            hef_model = hpf.HEF(self.model_path)

            # Configure network group
            configure_params = hpf.ConfigureParams.create_from_hef(hef_model, interface=hpf.HailoStreamInterface.PCIe)
            network_groups = self.vdevice.configure(hef_model, configure_params)
            self.network_group = network_groups[0]  # Get first network group

            print(f"✅ Hailo initialized")

            # Create network group parameters
            self.network_group_params = self.network_group.create_params()

            # Get input/output info
            input_vstream_info = hef_model.get_input_vstream_infos()
            output_vstream_info = hef_model.get_output_vstream_infos()

            print(f"📊 Model info:")
            print(f"   Inputs: {len(input_vstream_info)}")
            for info in input_vstream_info:
                print(f"     {info.name}: {info.shape} ({info.format})")

            print(f"   Outputs: {len(output_vstream_info)}")
            for info in output_vstream_info:
                print(f"     {info.name}: {info.shape} ({info.format})")

        except Exception as e:
            print(f"❌ Hailo init failed: {e}")
            return False

        # Initialize camera
        if PICAMERA2_AVAILABLE:
            try:
                self.camera = Picamera2()
                config = self.camera.create_still_configuration(main={"size": (1920, 1080)})
                self.camera.configure(config)
                self.camera.start()
                time.sleep(1)
                print("✅ Camera ready")
                return True
            except Exception as e:
                print(f"❌ Camera failed: {e}")
                return False
        else:
            print("❌ Picamera2 not available")
            return False

    def test_detection(self):
        """Test detection with YOLOv8m"""
        print("\n🔍 Testing YOLOv8m Detection")

        # Capture frame
        frame = self.camera.capture_array()
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        print(f"📷 Input frame: {frame.shape}")

        # Save original frame
        cv2.imwrite(str(self.output_dir / "original_frame.jpg"), frame)

        # Resize to model input size (check what YOLOv8m expects)
        # Most YOLO models expect square input, try 640x640
        frame_resized = cv2.resize(frame, (640, 640))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        print(f"📏 Preprocessed: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")

        # Save preprocessed frame
        cv2.imwrite(str(self.output_dir / "preprocessed_input.jpg"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        try:
            # Get vstream parameters
            input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=hpf.FormatType.UINT8)
            output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(self.network_group, quantized=False, format_type=hpf.FormatType.FLOAT32)

            with self.network_group.activate(self.network_group_params):
                with hpf.InferVStreams(self.network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:

                    # Prepare input
                    input_tensor = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)
                    input_name = list(input_vstreams_params.keys())[0]
                    input_data = {input_name: input_tensor}

                    print(f"🚀 Running inference...")
                    start_time = time.time()
                    output_data = infer_pipeline.infer(input_data)
                    inference_time = (time.time() - start_time) * 1000

                    print(f"⏱️  Inference time: {inference_time:.1f}ms")
                    print(f"📊 Raw outputs:")

                    for output_name, output_tensor in output_data.items():
                        print(f"   {output_name}: {type(output_tensor)}")
                        if hasattr(output_tensor, 'shape'):
                            print(f"     Shape: {output_tensor.shape}")
                            print(f"     Dtype: {output_tensor.dtype}")
                            print(f"     Min/Max: {output_tensor.min():.6f}/{output_tensor.max():.6f}")

                            # Save raw output for analysis
                            np.save(str(self.output_dir / f"raw_output_{output_name}.npy"), output_tensor)

                            # Look for detection-like data
                            if len(output_tensor.shape) >= 2:
                                print(f"     🔍 Analyzing for detections...")

                                # For YOLO, detections are usually in format (batch, boxes, 5+classes)
                                # where 5 = x1,y1,x2,y2,confidence or cx,cy,w,h,confidence
                                if output_tensor.shape[-1] >= 5:
                                    print(f"       Potential detection tensor!")

                                    # Flatten batch dimension if present
                                    if len(output_tensor.shape) == 3:
                                        detections = output_tensor[0]  # Remove batch dim
                                    else:
                                        detections = output_tensor

                                    print(f"       Detection shape: {detections.shape}")

                                    # Look at confidence values (assuming last column or 5th column)
                                    if detections.shape[1] >= 5:
                                        conf_idx = 4  # 5th column (0-indexed)
                                        confidences = detections[:, conf_idx]
                                        print(f"       Confidence stats:")
                                        print(f"         Range: {confidences.min():.6f} to {confidences.max():.6f}")
                                        print(f"         > 0.1: {np.sum(confidences > 0.1)}")
                                        print(f"         > 0.3: {np.sum(confidences > 0.3)}")
                                        print(f"         > 0.5: {np.sum(confidences > 0.5)}")

                                        # Show top detections
                                        top_indices = np.argsort(confidences)[-5:]
                                        print(f"       Top 5 detections:")
                                        for idx in reversed(top_indices):
                                            if conf_idx < detections.shape[1]:
                                                x1, y1, x2, y2, conf = detections[idx, :5]
                                                print(f"         [{idx}] ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) conf={conf:.6f}")

                        elif isinstance(output_tensor, list):
                            print(f"     List length: {len(output_tensor)}")
                            if len(output_tensor) > 0:
                                print(f"     First element type: {type(output_tensor[0])}")

                                # Handle nested list structure
                                first_elem = None
                                if isinstance(output_tensor[0], list) and len(output_tensor[0]) > 0:
                                    print(f"     Nested list! Second level length: {len(output_tensor[0])}")
                                    if hasattr(output_tensor[0][0], 'shape'):
                                        first_elem = output_tensor[0][0]  # Get the actual tensor
                                        print(f"     Actual tensor type: {type(first_elem)}")
                                    else:
                                        print(f"     No tensor found in nested structure")
                                elif hasattr(output_tensor[0], 'shape'):
                                    first_elem = output_tensor[0]

                                if first_elem is not None and hasattr(first_elem, 'shape'):
                                    print(f"     Shape: {first_elem.shape}")
                                    print(f"     Dtype: {first_elem.dtype}")
                                    print(f"     Min/Max: {first_elem.min():.6f}/{first_elem.max():.6f}")

                                    # Save raw output for analysis
                                    np.save(str(self.output_dir / f"raw_output_{output_name}.npy"), first_elem)

                                    # Analyze detection format
                                    # YOLOv8 NMS output is usually (classes, 5, detections)
                                    # where 5 = [x1, y1, x2, y2, confidence]
                                    if len(first_elem.shape) == 3:
                                        classes, coords, max_dets = first_elem.shape
                                        print(f"     🎯 NMS output format: {classes} classes, {coords} coords, {max_dets} max detections")

                                        if coords == 5:  # x1,y1,x2,y2,conf
                                            # Look across all classes for detections
                                            total_detections = 0
                                            valid_detections = []

                                            for class_idx in range(classes):
                                                class_dets = first_elem[class_idx]  # (5, 100)
                                                confidences = class_dets[4, :]  # confidence row

                                                # Find valid detections (confidence > 0)
                                                valid_mask = confidences > 0.0
                                                valid_count = np.sum(valid_mask)

                                                if valid_count > 0:
                                                    total_detections += valid_count
                                                    valid_indices = np.where(valid_mask)[0]

                                                    for det_idx in valid_indices[:5]:  # Show top 5 per class
                                                        x1 = class_dets[0, det_idx]
                                                        y1 = class_dets[1, det_idx]
                                                        x2 = class_dets[2, det_idx]
                                                        y2 = class_dets[3, det_idx]
                                                        conf = class_dets[4, det_idx]
                                                        valid_detections.append((class_idx, x1, y1, x2, y2, conf))

                                            print(f"     📊 Detection analysis:")
                                            print(f"       Total valid detections: {total_detections}")
                                            print(f"       Detections per confidence threshold:")
                                            all_confs = first_elem[:, 4, :].flatten()
                                            print(f"         > 0.1: {np.sum(all_confs > 0.1)}")
                                            print(f"         > 0.3: {np.sum(all_confs > 0.3)}")
                                            print(f"         > 0.5: {np.sum(all_confs > 0.5)}")

                                            if valid_detections:
                                                print(f"       Top detections:")
                                                # Sort by confidence
                                                valid_detections.sort(key=lambda x: x[5], reverse=True)
                                                for i, (cls, x1, y1, x2, y2, conf) in enumerate(valid_detections[:5]):
                                                    print(f"         [{i}] Class {cls}: ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) conf={conf:.6f}")
                                            else:
                                                print(f"       ❌ No valid detections found!")

        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()

    def cleanup(self):
        """Cleanup resources"""
        if self.camera:
            self.camera.stop()
        print(f"\n✅ Test complete! Results in: {self.output_dir}")

def main():
    tester = YOLOv8BaselineTest()

    if not tester.initialize():
        print("❌ Initialization failed")
        return

    print("\n📋 Testing YOLOv8m model to establish baseline")
    print("This will help determine if the issue is with dogdetector_14.hef or our pipeline")

    tester.test_detection()
    tester.cleanup()

if __name__ == "__main__":
    main()