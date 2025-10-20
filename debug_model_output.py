#!/usr/bin/env python3
"""
Deep debug of model inference outputs
Check raw model outputs to see what's actually happening
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

import cv2
from core.ai_controller_3stage_fixed import AI3StageControllerFixed

class ModelOutputDebugger:
    """Debug raw model outputs"""

    def __init__(self):
        self.output_dir = Path(f"debug_model_output_{time.strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(exist_ok=True)

        self.ai = AI3StageControllerFixed()
        self.camera = None

    def initialize(self):
        """Initialize"""
        print("🔬 Model Output Debugger")
        print("=" * 30)

        if not self.ai.initialize():
            print("❌ AI failed")
            return False
        print("✅ AI ready")

        # Camera
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

        return False

    def debug_model_outputs(self):
        """Debug raw model outputs"""
        print("\n🔍 Debugging Model Outputs")

        # Capture frame
        frame = self.camera.capture_array()
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        print(f"📷 Frame: {frame.shape}")

        # Save input frame
        cv2.imwrite(str(self.output_dir / "input_frame.jpg"), frame)

        # Process through detection manually to see raw outputs
        print("\n🤖 Running Detection Inference...")

        # Downsample like the AI does
        h_4k, w_4k = frame.shape[:2]
        frame_640 = cv2.resize(frame, (640, 640))

        # Convert BGR to RGB like the AI does
        frame_rgb = cv2.cvtColor(frame_640, cv2.COLOR_BGR2RGB)

        print(f"   Input to model: {frame_rgb.shape}, dtype: {frame_rgb.dtype}")
        print(f"   Min/Max: {frame_rgb.min()}/{frame_rgb.max()}")

        # Save preprocessed input
        cv2.imwrite(str(self.output_dir / "preprocessed_input.jpg"), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        # Run actual inference and intercept outputs
        try:
            with self.ai.detection_network_group.activate(self.ai.detection_network_group_params):
                import hailo_platform as hpf
                with hpf.InferVStreams(self.ai.detection_network_group,
                                      self.ai.detection_input_vstreams_params,
                                      self.ai.detection_output_vstreams_params) as infer_pipeline:

                    # Prepare input exactly like the AI does
                    input_tensor = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)
                    input_name = list(self.ai.detection_input_vstreams_params.keys())[0]
                    input_data = {input_name: input_tensor}

                    print(f"   Input tensor: {input_tensor.shape}, dtype: {input_tensor.dtype}")

                    # Run inference
                    output_data = infer_pipeline.infer(input_data)

                    print(f"\n📊 Raw Model Outputs:")
                    print(f"   Number of outputs: {len(output_data)}")

                    for output_name, output_tensor in output_data.items():
                        print(f"\n   Output: {output_name}")
                        print(f"     Type: {type(output_tensor)}")

                        if isinstance(output_tensor, list):
                            print(f"     List length: {len(output_tensor)}")
                            if len(output_tensor) > 0:
                                print(f"     First element type: {type(output_tensor[0])}")
                                if hasattr(output_tensor[0], 'shape'):
                                    print(f"     First element shape: {output_tensor[0].shape}")
                                    print(f"     First element dtype: {output_tensor[0].dtype}")
                                    print(f"     First element min/max: {output_tensor[0].min():.6f}/{output_tensor[0].max():.6f}")

                                    # Save raw output values to file for analysis
                                    np.save(str(self.output_dir / f"raw_output_{output_name}.npy"), output_tensor[0])

                                    # Print some actual values
                                    flat_vals = output_tensor[0].flatten()
                                    print(f"     Sample values: {flat_vals[:10]}")
                                    print(f"     Non-zero count: {np.count_nonzero(flat_vals)}/{len(flat_vals)}")

                                    # Check if this looks like the NMS output format (should be 1,5,100)
                                    if len(output_tensor[0].shape) == 3 and output_tensor[0].shape[1] == 5:
                                        print(f"     ✅ This looks like NMS output format!")
                                        detections_data = output_tensor[0][0].T  # (5,100) -> (100,5)
                                        print(f"     Detection data shape: {detections_data.shape}")

                                        # Check confidence values
                                        confidences = detections_data[:, 4]  # Last column should be confidence
                                        print(f"     Confidence range: {confidences.min():.6f} to {confidences.max():.6f}")
                                        print(f"     Confidences > 0.1: {np.sum(confidences > 0.1)}")
                                        print(f"     Confidences > 0.3: {np.sum(confidences > 0.3)}")
                                        print(f"     Top 10 confidences: {np.sort(confidences)[-10:]}")

                        elif hasattr(output_tensor, 'shape'):
                            print(f"     Shape: {output_tensor.shape}")
                            print(f"     Dtype: {output_tensor.dtype}")
                            print(f"     Min/Max: {output_tensor.min():.6f}/{output_tensor.max():.6f}")

        except Exception as e:
            print(f"❌ Inference failed: {e}")
            import traceback
            traceback.print_exc()

        self.camera.stop()
        print(f"\n✅ Debug complete! Check: {self.output_dir}")

def main():
    debugger = ModelOutputDebugger()

    if not debugger.initialize():
        print("❌ Failed to initialize")
        return

    print("\n📋 This will show raw model outputs")
    print("Make sure dogs are visible in camera view!")

    debugger.debug_model_outputs()

if __name__ == "__main__":
    main()