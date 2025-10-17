#!/usr/bin/env python3
"""
Debug the exact output format from AI controller vs working debug script
"""

import sys
import os
import numpy as np
import cv2
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ai_controller_3stage_fixed import AI3StageControllerFixed
import hailo_platform as hpf

def debug_ai_controller_parsing():
    """Debug the AI controller's actual parsing"""

    # Use the 1-dog frame
    frame_path = "simple_test_results_20251016_174201/frame_00300_174245.jpg"

    if not os.path.exists(frame_path):
        print(f"❌ Frame not found: {frame_path}")
        return

    frame = cv2.imread(frame_path)
    print(f"📷 Testing with: {frame_path}")
    print(f"📏 Frame shape: {frame.shape}")

    # Initialize AI controller
    ai = AI3StageControllerFixed()
    if not ai.initialize():
        print("❌ AI init failed")
        return

    print("✅ AI controller initialized")

    # Now let's manually run the detection and capture the raw output
    h_4k, w_4k = frame.shape[:2]
    frame_640 = cv2.resize(frame, (640, 640))
    frame_bgr = frame_640  # Use BGR as we determined

    print(f"📐 Resized to: {frame_640.shape}")

    try:
        with ai.detection_network_group.activate(ai.detection_network_group_params):
            with hpf.InferVStreams(ai.detection_network_group,
                                  ai.detection_input_vstreams_params,
                                  ai.detection_output_vstreams_params) as infer_pipeline:

                # Prepare input exactly like AI controller does
                input_tensor = np.expand_dims(frame_bgr, axis=0).astype(np.uint8)
                input_name = list(ai.detection_input_vstreams_params.keys())[0]
                input_data = {input_name: input_tensor}

                print(f"🚀 Running inference...")
                output_data = infer_pipeline.infer(input_data)

                print(f"\n🔍 RAW OUTPUT ANALYSIS:")
                for output_name, output_tensor in output_data.items():
                    print(f"\nOutput: {output_name}")
                    print(f"  Type: {type(output_tensor)}")

                    if isinstance(output_tensor, list):
                        print(f"  List length: {len(output_tensor)}")
                        if len(output_tensor) > 0:
                            print(f"  First element type: {type(output_tensor[0])}")

                            if isinstance(output_tensor[0], list):
                                print(f"  Nested list! Second level length: {len(output_tensor[0])}")
                                if len(output_tensor[0]) > 0:
                                    actual_data = output_tensor[0][0]
                                    print(f"  Actual tensor type: {type(actual_data)}")
                                    if hasattr(actual_data, 'shape'):
                                        print(f"  Actual tensor shape: {actual_data.shape}")
                                        print(f"  Data type: {actual_data.dtype}")
                                        print(f"  Sample values: {actual_data.flatten()[:10]}")

                                        # Show detection count and confidence values
                                        if len(actual_data.shape) == 2 and actual_data.shape[1] == 5:
                                            print(f"  🎯 Detection format confirmed: {actual_data.shape[0]} detections")
                                            if actual_data.shape[0] > 0:
                                                confidences = actual_data[:, 4]
                                                print(f"  Confidences: {confidences}")
                                                print(f"  Max confidence: {confidences.max():.6f}")
                                                print(f"  Detections > 0.1: {np.sum(confidences > 0.1)}")
                                                print(f"  Detections > 0.05: {np.sum(confidences > 0.05)}")
                                                print(f"  Detections > 0.01: {np.sum(confidences > 0.01)}")
                                        else:
                                            print(f"  ❌ Unexpected shape for detections")

                            elif hasattr(output_tensor[0], 'shape'):
                                print(f"  Direct array shape: {output_tensor[0].shape}")

                    elif hasattr(output_tensor, 'shape'):
                        print(f"  Direct tensor shape: {output_tensor.shape}")

                # Now test our parsing method on this raw output
                print(f"\n🔧 TESTING AI CONTROLLER PARSING:")
                detections = ai._parse_detection_output(output_data, w_4k, h_4k)
                print(f"  Parsed detections: {len(detections)}")
                for i, det in enumerate(detections):
                    print(f"    Det {i}: conf={det.confidence:.6f}, box=({det.x1},{det.y1},{det.x2},{det.y2})")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

    ai.cleanup()

if __name__ == "__main__":
    debug_ai_controller_parsing()