#!/usr/bin/env python3
"""
Debug raw detection model output format
Print everything we can about the model output structure
"""

import sys
import os
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from picamera2 import Picamera2
except ImportError:
    print("No Picamera2")
    exit()

import cv2
from core.ai_controller_3stage_fixed import AI3StageControllerFixed

def deep_inspect(obj, name="object", max_depth=5, current_depth=0):
    """Recursively inspect object structure"""
    indent = "  " * current_depth

    if current_depth >= max_depth:
        print(f"{indent}{name}: <max depth reached>")
        return

    print(f"{indent}{name}: {type(obj)}")

    if isinstance(obj, list):
        print(f"{indent}  Length: {len(obj)}")
        if len(obj) > 0:
            print(f"{indent}  First element:")
            deep_inspect(obj[0], "obj[0]", max_depth, current_depth + 2)
        if len(obj) > 1:
            print(f"{indent}  Second element:")
            deep_inspect(obj[1], "obj[1]", max_depth, current_depth + 2)

    elif isinstance(obj, np.ndarray):
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
        print(f"{indent}  Min/Max: {obj.min():.6f}/{obj.max():.6f}")
        print(f"{indent}  Non-zero: {np.count_nonzero(obj)}/{obj.size}")

        # Show some actual values
        flat = obj.flatten()
        if len(flat) <= 20:
            print(f"{indent}  Values: {flat}")
        else:
            print(f"{indent}  First 10: {flat[:10]}")
            print(f"{indent}  Last 10: {flat[-10:]}")

    elif hasattr(obj, '__len__') and not isinstance(obj, str):
        try:
            print(f"{indent}  Length: {len(obj)}")
        except:
            pass

    # Try to show first few attributes/items
    if hasattr(obj, '__dict__'):
        items = list(obj.__dict__.items())[:3]
        for key, val in items:
            print(f"{indent}  .{key}:")
            deep_inspect(val, f"{key}", max_depth, current_depth + 2)

def main():
    print("🔬 Raw Detection Output Debugger")

    # Initialize
    ai = AI3StageControllerFixed()
    if not ai.initialize():
        print("❌ AI failed")
        return

    camera = Picamera2()
    camera.configure(camera.create_still_configuration(main={"size": (1920, 1080)}))
    camera.start()

    print("✅ Ready")

    # Capture frame
    frame = camera.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Process exactly like the AI does
    frame_640 = cv2.resize(frame, (640, 640))
    frame_rgb = cv2.cvtColor(frame_640, cv2.COLOR_BGR2RGB)

    print(f"\n📷 Input: {frame_rgb.shape}, {frame_rgb.dtype}")

    # Run inference and capture raw output
    try:
        with ai.detection_network_group.activate(ai.detection_network_group_params):
            import hailo_platform as hpf
            with hpf.InferVStreams(ai.detection_network_group,
                                  ai.detection_input_vstreams_params,
                                  ai.detection_output_vstreams_params) as infer_pipeline:

                input_tensor = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)
                input_name = list(ai.detection_input_vstreams_params.keys())[0]
                input_data = {input_name: input_tensor}

                print(f"🚀 Running inference...")
                output_data = infer_pipeline.infer(input_data)

                print(f"\n🔍 RAW OUTPUT STRUCTURE:")
                print(f"Output dict keys: {list(output_data.keys())}")

                for output_name, output_tensor in output_data.items():
                    print(f"\n" + "="*50)
                    print(f"OUTPUT: {output_name}")
                    print("="*50)
                    deep_inspect(output_tensor, f"output_data['{output_name}']")

                    # Try to extract meaningful data
                    print(f"\n🧪 ATTEMPTING DATA EXTRACTION:")

                    try:
                        if isinstance(output_tensor, list):
                            if len(output_tensor) > 0:
                                if isinstance(output_tensor[0], list):
                                    print("   Double nested list detected")
                                    if len(output_tensor[0]) > 0:
                                        data = np.array(output_tensor[0])
                                        print(f"   Converted to array: {data.shape}")
                                        if data.size > 0:
                                            print(f"   Data range: {data.min():.6f} to {data.max():.6f}")

                                            # Check if this could be detection data
                                            if len(data.shape) == 2 and data.shape[1] >= 5:
                                                print(f"   🎯 Looks like detection format! Shape: {data.shape}")
                                                confidences = data[:, 4] if data.shape[1] > 4 else data[:, -1]
                                                print(f"   Confidence column stats:")
                                                print(f"     Min: {confidences.min():.6f}")
                                                print(f"     Max: {confidences.max():.6f}")
                                                print(f"     Mean: {confidences.mean():.6f}")
                                                print(f"     > 0.1: {np.sum(confidences > 0.1)}")
                                                print(f"     > 0.3: {np.sum(confidences > 0.3)}")
                                                print(f"     > 0.5: {np.sum(confidences > 0.5)}")

                                                # Show top detections
                                                top_indices = np.argsort(confidences)[-5:]
                                                print(f"   Top 5 detections:")
                                                for idx in reversed(top_indices):
                                                    if data.shape[1] >= 5:
                                                        x1, y1, x2, y2, conf = data[idx, :5]
                                                        print(f"     [{idx}] ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}) conf={conf:.6f}")
                    except Exception as e:
                        print(f"   ❌ Extraction failed: {e}")

    except Exception as e:
        print(f"❌ Inference error: {e}")
        import traceback
        traceback.print_exc()

    camera.stop()
    print(f"\n✅ Debug complete!")

if __name__ == "__main__":
    main()