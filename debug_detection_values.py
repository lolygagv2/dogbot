#!/usr/bin/env python3
"""
Debug Detection Values - Add confidence logging to understand what's being detected
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_pi_1024x768 import *

# Patch the decode function to add detailed logging
original_decode = decode_hailo_pose_outputs

def debug_decode_hailo_pose_outputs(raw_outputs):
    """Debug version with detailed confidence logging"""

    print("\n=== DETAILED DECODE DEBUG ===")

    detections = []
    scales = {
        '128x96': {'bbox': None, 'kpts': None, 'conf': None},
        '64x48': {'bbox': None, 'kpts': None, 'conf': None},
        '32x24': {'bbox': None, 'kpts': None, 'conf': None}
    }

    # Map conv layers
    for layer_name, output in raw_outputs.items():
        h, w = output.shape[1], output.shape[2] if len(output.shape) == 4 else (0, 0)
        channels = output.shape[3] if len(output.shape) == 4 else 0

        scale_name = None
        output_type = None

        if (h, w) == (128, 96) or (h, w) == (96, 128):
            scale_name = '128x96'
        elif (h, w) == (64, 48) or (h, w) == (48, 64):
            scale_name = '64x48'
        elif (h, w) == (32, 24) or (h, w) == (24, 32):
            scale_name = '32x24'

        if scale_name and channels == 64:
            output_type = 'bbox'
        elif scale_name and channels == 72:
            output_type = 'kpts'
        elif scale_name and channels == 1:
            output_type = 'conf'

        if scale_name and output_type:
            scales[scale_name][output_type] = output

    # Process each scale with detailed logging
    all_predictions = []
    strides = [8, 16, 32]
    scale_names = ['128x96', '64x48', '32x24']

    total_cells_checked = 0
    cells_above_threshold = 0
    confidence_values = []

    for scale_idx, scale_name in enumerate(scale_names):
        scale_data = scales[scale_name]

        if not all(scale_data[key] is not None for key in ['bbox', 'kpts', 'conf']):
            continue

        bbox_out = scale_data['bbox']
        kpts_out = scale_data['kpts']
        conf_out = scale_data['conf']

        _, h, w, _ = bbox_out.shape
        h, w = int(h), int(w)
        stride = strides[scale_idx]

        print(f"\nScale {scale_name} ({h}x{w}), stride={stride}:")

        # Sample confidence values across the grid
        conf_sample = []
        for i in range(0, h, max(1, h//5)):  # Sample 5 rows
            for j in range(0, w, max(1, w//5)):  # Sample 5 columns
                conf_raw = conf_out[0, i, j, 0]
                conf = 1.0 / (1.0 + np.exp(-conf_raw))
                conf_sample.append(conf)
                confidence_values.append(conf)
                total_cells_checked += 1

                if conf >= 0.25:
                    cells_above_threshold += 1
                    print(f"  HIGH CONF at ({i},{j}): {conf:.4f}")

        max_conf = max(conf_sample)
        min_conf = min(conf_sample)
        avg_conf = sum(conf_sample) / len(conf_sample)

        print(f"  Confidence range: {min_conf:.6f} to {max_conf:.6f}, avg: {avg_conf:.6f}")

    print(f"\nOVERALL STATS:")
    print(f"Total cells checked: {total_cells_checked}")
    print(f"Cells above 0.25 threshold: {cells_above_threshold}")
    print(f"Global confidence range: {min(confidence_values):.6f} to {max(confidence_values):.6f}")
    print(f"Average confidence: {sum(confidence_values)/len(confidence_values):.6f}")

    # Suggest threshold adjustments
    if max(confidence_values) < 0.1:
        print(f"🚨 ISSUE: Max confidence {max(confidence_values):.4f} is very low - model may not be detecting dogs")
    elif max(confidence_values) < 0.25:
        print(f"⚠️  SUGGESTION: Lower threshold from 0.25 to {max(confidence_values)*0.8:.3f}")

    # Call original decode to get actual detections
    return original_decode(raw_outputs)

# Monkey patch the decode function
decode_hailo_pose_outputs = debug_decode_hailo_pose_outputs

print("Debug decode function installed. Run your pose detection test to see detailed confidence analysis.")