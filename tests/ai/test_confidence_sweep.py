#!/usr/bin/env python3
"""
Test Different Confidence Thresholds
Find what confidence levels produce reasonable detections
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_pi_1024x768 import PoseDetectionApp
import time

def test_confidence_threshold(threshold, duration=10):
    """Test pose detection with specific confidence threshold"""

    print(f"\n=== TESTING CONFIDENCE THRESHOLD: {threshold} ===")

    # Temporarily modify the threshold in the decode function
    import run_pi_1024x768

    # Create a custom decode function with the specified threshold
    def custom_decode(raw_outputs):
        # Copy of original decode but with custom threshold
        detections = []
        scales = {
            '128x96': {'bbox': None, 'kpts': None, 'conf': None},
            '64x48': {'bbox': None, 'kpts': None, 'conf': None},
            '32x24': {'bbox': None, 'kpts': None, 'conf': None}
        }

        # Map outputs (same as original)
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

        # Process with custom threshold
        all_predictions = []
        strides = [8, 16, 32]
        scale_names = ['128x96', '64x48', '32x24']

        detection_stats = {'total_checked': 0, 'above_threshold': 0, 'max_conf': 0}

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

            for i in range(h):
                for j in range(w):
                    conf_raw = conf_out[0, i, j, 0]
                    conf = 1.0 / (1.0 + np.exp(-conf_raw))

                    detection_stats['total_checked'] += 1
                    detection_stats['max_conf'] = max(detection_stats['max_conf'], conf)

                    if conf < threshold:  # CUSTOM THRESHOLD HERE
                        continue

                    detection_stats['above_threshold'] += 1

                    # Decode bounding box
                    bbox_raw = bbox_out[0, i, j, :4]
                    cx = (bbox_raw[0] + j) * stride
                    cy = (bbox_raw[1] + i) * stride
                    bbox_w = np.exp(bbox_raw[2]) * stride
                    bbox_h = np.exp(bbox_raw[3]) * stride

                    x1 = cx - bbox_w / 2
                    y1 = cy - bbox_h / 2
                    x2 = cx + bbox_w / 2
                    y2 = cy + bbox_h / 2

                    # Decode keypoints
                    kpts_raw = kpts_out[0, i, j, :]
                    kpts = np.zeros((24, 3), dtype=np.float32)

                    for k in range(24):
                        kpts[k, 0] = (kpts_raw[k * 3] + j) * stride
                        kpts[k, 1] = (kpts_raw[k * 3 + 1] + i) * stride
                        kpts[k, 2] = 1.0 / (1.0 + np.exp(-kpts_raw[k * 3 + 2]))

                    all_predictions.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'keypoints': kpts,
                        'confidence': conf,
                        'bbox_area': bbox_w * bbox_h
                    })

        # Apply NMS (simplified)
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        detections = all_predictions[:10]  # Keep top 10

        print(f"  Threshold {threshold}: {len(detections)} detections")
        print(f"  Max confidence found: {detection_stats['max_conf']:.4f}")
        print(f"  Cells above threshold: {detection_stats['above_threshold']}/{detection_stats['total_checked']}")

        if detections:
            bbox_areas = [d['bbox_area'] for d in detections]
            print(f"  Bbox areas: {min(bbox_areas):.0f} to {max(bbox_areas):.0f} pixels")

            # Check if any detection looks reasonable for a dog
            reasonable_detections = [d for d in detections if d['bbox_area'] > 10000]  # At least 100x100 pixels
            print(f"  Reasonable sized detections: {len(reasonable_detections)}")

        return detections

    # Patch the decode function temporarily
    original_decode = run_pi_1024x768.decode_hailo_pose_outputs
    run_pi_1024x768.decode_hailo_pose_outputs = custom_decode

    try:
        # Run detection with this threshold
        app = PoseDetectionApp()
        if not app.initialize():
            print("Failed to initialize app")
            return

        # Take a few frames
        frame_count = 0
        start_time = time.time()

        while time.time() - start_time < duration and frame_count < 5:
            frame = app.camera.capture_array()
            if frame is not None:
                results = app.process_frame(frame)
                frame_count += 1
                time.sleep(0.5)

    finally:
        # Restore original decode function
        run_pi_1024x768.decode_hailo_pose_outputs = original_decode
        app.cleanup()

def main():
    """Test multiple confidence thresholds"""

    print("=== CONFIDENCE THRESHOLD SWEEP ===")
    print("Testing different thresholds to find usable detections...")

    # Test a range of thresholds
    thresholds = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25]

    for threshold in thresholds:
        test_confidence_threshold(threshold, duration=8)
        time.sleep(2)  # Brief pause between tests

    print("\n=== RECOMMENDATIONS ===")
    print("Look for threshold that gives:")
    print("- Reasonable bbox areas (>10,000 pixels)")
    print("- Not too many false positives")
    print("- Max confidence >0.5 for good detections")

if __name__ == "__main__":
    main()