#!/usr/bin/env python3
"""
Test with very low confidence threshold to see if we get any detections
"""

import sys
sys.path.insert(0, '/home/morgan/dogbot')

from test_pose_headless import HeadlessPoseApp
import numpy as np

class LowThresholdApp(HeadlessPoseApp):
    """App with very low confidence threshold for testing"""

    def __init__(self):
        super().__init__()

    def process_frame(self, frame):
        """Process frame with lower threshold"""
        # Temporarily monkey-patch the decode function to use lower threshold
        import run_pi_1024x768

        # Store original function
        original_decode = run_pi_1024x768.decode_hailo_pose_outputs

        def low_threshold_decode(raw_outputs):
            """Modified decoder with very low threshold"""
            if not isinstance(raw_outputs, dict):
                return []

            print(f"\n[LOW_DECODE] Processing {len(raw_outputs)} outputs")
            detections = []

            # Group outputs by scale
            scales = {
                '128x96': {'bbox': None, 'kpts': None, 'conf': None},
                '64x48': {'bbox': None, 'kpts': None, 'conf': None},
                '32x24': {'bbox': None, 'kpts': None, 'conf': None}
            }

            # Map conv layers based on shape patterns
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

            # Process each scale with VERY low threshold
            all_predictions = []
            strides = [8, 16, 32]
            scale_names = ['128x96', '64x48', '32x24']

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

                detection_count = 0
                max_per_scale = 5  # Limit detections per scale

                # Process with VERY low threshold
                for i in range(h):
                    for j in range(w):
                        if detection_count >= max_per_scale:
                            break

                        conf_raw = conf_out[0, i, j, 0]
                        conf = 1.0 / (1.0 + np.exp(-conf_raw))  # Sigmoid

                        # VERY low threshold - almost anything
                        if conf < 0.00001:  # Extremely low threshold
                            continue

                        print(f"[LOW_DECODE] Scale {scale_name} cell ({i},{j}): conf={conf:.8f}")

                        # Decode bounding box
                        bbox_raw = bbox_out[0, i, j, :4]
                        cx = (bbox_raw[0] + j) * stride
                        cy = (bbox_raw[1] + i) * stride
                        w_box = np.exp(bbox_raw[2]) * stride
                        h_box = np.exp(bbox_raw[3]) * stride

                        x1 = cx - w_box / 2
                        y1 = cy - h_box / 2
                        x2 = cx + w_box / 2
                        y2 = cy + h_box / 2

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
                            'confidence': conf
                        })
                        detection_count += 1

                print(f"[LOW_DECODE] Scale {scale_name}: found {detection_count} detections")

            print(f"[LOW_DECODE] Total raw predictions: {len(all_predictions)}")

            # Simple filtering - no NMS, just take top 3
            if all_predictions:
                all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                detections = all_predictions[:3]  # Top 3

            print(f"[LOW_DECODE] Final detections: {len(detections)}")
            return detections

        # Temporarily replace the function
        run_pi_1024x768.decode_hailo_pose_outputs = low_threshold_decode

        try:
            # Process with modified decoder
            result = super().process_frame(frame)
        finally:
            # Restore original function
            run_pi_1024x768.decode_hailo_pose_outputs = original_decode

        return result

def main():
    print("=" * 50)
    print("LOW THRESHOLD DETECTION TEST")
    print("=" * 50)

    app = LowThresholdApp()

    if not app.initialize():
        print("Failed to initialize")
        return 1

    try:
        app.run_headless(duration=10)
    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    return 0

if __name__ == "__main__":
    exit(main())