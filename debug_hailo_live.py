#!/usr/bin/env python3
"""
Debug Hailo detection with live camera capture
Captures a frame and saves raw model outputs for analysis
"""

import sys
import cv2
import numpy as np
import logging
import json
from datetime import datetime

# Add project to path
sys.path.insert(0, '/home/morgan/dogbot')

from core.vision.detection_plugins.hailo_detector import HailoDetector

logging.basicConfig(level=logging.DEBUG,
                   format='%(levelname)s:%(name)s:%(message)s')

def capture_frame():
    """Capture a frame from the camera"""
    print("📸 Capturing frame from camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Failed to open camera")
        return None

    # Let camera stabilize
    for _ in range(5):
        ret, frame = cap.read()

    ret, frame = cap.read()
    cap.release()

    if ret:
        print(f"✅ Captured frame: {frame.shape}")
        # Save the raw frame for inspection
        cv2.imwrite("debug_capture.jpg", frame)
        print("   Saved as debug_capture.jpg")
        return frame
    else:
        print("❌ Failed to capture frame")
        return None

def debug_hailo_detection():
    """Debug Hailo detection with real camera frame"""
    print("=" * 60)
    print("Hailo Detection Debug - Live Camera")
    print("=" * 60)

    # Capture frame
    frame = capture_frame()
    if frame is None:
        print("❌ No frame to process")
        return

    # Initialize detector with very low threshold for debugging
    print("\n🔧 Initializing Hailo detector...")
    detector = HailoDetector({
        'hailo_model_path': '/home/morgan/dogbot/ai/models/best_dogbot.hef',
        'confidence_threshold': 0.1  # Very low threshold to see all detections
    })

    if not detector.initialized:
        print("❌ Failed to initialize Hailo detector")
        return

    print("✅ Hailo detector initialized")
    print(f"   Model: best_dogbot.hef")
    print(f"   Input shape: {detector.input_shape}")
    print(f"   Confidence threshold: 0.1 (debug mode)")

    # Run detection
    print("\n🔍 Running detection...")

    # Monkey-patch the postprocess to capture raw outputs
    original_postprocess = detector._postprocess_outputs
    raw_outputs = {}

    def debug_postprocess(output_dict, original_shape):
        # Save raw outputs for analysis
        nonlocal raw_outputs
        raw_outputs = output_dict.copy()

        print("\n📊 Raw Model Outputs:")
        for key, value in output_dict.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}:")
                print(f"     - Shape: {value.shape}")
                print(f"     - dtype: {value.dtype}")
                print(f"     - Min: {value.min():.6f}")
                print(f"     - Max: {value.max():.6f}")
                print(f"     - Mean: {value.mean():.6f}")

                # Save raw output for detailed inspection
                np.save(f"debug_output_{key}.npy", value)

                # Show first few values
                flat = value.flatten()
                print(f"     - First 10 values: {flat[:10]}")

                # Check if there are any high confidence values
                if len(value.shape) == 2 and value.shape[1] >= 5:
                    # Assuming YOLO format: [x, y, w, h, conf, classes...]
                    confidences = value[:, 4]
                    high_conf = confidences[confidences > 0.1]
                    print(f"     - Detections with conf > 0.1: {len(high_conf)}")
                    if len(high_conf) > 0:
                        print(f"     - Top 5 confidences: {sorted(high_conf, reverse=True)[:5]}")

        # Call original postprocess
        return original_postprocess(output_dict, original_shape)

    detector._postprocess_outputs = debug_postprocess

    try:
        detections = detector.detect(frame)

        print(f"\n✅ Detection completed")
        print(f"   Number of detections: {len(detections)}")

        if detections:
            print("\n📦 Detected Objects:")
            for i, det in enumerate(detections):
                print(f"\n   Detection {i+1}:")
                print(f"   - Confidence: {det['confidence']:.4f}")
                print(f"   - BBox: {det['bbox']}")
                print(f"   - Center: {det['center']}")
                print(f"   - Class ID: {det.get('class_id', 'N/A')}")

                # Draw on frame
                x, y, w, h = det['bbox']
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{det['confidence']:.2f}",
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)

            # Save annotated frame
            cv2.imwrite("debug_detections.jpg", frame)
            print("\n   Annotated frame saved as debug_detections.jpg")
        else:
            print("\n⚠️ No objects detected")
            print("   This could mean:")
            print("   1. The model confidence is too low")
            print("   2. The model isn't trained for the objects in view")
            print("   3. The model output format is different than expected")
            print("   4. There's an issue with preprocessing or postprocessing")

        # Save debug info
        debug_info = {
            'timestamp': datetime.now().isoformat(),
            'frame_shape': frame.shape,
            'input_shape': detector.input_shape,
            'num_detections': len(detections),
            'detections': detections,
            'raw_output_shapes': {k: v.shape if isinstance(v, np.ndarray) else str(v)
                                 for k, v in raw_outputs.items()}
        }

        with open('debug_info.json', 'w') as f:
            json.dump(debug_info, f, indent=2, default=str)
        print("\n📝 Debug info saved to debug_info.json")

    except Exception as e:
        print(f"\n❌ Detection failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n🧹 Cleaning up...")
        detector.cleanup()
        print("✅ Done")

    print("\n" + "=" * 60)
    print("Debug files created:")
    print("  - debug_capture.jpg (raw camera frame)")
    print("  - debug_detections.jpg (annotated frame if detections found)")
    print("  - debug_output_*.npy (raw model outputs)")
    print("  - debug_info.json (detection summary)")
    print("=" * 60)

if __name__ == "__main__":
    debug_hailo_detection()