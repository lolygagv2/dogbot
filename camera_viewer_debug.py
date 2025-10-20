#!/usr/bin/env python3
"""
Debug version of camera viewer that shows raw detection outputs
"""

import cv2
import numpy as np
import time
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.vision.detection_plugins.hailo_detector import HailoDetector

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG,
                   format='%(levelname)s:%(name)s:%(message)s')

def main():
    """Run debug camera viewer"""
    print("=" * 60)
    print("🐕 DogBot Debug Camera Viewer")
    print("=" * 60)

    # Initialize Hailo detector with very low threshold
    print("\n🔧 Initializing Hailo detector...")
    detector = HailoDetector({
        'hailo_model_path': '/home/morgan/dogbot/ai/models/best_dogbot.hef',
        'confidence_threshold': 0.05  # VERY low threshold for debugging
    })

    if not detector.initialized:
        print("❌ Failed to initialize Hailo detector")
        return

    print("✅ Hailo detector initialized")
    print(f"   Confidence threshold: 0.05 (debug mode)")
    print("\nPress 'q' to quit, 's' to save current frame")
    print("Press 'd' to dump raw detection data")
    print("=" * 60)

    # Try different camera indices
    cap = None
    for idx in [0, 1, 2]:
        print(f"\n🎥 Trying camera index {idx}...")
        test_cap = cv2.VideoCapture(idx)
        if test_cap.isOpened():
            ret, frame = test_cap.read()
            if ret:
                print(f"✅ Camera {idx} works!")
                cap = test_cap
                break
            test_cap.release()

    if cap is None:
        # Try Picamera2 if available
        print("\n🎥 Trying Picamera2...")
        try:
            from picamera2 import Picamera2
            picam2 = Picamera2()
            config = picam2.create_preview_configuration(main={"size": (640, 480)})
            picam2.configure(config)
            picam2.start()
            time.sleep(2)  # Let camera stabilize
            print("✅ Picamera2 initialized!")
            use_picam = True
        except Exception as e:
            print(f"❌ No camera available: {e}")
            return
    else:
        use_picam = False

    frame_count = 0
    last_debug_time = time.time()
    saved_raw_outputs = None

    # Add monkey-patch to capture raw outputs
    original_postprocess = detector._postprocess_outputs

    def debug_postprocess(output_dict, original_shape):
        nonlocal saved_raw_outputs
        saved_raw_outputs = output_dict.copy()

        # Only print debug every 2 seconds
        current_time = time.time()
        if current_time - last_debug_time > 2.0:
            print(f"\n📊 Frame {frame_count} - Raw outputs:")
            for key, value in output_dict.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, "
                          f"min={value.min():.4f}, max={value.max():.4f}")

                    # Check for any significant activations
                    if len(value.shape) == 2 and value.shape[1] >= 5:
                        confidences = value[:, 4]
                        high_conf = confidences > 0.05
                        if high_conf.any():
                            print(f"    ⚠️ {high_conf.sum()} boxes with conf > 0.05")
                            top_conf_idx = np.argmax(confidences)
                            print(f"    Max confidence: {confidences[top_conf_idx]:.4f}")
                            print(f"    Box: {value[top_conf_idx, :4]}")

        return original_postprocess(output_dict, original_shape)

    detector._postprocess_outputs = debug_postprocess

    try:
        while True:
            # Get frame
            if use_picam:
                frame = picam2.capture_array("main")
                # Convert from RGB to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    continue

            frame_count += 1

            # Run detection
            detections = detector.detect(frame)

            # Draw detections
            display_frame = frame.copy()

            # Add frame info
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Detections: {len(detections)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if detections:
                print(f"\n🎯 Frame {frame_count}: {len(detections)} detection(s)")
                for i, det in enumerate(detections):
                    x, y, w, h = det['bbox']
                    conf = det['confidence']
                    class_id = det.get('class_id', -1)

                    # Draw bounding box
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    # Draw label
                    label = f"Dog {i+1}: {conf:.2f} (class={class_id})"
                    cv2.putText(display_frame, label, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"  Det {i+1}: conf={conf:.3f}, bbox={det['bbox']}, class={class_id}")

            # Show frame
            cv2.imshow("DogBot Debug Viewer", display_frame)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"debug_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved frame as {filename}")
            elif key == ord('d'):
                # Dump raw detection data
                if saved_raw_outputs:
                    for key, value in saved_raw_outputs.items():
                        if isinstance(value, np.ndarray):
                            filename = f"raw_output_{key}_{frame_count}.npy"
                            np.save(filename, value)
                            print(f"💾 Saved raw output as {filename}")

            last_debug_time = time.time()

    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")

    finally:
        print("\n🧹 Cleaning up...")
        if use_picam:
            picam2.stop()
        else:
            cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        print("✅ Done")

if __name__ == "__main__":
    main()