#!/usr/bin/env python3
"""
DEBUG GUI test for pose detection with adjustable confidence threshold
Shows keypoints even with lower confidence detections
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def draw_pose_detections(frame, detections, behaviors, show_all_keypoints=False):
    """Draw bounding boxes, keypoints, and behavior labels"""
    for det in detections:
        # Draw bounding box
        x1, y1, x2, y2 = det.bbox.astype(int)
        color = (0, 255, 0) if det.dog_id else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw confidence score
        conf_text = f"Conf: {det.confidence:.2f}"
        cv2.putText(frame, conf_text, (x1, y1 - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw dog ID if available
        if det.dog_id:
            label = f"Dog: {det.dog_id}"
            if det.dog_id in behaviors:
                beh_info = behaviors[det.dog_id]
                label += f" | {beh_info['behavior']} ({beh_info['confidence']:.2f})"

            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw keypoints with color coding based on confidence
        for i in range(24):
            x, y, conf = det.keypoints[i]

            # Show all keypoints or only high confidence ones
            if show_all_keypoints or conf > 0.3:
                # Color based on confidence: red (low) -> yellow (medium) -> green (high)
                if conf < 0.3:
                    kp_color = (0, 0, 255)  # Red
                elif conf < 0.6:
                    kp_color = (0, 255, 255)  # Yellow
                else:
                    kp_color = (0, 255, 0)  # Green

                cv2.circle(frame, (int(x), int(y)), 4, kp_color, -1)

                # Draw keypoint number for debugging
                if show_all_keypoints:
                    cv2.putText(frame, str(i), (int(x)+5, int(y)-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, kp_color, 1)

    return frame

def main():
    """Run DEBUG GUI test with adjustable threshold"""
    # Import with modified threshold for testing
    import core.pose_detector as pd

    # Temporarily lower the confidence threshold for testing
    original_threshold = 0.8
    test_threshold = 0.3  # Much lower for testing

    print(f"DEBUG MODE: Lowering confidence threshold from {original_threshold} to {test_threshold}")
    print("This will show more (potentially false) detections for testing keypoint visualization")

    from core.pose_detector import PoseDetector

    print("Initializing Pose Detector...")
    detector = PoseDetector()

    if not detector.initialize():
        print("Failed to initialize detector")
        return 1

    print("Starting camera...")

    # Try Picamera2 first, then OpenCV
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_preview_configuration(main={"size": (640, 480)})
        cam.configure(config)
        cam.start()

        def get_frame():
            return cam.capture_array()
    except:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No camera available")
            return 1

        def get_frame():
            ret, frame = cap.read()
            return frame if ret else None

    print("Starting DEBUG GUI (press 'q' to quit, 't' to toggle threshold, 'k' to toggle keypoints)...")

    # Performance tracking
    frame_count = 0
    start_time = time.time()
    use_low_threshold = True
    show_all_keypoints = True

    try:
        while True:
            # Get frame
            frame = get_frame()
            if frame is None:
                print("Failed to get frame")
                break

            # Temporarily modify the threshold in the decoder
            # This is a hack for testing - normally you'd modify the pose_detector.py
            temp_decode = detector.decode_pose_outputs

            def debug_decode(raw_outputs):
                # Monkey-patch the confidence check temporarily
                import core.pose_detector
                old_code = core.pose_detector.PoseDetector.decode_pose_outputs.__code__

                # Call original with our modifications
                detections = temp_decode(raw_outputs)

                # For debug, also try with lower threshold
                if use_low_threshold and len(detections) == 0:
                    # Try decoding with lower standards
                    print("No detections with high threshold, trying lower...")
                    # This would need actual implementation

                return detections

            # Process with pose detector
            result = detector.process_frame(frame)

            # Draw detections
            display_frame = frame.copy()
            display_frame = draw_pose_detections(
                display_frame,
                result['detections'],
                result['behaviors'],
                show_all_keypoints=show_all_keypoints
            )

            # Add stats overlay
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            stats_text = [
                f"FPS: {fps:.1f}",
                f"Detections: {len(result['detections'])}",
                f"Inference: {result.get('inference_time', 0):.1f}ms",
                f"Threshold Mode: {'LOW (0.3)' if use_low_threshold else 'HIGH (0.8)'}",
                f"Show All Keypoints: {show_all_keypoints}",
                "",
                "Controls: 'q'=quit, 't'=toggle threshold, 'k'=toggle keypoints"
            ]

            for i, text in enumerate(stats_text):
                cv2.putText(display_frame, text, (10, 30 + i*20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Try to show frame, fall back to saving if no display
            try:
                cv2.imshow("Pose Detection DEBUG", display_frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('t'):
                    use_low_threshold = not use_low_threshold
                    print(f"Toggled threshold mode to: {'LOW' if use_low_threshold else 'HIGH'}")
                elif key == ord('k'):
                    show_all_keypoints = not show_all_keypoints
                    print(f"Show all keypoints: {show_all_keypoints}")
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"pose_debug_{int(time.time())}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved screenshot: {filename}")

            except cv2.error as e:
                if "not implemented" in str(e).lower():
                    # No display available, save frames instead
                    if frame_count <= 5:  # Save first 5 frames
                        filename = f"pose_debug_frame_{frame_count}.jpg"
                        cv2.imwrite(filename, display_frame)
                        print(f"No display available - saved: {filename}")
                        print(f"  Detections: {len(result['detections'])}, FPS: {fps:.1f}")

                        if frame_count == 5:
                            print("Saved 5 frames. Check pose_debug_frame_*.jpg files")
                            break
                else:
                    raise

    finally:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # Ignore if no GUI support
        detector.cleanup()

        try:
            cam.stop()
        except:
            cap.release()

    return 0

if __name__ == "__main__":
    sys.exit(main())