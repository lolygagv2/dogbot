#!/usr/bin/env python3
"""
GUI test interface for pose detection system
Displays live camera feed with pose keypoints and behavior predictions
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def draw_pose_detections(frame, detections, behaviors):
    """Draw bounding boxes, keypoints, and behavior labels"""
    for det in detections:
        # Draw bounding box
        x1, y1, x2, y2 = det.bbox.astype(int)
        color = (0, 255, 0) if det.dog_id else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw dog ID if available
        if det.dog_id:
            label = f"Dog: {det.dog_id}"
            if det.dog_id in behaviors:
                beh_info = behaviors[det.dog_id]
                label += f" | {beh_info['behavior']} ({beh_info['confidence']:.2f})"

            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw keypoints
        for i in range(24):
            x, y, conf = det.keypoints[i]
            if conf > 0.5:  # Only draw high confidence keypoints
                cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

    return frame

def main():
    """Run GUI test with pose detection"""
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

    print("Starting GUI (press 'q' to quit)...")

    # Performance tracking
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Get frame
            frame = get_frame()
            if frame is None:
                print("Failed to get frame")
                break

            # Process with pose detector
            result = detector.process_frame(frame)

            # Draw detections
            display_frame = frame.copy()
            display_frame = draw_pose_detections(
                display_frame,
                result['detections'],
                result['behaviors']
            )

            # Add stats overlay
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            stats_text = [
                f"FPS: {fps:.1f}",
                f"Detections: {len(result['detections'])}",
                f"Inference: {result.get('inference_time', 0):.1f}ms"
            ]

            for i, text in enumerate(stats_text):
                cv2.putText(display_frame, text, (10, 30 + i*25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Try to show frame, fall back to saving if no display
            try:
                cv2.imshow("Pose Detection Test", display_frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save screenshot
                    filename = f"pose_test_{int(time.time())}.jpg"
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved screenshot: {filename}")

            except cv2.error as e:
                if "not implemented" in str(e).lower():
                    # No display available, save frames instead
                    if frame_count <= 5:  # Save first 5 frames
                        filename = f"pose_gui_frame_{frame_count}.jpg"
                        cv2.imwrite(filename, display_frame)
                        print(f"No display available - saved: {filename}")
                        print(f"  Detections: {len(result['detections'])}, FPS: {fps:.1f}")

                        if frame_count == 5:
                            print("Saved 5 frames. Check pose_gui_frame_*.jpg files")
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