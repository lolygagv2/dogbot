#!/usr/bin/env python3
"""
Test OpenCV Detection
Direct test of OpenCV detector to see if it finds dogs
"""

import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.vision.detection_plugins.opencv_detector import OpenCVDetector

def main():
    print("🔍 Testing OpenCV Detection")
    print("="*40)

    try:
        # Initialize detector
        print("Initializing OpenCV detector...")
        config = {
            'confidence_threshold': 0.3,
            'target_classes': ['dog', 'person', 'cat']  # Include person for testing
        }

        detector = OpenCVDetector(config)

        if not detector.initialized:
            print("❌ Detector failed to initialize!")
            return

        print("✅ Detector initialized successfully")

        # Initialize camera
        print("Initializing camera...")
        try:
            from picamera2 import Picamera2
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            camera.configure(config)
            camera.start()
            picamera_mode = True
            print("✅ Picamera2 initialized")
        except ImportError:
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("❌ Failed to open camera!")
                return
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            picamera_mode = False
            print("✅ OpenCV camera initialized")

        print("\n👀 Starting detection test...")
        print("Point camera at dogs, people, or cats for 20 seconds...")

        detection_count = 0
        dog_count = 0

        for i in range(60):  # 20 seconds at ~3 FPS
            try:
                # Capture frame
                if picamera_mode:
                    frame = camera.capture_array()
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    ret, frame_bgr = camera.read()
                    if not ret:
                        continue

                # Run detection
                detections = detector.detect(frame_bgr)

                # Print results
                if detections:
                    detection_count += 1
                    print(f"\n📍 Frame {i+1}: {len(detections)} objects detected:")

                    for det in detections:
                        class_name = det['class_name']
                        confidence = det['confidence']
                        bbox = det['bbox']

                        print(f"  - {class_name}: {confidence:.1%} at {bbox}")

                        if class_name == 'dog':
                            dog_count += 1
                            print(f"  🐕 DOG FOUND! (#{dog_count})")
                else:
                    print(f"Frame {i+1}: No detections", end='\r')

                # Wait between frames
                import time
                time.sleep(0.33)  # ~3 FPS

            except Exception as e:
                print(f"❌ Detection error: {e}")

        print(f"\n\n📊 Test Results:")
        print(f"Total detections: {detection_count}")
        print(f"Dog detections: {dog_count}")

        if dog_count > 0:
            print("✅ OpenCV detection is working - dogs found!")
        elif detection_count > 0:
            print("⚠️ OpenCV detects objects but no dogs found")
        else:
            print("❌ OpenCV detection not working - no objects detected")

    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
    except Exception as e:
        print(f"❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'camera' in locals():
            try:
                if picamera_mode:
                    camera.stop()
                    camera.close()
                else:
                    camera.release()
            except:
                pass

if __name__ == "__main__":
    main()