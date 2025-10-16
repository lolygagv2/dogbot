#!/usr/bin/env python3
"""
Test Hailo detection with live camera - no GUI
"""

import sys
import time
import numpy as np
import logging

# Add project to path
sys.path.insert(0, '/home/morgan/dogbot')

from core.vision.detection_plugins.hailo_detector import HailoDetector

logging.basicConfig(level=logging.INFO,
                   format='%(levelname)s:%(name)s:%(message)s')

def main():
    """Test Hailo with live camera"""
    print("=" * 60)
    print("🐕 Testing Hailo Detection with Live Camera")
    print("=" * 60)

    # Initialize Hailo detector
    print("\n🔧 Initializing Hailo detector...")
    detector = HailoDetector({
        'hailo_model_path': '/home/morgan/dogbot/ai/models/best_dogbot.hef',
        'confidence_threshold': 0.1  # Low threshold for testing
    })

    if not detector.initialized:
        print("❌ Failed to initialize Hailo detector")
        return

    print("✅ Hailo detector ready")

    # Initialize camera
    print("\n📷 Initializing camera...")
    try:
        from picamera2 import Picamera2
        import cv2

        picam2 = Picamera2()
        config = picam2.create_preview_configuration(main={"size": (640, 480)})
        picam2.configure(config)
        picam2.start()
        time.sleep(2)  # Let camera stabilize
        print("✅ Camera ready")

    except Exception as e:
        print(f"❌ Camera failed: {e}")
        return

    print("\n🎯 Starting detection test...")
    print("Testing for 30 seconds - checking every 2 seconds")
    print("Press Ctrl+C to stop early")

    frame_count = 0
    detection_count = 0

    try:
        for i in range(15):  # 30 seconds / 2 seconds each
            # Capture frame
            frame = picam2.capture_array("main")
            # Convert from RGB to BGR for OpenCV compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_count += 1

            print(f"\n🔍 Frame {frame_count} - Running detection...")

            # Run detection
            detections = detector.detect(frame)

            if detections:
                detection_count += len(detections)
                print(f"🎯 FOUND {len(detections)} detection(s)!")

                for j, det in enumerate(detections):
                    conf = det['confidence']
                    bbox = det['bbox']
                    class_id = det.get('class_id', -1)
                    print(f"  Detection {j+1}: confidence={conf:.3f}, bbox={bbox}, class={class_id}")

                    # Save this frame for inspection
                    filename = f"detection_frame_{frame_count}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"  💾 Saved frame as {filename}")

            else:
                print("   No detections")

            # Wait before next test
            if i < 14:  # Don't wait after last iteration
                time.sleep(2)

    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")

    finally:
        print("\n📊 Test Results:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Total detections: {detection_count}")
        if detection_count > 0:
            print(f"   Detection rate: {detection_count/frame_count:.1%}")
            print("   ✅ SUCCESS: Dogs were detected!")
        else:
            print("   ⚠️ No detections found")
            print("   This could indicate:")
            print("     - Model threshold too high")
            print("     - Model not trained for visible objects")
            print("     - Preprocessing issues")

        print("\n🧹 Cleaning up...")
        picam2.stop()
        detector.cleanup()
        print("✅ Test complete")

if __name__ == "__main__":
    main()