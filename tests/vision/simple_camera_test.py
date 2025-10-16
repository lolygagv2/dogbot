#!/usr/bin/env python3
"""
Simple camera test to verify what the camera sees
Saves a snapshot every 2 seconds for manual inspection
"""

import cv2
import time
from datetime import datetime

try:
    from picamera2 import Picamera2

    print("Starting Picamera2...")
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"size": (640, 480)})
    cam.configure(config)
    cam.start()

    print("Camera started! Taking snapshots every 2 seconds...")
    print("Check camera_test_*.jpg files to see what the camera sees")
    print("Press Ctrl+C to stop")

    count = 0
    try:
        while True:
            frame = cam.capture_array()

            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Save frame
            filename = f"camera_test_{count:03d}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename} - Shape: {frame.shape}")

            count += 1
            time.sleep(2)

            if count >= 5:
                print("\nSaved 5 test frames. Check the images to see camera view.")
                break

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cam.stop()

except ImportError:
    print("Picamera2 not available, trying OpenCV...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No camera available!")
    else:
        print("Camera opened with OpenCV")
        for i in range(5):
            ret, frame = cap.read()
            if ret:
                filename = f"camera_test_opencv_{i:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved {filename}")
            time.sleep(1)
        cap.release()