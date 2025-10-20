#!/usr/bin/env python3
"""
Simple camera preview for focus adjustment
Saves frames periodically so you can check focus
"""

import cv2
import time
from picamera2 import Picamera2

def main():
    print("Starting camera preview for focus adjustment...")
    print("Will save preview frames every 2 seconds")
    print("Press Ctrl+C to stop")

    # Setup camera
    cam = Picamera2()
    config = cam.create_preview_configuration(main={"size": (1280, 720)})
    cam.configure(config)
    cam.start()
    time.sleep(2)

    frame_count = 0

    try:
        while True:
            # Capture frame
            frame = cam.capture_array()

            # Convert to BGR if needed
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Save preview frame
            filename = f"focus_preview_{frame_count:03d}.jpg"
            cv2.imwrite(filename, frame)

            print(f"Saved {filename} - adjust focus and check image quality")
            frame_count += 1

            time.sleep(2)  # Save every 2 seconds

    except KeyboardInterrupt:
        print("\nStopping preview...")

    finally:
        cam.stop()
        print(f"Saved {frame_count} preview frames")

if __name__ == "__main__":
    main()