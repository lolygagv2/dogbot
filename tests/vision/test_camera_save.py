#!/usr/bin/env python3
"""
Simple camera test - saves frames so you can see what camera captures
"""
import cv2
import json
import time

# Load config
CFG = json.load(open("config/config.json"))
CAM_ROT_DEG = int(CFG.get("camera_rotation_deg", 0))

def get_camera():
    # Try OpenCV first
    print("Trying OpenCV...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Test frame capture
        test_ok, test_frame = cap.read()
        if test_ok and test_frame is not None:
            print("✅ OpenCV working")
            def grab():
                ok, frame = cap.read()
                if not ok:
                    return None
                if CAM_ROT_DEG:
                    if CAM_ROT_DEG == 90:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    elif CAM_ROT_DEG == 180:
                        frame = cv2.rotate(frame, cv2.ROTATE_180)
                    elif CAM_ROT_DEG == 270:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                return frame
            return grab
        else:
            print("❌ OpenCV can't capture")
            cap.release()

    # Try picamera2
    try:
        print("Trying picamera2...")
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_video_configuration(main={"size": (1280, 720)})
        cam.configure(config)
        cam.start()
        time.sleep(2)

        print("✅ Picamera2 working")
        def grab():
            frame = cam.capture_array()
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            return frame
        return grab

    except Exception as e:
        print(f"❌ Picamera2 failed: {e}")
        return None

def main():
    print("=== Testing Camera & Saving Frames ===")

    grab = get_camera()
    if not grab:
        print("❌ No camera working")
        return

    print("Capturing and saving 3 frames...")

    for i in range(3):
        print(f"Capturing frame {i+1}...")
        frame = grab()

        if frame is not None:
            filename = f"test_frame_{i+1}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✅ Saved {filename} - size: {frame.shape}")
        else:
            print(f"❌ Frame {i+1} failed")

        time.sleep(1)

    print("\n=== Results ===")
    print("Check these files to see what camera captures:")
    import os
    for i in range(1, 4):
        filename = f"test_frame_{i}.jpg"
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✅ {filename} - {size} bytes")
        else:
            print(f"❌ {filename} - not found")

if __name__ == "__main__":
    main()