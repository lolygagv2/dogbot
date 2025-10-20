#!/usr/bin/env python3
"""
Quick diagnostic script to check camera and detection pipeline
"""
import cv2
import numpy as np
import json

# Load config
CFG = json.load(open("config/config.json"))
CAM_ROT_DEG = int(CFG.get("camera_rotation_deg", 0))

def get_camera():
    # Try OpenCV first
    print("Trying OpenCV camera...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Test if we can actually read a frame
        test_ok, test_frame = cap.read()
        if test_ok and test_frame is not None:
            print("✅ OpenCV camera working - can capture frames")

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
            print("❌ OpenCV camera opens but can't capture frames")
            cap.release()

    # Try picamera2
    try:
        print("Trying picamera2...")
        from picamera2 import Picamera2
        import time

        cam = Picamera2()
        config = cam.create_video_configuration(main={"size": (1280, 720)})
        cam.configure(config)
        cam.start()
        time.sleep(2)  # Let camera initialize

        # Test frame capture
        test_frame = cam.capture_array()
        if test_frame is not None:
            print("✅ Picamera2 working - can capture frames")

            def grab():
                frame = cam.capture_array()
                # Convert XBGR to BGR
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                return frame
            return grab
        else:
            print("❌ Picamera2 starts but can't capture frames")

    except Exception as e:
        print(f"❌ Picamera2 failed: {e}")

    print("❌ No working camera found")
    return None

def setup_aruco():
    dict_name = str(CFG.get("aruco_dict", "DICT_4X4_1000"))
    dconst = getattr(cv2.aruco, dict_name)
    dic = cv2.aruco.getPredefinedDictionary(dconst)
    try:
        det = cv2.aruco.ArucoDetector(dic, cv2.aruco.DetectorParameters())
    except AttributeError:
        det = (dic, cv2.aruco.DetectorParameters_create())
    return det

def detect_markers(bgr, aruco_det):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    try:
        corners, ids, _ = aruco_det.detectMarkers(gray)
    except Exception:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_det[0], parameters=aruco_det[1])

    markers = []
    if ids is not None:
        for c, id_ in zip(corners, ids.flatten()):
            pts = c[0]
            cx, cy = float(pts[:,0].mean()), float(pts[:,1].mean())
            markers.append((int(id_), cx, cy))
    return markers

def main():
    print("=== Camera & Detection Diagnostic ===")

    # Test camera
    grab = get_camera()
    if not grab:
        print("❌ CAMERA FAILED")
        return

    print("✅ Camera initialized")

    # Test ArUco
    aruco_det = setup_aruco()
    print(f"✅ ArUco detector setup: {CFG.get('aruco_dict', 'DICT_4X4_1000')}")

    # Test frame capture and processing
    for i in range(5):
        print(f"\n--- Frame {i+1} ---")

        frame = grab()
        if frame is None:
            print("❌ Failed to capture frame")
            continue

        print(f"✅ Frame captured: {frame.shape} {frame.dtype}")
        print(f"   Pixel range: {frame.min()} to {frame.max()}")

        # Test ArUco detection
        markers = detect_markers(frame, aruco_det)
        print(f"✅ ArUco detection: found {len(markers)} markers")
        if markers:
            for marker_id, cx, cy in markers:
                print(f"   Marker {marker_id} at ({cx:.0f}, {cy:.0f})")

        # Save a test frame
        if i == 0:
            cv2.imwrite("debug_frame.jpg", frame)
            print("✅ Saved debug_frame.jpg")

        import time
        time.sleep(1)

    print("\n=== Diagnostic Complete ===")
    print("Check debug_frame.jpg to see what camera captures")

if __name__ == "__main__":
    main()