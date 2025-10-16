#!/usr/bin/env python3
import cv2
import numpy as np
from hailo_platform import (HailoRT, VDevice, HailoStreamInterface, 
                           InferVStreams, ConfigureParams)
import time

def test_camera_hailo():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    
    print("🎥 Camera initialized")
    
    # Test basic capture
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"✅ Frame {i+1}: {frame.shape}")
            time.sleep(0.1)
        else:
            print("❌ Failed to capture frame")
            
    cap.release()
    print("🏁 Camera test complete")

if __name__ == "__main__":
    test_camera_hailo()