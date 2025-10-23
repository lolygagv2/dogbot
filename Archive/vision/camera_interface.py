#!/usr/bin/env python3
"""
IMX500 Camera Interface Module for TreatBot
"""

import logging
import threading
import time
from typing import Optional, Tuple, Callable
import numpy as np
from picamera2 import Picamera2
import cv2

logger = logging.getLogger(__name__)

class CameraInterface:
    def __init__(self, resolution: Tuple[int, int] = (2028, 1520), 
                 framerate: int = 30):
        """
        Initialize IMX500 camera
        Using 2028x1520 @ 30fps for good performance
        """
        self.resolution = resolution
        self.framerate = framerate
        self.camera = None
        self.is_running = False
        
    def initialize(self) -> bool:
        """Initialize camera hardware"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"}
            )
            self.camera.configure(config)
            logger.info(f"IMX500 initialized: {self.resolution}@{self.framerate}fps")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def start(self):
        """Start camera capture"""
        if not self.camera:
            if not self.initialize():
                return False
        self.camera.start()
        self.is_running = True
        return True
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture single frame"""
        if self.is_running:
            return self.camera.capture_array()
        return None
    
    def stop(self):
        """Stop camera"""
        if self.camera and self.is_running:
            self.camera.stop()
            self.is_running = False

# Quick test
if __name__ == "__main__":
    cam = CameraInterface()
    if cam.start():
        print("Camera started!")
        frame = cam.capture_frame()
        if frame is not None:
            print(f"Captured frame: {frame.shape}")
        cam.stop()
