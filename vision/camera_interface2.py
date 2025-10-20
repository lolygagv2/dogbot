#!/usr/bin/env python3
"""
IMX500 Camera Interface Module for TreatBot
Handles camera initialization, capture, and AI model integration
"""

import logging
import threading
import time
from typing import Optional, Tuple, Callable
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import H264Encoder
from picamera2.outputs import FileOutput
import cv2

logger = logging.getLogger(__name__)

class CameraInterface:
    """Interface for IMX500 AI Camera with Hailo integration"""
    
    def __init__(self, resolution: Tuple[int, int] = (640, 480), 
                 framerate: int = 30, use_ai: bool = True):
        """
        Initialize camera interface
        
        Args:
            resolution: Camera resolution (width, height)
            framerate: Target framerate
            use_ai: Enable AI features on IMX500
        """
        self.resolution = resolution
        self.framerate = framerate
        self.use_ai = use_ai
        self.camera = None
        self.is_running = False
        self.capture_thread = None
        self.frame_callback = None
        self.last_frame = None
        self.frame_lock = threading.Lock()
        
        # AI detection results
        self.detections = []
        self.detection_lock = threading.Lock()
        
    def initialize(self) -> bool:
        """Initialize camera hardware"""
        try:
            # Initialize Picamera2
            self.camera = Picamera2()
            
            # Configure camera
            config = self.camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={"FrameRate": self.framerate}
            )
            
            # Enable IMX500 AI features if available
            if self.use_ai:
                # This would integrate with Hailo through the AI HAT
                config["controls"]["AeEnable"] = True
                config["controls"]["AwbEnable"] = True
                
            self.camera.configure(config)
            
            logger.info(f"Camera initialized: {self.resolution}@{self.framerate}fps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
    
    def start(self, frame_callback: Optional[Callable] = None) -> bool:
        """
        Start camera capture
        
        Args:
            frame_callback: Optional callback for each frame
        """
        if not self.camera:
            if not self.initialize():
                return False
        
        try:
            self.camera.start()
            self.is_running = True
            self.frame_callback = frame_callback
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            logger.info("Camera capture started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            return False
    
    def stop(self):
        """Stop camera capture"""
        self.is_running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=2.0)
        
        if self.camera:
            self.camera.stop()
            self.camera.close()
            self.camera = None
            
        logger.info("Camera stopped")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        while self.is_running:
            try:
                # Capture frame
                frame = self.camera.capture_array()
                
                # Store latest frame
                with self.frame_lock:
                    self.last_frame = frame.copy()
                
                # Process with callback if provided
                if self.frame_callback:
                    self.frame_callback(frame)
                
                # Small delay to control framerate
                time.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame"""
        with self.frame_lock:
            return self.last_frame.copy() if self.last_frame is not None else None
    
    def capture_image(self, filename: str = "capture.jpg") -> bool:
        """
        Capture and save a single image
        
        Args:
            filename: Output filename
        """
        try:
            frame = self.get_frame()
            if frame is not None:
                cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                logger.info(f"Image saved: {filename}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            return False
    
    def start_recording(self, filename: str = "video.h264", duration: int = 0):
        """
        Start video recording
        
        Args:
            filename: Output video filename
            duration: Recording duration in seconds (0 for continuous)
        """
        try:
            encoder = H264Encoder()
            output = FileOutput(filename)
            self.camera.start_recording(encoder, output)
            
            if duration > 0:
                threading.Timer(duration, self.stop_recording).start()
                
            logger.info(f"Recording started: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
    
    def stop_recording(self):
        """Stop video recording"""
        try:
            self.camera.stop_recording()
            logger.info("Recording stopped")
        except:
            pass
    
    def get_detections(self) -> list:
        """Get latest AI detections"""
        with self.detection_lock:
            return self.detections.copy()
    
    def set_ai_model(self, model_path: str):
        """
        Load AI model for on-device inference
        
        Args:
            model_path: Path to Hailo-compatible model
        """
        # This would integrate with Hailo Runtime
        logger.info(f"Loading AI model: {model_path}")
        # Implementation depends on Hailo SDK
    
    def enable_dog_detection(self):
        """Enable dog detection mode"""
        self.set_ai_model("/home/morgan/dogbot/ai/models/dog_detection.hef")
        logger.info("Dog detection enabled")
    
    def enable_pose_estimation(self):
        """Enable pose estimation mode"""
        self.set_ai_model("/home/morgan/dogbot/ai/models/pose_estimation.hef")
        logger.info("Pose estimation enabled")

# Test function
def test_camera():
    """Test camera functionality"""
    logging.basicConfig(level=logging.INFO)
    
    cam = CameraInterface(resolution=(640, 480))
    
    if cam.initialize():
        print("Camera initialized successfully")
        
        # Test capture
        cam.start()
        time.sleep(2)
        
        # Capture test image
        cam.capture_image("test_capture.jpg")
        
        # Get a frame
        frame = cam.get_frame()
        if frame is not None:
            print(f"Frame captured: {frame.shape}")
        
        cam.stop()
    else:
        print("Camera initialization failed")

if __name__ == "__main__":
    test_camera()
