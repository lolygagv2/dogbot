#!/usr/bin/env python3
"""
IMX500 Camera Interface - Properly configured for Raspberry Pi 5
"""

import numpy as np
import cv2
import time
import threading
import logging
from picamera2 import Picamera2
from libcamera import controls
import json

logger = logging.getLogger(__name__)

class IMX500Camera:
    """IMX500 AI Camera with on-sensor processing capabilities"""
    
    def __init__(self):
        self.camera = None
        self.config = None
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # IMX500 specific - on-sensor AI results
        self.ai_results = None
        self.ai_lock = threading.Lock()
        
    def initialize(self):
        """Initialize IMX500 with proper configuration for Pi 5"""
        try:
            self.camera = Picamera2()
            
            # Check if IMX500 is detected
            camera_info = self.camera.camera_properties
            model = camera_info.get('Model', 'unknown')
            
            if 'imx500' not in model.lower():
                logger.warning(f"Camera model {model} may not be IMX500")
            
            # Configure for IMX500 with AI metadata stream
            self.config = self.camera.create_preview_configuration(
                main={"size": (1536, 864), "format": "RGB888"},
                lores={"size": (768, 432), "format": "YUV420"},  # For AI processing
                controls={
                    "FrameRate": 30,
                    "AfMode": controls.AfModeEnum.Continuous,  # Continuous autofocus
                    "AeEnable": True,
                    "AwbEnable": True
                }
            )
            
            # Enable AI inference output from IMX500
            self.config["controls"]["AiInference"] = True  # If supported
            
            self.camera.configure(self.config)
            
            logger.info(f"IMX500 initialized on Pi 5 (Kernel 6.12)")
            return True
            
        except Exception as e:
            logger.error(f"IMX500 initialization failed: {e}")
            return False
    
    def start(self):
        """Start camera with metadata capture for AI results"""
        if not self.camera:
            if not self.initialize():
                return False
                
        try:
            self.camera.start()
            self.running = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            
            logger.info("IMX500 capture started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start IMX500: {e}")
            return False
    
    def _capture_loop(self):
        """Capture loop with AI metadata extraction"""
        while self.running:
            try:
                # Capture with metadata
                request = self.camera.capture_request()
                
                # Get the main frame
                frame = request.make_array("main")
                
                # Get AI inference results from metadata (if available)
                metadata = request.get_metadata()
                
                # IMX500 AI results would be in metadata
                if 'AiDetection' in metadata:
                    with self.ai_lock:
                        self.ai_results = metadata['AiDetection']
                
                # Store frame
                with self.frame_lock:
                    self.current_frame = frame
                
                request.release()
                
            except Exception as e:
                logger.error(f"Capture error: {e}")
                time.sleep(0.1)
    
    def get_frame_with_ai(self):
        """Get frame with any on-sensor AI results"""
        with self.frame_lock:
            frame = self.current_frame.copy() if self.current_frame is not None else None
            
        with self.ai_lock:
            ai_data = self.ai_results.copy() if self.ai_results else None
            
        return frame, ai_data
    
    def capture_frame(self):
        """Get current frame for compatibility"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def stop(self):
        """Stop camera"""
        self.running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=2)
        if self.camera:
            self.camera.stop()
            self.camera.close()