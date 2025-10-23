#!/usr/bin/env python3
"""
Enhanced Hailo Interface with proper behavior recognition
"""

import numpy as np
import cv2
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Hailo imports
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, 
                                InferVStreams, ConfigureParams, InputVStreamParams, 
                                OutputVStreamParams, FormatType)
    HAILO_AVAILABLE = True
except ImportError:
    logger.warning("Hailo SDK not available - using fallback detection")
    HAILO_AVAILABLE = False

logger = logging.getLogger(__name__)

class DogBehavior(Enum):
    """Extended dog behaviors"""
    IDLE = "idle"
    SITTING = "sitting"
    LYING = "lying"
    STANDING = "standing"
    SPINNING = "spinning"
    BARKING = "barking"
    COME = "approaching"
    STAY = "staying"
    HIGH_FIVE = "high_five"

@dataclass
class BehaviorDetection:
    """Behavior detection result"""
    behavior: DogBehavior
    confidence: float
    duration: float  # How long behavior has been held
    bbox: Optional[Tuple[int, int, int, int]] = None
    keypoints: Optional[np.ndarray] = None

class HailoInterface:
    """Enhanced Hailo interface for DogBot"""
    
    def __init__(self):
        self.device = None
        self.detection_network = None
        self.behavior_network = None
        
        # Behavior tracking
        self.behavior_history = []
        self.behavior_start_time = {}
        self.min_behavior_frames = 15  # 0.5 seconds at 30fps
        
        if HAILO_AVAILABLE:
            self._initialize_hailo()
    
    def _initialize_hailo(self):
        """Initialize Hailo-8L device"""
        try:
            # Create VDevice (Virtual Device)
            self.device = VDevice()
            
            # Load models
            self._load_yolo_model()
            self._load_behavior_model()
            
            logger.info("Hailo-8L initialized successfully")
            
        except Exception as e:
            logger.error(f"Hailo initialization failed: {e}")
            self.device = None
    
    def _load_yolo_model(self):
        """Load YOLOv8 for dog detection"""
        try:
            model_path = Path("/home/morgan/dogbot/ai/models/yolov8m.hef")
            
            if not model_path.exists():
                # Download from Hailo Model Zoo if needed
                logger.info("Downloading YOLOv8 model...")
                # Implementation for model download
                
            hef = HEF(str(model_path))
            
            # Configure network
            input_vstream_info = hef.get_input_vstream_infos()[0]
            output_vstream_info = hef.get_output_vstream_infos()[0]
            
            network_name = hef.get_network_group_names()[0]
            
            configure_params = ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe
            )
            
            network_group = self.device.configure(hef, configure_params)[0]
            
            self.detection_network = {
                'network': network_group,
                'input_info': input_vstream_info,
                'output_info': output_vstream_info
            }
            
            logger.info("YOLOv8 model loaded on Hailo")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
    
    def detect_and_classify(self, frame: np.ndarray) -> BehaviorDetection:
        """
        Detect dog and classify behavior in one pass
        """
        # First detect dog
        detections = self.detect_dogs(frame)
        
        if not detections:
            return None
        
        # Get best detection
        best_det = max(detections, key=lambda x: x['confidence'])
        
        # Extract dog region
        x, y, w, h = best_det['bbox']
        dog_region = frame[y:y+h, x:x+w]
        
        # Classify behavior
        behavior = self._classify_dog_behavior(dog_region, frame)
        
        # Track behavior duration
        current_time = time.time()
        if behavior != DogBehavior.IDLE:
            if behavior not in self.behavior_start_time:
                self.behavior_start_time[behavior] = current_time
            duration = current_time - self.behavior_start_time[behavior]
        else:
            duration = 0
            self.behavior_start_time.clear()
        
        return BehaviorDetection(
            behavior=behavior,
            confidence=best_det['confidence'],
            duration=duration,
            bbox=best_det['bbox']
        )
    
    def _classify_dog_behavior(self, dog_region: np.ndarray, full_frame: np.ndarray) -> DogBehavior:
        """
        Classify dog behavior using visual analysis
        """
        if dog_region.size == 0:
            return DogBehavior.IDLE
        
        # Analyze dog position and pose
        h, w = dog_region.shape[:2]
        
        # Simple heuristics (enhance with trained model)
        
        # 1. Check if sitting (height < width typically)
        aspect_ratio = h / w if w > 0 else 1
        if 0.8 < aspect_ratio < 1.2:
            return DogBehavior.SITTING
        
        # 2. Check if lying down (very low aspect ratio)
        if aspect_ratio < 0.6:
            return DogBehavior.LYING
        
        # 3. Check for high five (paws near camera)
        # Look for paw patterns in upper portion of dog region
        upper_third = dog_region[:h//3, :]
        # Implement paw detection logic here
        
        # 4. Check motion for spinning
        if len(self.behavior_history) > 10:
            # Analyze rotation in position history
            pass
        
        return DogBehavior.STANDING
    
    def detect_barking(self, audio_level: float, frequency_profile: np.ndarray) -> bool:
        """
        Detect barking from audio characteristics
        """
        # Barking typically has:
        # - High amplitude bursts
        # - Frequency range 500-2000 Hz
        # - Repetitive pattern
        
        if audio_level > 0.7:  # Threshold
            # Check frequency profile
            bark_freq_range = frequency_profile[500:2000]
            if np.mean(bark_freq_range) > 0.5:
                return True
        
        return False