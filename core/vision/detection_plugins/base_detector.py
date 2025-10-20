#!/usr/bin/env python3
"""
Base Detector Interface - Unified detection plugin system
Allows multiple AI backends to be used interchangeably
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List, Any, Optional

class BaseDetector(ABC):
    """Abstract base class for all detection plugins"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize detector with configuration

        Args:
            config: Detection configuration dictionary
        """
        self.config = config
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.nms_threshold = config.get('nms_threshold', 0.4)
        self.target_classes = config.get('target_classes', [16])  # Default: dog class
        self.initialized = False

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in frame

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            List of detection dictionaries with format:
            {
                'bbox': [x, y, width, height],
                'confidence': float,
                'class_id': int,
                'class_name': str (optional)
            }
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if detector backend is available"""
        pass

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Default preprocessing - can be overridden"""
        return frame

    def postprocess_detections(self, raw_detections: Any) -> List[Dict[str, Any]]:
        """Default postprocessing - can be overridden"""
        return raw_detections

    def filter_detections(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter detections by confidence and target classes"""
        filtered = []

        for detection in detections:
            confidence = detection.get('confidence', 0.0)
            class_id = detection.get('class_id', -1)

            if (confidence >= self.confidence_threshold and
                class_id in self.target_classes):
                filtered.append(detection)

        return filtered

    def get_status(self) -> Dict[str, Any]:
        """Get detector status"""
        return {
            'name': self.__class__.__name__,
            'initialized': self.initialized,
            'available': self.is_available(),
            'confidence_threshold': self.confidence_threshold,
            'target_classes': self.target_classes
        }

    def cleanup(self):
        """Cleanup resources - override if needed"""
        pass