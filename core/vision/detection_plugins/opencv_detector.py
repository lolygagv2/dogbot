#!/usr/bin/env python3
"""
OpenCV YOLO Detector - Reliable fallback detection
Consolidates OpenCV YOLO implementations from multiple files
"""

import cv2
import numpy as np
import os
import requests
import logging
from pathlib import Path
from typing import Dict, List, Any
from .base_detector import BaseDetector

class OpenCVDetector(BaseDetector):
    """OpenCV YOLO detector for reliable dog detection"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger('OpenCVDetector')

        # Model paths
        self.model_dir = Path("/home/morgan/dogbot/models")
        self.model_dir.mkdir(exist_ok=True)

        self.weights_path = self.model_dir / "yolov4-tiny.weights"
        self.config_path = self.model_dir / "yolov4-tiny.cfg"

        # OpenCV DNN network
        self.net = None
        self.output_layers = None

        # COCO class names
        self.class_names = self._get_coco_classes()

        # Initialize
        self._initialize()

    def _get_coco_classes(self) -> List[str]:
        """Get COCO class names"""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]

    def _initialize(self):
        """Initialize YOLO model"""
        try:
            # Download models if needed
            self._download_models()

            # Load YOLO
            self.net = cv2.dnn.readNet(str(self.weights_path), str(self.config_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

            # Get output layers
            layer_names = self.net.getLayerNames()
            self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

            self.initialized = True
            self.logger.info("OpenCV YOLO detector initialized successfully")

        except Exception as e:
            self.logger.error(f"OpenCV detector initialization failed: {e}")
            self.initialized = False

    def _download_models(self):
        """Download YOLO models if not present"""
        models = [
            {
                'url': 'https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights',
                'path': self.weights_path
            },
            {
                'url': 'https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg',
                'path': self.config_path
            }
        ]

        for model in models:
            if not model['path'].exists():
                self.logger.info(f"Downloading {model['path'].name}...")
                try:
                    response = requests.get(model['url'], timeout=30)
                    response.raise_for_status()

                    with open(model['path'], 'wb') as f:
                        f.write(response.content)

                    self.logger.info(f"Downloaded {model['path'].name}")

                except Exception as e:
                    self.logger.error(f"Failed to download {model['path'].name}: {e}")
                    raise

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects using YOLO

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            List of detection dictionaries
        """
        if not self.initialized or self.net is None:
            return []

        try:
            height, width = frame.shape[:2]

            # Prepare input blob
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (416, 416),
                swapRB=True, crop=False
            )
            self.net.setInput(blob)

            # Run inference
            outputs = self.net.forward(self.output_layers)

            # Process outputs
            detections = self._process_outputs(outputs, width, height)

            # Apply NMS
            detections = self._apply_nms(detections)

            # Filter by confidence and target classes
            detections = self.filter_detections(detections)

            return detections

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return []

    def _process_outputs(self, outputs: List[np.ndarray], width: int, height: int) -> List[Dict[str, Any]]:
        """Process YOLO outputs into detection format"""
        detections = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.1:  # Low threshold for initial filtering
                    # Get bbox coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Convert to top-left coordinates
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)

                    # Get class name
                    class_name = self.class_names[class_id] if class_id < len(self.class_names) else 'unknown'

                    detections.append({
                        'bbox': [x, y, w, h],
                        'confidence': float(confidence),
                        'class_id': int(class_id),
                        'class_name': class_name,
                        'center': (center_x, center_y)
                    })

        return detections

    def _apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []

        # Extract boxes, confidences, and class_ids
        boxes = []
        confidences = []
        class_ids = []

        for detection in detections:
            bbox = detection['bbox']
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            confidences.append(detection['confidence'])
            class_ids.append(detection['class_id'])

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences,
            self.confidence_threshold,
            self.nms_threshold
        )

        # Return filtered detections
        if len(indices) > 0:
            return [detections[i] for i in indices.flatten()]
        else:
            return []

    def is_available(self) -> bool:
        """Check if OpenCV detector is available"""
        try:
            return (cv2.__version__ is not None and
                    self.weights_path.exists() and
                    self.config_path.exists())
        except:
            return False

    def cleanup(self):
        """Cleanup resources"""
        self.net = None
        self.output_layers = None
        self.logger.info("OpenCV detector cleaned up")