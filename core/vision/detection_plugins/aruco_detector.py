#!/usr/bin/env python3
"""
ArUco Detector - Marker-based dog identification
Consolidates ArUco detection from multiple implementations
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from .base_detector import BaseDetector

class ArUcoDetector(BaseDetector):
    """ArUco marker detector for dog identification"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger('ArUcoDetector')

        # ArUco configuration
        self.dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters_create()

        # Dog ID mapping (from marker IDs to dog names)
        self.dog_id_mapping = {
            1: "elsa",
            2: "bezik"
        }

        self.initialized = True
        self.logger.info("ArUco detector initialized")

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect ArUco markers in frame

        Args:
            frame: Input image as numpy array (BGR format)

        Returns:
            Dictionary with marker information
        """
        try:
            # Convert to grayscale for better detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters
            )

            # Process detections
            markers = []
            if ids is not None:
                for i, marker_id in enumerate(ids.flatten()):
                    corner = corners[i][0]  # Get first corner set

                    # Calculate center and size
                    center_x = int(np.mean(corner[:, 0]))
                    center_y = int(np.mean(corner[:, 1]))

                    # Calculate marker size (average of width and height)
                    width = np.max(corner[:, 0]) - np.min(corner[:, 0])
                    height = np.max(corner[:, 1]) - np.min(corner[:, 1])
                    size = (width + height) / 2

                    # Get dog name if mapped
                    dog_name = self.dog_id_mapping.get(int(marker_id), f"unknown_dog_{marker_id}")

                    marker_info = {
                        'id': int(marker_id),
                        'dog_name': dog_name,
                        'center': (center_x, center_y),
                        'size': float(size),
                        'corners': corner.tolist(),
                        'confidence': 1.0  # ArUco detection is binary
                    }

                    markers.append(marker_info)

            return {
                'markers': markers,
                'marker_count': len(markers),
                'frame_shape': frame.shape,
                'detected_dogs': [m['dog_name'] for m in markers if m['dog_name'] != f"unknown_dog_{m['id']}"]
            }

        except Exception as e:
            self.logger.error(f"ArUco detection failed: {e}")
            return {'markers': [], 'marker_count': 0}

    def detect_with_pose(self, frame: np.ndarray, camera_matrix: np.ndarray,
                        dist_coeffs: np.ndarray, marker_size: float = 0.05) -> Dict[str, Any]:
        """
        Detect markers with pose estimation

        Args:
            frame: Input image
            camera_matrix: Camera calibration matrix
            dist_coeffs: Distortion coefficients
            marker_size: Physical marker size in meters

        Returns:
            Dictionary with marker and pose information
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.dictionary, parameters=self.parameters
            )

            markers = []
            if ids is not None:
                # Estimate pose for each marker
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_size, camera_matrix, dist_coeffs
                )

                for i, marker_id in enumerate(ids.flatten()):
                    corner = corners[i][0]
                    rvec = rvecs[i]
                    tvec = tvecs[i]

                    # Calculate center
                    center_x = int(np.mean(corner[:, 0]))
                    center_y = int(np.mean(corner[:, 1]))

                    # Get dog name
                    dog_name = self.dog_id_mapping.get(int(marker_id), f"unknown_dog_{marker_id}")

                    # Distance from camera
                    distance = np.linalg.norm(tvec)

                    marker_info = {
                        'id': int(marker_id),
                        'dog_name': dog_name,
                        'center': (center_x, center_y),
                        'corners': corner.tolist(),
                        'rvec': rvec.tolist(),
                        'tvec': tvec.tolist(),
                        'distance': float(distance),
                        'confidence': 1.0
                    }

                    markers.append(marker_info)

            return {
                'markers': markers,
                'marker_count': len(markers),
                'detected_dogs': [m['dog_name'] for m in markers if m['dog_name'] != f"unknown_dog_{m['id']}"]
            }

        except Exception as e:
            self.logger.error(f"ArUco pose detection failed: {e}")
            return {'markers': [], 'marker_count': 0}

    def match_behavior_to_dog(self, behavior_detection: Dict[str, Any],
                             aruco_markers: List[Dict[str, Any]],
                             proximity_threshold: float = 100) -> Optional[str]:
        """
        Match a behavior detection to a specific dog using ArUco proximity

        Args:
            behavior_detection: Behavior detection with bbox
            aruco_markers: List of detected ArUco markers
            proximity_threshold: Maximum distance in pixels

        Returns:
            Dog name if match found, None otherwise
        """
        if not aruco_markers or 'bbox' not in behavior_detection:
            return None

        # Get behavior bounding box center
        bbox = behavior_detection['bbox']
        behavior_center = (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)

        # Find closest marker
        closest_dog = None
        min_distance = float('inf')

        for marker in aruco_markers:
            marker_center = marker['center']

            # Calculate distance
            distance = np.sqrt(
                (behavior_center[0] - marker_center[0])**2 +
                (behavior_center[1] - marker_center[1])**2
            )

            if distance < min_distance and distance <= proximity_threshold:
                min_distance = distance
                closest_dog = marker['dog_name']

        return closest_dog

    def draw_markers(self, frame: np.ndarray, markers: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw detected markers on frame

        Args:
            frame: Input frame
            markers: List of detected markers

        Returns:
            Frame with markers drawn
        """
        output_frame = frame.copy()

        for marker in markers:
            # Draw marker outline
            corners = np.array(marker['corners'], dtype=np.int32)
            cv2.polylines(output_frame, [corners], True, (0, 255, 0), 2)

            # Draw center point
            center = marker['center']
            cv2.circle(output_frame, center, 5, (0, 0, 255), -1)

            # Draw text
            text = f"{marker['dog_name']} (ID:{marker['id']})"
            cv2.putText(output_frame, text, (center[0] - 50, center[1] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return output_frame

    def is_available(self) -> bool:
        """Check if ArUco detector is available"""
        try:
            return cv2.__version__ is not None
        except:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get detector status"""
        status = super().get_status()
        status.update({
            'dictionary': 'DICT_6X6_250',
            'mapped_dogs': list(self.dog_id_mapping.values())
        })
        return status