#!/usr/bin/env python3
"""
Dog Tracking System with ArUco Persistence Rules
Handles identification and tracking of multiple dogs using ArUco markers
with smart persistence and fallback rules.
"""

import time
from typing import Dict, List, Tuple, Optional
import numpy as np

class DogTracker:
    """Smart dog tracking with ArUco persistence rules"""

    def __init__(self, config: dict):
        """Initialize dog tracker with config"""
        # Rule 1: Confined list of valid dog IDs
        self.valid_dog_ids = {dog['marker_id']: dog['id'] for dog in config.get('dogs', [])}
        self.dog_names = {v: k for k, v in self.valid_dog_ids.items()}  # Reverse mapping

        # Rule 4: Default dog (Bezik 832)
        self.default_dog_id = 832
        self.default_dog_name = self.valid_dog_ids.get(832, "bezik")

        # Rule 6: Maximum dogs on screen
        self.max_dogs = config.get('max_dogs_on_screen', 2)

        # Rule 2: Persistence tracking (30 seconds)
        self.persistence_time = config.get('persistence_seconds', 30)
        self.last_known_positions = {}  # {dog_id: {'time': timestamp, 'bbox': [x1,y1,x2,y2], 'confidence': float}}

        # Tracking state
        self.frame_count = 0
        self.active_dogs = set()

    def update_valid_ids(self, dog_list: List[dict]):
        """Update the list of valid dog IDs (Rule 1 - user configurable)"""
        self.valid_dog_ids = {dog['marker_id']: dog['id'] for dog in dog_list}
        self.dog_names = {v: k for k, v in self.valid_dog_ids.items()}

    def process_frame(self, detections: List, aruco_markers: List[Tuple[int, float, float]]) -> Dict:
        """
        Process a frame with detections and ArUco markers

        Args:
            detections: List of AI detection boxes
            aruco_markers: List of (marker_id, cx, cy) tuples

        Returns:
            Dict mapping detection indices to dog identities
        """
        self.frame_count += 1
        current_time = time.time()
        assignments = {}  # {detection_idx: dog_name}

        # Rule 1: Filter out invalid ArUco IDs
        valid_markers = [(id, x, y) for id, x, y in aruco_markers if id in self.valid_dog_ids]

        # Clean up old persistence data
        self._cleanup_old_tracking(current_time)

        # Process each detection box
        for idx, detection in enumerate(detections):
            if idx >= self.max_dogs:  # Rule 6: Max dogs limit
                break

            bbox = self._get_bbox(detection)
            if not bbox:
                continue

            # Try to find ArUco marker in this bbox
            dog_id = self._find_marker_in_bbox(bbox, valid_markers)

            if dog_id:
                # Direct ArUco detection - highest confidence
                dog_name = self.valid_dog_ids[dog_id]
                assignments[idx] = dog_name
                # Rule 2: Update persistence tracking
                self._update_tracking(dog_id, bbox, current_time, confidence=1.0)

            else:
                # No marker found, try persistence rules
                dog_name = self._apply_persistence_rules(bbox, valid_markers, len(detections), current_time)
                if dog_name:
                    assignments[idx] = dog_name

        return assignments

    def _find_marker_in_bbox(self, bbox: List[float], markers: List[Tuple]) -> Optional[int]:
        """Find ArUco marker inside bounding box"""
        x1, y1, x2, y2 = bbox
        for marker_id, cx, cy in markers:
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return marker_id
        return None

    def _apply_persistence_rules(self, bbox: List[float], valid_markers: List,
                                  total_detections: int, current_time: float) -> Optional[str]:
        """Apply persistence rules to identify dog without direct marker detection"""

        # Rule 3: Proximity matching with persistence
        closest_dog = self._find_closest_tracked_dog(bbox, current_time)
        if closest_dog:
            return self.valid_dog_ids.get(closest_dog)

        # Rule 5: Mutual exclusion (if one dog detected, other must be the other)
        if total_detections == 2 and len(valid_markers) == 1:
            detected_id = valid_markers[0][0]
            # Get the other dog ID
            for dog_id in self.valid_dog_ids:
                if dog_id != detected_id and dog_id in self.active_dogs:
                    return self.valid_dog_ids[dog_id]

        # Rule 4: Single dog mode - use default
        if total_detections == 1 and len(self.active_dogs) <= 1:
            return self.default_dog_name

        return None

    def _find_closest_tracked_dog(self, bbox: List[float], current_time: float) -> Optional[int]:
        """Find closest dog from persistence tracking (Rule 3)"""
        best_match = None
        best_distance = float('inf')

        for dog_id, tracking in self.last_known_positions.items():
            # Check if tracking is still valid (within persistence time)
            if current_time - tracking['time'] > self.persistence_time:
                continue

            # Calculate bbox center distance
            old_bbox = tracking['bbox']
            distance = self._bbox_distance(bbox, old_bbox)

            # Weight by time decay (more recent = better)
            time_factor = 1.0 - (current_time - tracking['time']) / self.persistence_time
            weighted_distance = distance / (time_factor + 0.1)

            if weighted_distance < best_distance and weighted_distance < 200:  # Max 200 pixels
                best_distance = weighted_distance
                best_match = dog_id

        return best_match

    def _update_tracking(self, dog_id: int, bbox: List[float], timestamp: float, confidence: float):
        """Update persistence tracking for a dog (Rule 2)"""
        self.last_known_positions[dog_id] = {
            'time': timestamp,
            'bbox': bbox,
            'confidence': confidence
        }
        self.active_dogs.add(dog_id)

    def _cleanup_old_tracking(self, current_time: float):
        """Remove expired tracking data"""
        expired = []
        for dog_id, tracking in self.last_known_positions.items():
            if current_time - tracking['time'] > self.persistence_time:
                expired.append(dog_id)

        for dog_id in expired:
            del self.last_known_positions[dog_id]
            self.active_dogs.discard(dog_id)

    def _get_bbox(self, detection) -> List[float]:
        """Extract bbox from detection object"""
        if hasattr(detection, 'bbox'):
            return detection.bbox[:4] if len(detection.bbox) >= 4 else None
        elif isinstance(detection, dict):
            return detection.get('bbox', [])[:4] if len(detection.get('bbox', [])) >= 4 else None
        return None

    def _bbox_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate distance between two bbox centers"""
        cx1 = (bbox1[0] + bbox1[2]) / 2
        cy1 = (bbox1[1] + bbox1[3]) / 2
        cx2 = (bbox2[0] + bbox2[2]) / 2
        cy2 = (bbox2[1] + bbox2[3]) / 2
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    def get_status(self) -> Dict:
        """Get current tracking status"""
        return {
            'active_dogs': list(self.active_dogs),
            'tracked_count': len(self.last_known_positions),
            'max_dogs': self.max_dogs,
            'default_dog': self.default_dog_name,
            'frame_count': self.frame_count
        }