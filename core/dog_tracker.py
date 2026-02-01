#!/usr/bin/env python3
"""
Dog Tracking System with ArUco Persistence Rules
Handles identification and tracking of multiple dogs using ArUco markers
with smart persistence and fallback rules.

Identification Priority:
1. ARUCO marker (100% certain)
2. Color match with single dog (high confidence)
3. Color match with multiple dogs (uncertain - show generic)
4. No match (show "Unknown Dog")
"""

import time
from typing import Dict, List, Tuple, Optional
import numpy as np

# Import profile manager for color-based identification
try:
    from core.dog_profile_manager import (
        get_dog_profile_manager,
        IdentificationMethod,
        IdentificationResult
    )
    HAS_PROFILE_MANAGER = True
except ImportError:
    HAS_PROFILE_MANAGER = False
    IdentificationMethod = None

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

        # Rule 2: Persistence tracking (reduced from 30s to 15s for faster switching)
        self.persistence_time = config.get('persistence_seconds', 15)
        self.last_known_positions = {}  # {dog_id: {'time': timestamp, 'bbox': [x1,y1,x2,y2], 'confidence': float}}

        # Faster clearing when dog leaves frame
        self.frames_without_detection = {}  # {dog_id: frame_count}

        # Grace period: wait for ArUco identification before falling back to default
        # Reduced to 0 - coaching engine now handles late ArUco identification separately
        # ArUco runs in parallel; sessions start based on presence, not identity
        self.aruco_grace_period = config.get('aruco_grace_period', 0.0)  # seconds (was 10.0)
        self.unidentified_dogs = {}  # {detection_idx: first_seen_time}

        # Tracking state
        self.frame_count = 0
        self.active_dogs = set()

        # Profile manager for color-based identification
        self._profile_manager = None
        self._last_frame = None  # Cache for color extraction
        if HAS_PROFILE_MANAGER:
            try:
                self._profile_manager = get_dog_profile_manager()
            except Exception as e:
                print(f"Warning: Could not load profile manager: {e}")

    def update_valid_ids(self, dog_list: List[dict]):
        """Update the list of valid dog IDs (Rule 1 - user configurable)"""
        self.valid_dog_ids = {dog['marker_id']: dog['id'] for dog in dog_list}
        self.dog_names = {v: k for k, v in self.valid_dog_ids.items()}

    def set_frame(self, frame):
        """Set current frame for color extraction (call before process_frame)"""
        self._last_frame = frame

    def process_frame(self, detections: List, aruco_markers: List[Tuple[int, float, float]], frame=None) -> Dict:
        """
        Process a frame with detections and ArUco markers

        Args:
            detections: List of AI detection boxes
            aruco_markers: List of (marker_id, cx, cy) tuples
            frame: Optional camera frame for color-based identification

        Returns:
            Dict mapping detection indices to dog identities
        """
        # Store frame for color extraction
        if frame is not None:
            self._last_frame = frame
        self.frame_count += 1
        current_time = time.time()
        assignments = {}  # {detection_idx: dog_name}

        # Rule 1: Filter out invalid ArUco IDs
        valid_markers = [(id, x, y) for id, x, y in aruco_markers if id in self.valid_dog_ids]

        # Clean up old persistence data
        self._cleanup_old_tracking(current_time)

        # Track which detection indices are still in view
        current_detections = set()

        # Process each detection box
        for idx, detection in enumerate(detections):
            if idx >= self.max_dogs:  # Rule 6: Max dogs limit
                break

            bbox = self._get_bbox(detection)
            if not bbox:
                continue

            current_detections.add(idx)

            # Try to find ArUco marker in this bbox
            dog_id = self._find_marker_in_bbox(bbox, valid_markers)

            if dog_id:
                # Direct ArUco detection - highest confidence
                dog_name = self.valid_dog_ids[dog_id]
                assignments[idx] = dog_name
                # Rule 2: Update persistence tracking
                id_method = "aruco" if HAS_PROFILE_MANAGER else "aruco"
                self._update_tracking(dog_id, bbox, current_time, confidence=1.0, id_method=id_method)
                # Clear from unidentified tracking (ArUco found!)
                if idx in self.unidentified_dogs:
                    del self.unidentified_dogs[idx]

            else:
                # No marker found - try color-based identification first
                id_result = None
                if self._profile_manager and self._last_frame is not None:
                    id_result = self._profile_manager.identify_dog(
                        bbox=bbox,
                        frame=self._last_frame,
                        aruco_markers=aruco_markers
                    )

                    if id_result and id_result.method.value in ("aruco", "color"):
                        # High confidence identification (ARUCO or unique color)
                        dog_name = id_result.dog_name.lower()
                        assignments[idx] = dog_name
                        # Find marker_id for this dog if we have it
                        marker_id = self.dog_names.get(dog_name, hash(dog_name) % 10000)
                        self._update_tracking(
                            marker_id, bbox, current_time,
                            confidence=id_result.confidence,
                            id_method=id_result.method.value
                        )
                        if idx in self.unidentified_dogs:
                            del self.unidentified_dogs[idx]
                        continue  # Skip fallback rules

                # Fallback to persistence rules
                dog_name = self._apply_persistence_rules(bbox, valid_markers, len(detections), current_time)

                if dog_name:
                    # Check if this is the default name (bezik)
                    if dog_name == self.default_dog_name:
                        # Apply grace period - only use default after waiting
                        if idx not in self.unidentified_dogs:
                            # First time seeing this unidentified dog
                            self.unidentified_dogs[idx] = current_time

                        time_waiting = current_time - self.unidentified_dogs[idx]
                        if time_waiting < self.aruco_grace_period:
                            # Still within grace period - don't assign default yet
                            # Return None to let coaching engine wait
                            pass
                        else:
                            # Grace period expired - use default
                            assignments[idx] = dog_name
                    else:
                        # Not the default - it's a real identification
                        assignments[idx] = dog_name
                        # Clear from unidentified
                        if idx in self.unidentified_dogs:
                            del self.unidentified_dogs[idx]
                else:
                    # BUILD 38: Store unidentified dogs with generic name so bounding boxes can be drawn
                    # Previously, unidentified dogs weren't stored, causing no boxes on WebRTC overlay
                    generic_id = f"dog_{idx}"
                    assignments[idx] = generic_id
                    # Store in tracking with generic marker ID (negative to avoid collision with real ArUco)
                    self._update_tracking(
                        -(idx + 1000),  # Negative ID for unidentified dogs
                        bbox,
                        current_time,
                        confidence=0.5,
                        id_method="unknown"
                    )
                    # Store the generic name mapping
                    self.valid_dog_ids[-(idx + 1000)] = generic_id

        # Clean up unidentified dogs that are no longer in view
        expired_unidentified = [idx for idx in self.unidentified_dogs if idx not in current_detections]
        for idx in expired_unidentified:
            del self.unidentified_dogs[idx]

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
        """Apply persistence rules to identify dog without direct marker detection

        BUILD 34 FIX: More conservative identification to prevent wrong dog labels.
        - Only use proximity if ArUco was seen in same bbox recently
        - Don't default to specific dog names - return None for unknown
        - Let caller display "Dog" for unidentified detections
        """

        # Rule 0: Single dog + single marker = same dog (collar might be outside bbox)
        # This is the most reliable fallback - ArUco visible means we know which dog
        if total_detections == 1 and len(valid_markers) == 1:
            marker_id = valid_markers[0][0]
            dog_name = self.valid_dog_ids.get(marker_id)
            if dog_name:
                # Update tracking for this dog
                self._update_tracking(marker_id, bbox, current_time, confidence=0.9)
                return dog_name

        # Rule 3: Proximity matching - ONLY if very close to recent ArUco detection
        # Reduced threshold from 200px to 80px to prevent false matches
        closest_dog = self._find_closest_tracked_dog(bbox, current_time, max_distance=80)
        if closest_dog:
            # Additional check: only if last ID was ArUco (not persistence)
            tracking = self.last_known_positions.get(closest_dog)
            if tracking and tracking.get('id_method') == 'aruco':
                return self.valid_dog_ids.get(closest_dog)

        # Rule 5: Mutual exclusion - DISABLED in Build 34
        # This rule was too aggressive and labeled wrong dogs
        # if total_detections == 2 and len(valid_markers) == 1:
        #     ... removed

        # BUILD 34: Don't use default dog name for unknown detections
        # Return None and let the caller display "Dog" generically
        # This prevents wrong dog names from appearing
        return None

    def _find_closest_tracked_dog(self, bbox: List[float], current_time: float, max_distance: float = 200) -> Optional[int]:
        """Find closest dog from persistence tracking (Rule 3)

        Args:
            bbox: Current bounding box
            current_time: Current timestamp
            max_distance: Maximum pixel distance to consider a match (default 200, reduced to 80 for stricter matching)
        """
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

            if weighted_distance < best_distance and weighted_distance < max_distance:
                best_distance = weighted_distance
                best_match = dog_id

        return best_match

    def _update_tracking(self, dog_id: int, bbox: List[float], timestamp: float, confidence: float, id_method: str = "persistence"):
        """Update persistence tracking for a dog (Rule 2)"""
        self.last_known_positions[dog_id] = {
            'time': timestamp,
            'bbox': bbox,
            'confidence': confidence,
            'id_method': id_method
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
            if dog_id in self.frames_without_detection:
                del self.frames_without_detection[dog_id]

    def clear_dog_tracking(self, dog_name: str = None):
        """
        Manually clear tracking for a specific dog or all dogs.
        Call this when you want to force re-identification.

        Args:
            dog_name: Name to clear, or None to clear all
        """
        if dog_name:
            # Find marker ID for this dog name
            marker_id = self.dog_names.get(dog_name)
            if marker_id:
                if marker_id in self.last_known_positions:
                    del self.last_known_positions[marker_id]
                self.active_dogs.discard(marker_id)
                if marker_id in self.frames_without_detection:
                    del self.frames_without_detection[marker_id]
        else:
            # Clear all tracking
            self.last_known_positions.clear()
            self.active_dogs.clear()
            self.frames_without_detection.clear()
            self.unidentified_dogs.clear()

    def _get_bbox(self, detection) -> List[float]:
        """Extract bbox from detection object"""
        # Handle Detection dataclass with x1, y1, x2, y2 attributes
        if hasattr(detection, 'x1') and hasattr(detection, 'y2'):
            return [detection.x1, detection.y1, detection.x2, detection.y2]
        # Handle objects with bbox attribute
        if hasattr(detection, 'bbox'):
            return detection.bbox[:4] if len(detection.bbox) >= 4 else None
        # Handle dict format
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

    def get_tracked_dogs(self) -> Dict:
        """Get all currently tracked dogs with their data for overlay rendering

        Returns:
            Dict mapping dog_id to dog data:
            {
                'dog_name': {
                    'bbox': [x1, y1, x2, y2],
                    'name': 'bezik',
                    'confidence': 0.85,
                    'id_method': 'aruco',  # aruco, color, persistence, unknown
                    'behavior': '',  # Set by behavior interpreter
                    'keypoints': []  # Set by pose detector
                }
            }
        """
        current_time = time.time()
        result = {}

        for marker_id, tracking in self.last_known_positions.items():
            # Only include dogs with recent tracking (within persistence time)
            if current_time - tracking['time'] > self.persistence_time:
                continue

            dog_name = self.valid_dog_ids.get(marker_id, f"dog_{marker_id}")
            # BUILD 38: Display "Dog" for unidentified dogs instead of "dog_0", "dog_-1000" etc
            id_method = tracking.get('id_method', 'persistence')
            display_name = dog_name
            if id_method == "unknown" or dog_name.startswith("dog_"):
                display_name = "Dog"

            result[dog_name] = {
                'bbox': tracking.get('bbox', []),
                'name': display_name,
                'confidence': tracking.get('confidence', 0.0),
                'id_method': id_method,
                'behavior': tracking.get('behavior', ''),
                'keypoints': tracking.get('keypoints', [])
            }

        return result

    def update_dog_behavior(self, dog_name: str, behavior: str, confidence: float, keypoints: list = None):
        """Update behavior and keypoints for a tracked dog (called by behavior interpreter)"""
        marker_id = self.dog_names.get(dog_name)
        if marker_id and marker_id in self.last_known_positions:
            self.last_known_positions[marker_id]['behavior'] = behavior
            self.last_known_positions[marker_id]['confidence'] = confidence
            if keypoints:
                self.last_known_positions[marker_id]['keypoints'] = keypoints