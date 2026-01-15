"""
Geometric Behavior Classifier for Dogs
=======================================

Uses simple geometric heuristics instead of ML when keypoint quality is poor.
This is a fallback for the LSTM behavior model which requires good keypoint data.

Heuristics:
- SIT: Tall bbox (height > width), head at top, compact body
- STAND: Wide bbox (width > height), legs visible below body
- LIE: Very wide/flat bbox, body low and spread
- CROSS: Lying with paws close together (requires good paw keypoints)
- SPIN: Detected via rotation (requires temporal tracking)

Usage:
    from services.ai.geometric_classifier import GeometricClassifier

    classifier = GeometricClassifier()
    behavior, confidence, method = classifier.classify(keypoints, bbox)
"""

import numpy as np
import logging
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from collections import deque
import threading

logger = logging.getLogger(__name__)


@dataclass
class GeometricConfig:
    """Configuration for geometric classification thresholds"""
    # Aspect ratio thresholds (height / width)
    sit_min_aspect: float = 0.9       # Sitting dogs are tall
    sit_max_aspect: float = 2.0       # But not too tall
    stand_min_aspect: float = 0.4     # Standing dogs are wide
    stand_max_aspect: float = 0.9     # But not too wide
    lie_max_aspect: float = 0.5       # Lying dogs are flat/wide

    # Keypoint position thresholds (normalized 0-1 within bbox)
    sit_head_max_y: float = 0.35      # Head should be in top 35% when sitting
    stand_head_max_y: float = 0.4     # Head in top 40% when standing
    lie_head_min_y: float = 0.3       # Head can be lower when lying

    # Paw position thresholds for cross detection
    cross_paw_max_distance: float = 0.15  # Paws within 15% of bbox width

    # Confidence thresholds
    min_keypoint_conf: float = 0.25   # Minimum keypoint confidence to use
    high_conf_keypoints: int = 8      # Need this many for high confidence

    # Spin detection
    spin_rotation_threshold: float = 45.0  # Degrees of rotation to detect spin
    spin_history_frames: int = 10     # Frames to track for spin


class GeometricClassifier:
    """
    Classifies dog behavior using geometric heuristics.

    Works when ML model fails due to poor keypoint quality.
    """

    # Keypoint indices (YOLOv8 dog pose - 24 keypoints)
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9      # Front left paw
    RIGHT_WRIST = 10    # Front right paw
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15     # Back left paw
    RIGHT_ANKLE = 16    # Back right paw
    TAIL_BASE = 17

    def __init__(self, config: GeometricConfig = None):
        self.config = config or GeometricConfig()

        # Spin detection history per dog
        self._orientation_history: Dict[str, deque] = {}
        self._lock = threading.Lock()

        logger.info("GeometricClassifier initialized")

    def classify(self, keypoints: np.ndarray, bbox: Tuple[float, float, float, float],
                 dog_id: str = "default") -> Tuple[str, float, str]:
        """
        Classify behavior using geometric heuristics.

        Args:
            keypoints: Shape (24, 3) - x, y, confidence per keypoint
            bbox: (x1, y1, x2, y2) bounding box
            dog_id: Identifier for temporal tracking

        Returns:
            (behavior, confidence, method) where method is 'geometric' or 'aspect_only'
        """
        x1, y1, x2, y2 = bbox
        bbox_width = max(x2 - x1, 1)
        bbox_height = max(y2 - y1, 1)
        aspect_ratio = bbox_height / bbox_width

        # Count confident keypoints
        if keypoints.shape[1] >= 3:
            confident_mask = keypoints[:, 2] >= self.config.min_keypoint_conf
            num_confident = np.sum(confident_mask)
        else:
            confident_mask = np.ones(len(keypoints), dtype=bool)
            num_confident = 0

        # Normalize keypoints to bbox (0-1)
        norm_kpts = self._normalize_keypoints(keypoints, bbox)

        # Check for spin first (requires temporal data)
        spin_detected, spin_conf = self._check_spin(norm_kpts, dog_id, confident_mask)
        if spin_detected:
            return "spin", spin_conf, "geometric_temporal"

        # If we have enough keypoints, use full geometric analysis
        if num_confident >= self.config.high_conf_keypoints:
            behavior, conf = self._classify_with_keypoints(norm_kpts, aspect_ratio, confident_mask)
            return behavior, conf, "geometric"

        # Fallback to aspect ratio only
        behavior, conf = self._classify_by_aspect(aspect_ratio)
        return behavior, conf * 0.6, "aspect_only"  # Lower confidence for aspect-only

    def _normalize_keypoints(self, keypoints: np.ndarray, bbox: Tuple) -> np.ndarray:
        """Normalize keypoints to 0-1 within bounding box"""
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1e-6)
        h = max(y2 - y1, 1e-6)

        norm = keypoints.copy()
        norm[:, 0] = (keypoints[:, 0] - x1) / w
        norm[:, 1] = (keypoints[:, 1] - y1) / h
        return norm

    def _classify_with_keypoints(self, norm_kpts: np.ndarray, aspect_ratio: float,
                                  confident_mask: np.ndarray) -> Tuple[str, float]:
        """Classify using keypoint positions + aspect ratio"""
        cfg = self.config

        # Get head position (average of nose, eyes, ears if confident)
        head_indices = [self.NOSE, self.LEFT_EYE, self.RIGHT_EYE, self.LEFT_EAR, self.RIGHT_EAR]
        head_y = self._get_average_y(norm_kpts, head_indices, confident_mask)

        # Get front paw positions
        front_paw_indices = [self.LEFT_WRIST, self.RIGHT_WRIST]
        left_paw = norm_kpts[self.LEFT_WRIST] if confident_mask[self.LEFT_WRIST] else None
        right_paw = norm_kpts[self.RIGHT_WRIST] if confident_mask[self.RIGHT_WRIST] else None

        # Get body center (shoulders + hips)
        body_indices = [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, self.LEFT_HIP, self.RIGHT_HIP]
        body_y = self._get_average_y(norm_kpts, body_indices, confident_mask)

        # Get back paw positions
        back_paw_y = self._get_average_y(norm_kpts, [self.LEFT_ANKLE, self.RIGHT_ANKLE], confident_mask)

        scores = {
            "sit": 0.0,
            "stand": 0.0,
            "lie": 0.0,
            "cross": 0.0
        }

        # --- SIT detection ---
        # Tall aspect, head at top, body compact
        if cfg.sit_min_aspect <= aspect_ratio <= cfg.sit_max_aspect:
            scores["sit"] += 0.4
        if head_y is not None and head_y < cfg.sit_head_max_y:
            scores["sit"] += 0.3
        # Sitting dogs have paws in front, roughly same Y as body center
        if body_y is not None and back_paw_y is not None:
            if abs(body_y - back_paw_y) < 0.3:  # Back paws close to body (tucked)
                scores["sit"] += 0.2

        # --- STAND detection ---
        # Wide aspect, legs below body
        if cfg.stand_min_aspect <= aspect_ratio <= cfg.stand_max_aspect:
            scores["stand"] += 0.4
        if head_y is not None and head_y < cfg.stand_head_max_y:
            scores["stand"] += 0.2
        # Standing dogs have paws at bottom of bbox
        if back_paw_y is not None and back_paw_y > 0.7:
            scores["stand"] += 0.3

        # --- LIE detection ---
        # Very wide/flat aspect, body spread out
        if aspect_ratio < cfg.lie_max_aspect:
            scores["lie"] += 0.5
        if head_y is not None and head_y > cfg.lie_head_min_y:
            scores["lie"] += 0.2
        # Lying dogs have body low in frame
        if body_y is not None and body_y > 0.4:
            scores["lie"] += 0.2

        # --- CROSS detection ---
        # Must be lying + paws close together
        if scores["lie"] > 0.4 and left_paw is not None and right_paw is not None:
            paw_distance = abs(left_paw[0] - right_paw[0])
            if paw_distance < cfg.cross_paw_max_distance:
                scores["cross"] = scores["lie"] + 0.3
                scores["lie"] *= 0.5  # Reduce lie score

        # Find best match
        best_behavior = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_behavior]

        # Convert score to confidence (0.5-0.9 range)
        confidence = min(0.9, 0.5 + best_score * 0.4)

        return best_behavior, confidence

    def _classify_by_aspect(self, aspect_ratio: float) -> Tuple[str, float]:
        """Simple classification by aspect ratio only"""
        cfg = self.config

        if aspect_ratio < cfg.lie_max_aspect:
            return "lie", 0.7
        elif aspect_ratio < cfg.stand_max_aspect:
            return "stand", 0.6
        elif aspect_ratio < cfg.sit_max_aspect:
            return "sit", 0.7
        else:
            return "sit", 0.5  # Very tall = probably sitting

    def _get_average_y(self, norm_kpts: np.ndarray, indices: List[int],
                       confident_mask: np.ndarray) -> Optional[float]:
        """Get average Y position of specified keypoints"""
        valid_ys = []
        for idx in indices:
            if idx < len(norm_kpts) and idx < len(confident_mask) and confident_mask[idx]:
                valid_ys.append(norm_kpts[idx, 1])

        if valid_ys:
            return np.mean(valid_ys)
        return None

    def _check_spin(self, norm_kpts: np.ndarray, dog_id: str,
                    confident_mask: np.ndarray) -> Tuple[bool, float]:
        """
        Detect spin by tracking body orientation over time.

        A spin is detected when the dog rotates significantly.
        """
        # Calculate body orientation from shoulders and hips
        orientation = self._calculate_orientation(norm_kpts, confident_mask)

        if orientation is None:
            return False, 0.0

        with self._lock:
            if dog_id not in self._orientation_history:
                self._orientation_history[dog_id] = deque(maxlen=self.config.spin_history_frames)

            history = self._orientation_history[dog_id]
            history.append(orientation)

            if len(history) < 5:
                return False, 0.0

            # Calculate total rotation over history
            total_rotation = 0.0
            for i in range(1, len(history)):
                diff = history[i] - history[i-1]
                # Handle wraparound at 180/-180
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                total_rotation += abs(diff)

            # Spin detected if total rotation exceeds threshold
            if total_rotation > self.config.spin_rotation_threshold:
                confidence = min(0.9, 0.5 + total_rotation / 180.0)
                # Clear history after detection to prevent repeated triggers
                history.clear()
                return True, confidence

        return False, 0.0

    def _calculate_orientation(self, norm_kpts: np.ndarray,
                               confident_mask: np.ndarray) -> Optional[float]:
        """Calculate body orientation angle from keypoints"""
        # Use shoulders or hips to calculate orientation
        left_shoulder = norm_kpts[self.LEFT_SHOULDER] if confident_mask[self.LEFT_SHOULDER] else None
        right_shoulder = norm_kpts[self.RIGHT_SHOULDER] if confident_mask[self.RIGHT_SHOULDER] else None

        if left_shoulder is not None and right_shoulder is not None:
            dx = right_shoulder[0] - left_shoulder[0]
            dy = right_shoulder[1] - left_shoulder[1]
            return np.degrees(np.arctan2(dy, dx))

        # Fallback to hips
        left_hip = norm_kpts[self.LEFT_HIP] if confident_mask[self.LEFT_HIP] else None
        right_hip = norm_kpts[self.RIGHT_HIP] if confident_mask[self.RIGHT_HIP] else None

        if left_hip is not None and right_hip is not None:
            dx = right_hip[0] - left_hip[0]
            dy = right_hip[1] - left_hip[1]
            return np.degrees(np.arctan2(dy, dx))

        return None

    def reset_history(self, dog_id: str = None):
        """Clear orientation history for spin detection"""
        with self._lock:
            if dog_id:
                if dog_id in self._orientation_history:
                    self._orientation_history[dog_id].clear()
            else:
                self._orientation_history.clear()


# Singleton instance
_classifier_instance: Optional[GeometricClassifier] = None
_classifier_lock = threading.Lock()


def get_geometric_classifier(config: GeometricConfig = None) -> GeometricClassifier:
    """Get or create singleton GeometricClassifier instance"""
    global _classifier_instance

    if _classifier_instance is None:
        with _classifier_lock:
            if _classifier_instance is None:
                _classifier_instance = GeometricClassifier(config)

    return _classifier_instance
