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
    # Aspect ratio thresholds (height / width) - tuned from real testing
    # NOTE: Front-facing dogs (crowding robot for treats) look different than side-view!
    # - Front-facing lying dog (sphinx pose): aspect ~0.8-1.1 (nearly square)
    # - Side-view lying dog: aspect < 0.6 (flat/wide)
    # - Sitting dog: aspect > 1.15 (tall)
    # - Standing dog: fills the gap
    sit_min_aspect: float = 1.15      # Sitting dogs are tall (raised from 1.05)
    sit_max_aspect: float = 3.0       # Very tall when sitting upright
    stand_min_aspect: float = 0.95    # Standing - narrowed range
    stand_max_aspect: float = 1.15    # Up to sit threshold
    lie_max_aspect: float = 0.95      # RAISED from 0.75 - front-facing sphinx pose is ~0.9

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
        self._center_history: Dict[str, deque] = {}  # Track bbox center movement
        self._orientation_history: Dict[str, deque] = {}  # Legacy - keep for now
        self._handedness_history: Dict[str, deque] = {}  # Track left/right keypoint flips
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
        spin_detected, spin_conf = self._check_spin(norm_kpts, dog_id, confident_mask, bbox)
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
            # "cross" REMOVED - unreliable detection, dogs don't cross paws reliably
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

        # CROSS detection REMOVED - unreliable, dogs don't cross paws on command reliably

        # Find best match
        best_behavior = max(scores.keys(), key=lambda k: scores[k])
        best_score = scores[best_behavior]

        # Convert score to confidence (0.5-0.9 range)
        confidence = min(0.9, 0.5 + best_score * 0.4)

        return best_behavior, confidence

    def _classify_by_aspect(self, aspect_ratio: float) -> Tuple[str, float]:
        """
        Simple classification by aspect ratio only (height/width).

        Typical ranges for FRONT-FACING dogs (crowding robot for treats):
        - Lying (sphinx pose): aspect < 0.95 (wider than tall, but not as flat as side-view)
        - Standing: aspect 0.95 - 1.15 (roughly square)
        - Sitting: aspect > 1.15 (taller than wide)
        """
        cfg = self.config

        # Log aspect ratio for tuning
        logger.info(f"📐 Aspect ratio: {aspect_ratio:.2f} (lie<{cfg.lie_max_aspect:.2f}, stand<{cfg.stand_max_aspect:.2f}, sit>{cfg.sit_min_aspect:.2f})")

        if aspect_ratio < cfg.lie_max_aspect:
            # Lower aspect = more confident it's lying
            conf = 0.80 if aspect_ratio < 0.75 else 0.70
            return "lie", conf
        elif aspect_ratio >= cfg.sit_min_aspect:
            # Higher aspect = more confident it's sitting
            conf = 0.80 if aspect_ratio > 1.3 else 0.70
            return "sit", conf
        else:
            # Middle range = standing (lower confidence)
            return "stand", 0.60

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
                    confident_mask: np.ndarray, bbox: Tuple[float, float, float, float] = None) -> Tuple[bool, float]:
        """
        Detect FAST spins (0.25 sec = 2-3 frames at 10fps).

        Key insight: During fast spin, keypoints are garbage (motion blur).
        But BBOX changes rapidly:
        - Aspect ratio swings (side->front->side = narrow->wide->narrow)
        - Center position moves

        We track FRAME-TO-FRAME DELTAS, not long-term statistics.
        High deltas in aspect + center over 3-5 frames = spin.
        """
        if bbox is None:
            return False, 0.0

        with self._lock:
            if dog_id not in self._handedness_history:
                self._handedness_history[dog_id] = deque(maxlen=15)

            history = self._handedness_history[dog_id]

            # Extract bbox metrics
            x1, y1, x2, y2 = bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = max(x2 - x1, 1)
            h = max(y2 - y1, 1)
            aspect = h / w

            # Store current frame
            import time
            history.append({
                'time': time.time(),
                'cx': cx,
                'cy': cy,
                'aspect': aspect,
                'w': w,
                'h': h
            })

            # Need at least 4 frames (catches 0.3-0.4 sec spins at 10fps)
            if len(history) < 4:
                return False, 0.0

            # Calculate RAPID CHANGES over recent frames
            recent = list(history)[-8:]  # Last 8 frames max

            # 1. Frame-to-frame aspect ratio changes (sum of absolute deltas)
            aspect_deltas = []
            for i in range(1, len(recent)):
                delta = abs(recent[i]['aspect'] - recent[i-1]['aspect'])
                aspect_deltas.append(delta)
            total_aspect_change = sum(aspect_deltas)
            max_aspect_delta = max(aspect_deltas) if aspect_deltas else 0

            # 2. Frame-to-frame center movement
            center_deltas = []
            for i in range(1, len(recent)):
                dx = abs(recent[i]['cx'] - recent[i-1]['cx'])
                dy = abs(recent[i]['cy'] - recent[i-1]['cy'])
                center_deltas.append(dx + dy)
            total_center_movement = sum(center_deltas)
            max_center_delta = max(center_deltas) if center_deltas else 0

            # 3. Aspect ratio range (min to max)
            aspects = [f['aspect'] for f in recent]
            aspect_range = max(aspects) - min(aspects)

            # 4. Width variation (dog appears wider when sideways during spin)
            widths = [f['w'] for f in recent]
            width_range = max(widths) - min(widths)

            # Calculate spin score based on rapid changes
            # LOWERED THRESHOLDS - fast spins at 10fps need sensitivity
            spin_score = 0.0

            # Large aspect ratio swings (side-to-front rotation)
            if total_aspect_change > 0.25:
                spin_score += 0.25
            if total_aspect_change > 0.45:
                spin_score += 0.2
            if max_aspect_delta > 0.12:  # Single big change
                spin_score += 0.15

            # Center movement (LOWERED from 50/100)
            if total_center_movement > 25:
                spin_score += 0.2
            if total_center_movement > 50:
                spin_score += 0.15

            # Aspect range (spinning shows different profiles)
            if aspect_range > 0.18:
                spin_score += 0.2
            if aspect_range > 0.35:
                spin_score += 0.15

            # Width variation (spinning dog width changes)
            if width_range > 25:
                spin_score += 0.15

            # Log for debugging
            if spin_score > 0.25:
                logger.debug(f"Spin check: aspect_change={total_aspect_change:.2f}, "
                            f"center_move={total_center_movement:.0f}, "
                            f"aspect_range={aspect_range:.2f}, score={spin_score:.2f}")

            # Detect spin with lower threshold (fast spins need sensitivity)
            if spin_score >= 0.40:
                confidence = min(0.80, 0.55 + spin_score * 0.3)
                # Clear history after detection
                history.clear()
                logger.info(f"🔄 SPIN detected! aspect_change={total_aspect_change:.2f}, "
                           f"center_move={total_center_movement:.0f}, aspect_range={aspect_range:.2f}, "
                           f"width_range={width_range:.0f}, score={spin_score:.2f}")
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
        """Clear all spin detection history"""
        with self._lock:
            if dog_id:
                if dog_id in self._orientation_history:
                    self._orientation_history[dog_id].clear()
                if dog_id in self._handedness_history:
                    self._handedness_history[dog_id].clear()
                if dog_id in self._center_history:
                    self._center_history[dog_id].clear()
            else:
                self._orientation_history.clear()
                self._handedness_history.clear()
                self._center_history.clear()


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
