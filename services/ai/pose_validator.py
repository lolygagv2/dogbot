"""
Pose Validation & Filtering Module for WIM-Z Dog Behavior Classification
=========================================================================

This module sits between YOLOv8 pose estimation and the behavior classification model.
It filters out bad/unreliable pose estimations before they can cause misclassifications.

Filters implemented:
1. Keypoint confidence filtering - reject low-confidence pose detections
2. Motion blur detection - skip blurry frames
3. Skeleton geometry validation - reject physically impossible poses
4. Detection region quality - reject empty/featureless detections
5. Temporal voting - require consistency across frames for stable predictions

Integration:
    from services.ai.pose_validator import get_pose_validator

    validator = get_pose_validator()

    # Before running behavior model:
    validation = validator.validate_pose(frame, pose_keypoints)
    if not validation.is_valid:
        skip this frame

    # After getting raw prediction:
    stable_behavior, confidence, is_stable = validator.temporal_vote(behavior, conf)
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import logging
import threading

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of pose validation"""
    is_valid: bool
    confidence: float
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PoseValidatorConfig:
    """Configuration for pose validation thresholds"""
    # Keypoint confidence thresholds
    min_keypoint_confidence: float = 0.35      # Minimum confidence for a keypoint to be valid
    min_visible_keypoints: int = 6             # Need at least this many valid keypoints
    min_leg_keypoints: int = 2                 # Need at least this many leg keypoints

    # Blur/quality thresholds
    blur_threshold: float = 80.0               # Laplacian variance threshold (lower = blurry)
    min_edge_density: float = 0.025            # Minimum edge pixel ratio in detection

    # Bounding box validation
    min_bbox_area: int = 4000                  # Minimum detection size in pixels
    max_aspect_ratio: float = 3.5              # Max height/width ratio
    min_aspect_ratio: float = 0.25             # Min height/width ratio
    min_frame_coverage: float = 0.01           # Min percentage of frame covered
    max_frame_coverage: float = 0.95           # Max percentage (reject full-frame detections)

    # Skeleton geometry
    max_limb_ratio: float = 0.65               # Max limb length as fraction of bbox diagonal
    min_limb_ratio: float = 0.015              # Min limb length as fraction of bbox diagonal
    max_keypoints_outside_bbox: int = 2        # Max keypoints allowed outside bbox

    # Temporal voting
    temporal_window: int = 5                   # Frames for temporal smoothing
    temporal_threshold: float = 0.6            # Fraction of frames that must agree

    # Behavior-specific adjustments
    cross_confidence_penalty: float = 0.15     # Reduce cross confidence by this amount
    spin_min_frames: int = 3                   # Spin needs more temporal consistency


class PoseValidator:
    """
    Validates pose estimations before behavior classification.
    Catches garbage poses that would cause misclassification.
    """

    # YOLOv8 pose keypoint indices for dogs (24 keypoints)
    KEYPOINT_NAMES = {
        0: 'nose', 1: 'left_eye', 2: 'right_eye',
        3: 'left_ear', 4: 'right_ear',
        5: 'left_shoulder', 6: 'right_shoulder',  # front
        7: 'left_elbow', 8: 'right_elbow',
        9: 'left_wrist', 10: 'right_wrist',       # front paws
        11: 'left_hip', 12: 'right_hip',          # back
        13: 'left_knee', 14: 'right_knee',
        15: 'left_ankle', 16: 'right_ankle',      # back paws
        17: 'tail_base', 18: 'tail_mid', 19: 'tail_tip',
        20: 'neck', 21: 'withers', 22: 'spine_mid', 23: 'rump'
    }

    # Keypoint groups for validation
    FRONT_LEG_KEYPOINTS = [5, 6, 7, 8, 9, 10]   # shoulders, elbows, paws
    BACK_LEG_KEYPOINTS = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
    HEAD_KEYPOINTS = [0, 1, 2, 3, 4]
    BODY_KEYPOINTS = [17, 18, 19, 20, 21, 22, 23]

    # Limb pairs for geometry validation
    LIMB_PAIRS = [
        (5, 7), (7, 9),    # left front leg
        (6, 8), (8, 10),   # right front leg
        (11, 13), (13, 15), # left back leg
        (12, 14), (14, 16), # right back leg
    ]

    def __init__(self, config: PoseValidatorConfig = None):
        """Initialize validator with configuration"""
        self.config = config or PoseValidatorConfig()

        # Per-dog temporal history: dog_id -> deque of (behavior, confidence)
        self._prediction_history: Dict[str, deque] = {}
        self._lock = threading.Lock()

        # Validation statistics
        self.stats = {
            'total_frames': 0,
            'rejected_blur': 0,
            'rejected_keypoints': 0,
            'rejected_geometry': 0,
            'rejected_bbox': 0,
            'rejected_quality': 0,
            'passed': 0
        }

        logger.info(f"PoseValidator initialized with config: blur_th={self.config.blur_threshold}, "
                   f"min_kp={self.config.min_visible_keypoints}, temporal_window={self.config.temporal_window}")

    def check_motion_blur(self, frame: np.ndarray, bbox: Tuple[float, float, float, float] = None) -> Tuple[bool, float]:
        """
        Detect motion blur using Laplacian variance.

        Args:
            frame: BGR image
            bbox: Optional (x1, y1, x2, y2) to check only the detection region

        Returns:
            (is_sharp, blur_score) - True if image is sharp enough
        """
        try:
            if bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                roi = frame[y1:y2, x1:x2]
            else:
                roi = frame

            if roi.size == 0:
                return False, 0.0

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

            is_sharp = laplacian_var > self.config.blur_threshold
            return is_sharp, float(laplacian_var)

        except Exception as e:
            logger.debug(f"Blur check error: {e}")
            return True, 100.0  # Assume sharp on error

    def check_keypoint_confidence(self, keypoints: np.ndarray) -> Tuple[bool, Dict]:
        """
        Check if enough keypoints have sufficient confidence.

        Args:
            keypoints: Array shape (N, 3) with x, y, confidence per keypoint

        Returns:
            (is_valid, details)
        """
        if keypoints.shape[1] < 3:
            return False, {'error': 'no confidence values'}

        confidences = keypoints[:, 2]
        valid_mask = confidences >= self.config.min_keypoint_confidence
        num_valid = np.sum(valid_mask)

        # Check specifically for leg keypoints
        front_leg_valid = sum(1 for i in self.FRONT_LEG_KEYPOINTS
                             if i < len(confidences) and confidences[i] >= self.config.min_keypoint_confidence)
        back_leg_valid = sum(1 for i in self.BACK_LEG_KEYPOINTS
                            if i < len(confidences) and confidences[i] >= self.config.min_keypoint_confidence)

        details = {
            'total_valid': int(num_valid),
            'total_keypoints': len(confidences),
            'front_leg_valid': front_leg_valid,
            'back_leg_valid': back_leg_valid,
            'mean_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)) if len(confidences) > 0 else 0,
        }

        # Need enough total keypoints AND at least some leg keypoints
        has_enough_keypoints = num_valid >= self.config.min_visible_keypoints
        has_enough_legs = (front_leg_valid >= self.config.min_leg_keypoints or
                         back_leg_valid >= self.config.min_leg_keypoints)

        is_valid = has_enough_keypoints and has_enough_legs
        return is_valid, details

    def check_skeleton_geometry(self, keypoints: np.ndarray, bbox: Tuple) -> Tuple[bool, Dict]:
        """
        Check if the skeleton geometry is physically plausible.

        Catches cases where:
        - Limbs are impossibly long/short
        - Keypoints are outside the bounding box
        - Skeleton is totally scrambled
        """
        x1, y1, x2, y2 = bbox
        bbox_width = max(x2 - x1, 1)
        bbox_height = max(y2 - y1, 1)
        bbox_diag = np.sqrt(bbox_width**2 + bbox_height**2)

        issues = []
        confidences = keypoints[:, 2] if keypoints.shape[1] >= 3 else np.ones(len(keypoints))

        # Check if keypoints are within/near bounding box
        margin = 0.25  # Allow 25% outside bbox
        expanded_x1 = x1 - bbox_width * margin
        expanded_y1 = y1 - bbox_height * margin
        expanded_x2 = x2 + bbox_width * margin
        expanded_y2 = y2 + bbox_height * margin

        keypoints_outside = 0
        for i, kp in enumerate(keypoints):
            if i < len(confidences) and confidences[i] >= self.config.min_keypoint_confidence:
                if not (expanded_x1 <= kp[0] <= expanded_x2 and
                       expanded_y1 <= kp[1] <= expanded_y2):
                    keypoints_outside += 1

        if keypoints_outside > self.config.max_keypoints_outside_bbox:
            issues.append(f"{keypoints_outside} keypoints outside bbox")

        # Check limb lengths
        max_limb_length = bbox_diag * self.config.max_limb_ratio
        min_limb_length = bbox_diag * self.config.min_limb_ratio

        impossible_limbs = 0
        for i, j in self.LIMB_PAIRS:
            if i < len(keypoints) and j < len(keypoints):
                if (i < len(confidences) and j < len(confidences) and
                    confidences[i] >= self.config.min_keypoint_confidence and
                    confidences[j] >= self.config.min_keypoint_confidence):
                    length = np.linalg.norm(keypoints[i][:2] - keypoints[j][:2])
                    if length > max_limb_length or length < min_limb_length:
                        impossible_limbs += 1

        if impossible_limbs > 2:
            issues.append(f"{impossible_limbs} impossible limb lengths")

        # Check if skeleton is inverted (shoulders below paws)
        if all(i < len(keypoints) for i in [5, 6, 9, 10]):
            shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
            paw_y = (keypoints[9][1] + keypoints[10][1]) / 2
            # In image coords, y increases downward
            if shoulder_y > paw_y + bbox_height * 0.35:
                issues.append("skeleton inverted")

        details = {
            'keypoints_outside_bbox': keypoints_outside,
            'impossible_limbs': impossible_limbs,
            'issues': issues,
            'bbox_diagonal': float(bbox_diag)
        }

        is_valid = len(issues) == 0
        return is_valid, details

    def check_bbox_validity(self, bbox: Tuple, frame_shape: Tuple) -> Tuple[bool, Dict]:
        """Check if bounding box is reasonable"""
        x1, y1, x2, y2 = bbox
        width = max(x2 - x1, 1)
        height = max(y2 - y1, 1)
        area = width * height
        aspect_ratio = height / width

        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_height * frame_width
        coverage = area / frame_area

        issues = []

        if area < self.config.min_bbox_area:
            issues.append(f"bbox too small ({area}px)")

        if aspect_ratio > self.config.max_aspect_ratio:
            issues.append(f"aspect too tall ({aspect_ratio:.2f})")
        elif aspect_ratio < self.config.min_aspect_ratio:
            issues.append(f"aspect too wide ({aspect_ratio:.2f})")

        if coverage < self.config.min_frame_coverage:
            issues.append(f"detection too small ({coverage:.1%})")
        elif coverage > self.config.max_frame_coverage:
            issues.append(f"detection covers entire frame ({coverage:.1%})")

        details = {
            'area': int(area),
            'aspect_ratio': float(aspect_ratio),
            'frame_coverage': float(coverage),
            'issues': issues
        }

        is_valid = len(issues) == 0
        return is_valid, details

    def check_detection_region_quality(self, frame: np.ndarray, bbox: Tuple) -> Tuple[bool, Dict]:
        """
        Check if the detection region contains actual meaningful content.
        Catches empty/blurry detection regions.
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return False, {'issues': ['empty ROI']}

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Check sharpness within the detection
        roi_blur = cv2.Laplacian(gray_roi, cv2.CV_64F).var()

        # Check edge density (is there actual structure?)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size

        issues = []

        if roi_blur < self.config.blur_threshold * 0.8:  # Slightly more lenient for ROI
            issues.append(f"detection region blurry ({roi_blur:.1f})")

        if edge_density < self.config.min_edge_density:
            issues.append(f"no detail in detection ({edge_density:.1%} edges)")

        # Quality score for logging
        quality_score = min(1.0, (roi_blur / 300) * (edge_density / 0.05))

        details = {
            'roi_blur': float(roi_blur),
            'edge_density': float(edge_density),
            'quality_score': float(quality_score),
            'issues': issues
        }

        is_valid = len(issues) == 0
        return is_valid, details

    def validate_pose(self, frame: np.ndarray, keypoints: np.ndarray,
                      bbox: Tuple[float, float, float, float],
                      check_blur: bool = True) -> ValidationResult:
        """
        Run all validation checks on a pose detection.

        Args:
            frame: BGR image
            keypoints: (N, 3) array with x, y, confidence per keypoint
            bbox: (x1, y1, x2, y2) detection bounding box
            check_blur: Whether to run blur detection (can be slower)

        Returns:
            ValidationResult with is_valid, confidence, reason, and details
        """
        self.stats['total_frames'] += 1
        all_details = {}
        reasons = []

        # 1. Check bounding box
        bbox_valid, bbox_details = self.check_bbox_validity(bbox, frame.shape)
        all_details['bbox'] = bbox_details
        if not bbox_valid:
            reasons.extend(bbox_details['issues'])
            self.stats['rejected_bbox'] += 1

        # 2. Check detection region quality
        roi_valid, roi_details = self.check_detection_region_quality(frame, bbox)
        all_details['roi_quality'] = roi_details
        if not roi_valid:
            reasons.extend(roi_details['issues'])
            self.stats['rejected_quality'] += 1

        # 3. Check motion blur (optional)
        if check_blur:
            is_sharp, blur_score = self.check_motion_blur(frame, bbox)
            all_details['blur'] = {'is_sharp': is_sharp, 'score': blur_score}
            if not is_sharp:
                reasons.append(f"motion blur ({blur_score:.1f})")
                self.stats['rejected_blur'] += 1

        # 4. Check keypoint confidence
        kp_valid, kp_details = self.check_keypoint_confidence(keypoints)
        all_details['keypoints'] = kp_details
        if not kp_valid:
            reasons.append(f"keypoints ({kp_details['total_valid']}/{kp_details['total_keypoints']} valid)")
            self.stats['rejected_keypoints'] += 1

        # 5. Check skeleton geometry
        geom_valid, geom_details = self.check_skeleton_geometry(keypoints, bbox)
        all_details['geometry'] = geom_details
        if not geom_valid:
            reasons.extend(geom_details['issues'])
            self.stats['rejected_geometry'] += 1

        # Calculate overall confidence from validation factors
        confidence_factors = [
            kp_details.get('mean_confidence', 0.5),
            1.0 if bbox_valid else 0.4,
            1.0 if geom_valid else 0.3,
            1.0 if roi_valid else 0.5,
        ]
        if check_blur:
            blur_factor = min(1.0, all_details['blur']['score'] / 150)
            confidence_factors.append(blur_factor)

        overall_confidence = np.prod(confidence_factors) ** (1/len(confidence_factors))

        is_valid = len(reasons) == 0
        reason = "; ".join(reasons) if reasons else "passed"

        if is_valid:
            self.stats['passed'] += 1

        return ValidationResult(
            is_valid=is_valid,
            confidence=float(overall_confidence),
            reason=reason,
            details=all_details
        )

    def temporal_vote(self, dog_id: str, behavior: str, confidence: float) -> Tuple[str, float, bool]:
        """
        Apply temporal smoothing to predictions for a specific dog.

        Requires multiple consecutive frames to agree before changing prediction.
        Prevents flickering between behaviors.

        Args:
            dog_id: Identifier for the dog (for per-dog tracking)
            behavior: Current frame's behavior prediction
            confidence: Confidence of current prediction

        Returns:
            (smoothed_behavior, smoothed_confidence, is_stable)
        """
        with self._lock:
            # Get or create history for this dog
            if dog_id not in self._prediction_history:
                self._prediction_history[dog_id] = deque(maxlen=self.config.temporal_window)

            history = self._prediction_history[dog_id]

            # Apply behavior-specific confidence adjustments
            adjusted_confidence = confidence
            if behavior == 'cross':
                adjusted_confidence = confidence - self.config.cross_confidence_penalty

            history.append((behavior, adjusted_confidence))

            # Not enough history yet
            if len(history) < self.config.temporal_window:
                return behavior, adjusted_confidence, False

            # Count votes
            votes: Dict[str, int] = {}
            weighted_conf: Dict[str, float] = {}

            for pred, conf in history:
                votes[pred] = votes.get(pred, 0) + 1
                weighted_conf[pred] = weighted_conf.get(pred, 0) + conf

            # Find majority
            majority_behavior = max(votes.keys(), key=lambda k: votes[k])
            majority_fraction = votes[majority_behavior] / len(history)

            # Calculate average confidence for majority behavior
            avg_confidence = weighted_conf[majority_behavior] / votes[majority_behavior]

            # Check stability threshold
            is_stable = majority_fraction >= self.config.temporal_threshold

            # Spin requires extra stability
            if majority_behavior == 'spin':
                is_stable = is_stable and votes[majority_behavior] >= self.config.spin_min_frames

            if is_stable:
                return majority_behavior, avg_confidence, True
            else:
                # Return majority but flag as unstable (reduce confidence)
                return majority_behavior, avg_confidence * 0.6, False

    def reset_temporal(self, dog_id: str = None):
        """Clear temporal history for a dog (or all dogs if dog_id is None)"""
        with self._lock:
            if dog_id:
                if dog_id in self._prediction_history:
                    self._prediction_history[dog_id].clear()
            else:
                self._prediction_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total = self.stats['total_frames']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'pass_rate': self.stats['passed'] / total,
            'blur_reject_rate': self.stats['rejected_blur'] / total,
            'keypoint_reject_rate': self.stats['rejected_keypoints'] / total,
            'geometry_reject_rate': self.stats['rejected_geometry'] / total,
        }

    def reset_stats(self):
        """Reset validation statistics"""
        for key in self.stats:
            self.stats[key] = 0


# Singleton instance
_validator_instance: Optional[PoseValidator] = None
_validator_lock = threading.Lock()


def get_pose_validator(config: PoseValidatorConfig = None) -> PoseValidator:
    """Get or create singleton PoseValidator instance"""
    global _validator_instance

    if _validator_instance is None:
        with _validator_lock:
            if _validator_instance is None:
                _validator_instance = PoseValidator(config)

    return _validator_instance


def reset_pose_validator():
    """Reset the singleton validator (for testing)"""
    global _validator_instance
    with _validator_lock:
        _validator_instance = None
