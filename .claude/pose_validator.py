"""
Pose Validation & Filtering Module for Dog Pose Classification
==============================================================

This module sits between YOLOv8 pose estimation and your classification model.
It filters out bad/unreliable pose estimations before they can cause 
misclassifications.

Filters implemented:
1. Keypoint confidence filtering - reject low-confidence pose detections
2. Motion blur detection - skip blurry frames
3. Skeleton geometry validation - reject physically impossible poses
4. Temporal voting - require consistency across frames
5. Bounding box validation - reject detections that are too small/wrong aspect ratio

Usage:
    from pose_validator import PoseValidator
    
    validator = PoseValidator()
    
    # For single frame:
    is_valid, reason = validator.validate_frame(frame, keypoints, confidences, bbox)
    
    # For temporal sequence:
    predictions = [...]  # your raw predictions per frame
    smoothed = validator.temporal_vote(predictions, window=5, threshold=0.6)
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of pose validation"""
    is_valid: bool
    confidence: float
    reason: str
    details: Dict


class PoseValidator:
    """
    Validates pose estimations before classification.
    Catches garbage poses that would cause misclassification.
    """
    
    # YOLOv8 pose keypoint indices for dogs/quadrupeds
    # Adjust these based on your actual model's keypoint definition
    KEYPOINT_NAMES = {
        0: 'nose',
        1: 'left_eye', 
        2: 'right_eye',
        3: 'left_ear',
        4: 'right_ear',
        5: 'left_shoulder',  # front left
        6: 'right_shoulder', # front right
        7: 'left_elbow',
        8: 'right_elbow',
        9: 'left_wrist',     # front left paw
        10: 'right_wrist',   # front right paw
        11: 'left_hip',      # back left
        12: 'right_hip',     # back right
        13: 'left_knee',
        14: 'right_knee',
        15: 'left_ankle',    # back left paw
        16: 'right_ankle',   # back right paw
    }
    
    # Keypoints that matter for "crossed legs" detection
    FRONT_LEG_KEYPOINTS = [5, 6, 7, 8, 9, 10]  # shoulders, elbows, wrists
    BACK_LEG_KEYPOINTS = [11, 12, 13, 14, 15, 16]  # hips, knees, ankles
    
    def __init__(
        self,
        min_keypoint_confidence: float = 0.4,
        min_visible_keypoints: int = 6,
        blur_threshold: float = 100.0,
        min_bbox_area: int = 5000,
        max_aspect_ratio: float = 3.0,
        min_aspect_ratio: float = 0.3,
        temporal_window: int = 5,
        temporal_threshold: float = 0.6,
    ):
        """
        Initialize validator with thresholds.
        
        Args:
            min_keypoint_confidence: Minimum confidence for a keypoint to be considered valid
            min_visible_keypoints: Minimum number of valid keypoints required
            blur_threshold: Laplacian variance threshold (lower = more blurry)
            min_bbox_area: Minimum bounding box area in pixels
            max_aspect_ratio: Maximum height/width ratio for valid detection
            min_aspect_ratio: Minimum height/width ratio for valid detection
            temporal_window: Number of frames for temporal smoothing
            temporal_threshold: Fraction of frames that must agree
        """
        self.min_keypoint_confidence = min_keypoint_confidence
        self.min_visible_keypoints = min_visible_keypoints
        self.blur_threshold = blur_threshold
        self.min_bbox_area = min_bbox_area
        self.max_aspect_ratio = max_aspect_ratio
        self.min_aspect_ratio = min_aspect_ratio
        self.temporal_window = temporal_window
        self.temporal_threshold = temporal_threshold
        
        # For temporal smoothing
        self.prediction_history = deque(maxlen=temporal_window)
        self.confidence_history = deque(maxlen=temporal_window)
        
    def check_motion_blur(self, frame: np.ndarray, bbox: Optional[Tuple] = None) -> Tuple[bool, float]:
        """
        Detect motion blur using Laplacian variance.
        
        Args:
            frame: BGR image
            bbox: Optional (x1, y1, x2, y2) to check only the detection region
            
        Returns:
            (is_sharp, blur_score) - True if image is sharp enough
        """
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            roi = frame[y1:y2, x1:x2]
        else:
            roi = frame
            
        if roi.size == 0:
            return False, 0.0
            
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        is_sharp = laplacian_var > self.blur_threshold
        return is_sharp, laplacian_var
    
    def check_keypoint_confidence(
        self, 
        keypoints: np.ndarray, 
        confidences: np.ndarray
    ) -> Tuple[bool, Dict]:
        """
        Check if enough keypoints have sufficient confidence.
        
        Args:
            keypoints: Array of (x, y) coordinates, shape (N, 2)
            confidences: Array of confidence scores, shape (N,)
            
        Returns:
            (is_valid, details) - True if enough keypoints are confident
        """
        valid_mask = confidences >= self.min_keypoint_confidence
        num_valid = np.sum(valid_mask)
        
        # Check specifically for leg keypoints
        front_leg_valid = sum(1 for i in self.FRONT_LEG_KEYPOINTS 
                             if i < len(confidences) and confidences[i] >= self.min_keypoint_confidence)
        back_leg_valid = sum(1 for i in self.BACK_LEG_KEYPOINTS 
                            if i < len(confidences) and confidences[i] >= self.min_keypoint_confidence)
        
        details = {
            'total_valid': int(num_valid),
            'total_keypoints': len(confidences),
            'front_leg_valid': front_leg_valid,
            'back_leg_valid': back_leg_valid,
            'mean_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'valid_keypoints': [int(i) for i in np.where(valid_mask)[0]]
        }
        
        # Need enough total keypoints AND at least some leg keypoints
        is_valid = (num_valid >= self.min_visible_keypoints and 
                   (front_leg_valid >= 2 or back_leg_valid >= 2))
        
        return is_valid, details
    
    def check_skeleton_geometry(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray,
        bbox: Tuple
    ) -> Tuple[bool, Dict]:
        """
        Check if the skeleton geometry is physically plausible.
        
        Catches cases where:
        - Limbs are impossibly long/short
        - Keypoints are outside the bounding box
        - Skeleton is totally scrambled
        
        Args:
            keypoints: Array of (x, y) coordinates
            confidences: Array of confidence scores
            bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            (is_valid, details)
        """
        x1, y1, x2, y2 = bbox
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        bbox_diag = np.sqrt(bbox_width**2 + bbox_height**2)
        
        issues = []
        
        # Check if keypoints are within/near bounding box
        margin = 0.2  # Allow 20% outside bbox
        expanded_x1 = x1 - bbox_width * margin
        expanded_y1 = y1 - bbox_height * margin
        expanded_x2 = x2 + bbox_width * margin
        expanded_y2 = y2 + bbox_height * margin
        
        keypoints_outside = 0
        for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
            if conf >= self.min_keypoint_confidence:
                if not (expanded_x1 <= kp[0] <= expanded_x2 and 
                       expanded_y1 <= kp[1] <= expanded_y2):
                    keypoints_outside += 1
                    
        if keypoints_outside > 2:
            issues.append(f"{keypoints_outside} keypoints outside bbox")
        
        # Check limb lengths (shouldn't be > 60% of bbox diagonal)
        max_limb_length = bbox_diag * 0.6
        min_limb_length = bbox_diag * 0.02
        
        # Define limb connections to check
        limb_pairs = [
            (5, 7), (7, 9),   # left front leg
            (6, 8), (8, 10),  # right front leg
            (11, 13), (13, 15),  # left back leg
            (12, 14), (14, 16),  # right back leg
        ]
        
        impossible_limbs = 0
        for i, j in limb_pairs:
            if i < len(keypoints) and j < len(keypoints):
                if confidences[i] >= self.min_keypoint_confidence and \
                   confidences[j] >= self.min_keypoint_confidence:
                    length = np.linalg.norm(keypoints[i] - keypoints[j])
                    if length > max_limb_length or length < min_limb_length:
                        impossible_limbs += 1
                        
        if impossible_limbs > 2:
            issues.append(f"{impossible_limbs} impossible limb lengths")
            
        # Check if skeleton is "scrambled" (e.g., left/right crossed impossibly)
        # Front shoulders should generally be above front paws
        if all(i < len(keypoints) for i in [5, 6, 9, 10]):
            shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
            paw_y = (keypoints[9][1] + keypoints[10][1]) / 2
            # In image coordinates, y increases downward
            if shoulder_y > paw_y + bbox_height * 0.3:  # shoulders way below paws
                issues.append("skeleton inverted (shoulders below paws)")
        
        details = {
            'keypoints_outside_bbox': keypoints_outside,
            'impossible_limbs': impossible_limbs,
            'issues': issues,
            'bbox_diagonal': float(bbox_diag)
        }
        
        is_valid = len(issues) == 0
        return is_valid, details
    
    def check_bbox_validity(self, bbox: Tuple, frame_shape: Tuple) -> Tuple[bool, Dict]:
        """
        Check if bounding box is reasonable.
        
        Args:
            bbox: (x1, y1, x2, y2)
            frame_shape: (height, width, channels)
            
        Returns:
            (is_valid, details)
        """
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = height / width if width > 0 else 0
        
        frame_height, frame_width = frame_shape[:2]
        frame_area = frame_height * frame_width
        coverage = area / frame_area
        
        issues = []
        
        if area < self.min_bbox_area:
            issues.append(f"bbox too small ({area} < {self.min_bbox_area})")
            
        if aspect_ratio > self.max_aspect_ratio:
            issues.append(f"aspect ratio too tall ({aspect_ratio:.2f})")
        elif aspect_ratio < self.min_aspect_ratio:
            issues.append(f"aspect ratio too wide ({aspect_ratio:.2f})")
            
        if coverage < 0.01:
            issues.append(f"detection too small relative to frame ({coverage:.1%})")
        elif coverage > 0.95:
            issues.append(f"detection covers entire frame ({coverage:.1%})")
            
        details = {
            'area': int(area),
            'aspect_ratio': float(aspect_ratio),
            'frame_coverage': float(coverage),
            'issues': issues
        }
        
        is_valid = len(issues) == 0
        return is_valid, details
    
    def check_detection_region_quality(
        self,
        frame: np.ndarray,
        bbox: Tuple,
        min_edge_density: float = 0.03,
        min_roi_blur: float = 100.0
    ) -> Tuple[bool, Dict]:
        """
        Check if the detection region contains actual meaningful content.
        
        This catches cases where:
        - The detection is on a blurry/empty region
        - There's no actual object detail in the bbox
        - The "dog" detection is actually background
        
        Args:
            frame: BGR image
            bbox: (x1, y1, x2, y2) detection bounding box
            min_edge_density: Minimum fraction of edge pixels required
            min_roi_blur: Minimum Laplacian variance in the ROI
            
        Returns:
            (is_valid, details)
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        
        # Clamp to frame boundaries
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
        
        # Check gradient magnitude (another structure indicator)
        gradient_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.mean(np.sqrt(gradient_x**2 + gradient_y**2))
        
        issues = []
        
        if roi_blur < min_roi_blur:
            issues.append(f"detection region too blurry ({roi_blur:.1f} < {min_roi_blur})")
            
        if edge_density < min_edge_density:
            issues.append(f"no detail in detection ({edge_density:.1%} edges < {min_edge_density:.1%})")
            
        # Combined quality score
        quality_score = (roi_blur / 500) * (edge_density / 0.10)  # Normalized
        
        details = {
            'roi_blur': float(roi_blur),
            'edge_density': float(edge_density),
            'gradient_mag': float(gradient_mag),
            'quality_score': float(min(1.0, quality_score)),
            'issues': issues
        }
        
        is_valid = len(issues) == 0
        return is_valid, details
    
    def validate_frame(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        confidences: np.ndarray,
        bbox: Tuple,
        check_blur: bool = True
    ) -> ValidationResult:
        """
        Run all validation checks on a single frame.
        
        Args:
            frame: BGR image
            keypoints: (N, 2) array of keypoint coordinates
            confidences: (N,) array of confidence scores
            bbox: (x1, y1, x2, y2) detection bounding box
            check_blur: Whether to run blur detection (slower)
            
        Returns:
            ValidationResult with is_valid, confidence, reason, and details
        """
        all_details = {}
        reasons = []
        
        # 1. Check bounding box
        bbox_valid, bbox_details = self.check_bbox_validity(bbox, frame.shape)
        all_details['bbox'] = bbox_details
        if not bbox_valid:
            reasons.extend(bbox_details['issues'])
        
        # 2. Check detection region quality (NEW - catches empty/blurry detections)
        roi_valid, roi_details = self.check_detection_region_quality(frame, bbox)
        all_details['roi_quality'] = roi_details
        if not roi_valid:
            reasons.extend(roi_details['issues'])
        
        # 3. Check motion blur (legacy - kept for compatibility)
        if check_blur:
            is_sharp, blur_score = self.check_motion_blur(frame, bbox)
            all_details['blur'] = {'is_sharp': is_sharp, 'score': float(blur_score)}
            if not is_sharp:
                reasons.append(f"motion blur detected (score={blur_score:.1f})")
        
        # 3. Check keypoint confidence
        kp_valid, kp_details = self.check_keypoint_confidence(keypoints, confidences)
        all_details['keypoints'] = kp_details
        if not kp_valid:
            reasons.append(f"insufficient keypoint confidence ({kp_details['total_valid']}/{kp_details['total_keypoints']} valid)")
        
        # 4. Check skeleton geometry
        geom_valid, geom_details = self.check_skeleton_geometry(keypoints, confidences, bbox)
        all_details['geometry'] = geom_details
        if not geom_valid:
            reasons.extend(geom_details['issues'])
        
        # Calculate overall confidence
        confidence_factors = [
            kp_details['mean_confidence'],
            1.0 if bbox_valid else 0.5,
            1.0 if geom_valid else 0.3,
        ]
        if check_blur:
            confidence_factors.append(min(1.0, all_details['blur']['score'] / 200))
            
        overall_confidence = np.prod(confidence_factors) ** (1/len(confidence_factors))
        
        is_valid = len(reasons) == 0
        reason = "; ".join(reasons) if reasons else "all checks passed"
        
        return ValidationResult(
            is_valid=is_valid,
            confidence=float(overall_confidence),
            reason=reason,
            details=all_details
        )
    
    def temporal_vote(
        self,
        current_prediction: str,
        current_confidence: float,
        reset: bool = False
    ) -> Tuple[str, float, bool]:
        """
        Apply temporal smoothing to predictions.
        
        Requires multiple consecutive frames to agree before
        changing the prediction. Helps prevent flickering.
        
        Args:
            current_prediction: Current frame's prediction (e.g., "CROSS", "NO_CROSS")
            current_confidence: Confidence of current prediction
            reset: If True, clear history and start fresh
            
        Returns:
            (smoothed_prediction, smoothed_confidence, is_stable)
        """
        if reset:
            self.prediction_history.clear()
            self.confidence_history.clear()
            
        self.prediction_history.append(current_prediction)
        self.confidence_history.append(current_confidence)
        
        if len(self.prediction_history) < self.temporal_window:
            # Not enough history yet, return current but mark unstable
            return current_prediction, current_confidence, False
        
        # Count votes
        votes = {}
        weighted_votes = {}
        for pred, conf in zip(self.prediction_history, self.confidence_history):
            votes[pred] = votes.get(pred, 0) + 1
            weighted_votes[pred] = weighted_votes.get(pred, 0) + conf
            
        # Find majority
        majority_pred = max(votes.keys(), key=lambda k: votes[k])
        majority_fraction = votes[majority_pred] / len(self.prediction_history)
        
        # Calculate confidence
        avg_confidence = weighted_votes[majority_pred] / votes[majority_pred]
        
        is_stable = majority_fraction >= self.temporal_threshold
        
        if is_stable:
            return majority_pred, avg_confidence, True
        else:
            # Return most common but flag as unstable
            return majority_pred, avg_confidence * 0.5, False
    
    def reset_temporal(self):
        """Clear temporal history (e.g., when switching to a new video/dog)"""
        self.prediction_history.clear()
        self.confidence_history.clear()


class HarnessTagFilter:
    """
    Specifically detects and masks harness name tags that confuse pose estimation.
    """
    
    def __init__(self):
        # Will be trained/tuned based on your specific harness tags
        pass
    
    def detect_tag_region(
        self, 
        frame: np.ndarray, 
        bbox: Tuple
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect the harness tag region within a dog detection.
        
        Uses color/pattern matching to find the tag area.
        
        Args:
            frame: BGR image
            bbox: Dog bounding box (x1, y1, x2, y2)
            
        Returns:
            Tag region (x1, y1, x2, y2) or None if not found
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Tags are usually high-contrast rectangular regions
        # Look for bright rectangular patches in the chest area
        
        # Focus on the middle-upper portion of bbox (chest area)
        chest_y_start = int((y2 - y1) * 0.3)
        chest_y_end = int((y2 - y1) * 0.7)
        chest_x_start = int((x2 - x1) * 0.2)
        chest_x_end = int((x2 - x1) * 0.8)
        
        chest_roi = gray[chest_y_start:chest_y_end, chest_x_start:chest_x_end]
        
        if chest_roi.size == 0:
            return None
        
        # Apply edge detection
        edges = cv2.Canny(chest_roi, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangular contours of appropriate size
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500 or area > 10000:
                continue
                
            # Approximate contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            
            # If it's roughly rectangular (4 corners)
            if len(approx) >= 4 and len(approx) <= 6:
                rx, ry, rw, rh = cv2.boundingRect(contour)
                # Convert back to full image coordinates
                tag_x1 = x1 + chest_x_start + rx
                tag_y1 = y1 + chest_y_start + ry
                tag_x2 = tag_x1 + rw
                tag_y2 = tag_y1 + rh
                
                return (tag_x1, tag_y1, tag_x2, tag_y2)
        
        return None
    
    def mask_tag_region(
        self,
        frame: np.ndarray,
        tag_bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Mask the tag region to prevent it from affecting pose estimation.
        
        Args:
            frame: BGR image
            tag_bbox: Tag region (x1, y1, x2, y2)
            
        Returns:
            Frame with tag region masked (blurred or filled)
        """
        x1, y1, x2, y2 = tag_bbox
        masked = frame.copy()
        
        # Option 1: Blur the region
        roi = masked[y1:y2, x1:x2]
        if roi.size > 0:
            blurred = cv2.GaussianBlur(roi, (21, 21), 0)
            masked[y1:y2, x1:x2] = blurred
        
        return masked
    
    def check_keypoints_on_tag(
        self,
        keypoints: np.ndarray,
        confidences: np.ndarray,
        tag_bbox: Tuple[int, int, int, int],
        leg_keypoint_indices: List[int]
    ) -> Tuple[bool, List[int]]:
        """
        Check if leg keypoints are incorrectly placed on the tag.
        
        Args:
            keypoints: (N, 2) array of keypoint coordinates
            confidences: (N,) confidence scores
            tag_bbox: Tag region
            leg_keypoint_indices: Indices of leg keypoints
            
        Returns:
            (has_problem, affected_keypoints)
        """
        x1, y1, x2, y2 = tag_bbox
        
        affected = []
        for idx in leg_keypoint_indices:
            if idx < len(keypoints) and confidences[idx] > 0.3:
                kp = keypoints[idx]
                if x1 <= kp[0] <= x2 and y1 <= kp[1] <= y2:
                    affected.append(idx)
        
        return len(affected) > 0, affected


def process_video_with_validation(
    video_path: str,
    pose_model,
    classifier,
    output_path: Optional[str] = None,
    show_debug: bool = True
) -> List[Dict]:
    """
    Process a video with validation filtering.
    
    Args:
        video_path: Path to input video
        pose_model: YOLOv8 pose model
        classifier: Your cross/no-cross classifier
        output_path: Optional path to save annotated video
        show_debug: Whether to draw debug info on frames
        
    Returns:
        List of frame results with validation info
    """
    validator = PoseValidator()
    tag_filter = HarnessTagFilter()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    results = []
    frame_idx = 0
    
    validator.reset_temporal()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run pose estimation
        pose_results = pose_model(frame)
        
        frame_result = {
            'frame_idx': frame_idx,
            'valid': False,
            'prediction': None,
            'confidence': 0,
            'validation': None
        }
        
        if len(pose_results) > 0 and pose_results[0].keypoints is not None:
            # Get keypoints and confidence
            kp_data = pose_results[0].keypoints
            keypoints = kp_data.xy[0].cpu().numpy()  # (N, 2)
            confidences = kp_data.conf[0].cpu().numpy()  # (N,)
            
            # Get bounding box
            if pose_results[0].boxes is not None and len(pose_results[0].boxes) > 0:
                bbox = pose_results[0].boxes.xyxy[0].cpu().numpy()
            else:
                bbox = (0, 0, width, height)
            
            # Validate the pose
            validation = validator.validate_frame(
                frame, keypoints, confidences, bbox
            )
            
            frame_result['validation'] = validation
            
            if validation.is_valid:
                # Check for tag interference
                tag_region = tag_filter.detect_tag_region(frame, bbox)
                if tag_region:
                    has_tag_problem, affected = tag_filter.check_keypoints_on_tag(
                        keypoints, confidences, tag_region,
                        validator.FRONT_LEG_KEYPOINTS
                    )
                    if has_tag_problem:
                        frame_result['validation'].reason = f"keypoints on tag: {affected}"
                        frame_result['validation'].is_valid = False
                
                if frame_result['validation'].is_valid:
                    # Run classifier
                    raw_prediction = classifier.predict(keypoints, confidences)
                    raw_confidence = classifier.confidence
                    
                    # Apply temporal smoothing
                    smoothed_pred, smoothed_conf, is_stable = validator.temporal_vote(
                        raw_prediction, raw_confidence
                    )
                    
                    frame_result['valid'] = True
                    frame_result['prediction'] = smoothed_pred
                    frame_result['confidence'] = smoothed_conf
                    frame_result['is_stable'] = is_stable
                    frame_result['raw_prediction'] = raw_prediction
        
        results.append(frame_result)
        
        # Draw debug visualization
        if show_debug and writer:
            debug_frame = draw_debug_info(frame, frame_result)
            writer.write(debug_frame)
        elif writer:
            writer.write(frame)
            
        frame_idx += 1
    
    cap.release()
    if writer:
        writer.release()
        
    return results


def draw_debug_info(frame: np.ndarray, result: Dict) -> np.ndarray:
    """Draw validation debug info on frame"""
    debug = frame.copy()
    
    y_offset = 30
    
    if result['valid']:
        color = (0, 255, 0)  # Green
        text = f"VALID: {result['prediction']} ({result['confidence']:.1%})"
        if not result.get('is_stable', True):
            color = (0, 255, 255)  # Yellow
            text += " [UNSTABLE]"
    else:
        color = (0, 0, 255)  # Red
        reason = result['validation'].reason if result['validation'] else "no detection"
        text = f"REJECTED: {reason}"
    
    cv2.putText(debug, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, color, 2)
    
    return debug


# Example usage and testing
if __name__ == "__main__":
    # Test with sample data
    print("PoseValidator module loaded successfully!")
    print("\nExample usage:")
    print("""
    from pose_validator import PoseValidator, HarnessTagFilter
    
    validator = PoseValidator(
        min_keypoint_confidence=0.4,
        blur_threshold=100,
        temporal_window=5
    )
    
    # For each frame from your video:
    result = validator.validate_frame(frame, keypoints, confidences, bbox)
    
    if result.is_valid:
        # Safe to run classifier
        prediction = your_classifier.predict(keypoints)
        
        # Apply temporal smoothing
        smooth_pred, smooth_conf, stable = validator.temporal_vote(
            prediction, confidence
        )
        
        if stable:
            # This is a reliable prediction
            print(f"Prediction: {smooth_pred}")
    else:
        print(f"Frame rejected: {result.reason}")
    """)
