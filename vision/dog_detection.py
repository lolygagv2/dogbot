#!/usr/bin/env python3
"""
Dog Detection and Pose Recognition Module for TreatBot
Uses MediaPipe and custom models for behavior recognition
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)

class DogDetector:
    """Detect dogs and recognize poses/behaviors"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        Initialize dog detector
        
        Args:
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        # MediaPipe pose detection (can work for quadrupeds with adaptation)
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Object detection for dogs
        self.mp_objectron = mp.solutions.objectron
        
        # Behavior states
        self.behaviors = {
            "sit": False,
            "lie_down": False,
            "stand": False,
            "play_bow": False,
            "spin": False,
            "jump": False
        }
        
        # Tracking
        self.last_detection_time = 0
        self.detection_history = []
        self.pose_history = []
        
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame for dog detection and pose
        
        Args:
            frame: Input image frame (RGB)
            
        Returns:
            Detection results dictionary
        """
        results = {
            "dog_detected": False,
            "confidence": 0.0,
            "bbox": None,
            "pose": None,
            "behavior": None,
            "keypoints": []
        }
        
        # Convert to RGB if needed
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        
        # Detect pose (works for general poses, can be adapted)
        pose_results = self.pose.process(frame)
        
        if pose_results.pose_landmarks:
            # Extract keypoints
            keypoints = []
            for landmark in pose_results.pose_landmarks.landmark:
                keypoints.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                    "visibility": landmark.visibility
                })
            
            results["keypoints"] = keypoints
            
            # Analyze pose for dog behaviors
            behavior = self._analyze_pose(keypoints, frame.shape)
            if behavior:
                results["behavior"] = behavior
                results["dog_detected"] = True
                results["confidence"] = self._calculate_confidence(keypoints)
                
                # Calculate bounding box from keypoints
                results["bbox"] = self._keypoints_to_bbox(keypoints, frame.shape)
        
        # Update tracking
        self.last_detection_time = time.time()
        self.detection_history.append(results["dog_detected"])
        if len(self.detection_history) > 30:  # Keep last 30 frames
            self.detection_history.pop(0)
        
        return results
    
    def _analyze_pose(self, keypoints: List[Dict], image_shape: Tuple) -> Optional[str]:
        """
        Analyze keypoints to determine dog behavior
        
        Args:
            keypoints: List of keypoint dictionaries
            image_shape: Shape of the input image
            
        Returns:
            Detected behavior or None
        """
        if not keypoints or len(keypoints) < 10:
            return None
        
        # Get image dimensions
        height, width = image_shape[:2]
        
        # Simple heuristics for dog poses (would need training for accuracy)
        # These are placeholder algorithms - real implementation would use trained model
        
        # Calculate average y-position of upper vs lower body
        upper_points = keypoints[:11]  # Upper body landmarks
        lower_points = keypoints[23:]  # Lower body landmarks
        
        avg_upper_y = np.mean([p["y"] for p in upper_points if p["visibility"] > 0.5])
        avg_lower_y = np.mean([p["y"] for p in lower_points if p["visibility"] > 0.5])
        
        # Detect sitting (hindquarters lower than shoulders)
        if avg_lower_y > avg_upper_y + 0.1:
            return "sit"
        
        # Detect lying down (all points at similar height)
        all_y = [p["y"] for p in keypoints if p["visibility"] > 0.5]
        if all_y and np.std(all_y) < 0.05:
            return "lie_down"
        
        # Detect play bow (front low, rear high)
        front_y = np.mean([p["y"] for p in keypoints[:5] if p["visibility"] > 0.5])
        rear_y = np.mean([p["y"] for p in keypoints[-5:] if p["visibility"] > 0.5])
        if front_y > rear_y + 0.15:
            return "play_bow"
        
        # Default standing
        return "stand"
    
    def _calculate_confidence(self, keypoints: List[Dict]) -> float:
        """Calculate overall confidence from keypoint visibilities"""
        if not keypoints:
            return 0.0
        
        visibilities = [p["visibility"] for p in keypoints]
        return np.mean(visibilities)
    
    def _keypoints_to_bbox(self, keypoints: List[Dict], image_shape: Tuple) -> Tuple[int, int, int, int]:
        """
        Convert keypoints to bounding box
        
        Returns:
            (x, y, width, height) in pixels
        """
        height, width = image_shape[:2]
        
        visible_points = [p for p in keypoints if p["visibility"] > 0.3]
        if not visible_points:
            return None
        
        x_coords = [int(p["x"] * width) for p in visible_points]
        y_coords = [int(p["y"] * height) for p in visible_points]
        
        x_min = max(0, min(x_coords) - 20)
        y_min = max(0, min(y_coords) - 20)
        x_max = min(width, max(x_coords) + 20)
        y_max = min(height, max(y_coords) + 20)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def draw_detection(self, frame: np.ndarray, results: Dict) -> np.ndarray:
        """
        Draw detection results on frame
        
        Args:
            frame: Input frame
            results: Detection results
            
        Returns:
            Frame with annotations
        """
        annotated = frame.copy()
        
        if results["dog_detected"]:
            # Draw bounding box
            if results["bbox"]:
                x, y, w, h = results["bbox"]
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw behavior label
            if results["behavior"]:
                label = f"{results['behavior']} ({results['confidence']:.2f})"
                cv2.putText(annotated, label, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw keypoints
            if results["keypoints"]:
                height, width = frame.shape[:2]
                for point in results["keypoints"]:
                    if point["visibility"] > 0.5:
                        x = int(point["x"] * width)
                        y = int(point["y"] * height)
                        cv2.circle(annotated, (x, y), 5, (255, 0, 0), -1)
        
        return annotated
    
    def is_dog_present(self) -> bool:
        """Check if dog is currently detected based on recent history"""
        if not self.detection_history:
            return False
        
        # Require detection in at least 50% of recent frames
        recent_detections = self.detection_history[-10:]
        return sum(recent_detections) >= len(recent_detections) * 0.5
    
    def get_behavior_duration(self, behavior: str) -> float:
        """Get duration of current behavior in seconds"""
        if not self.pose_history:
            return 0.0
        
        # Count consecutive frames with this behavior
        count = 0
        for pose in reversed(self.pose_history):
            if pose == behavior:
                count += 1
            else:
                break
        
        # Assume 30fps
        return count / 30.0
    
    def should_reward(self, target_behavior: str, min_duration: float = 2.0) -> bool:
        """
        Determine if dog should be rewarded for behavior
        
        Args:
            target_behavior: Desired behavior (sit, lie_down, etc)
            min_duration: Minimum duration in seconds
            
        Returns:
            True if reward criteria met
        """
        if not self.is_dog_present():
            return False
        
        duration = self.get_behavior_duration(target_behavior)
        return duration >= min_duration

# Bark detection using audio
class BarkDetector:
    """Simple bark detection using audio frequency analysis"""
    
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.7):
        """
        Initialize bark detector
        
        Args:
            sample_rate: Audio sample rate
            threshold: Detection threshold
        """
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.bark_detected = False
        self.last_bark_time = 0
        
    def process_audio(self, audio_data: np.ndarray) -> bool:
        """
        Process audio buffer for bark detection
        
        Args:
            audio_data: Audio samples
            
        Returns:
            True if bark detected
        """
        # Simple energy-based detection (placeholder)
        # Real implementation would use trained model
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_data**2))
        
        # Frequency analysis for bark characteristics
        # Barks typically have energy in 500-2000 Hz range
        fft = np.fft.rfft(audio_data)
        freqs = np.fft.rfftfreq(len(audio_data), 1/self.sample_rate)
        
        # Find energy in bark frequency range
        bark_freq_mask = (freqs >= 500) & (freqs <= 2000)
        bark_energy = np.sum(np.abs(fft[bark_freq_mask]))
        total_energy = np.sum(np.abs(fft))
        
        bark_ratio = bark_energy / (total_energy + 1e-10)
        
        # Detect if bark-like sound
        if rms > 0.1 and bark_ratio > self.threshold:
            self.bark_detected = True
            self.last_bark_time = time.time()
            return True
        
        return False
    
    def get_time_since_bark(self) -> float:
        """Get time since last bark in seconds"""
        if self.last_bark_time == 0:
            return float('inf')
        return time.time() - self.last_bark_time

# Test function
def test_detection():
    """Test detection with camera"""
    logging.basicConfig(level=logging.INFO)
    
    detector = DogDetector()
    
    # Test with webcam or image
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = detector.process_frame(rgb_frame)
        
        # Draw results
        annotated = detector.draw_detection(frame, results)
        
        # Display
        cv2.imshow("Dog Detection", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_detection()
