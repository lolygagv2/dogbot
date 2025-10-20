#!/usr/bin/env python3
"""
DogBot Behavior Detection System
Integrates YOLOv8 with Hailo-8L acceleration for real-time dog behavior recognition
"""

import cv2
import numpy as np
import onnxruntime as ort
import time
import json
from datetime import datetime
from hailo_platform import HailoRT, VDevice
import threading
import queue

class DogBehaviorDetector:
    def __init__(self, model_path="/home/morgan/dogbot/ai/models/best_dogbot.hef"):
        self.model_path = model_path
        self.device = None
        self.network_group = None
        self.input_streams = None
        self.output_streams = None
        
        # Behavior classes (matching your training)
        self.classes = {
            0: "elsa_sitting", 1: "elsa_lying", 2: "elsa_standing", 
            3: "elsa_spinning", 4: "elsa_playing",
            5: "bezik_sitting", 6: "bezik_lying", 7: "bezik_standing",
            8: "bezik_spinning", 9: "bezik_playing"
        }
        
        # Detection history for temporal analysis
        self.detection_history = []
        self.behavior_duration = {}
        self.last_treat_time = 0
        self.treat_cooldown = 30  # seconds
        
        self.initialize_hailo()
    
    def initialize_hailo(self):
        """Initialize Hailo device and load model"""
        try:
            print("🧠 Initializing Hailo AI accelerator...")
            self.device = VDevice()
            
            # Load and configure network
            hef = HailoRT.create_hef_file(self.model_path)
            configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
            self.network_group = self.device.configure(hef, configure_params)[0]
            
            # Get input/output stream info
            self.input_streams = self.network_group.get_input_streams()
            self.output_streams = self.network_group.get_output_streams()
            
            print("✅ Hailo initialization complete")
            return True
            
        except Exception as e:
            print(f"❌ Hailo initialization failed: {e}")
            return False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for YOLO input"""
        # Resize to model input size (640x640 for YOLOv8)
        input_size = 640
        frame_resized = cv2.resize(frame, (input_size, input_size))
        
        # Normalize to 0-1 range
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        
        # Convert BGR to RGB and add batch dimension
        frame_rgb = cv2.cvtColor(frame_normalized, cv2.COLOR_BGR2RGB)
        frame_batch = np.expand_dims(frame_rgb.transpose(2, 0, 1), axis=0)
        
        return frame_batch
    
    def postprocess_detections(self, outputs, conf_threshold=0.5):
        """Process YOLO outputs to extract detections"""
        detections = []
        
        # Process YOLO output format [batch, anchors, 5 + num_classes]
        for output in outputs:
            for detection in output[0]:  # Remove batch dimension
                confidence = detection[4]
                
                if confidence > conf_threshold:
                    # Extract bounding box (center_x, center_y, width, height)
                    cx, cy, w, h = detection[0:4]
                    
                    # Convert to corner coordinates
                    x1 = int((cx - w/2) * 640)
                    y1 = int((cy - h/2) * 640)
                    x2 = int((cx + w/2) * 640)
                    y2 = int((cy + h/2) * 640)
                    
                    # Get class probabilities
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    class_confidence = class_scores[class_id]
                    
                    total_confidence = confidence * class_confidence
                    
                    if total_confidence > conf_threshold:
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': total_confidence,
                            'class_id': class_id,
                            'class_name': self.classes.get(class_id, 'unknown'),
                            'timestamp': time.time()
                        })
        
        return detections
    
    def analyze_behavior_duration(self, detections):
        """Analyze behavior patterns over time"""
        current_time = time.time()
        
        # Update detection history
        self.detection_history.append({
            'timestamp': current_time,
            'detections': detections
        })
        
        # Keep only last 30 seconds of history
        cutoff_time = current_time - 30
        self.detection_history = [h for h in self.detection_history if h['timestamp'] > cutoff_time]
        
        # Analyze sustained behaviors
        behavior_analysis = {}
        
        for detection in detections:
            behavior = detection['class_name']
            
            # Count recent occurrences of this behavior
            recent_count = 0
            for hist in self.detection_history[-10:]:  # Last 10 frames
                for hist_det in hist['detections']:
                    if hist_det['class_name'] == behavior:
                        recent_count += 1
            
            # Calculate behavior stability (consistency over time)
            stability = recent_count / 10.0  # 0-1 score
            
            behavior_analysis[behavior] = {
                'stability': stability,
                'duration': self.calculate_behavior_duration(behavior),
                'confidence': detection['confidence'],
                'should_reward': self.should_reward_behavior(behavior, stability)
            }
        
        return behavior_analysis
    
    def calculate_behavior_duration(self, behavior):
        """Calculate how long a behavior has been sustained"""
        if not self.detection_history:
            return 0
        
        current_time = time.time()
        duration = 0
        
        # Look backwards from most recent
        for hist in reversed(self.detection_history):
            found_behavior = False
            for detection in hist['detections']:
                if detection['class_name'] == behavior:
                    found_behavior = True
                    break
            
            if found_behavior:
                duration = current_time - hist['timestamp']
            else:
                break
        
        return duration
    
    def should_reward_behavior(self, behavior, stability):
        """Determine if behavior warrants a treat reward"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_treat_time < self.treat_cooldown:
            return False
        
        # Reward criteria
        reward_behaviors = ['sitting', 'lying']
        
        # Check if it's a rewardable behavior
        is_rewardable = any(reward in behavior.lower() for reward in reward_behaviors)
        
        # Must be stable for at least 3 seconds with high confidence
        duration = self.calculate_behavior_duration(behavior)
        
        return (is_rewardable and 
                stability > 0.7 and 
                duration > 3.0)
    
    def detect_frame(self, frame):
        """Process single frame through AI pipeline"""
        if not self.network_group:
            return []
        
        try:
            # Preprocess
            input_data = self.preprocess_frame(frame)
            
            # Run inference on Hailo
            with self.network_group.activate():
                # Send data to input streams
                for input_stream, data in zip(self.input_streams, [input_data]):
                    input_stream.send(data)
                
                # Get results from output streams  
                outputs = []
                for output_stream in self.output_streams:
                    outputs.append(output_stream.recv())
            
            # Postprocess
            detections = self.postprocess_detections(outputs)
            
            # Analyze behavior patterns
            behavior_analysis = self.analyze_behavior_duration(detections)
            
            return detections, behavior_analysis
            
        except Exception as e:
            print(f"❌ Detection error: {e}")
            return [], {}

# Usage example
if __name__ == "__main__":
    detector = DogBehaviorDetector()
    
    # Test with camera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        detections, analysis = detector.detect_frame(frame)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['class_name']}: {det['confidence']:.2f}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imshow('DogBot Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()