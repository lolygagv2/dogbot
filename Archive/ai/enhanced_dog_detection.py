import cv2
import numpy as np
import onnxruntime as ort
import time
from ai.aruco_detector import ArUcoDetector

class EnhancedDogBehaviorDetector:
    def __init__(self, model_path="/home/morgan/dogbot/ai/models/best_dogbot_optimized.hef"):
        # Initialize YOLO behavior detection (or HEF if available)
        self.session = ort.InferenceSession(model_path.replace('.hef', '.onnx'))
        self.input_name = self.session.get_inputs()[0].name
        
        # Initialize ArUco detection
        self.aruco_detector = ArUcoDetector()
        
        # Behavior classes
        self.classes = {
            0: "elsa_sitting", 1: "elsa_lying", 2: "elsa_standing", 
            3: "elsa_spinning", 4: "elsa_playing",
            5: "bezik_sitting", 6: "bezik_lying", 7: "bezik_standing",
            8: "bezik_spinning", 9: "bezik_playing"
        }
        
        # Tracking
        self.detection_history = []
        self.last_treat_time = {"elsa": 0, "bezik": 0}
        self.treat_cooldown = 30
    
    def detect_frame(self, frame):
        """Combined ArUco + behavior detection"""
        # Step 1: Detect ArUco markers for dog identification
        aruco_detections, corners, ids = self.aruco_detector.detect_markers(frame)
        
        # Step 2: Detect behaviors using YOLO
        behavior_detections = self.detect_behaviors(frame)
        
        # Step 3: Match behaviors to specific dogs using ArUco proximity
        matched_detections = self.match_behaviors_to_dogs(
            behavior_detections, aruco_detections
        )
        
        return matched_detections, aruco_detections
    
    def detect_behaviors(self, frame):
        """Run YOLO behavior detection"""
        # Preprocess for YOLO
        input_image = cv2.resize(frame, (640, 640))
        input_image = input_image.astype(np.float32) / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_image = np.expand_dims(input_image, axis=0)
        
        # Run inference
        outputs = self.session.run([self.session.get_outputs()[0].name], 
                                 {self.input_name: input_image})
        
        # Postprocess detections
        detections = []
        output = outputs[0][0]
        
        for detection in output.T:
            confidence = detection[4]
            if confidence > 0.5:
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                total_confidence = confidence * class_confidence
                if total_confidence > 0.5:
                    x_center, y_center, width, height = detection[0:4]
                    x1 = int((x_center - width/2) * 640)
                    y1 = int((y_center - height/2) * 640)
                    x2 = int((x_center + width/2) * 640)
                    y2 = int((y_center + height/2) * 640)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'center': [int(x_center * 640), int(y_center * 640)],
                        'confidence': total_confidence,
                        'class_id': class_id,
                        'class_name': self.classes.get(class_id, 'unknown'),
                        'behavior': self.extract_behavior(self.classes.get(class_id, '')),
                        'timestamp': time.time()
                    })
        
        return detections
    
    def extract_behavior(self, class_name):
        """Extract behavior from class name (e.g., 'elsa_sitting' -> 'sitting')"""
        if '_' in class_name:
            return class_name.split('_')[1]
        return class_name
    
    def match_behaviors_to_dogs(self, behavior_detections, aruco_detections):
        """Match behavior detections to specific dogs using ArUco markers"""
        matched = []
        
        for behavior in behavior_detections:
            best_match = None
            min_distance = float('inf')
            
            # Find closest ArUco marker
            for aruco in aruco_detections:
                distance = self.calculate_distance(
                    behavior['center'], aruco['center']
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = aruco
            
            # If marker is close enough (within reasonable distance)
            if best_match and min_distance < 200:  # pixels
                matched.append({
                    'dog_id': best_match['dog_id'],
                    'behavior': behavior['behavior'],
                    'confidence': behavior['confidence'],
                    'bbox': behavior['bbox'],
                    'center': behavior['center'],
                    'marker_distance': min_distance,
                    'timestamp': behavior['timestamp'],
                    'should_reward': self.should_reward(
                        best_match['dog_id'], behavior['behavior']
                    )
                })
        
        return matched
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def should_reward(self, dog_id, behavior):
        """Determine if behavior should be rewarded"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_treat_time[dog_id] < self.treat_cooldown:
            return False
        
        # Reward sitting and lying behaviors
        reward_behaviors = ['sitting', 'lying']
        return behavior in reward_behaviors

# Test the enhanced detection
if __name__ == "__main__":
    detector = EnhancedDogBehaviorDetector()
    cap = cv2.VideoCapture(0)
    
    print("Enhanced detection test - use ArUco markers on dogs")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        matched_detections, aruco_detections = detector.detect_frame(frame)
        
        # Draw ArUco markers
        frame = detector.aruco_detector.draw_markers(frame, aruco_detections, None, None)
        
        # Draw behavior detections
        for detection in matched_detections:
            x1, y1, x2, y2 = detection['bbox']
            label = f"{detection['dog_id']}: {detection['behavior']} ({detection['confidence']:.2f})"
            
            color = (0, 255, 0) if detection['should_reward'] else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imshow('Enhanced Dog Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()