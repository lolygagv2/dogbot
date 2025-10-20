#!/usr/bin/env python3
"""
Simple Hailo inference without TAPPAS
Direct HailoRT usage with proper buffer handling
"""

import numpy as np
import cv2
import time
from pathlib import Path
from hailo_platform import (
    Device,
    ConfigureParams,
    InputVStreamParams,
    OutputVStreamParams,
    HailoStreamInterface,
    FormatType,
    HEF
)

class SimpleHailoInference:
    """Simple Hailo inference using raw HailoRT"""
    
    def __init__(self, hef_path):
        self.hef_path = Path(hef_path)
        
        # Class mapping
        self.classes = [
            "elsa_sitting", "elsa_lying", "elsa_standing", "elsa_spinning", "elsa_playing",
            "bezik_sitting", "bezik_lying", "bezik_standing", "bezik_spinning", "bezik_playing"
        ]
        
        # Initialize device
        self.device = Device()
        
        # Load HEF
        self.hef = HEF(str(self.hef_path))
        
        # Configure device
        self.configure_params = ConfigureParams.create_from_hef(
            self.hef, 
            interface=HailoStreamInterface.PCIe
        )
        
        self.network_groups = self.device.configure(self.hef, self.configure_params)
        
        if not self.network_groups:
            raise RuntimeError("Failed to configure device")
            
        self.network_group = self.network_groups[0]
        self.network_group_params = self.network_group.create_params()
        
        # Get input/output info
        self.input_vstreams_info = self.hef.get_input_vstream_infos()
        self.output_vstreams_info = self.hef.get_output_vstream_infos()
        
        # Setup input/output vstreams
        self.input_vstreams = InputVStreamParams.make_from_network_group(
            self.network_group, 
            quantized=True,
            format_type=FormatType.UINT8
        )
        
        self.output_vstreams = OutputVStreamParams.make_from_network_group(
            self.network_group,
            quantized=False, 
            format_type=FormatType.FLOAT32
        )
        
    def preprocess(self, image):
        """Preprocess image for inference"""
        # Resize to 640x640
        resized = cv2.resize(image, (640, 640))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Add batch dimension
        batched = np.expand_dims(rgb, axis=0)
        
        return batched.astype(np.uint8)
    
    def infer(self, image):
        """Run inference on single image"""
        # Preprocess
        input_data = self.preprocess(image)
        
        # Activate network group
        with self.network_group.activate(self.network_group_params):
            # Create input/output dictionaries
            input_data_dict = {
                self.input_vstreams_info[0].name: input_data
            }
            
            # Run inference
            raw_outputs = self.network_group.infer(
                input_data_dict,
                self.output_vstreams
            )
            
        return self.postprocess(raw_outputs)
    
    def postprocess(self, outputs):
        """Process model outputs"""
        # Get first output
        output_name = list(outputs.keys())[0]
        raw_output = outputs[output_name]
        
        # Expected shape: [1, 84, 8400] for YOLOv8
        # Reshape if needed
        if raw_output.size == 84 * 8400:
            output = raw_output.reshape(1, 84, -1)
        else:
            output = raw_output
        
        detections = []
        
        # Process detections
        for i in range(output.shape[2]):
            # Extract bbox and scores
            x, y, w, h = output[0, :4, i]
            obj_conf = output[0, 4, i]
            class_scores = output[0, 5:15, i]  # 10 classes
            
            # Get best class
            class_id = np.argmax(class_scores)
            class_conf = class_scores[class_id]
            
            # Combined confidence
            confidence = obj_conf * class_conf
            
            if confidence > 0.5:
                detections.append({
                    'bbox': [float(x), float(y), float(w), float(h)],
                    'class_id': int(class_id),
                    'class_name': self.classes[class_id],
                    'confidence': float(confidence)
                })
        
        return detections
    
    def run_camera(self, camera_index=0):
        """Run inference on camera stream"""
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        fps_start = time.time()
        frame_count = 0
        
        print("Starting camera inference. Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Run inference
            detections = self.infer(frame)
            
            # Draw detections
            for det in detections:
                x, y, w, h = det['bbox']
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                
                # Scale to original image size
                h_orig, w_orig = frame.shape[:2]
                x1 = int(x1 * w_orig / 640)
                y1 = int(y1 * h_orig / 640)
                x2 = int(x2 * w_orig / 640)
                y2 = int(y2 * h_orig / 640)
                
                # Draw bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - fps_start
                fps = frame_count / elapsed
                print(f"FPS: {fps:.2f}")
            
            # Display
            cv2.imshow("DogBot Detection", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 simple_hailo_inference.py <path_to_hef>")
        sys.exit(1)
    
    hef_path = sys.argv[1]
    
    # Create inference object
    inference = SimpleHailoInference(hef_path)
    
    # Run camera inference
    inference.run_camera(0)
