#!/usr/bin/env python3
"""
DogBot Main System - Complete Version with All Features
"""

import sys
import time
import logging
import threading
import os
import numpy as np
import cv2
from pathlib import Path

# Your WORKING modules from the uploaded files
from core.motor_controller import MotorController
from core.audio_controller import AudioController  
from core.led_controller import LEDController, LEDMode
from core.servo_controller import ServoController

# Camera
from picamera2 import Picamera2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DogBot:
    """Complete DogBot class with all features"""
    
    def __init__(self):
        print("Initializing DogBot hardware...")
        
        # Hardware that auto-initializes
        self.motors = MotorController()
        self.audio = AudioController()
        self.leds = LEDController()
        self.servos = ServoController()
        
        # Camera setup
        self.camera = None
        self.setup_camera()
        
        # Detection setup
        self.net = None
        self.output_layers = None
        
        # Behavior tracking
        self.pose_history = []
        self.behavior_timers = {}
        self.last_treat_time = 0
        
        # Check initialization status
        self.check_hardware_status()
        
    def setup_camera(self):
        """Simple working camera setup"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480), "format": "RGB888"}
            )
            self.camera.configure(config)
            logger.info("Camera ready")
        except Exception as e:
            logger.error(f"Camera setup failed: {e}")
    
    def check_hardware_status(self):
        """Check all hardware is initialized"""
        print("\nHardware Status:")
        print(f"  Motors: {'✓' if self.motors.is_initialized() else '✗'}")
        print(f"  Audio: {'✓' if self.audio.is_initialized() else '✗'}")
        print(f"  LEDs: {'✓' if self.leds.is_initialized() else '✗'}")
        print(f"  Servos: {'✓' if self.servos.is_initialized() else '✗'}")
        print(f"  Camera: {'✓' if self.camera else '✗'}")
        
    def set_initial_state(self):
        """Set robot to initial state"""
        try:
            # Center camera
            self.servos.center_camera()
            
            # Set LEDs to idle
            self.leds.set_mode(LEDMode.IDLE)
            
            # Play startup sound
            self.audio.switch_to_dfplayer()
            self.audio.play_sound("door_scan")
            
            print("Robot ready!")
            return True
            
        except Exception as e:
            logger.error(f"Initial state setup failed: {e}")
            return False
    
    def setup_yolo_detection(self):
        """Setup YOLO for actual dog detection"""
        try:
            # Try Hailo first
            from hailo_platform import VDevice
            self.use_hailo = True
            print("Using Hailo for detection")
        except:
            # Fallback to CPU detection
            self.use_hailo = False
            print("Using CPU detection (Hailo not available)")
            
        # For now, use OpenCV DNN as reliable fallback
        self.setup_opencv_yolo()

    def setup_opencv_yolo(self):
        """Setup OpenCV YOLO detection"""
        import requests
        
        # Download YOLOv4-tiny for fast detection
        weights_url = "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights"
        config_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
        
        weights_path = "/home/morgan/dogbot/models/yolov4-tiny.weights"
        config_path = "/home/morgan/dogbot/models/yolov4-tiny.cfg"
        
        os.makedirs("/home/morgan/dogbot/models", exist_ok=True)
        
        # Download if not exists
        if not os.path.exists(weights_path):
            print("Downloading YOLO model...")
            r = requests.get(weights_url)
            with open(weights_path, 'wb') as f:
                f.write(r.content)
        
        if not os.path.exists(config_path):
            r = requests.get(config_url)
            with open(config_path, 'wb') as f:
                f.write(r.content)
        
        # Load YOLO
        self.net = cv2.dnn.readNet(weights_path, config_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Get output layers
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        print("YOLO detector ready")

    def detect_dog_yolo(self, frame):
        """Detect dogs using YOLO"""
        if not hasattr(self, 'net') or self.net is None:
            return None
        
        height, width = frame.shape[:2]
        
        # Prepare input
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Run inference
        outputs = self.net.forward(self.output_layers)
        
        # Process outputs
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Class 16 is dog in COCO dataset
                if class_id == 16 and confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    x = int(center_x - w/2)
                    y = int(center_y - h/2)
                    
                    return {
                        "bbox": (x, y, w, h),
                        "confidence": float(confidence),
                        "center": (center_x, center_y)
                    }
        
        return None
    
    def detect_behavior(self, frame, dog_bbox):
        """Detect dog behavior from pose"""
        if not dog_bbox:
            return "idle"
        
        x, y, w, h = dog_bbox
        dog_region = frame[y:y+h, x:x+w]
        
        # Analyze dog pose
        aspect_ratio = h / w if w > 0 else 1
        
        # Store history for temporal behaviors
        current_pose = {
            'aspect_ratio': aspect_ratio,
            'bbox': dog_bbox,
            'time': time.time()
        }
        
        self.pose_history.append(current_pose)
        if len(self.pose_history) > 30:  # Keep last second at 30fps
            self.pose_history.pop(0)
        
        # Detect behaviors
        behavior = self.classify_behavior(aspect_ratio)
        
        # Check for spin (rotation movement)
        if self.detect_spinning():
            behavior = "spinning"
        
        # Check for stay (no movement for 3+ seconds)
        if self.detect_staying():
            behavior = "staying"
        
        return behavior

    def classify_behavior(self, aspect_ratio):
        """Classify static behaviors based on aspect ratio"""
        if 0.9 < aspect_ratio < 1.3:
            return "sitting"
        elif aspect_ratio < 0.7:
            return "lying"
        elif aspect_ratio > 1.3:
            return "standing"
        else:
            return "idle"

    def detect_spinning(self):
        """Detect spinning behavior from position history"""
        if len(self.pose_history) < 20:
            return False
        
        # Check if dog position is rotating
        positions = [(p['bbox'][0] + p['bbox'][2]//2, 
                     p['bbox'][1] + p['bbox'][3]//2) for p in self.pose_history[-20:]]
        
        # Calculate circular motion
        center_x = sum(p[0] for p in positions) / len(positions)
        center_y = sum(p[1] for p in positions) / len(positions)
        
        angles = []
        for x, y in positions:
            angle = np.arctan2(y - center_y, x - center_x)
            angles.append(angle)
        
        # Check if angles are increasing/decreasing consistently (rotation)
        angle_diffs = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
        if all(d > 0 for d in angle_diffs) or all(d < 0 for d in angle_diffs):
            return True
        
        return False

    def detect_staying(self):
        """Detect stay behavior (no movement)"""
        if len(self.pose_history) < 90:  # Need 3 seconds of history
            return False
        
        # Check last 3 seconds
        recent_positions = self.pose_history[-90:]
        
        # Calculate movement variance
        positions = [(p['bbox'][0], p['bbox'][1]) for p in recent_positions]
        x_variance = np.var([p[0] for p in positions])
        y_variance = np.var([p[1] for p in positions])
        
        # Low variance = staying still
        return x_variance < 100 and y_variance < 100
    
    def check_and_reward(self, behavior):
        """Check if behavior should be rewarded"""
        # Required duration for each behavior
        required_durations = {
            'sitting': 2.0,
            'lying': 3.0,
            'staying': 5.0,
            'spinning': 1.0
        }
        
        if behavior not in required_durations:
            self.behavior_timers.clear()
            return
        
        # Track behavior duration
        if behavior not in self.behavior_timers:
            self.behavior_timers[behavior] = time.time()
        
        duration = time.time() - self.behavior_timers[behavior]
        
        # Check if behavior held long enough and cooldown passed
        if duration >= required_durations[behavior]:
            if time.time() - self.last_treat_time > 10:  # 10 second cooldown
                self.dispense_treat(behavior)
                self.behavior_timers.clear()
                self.last_treat_time = time.time()

    def dispense_treat(self, behavior):
        """Dispense treat with feedback"""
        print(f"\n🎉 Good dog! Rewarding for: {behavior}")
        
        # Audio feedback
        self.audio.play_sound("good_dog")
        
        # Visual feedback
        self.leds.set_mode(LEDMode.TREAT_LAUNCHING)
        
        # Dispense treat
        self.servos.rotate_winch(direction='forward', duration=0.5)
        
        # Return to normal after 2 seconds
        threading.Timer(2.0, lambda: self.leds.set_mode(LEDMode.IDLE)).start()
    
    def track_and_treat(self):
        """Enhanced tracking with behavior detection"""
        if not self.camera:
            print("No camera available")
            return
        
        # Setup YOLO detection
        self.setup_yolo_detection()
        
        self.camera.start()
        print("\nStarting enhanced dog tracking with behavior detection...")
        print("Behaviors: sit (2s), lie down (3s), stay (5s), spin (1s)")
        print("Press Ctrl+C to stop\n")
        
        try:
            while True:
                frame = self.camera.capture_array()
                
                # Detect dog with YOLO
                detection = self.detect_dog_yolo(frame)
                
                if detection:
                    self.leds.set_mode(LEDMode.DOG_DETECTED)
                    
                    # Track with camera
                    center_x, center_y = detection['center']
                    if center_x < 320 - 50:
                        self.servos.set_camera_pan(max(10, self.servos.current_pan - 5))
                    elif center_x > 320 + 50:
                        self.servos.set_camera_pan(min(200, self.servos.current_pan + 5))
                    
                    # Detect behavior
                    behavior = self.detect_behavior(frame, detection['bbox'])
                    
                    # Check for reward
                    self.check_and_reward(behavior)
                    
                    # Display status
                    print(f"\rDog detected - Behavior: {behavior:10s} Confidence: {detection['confidence']:.2f}", end='')
                    
                else:
                    self.leds.set_mode(LEDMode.SEARCHING)
                    print("\rSearching for dog...                    ", end='')
                
                time.sleep(0.03)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.camera.stop()
    
    def manual_control_mode(self):
        """Manual control for testing"""
        print("\n=== Manual Control Mode ===")
        print("Commands:")
        print("  f/b/l/r - Forward/Back/Left/Right")
        print("  s - Stop motors")
        print("  c - Center camera")
        print("  scan - Scan camera left/right")
        print("  treat - Manual treat dispense")
        print("  led [mode] - Change LED mode")
        print("  track - Start auto tracking with behavior detection")
        print("  q - Quit")
        
        while True:
            cmd = input("> ").lower().strip()
            
            if cmd == 'q':
                break
            elif cmd == 'f':
                self.motors.move_forward(duration=1, speed=30)
            elif cmd == 'b':
                self.motors.move_backward(duration=1, speed=30)
            elif cmd == 'l':
                self.motors.turn_left(angle=45, speed=30)
            elif cmd == 'r':
                self.motors.turn_right(angle=45, speed=30)
            elif cmd == 's':
                self.motors.stop()
            elif cmd == 'c':
                self.servos.center_camera()
            elif cmd == 'scan':
                self.servos.scan_left_right(cycles=1)
            elif cmd == 'treat':
                self.dispense_treat("manual")
            elif cmd.startswith('led'):
                parts = cmd.split()
                if len(parts) == 2:
                    try:
                        mode = LEDMode(parts[1])
                        self.leds.set_mode(mode)
                    except:
                        print("Invalid LED mode")
            elif cmd == 'track':
                self.track_and_treat()
    
    def cleanup(self):
        """Clean shutdown"""
        print("\nCleaning up...")
        self.motors.cleanup()
        self.audio.cleanup()
        self.leds.cleanup()
        self.servos.cleanup()
        print("Cleanup complete")

def main():
    """Single entry point"""
    try:
        dogbot = DogBot()
        
        if dogbot.set_initial_state():
            # Start in manual control mode for testing
            dogbot.manual_control_mode()
        else:
            print("Failed to set initial state")
            
    except KeyboardInterrupt:
        print("\nShutdown requested")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'dogbot' in locals():
            dogbot.cleanup()

if __name__ == "__main__":
    main()q