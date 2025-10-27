#!/usr/bin/env python3
"""
WIM-Z Real-time GUI with ArUco Dog Identification
Shows camera feed with detection bounding boxes, poses, behaviors AND ArUco markers
"""

import sys
import os
import time
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import threading
import queue

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

from core.ai_controller_3stage_fixed import AI3StageControllerFixed
from detect_aruco_id import detect_ids  # Use existing ArUco implementation

# Try to import servo control (optional)
try:
    from servo_control_module import ServoController
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False

class WIMZDetectionGUI:
    """WIM-Z Real-time GUI with ArUco dog identification"""

    def __init__(self):
        # Display settings
        self.display_width = 1920
        self.display_height = 1080
        self.camera_width = 1920
        self.camera_height = 1080

        # Colors (BGR format)
        self.colors = {
            'detection_box': (0, 255, 0),      # Green
            'pose_keypoints': (255, 0, 0),     # Blue
            'behavior_text': (0, 255, 255),    # Yellow
            'stats_text': (255, 255, 255),     # White
            'confidence_text': (0, 255, 0),    # Green
            'aruco_box': (255, 0, 255),        # Magenta for ArUco
            'aruco_text': (255, 128, 255),     # Light magenta
            'background': (0, 0, 0)            # Black
        }

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_thickness = 2

        # AI and camera
        self.ai = AI3StageControllerFixed()
        self.camera = None

        # Servo control
        self.servo_controller = None
        if SERVO_AVAILABLE:
            try:
                self.servo_controller = ServoController()
                if self.servo_controller.initialize():
                    print("[INFO] Servo controller initialized")
            except Exception as e:
                print(f"[WARNING] Servo controller not available: {e}")

        # ArUco dog ID mapping
        self.dog_names = {
            1: "Max",
            2: "Bella",
            3: "Charlie",
            4: "Luna",
            5: "Cooper"
        }

        # Dog tracking data
        self.dog_profiles = {}  # Store per-dog data
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

        # Camera control
        self.pan_angle = 90
        self.tilt_angle = 90
        self.angle_step = 5

    def initialize_camera(self):
        """Initialize the camera"""
        if PICAMERA2_AVAILABLE:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (self.camera_width, self.camera_height), "format": "BGR888"}
                )
                self.camera.configure(config)
                self.camera.start()
                print(f"[INFO] Camera initialized at {self.camera_width}x{self.camera_height}")
                return True
            except Exception as e:
                print(f"[ERROR] Failed to initialize Picamera2: {e}")
                return False
        else:
            print("[WARNING] Picamera2 not available, using mock camera")
            self.camera = None
            return True

    def get_frame(self):
        """Get a frame from the camera"""
        if self.camera and PICAMERA2_AVAILABLE:
            return self.camera.capture_array()
        else:
            # Create a mock frame for testing
            frame = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            cv2.putText(frame, "MOCK CAMERA - No Picamera2",
                       (self.camera_width//2 - 200, self.camera_height//2),
                       self.font, 1, (255, 255, 255), 2)
            return frame

    def draw_aruco_markers(self, frame, markers):
        """Draw ArUco markers and dog IDs on frame"""
        for marker_id, cx, cy in markers:
            # Get dog name if known
            dog_name = self.dog_names.get(marker_id, f"Dog #{marker_id}")

            # Draw marker box (approximate size)
            box_size = 50
            cv2.rectangle(frame,
                         (int(cx - box_size), int(cy - box_size)),
                         (int(cx + box_size), int(cy + box_size)),
                         self.colors['aruco_box'], 3)

            # Draw dog ID and name
            label = f"ArUco {marker_id}: {dog_name}"
            cv2.putText(frame, label,
                       (int(cx - box_size), int(cy - box_size - 10)),
                       self.font, 0.7, self.colors['aruco_text'], 2)

            # Update dog profile
            if marker_id not in self.dog_profiles:
                self.dog_profiles[marker_id] = {
                    'name': dog_name,
                    'last_seen': time.time(),
                    'behaviors': [],
                    'treats_earned': 0
                }
            else:
                self.dog_profiles[marker_id]['last_seen'] = time.time()

    def draw_detections(self, frame, detections):
        """Draw AI detection results on frame"""
        if not detections:
            return

        for i, det in enumerate(detections):
            bbox = det.get('bbox', [])
            if len(bbox) >= 4:
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['detection_box'], 2)

                # Draw confidence
                conf = det.get('confidence', 0)
                label = f"Dog {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           self.font, 0.6, self.colors['confidence_text'], 2)

            # Draw pose keypoints
            keypoints = det.get('keypoints', {}).get('keypoints', [])
            if keypoints:
                for kp in keypoints:
                    if len(kp) >= 2:
                        x, y = int(kp[0]), int(kp[1])
                        cv2.circle(frame, (x, y), 5, self.colors['pose_keypoints'], -1)

            # Draw behavior if detected
            behavior = det.get('behavior')
            if behavior and behavior != 'unknown':
                text = f"BEHAVIOR: {behavior.upper()}"
                y_pos = 100 + (i * 40)
                cv2.putText(frame, text, (50, y_pos),
                           self.font, 1, self.colors['behavior_text'], 2)

    def draw_stats(self, frame):
        """Draw statistics on frame"""
        # FPS counter
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (self.display_width - 150, 30),
                   self.font, 0.7, self.colors['stats_text'], 2)

        # Frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (self.display_width - 150, 60),
                   self.font, 0.7, self.colors['stats_text'], 2)

        # WIM-Z branding
        cv2.putText(frame, "WIM-Z: Watchful Intelligent Mobile Zen",
                   (50, self.display_height - 30),
                   self.font, 0.6, self.colors['stats_text'], 1)

        # Dog profiles
        y_pos = 150
        for dog_id, profile in self.dog_profiles.items():
            age = time.time() - profile['last_seen']
            if age < 10:  # Show dogs seen in last 10 seconds
                text = f"{profile['name']}: Treats {profile['treats_earned']}"
                cv2.putText(frame, text, (self.display_width - 250, y_pos),
                           self.font, 0.6, self.colors['aruco_text'], 1)
                y_pos += 30

    def update_camera_position(self, key):
        """Update camera pan/tilt based on key press"""
        if not self.servo_controller:
            return

        moved = False
        if key == ord('w'):  # Tilt up
            self.tilt_angle = max(0, self.tilt_angle - self.angle_step)
            moved = True
        elif key == ord('s'):  # Tilt down
            self.tilt_angle = min(180, self.tilt_angle + self.angle_step)
            moved = True
        elif key == ord('a'):  # Pan left
            self.pan_angle = max(0, self.pan_angle - self.angle_step)
            moved = True
        elif key == ord('d'):  # Pan right
            self.pan_angle = min(180, self.pan_angle + self.angle_step)
            moved = True
        elif key == ord('h'):  # Home position
            self.pan_angle = 90
            self.tilt_angle = 90
            moved = True

        if moved:
            self.servo_controller.set_angle('pan', self.pan_angle)
            self.servo_controller.set_angle('tilt', self.tilt_angle)
            print(f"[SERVO] Pan: {self.pan_angle}°, Tilt: {self.tilt_angle}°")

    def run(self):
        """Main GUI loop"""
        if not self.initialize_camera():
            print("[ERROR] Failed to initialize camera")
            return

        window_name = "WIM-Z Detection with ArUco"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)

        print("\n=== WIM-Z Controls ===")
        print("W/S: Tilt camera up/down")
        print("A/D: Pan camera left/right")
        print("H: Home position")
        print("Q/ESC: Quit")
        print("=====================\n")

        try:
            while True:
                # Get frame
                frame = self.get_frame()

                # Detect ArUco markers
                aruco_markers = detect_ids(frame)
                if aruco_markers:
                    self.draw_aruco_markers(frame, aruco_markers)
                    print(f"[ARUCO] Detected markers: {aruco_markers}")

                # Run AI detection
                detections = self.ai.process_frame(frame)
                self.draw_detections(frame, detections)

                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    current_time = time.time()
                    self.fps = 30 / (current_time - self.last_time)
                    self.last_time = current_time

                # Draw stats
                self.draw_stats(frame)

                # Display frame
                cv2.imshow(window_name, frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key != 255:  # Any other key
                    self.update_camera_position(key)

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user")
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("[INFO] Cleaning up...")
        if self.camera and PICAMERA2_AVAILABLE:
            self.camera.stop()
        if self.servo_controller:
            # Return to home position
            self.servo_controller.set_angle('pan', 90)
            self.servo_controller.set_angle('tilt', 90)
            self.servo_controller.cleanup()
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")

if __name__ == "__main__":
    gui = WIMZDetectionGUI()
    gui.run()