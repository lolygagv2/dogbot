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

# Force DISPLAY for local Pi testing
if not os.environ.get('DISPLAY'):
    os.environ['DISPLAY'] = ':0'

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
    print("[INFO] Servo control module imported successfully")
except ImportError as e:
    SERVO_AVAILABLE = False
    print(f"[ERROR] Failed to import servo_control_module: {e}")
except Exception as e:
    SERVO_AVAILABLE = False
    print(f"[ERROR] Unexpected error importing servo module: {e}")

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
        print(f"[DEBUG] SERVO_AVAILABLE = {SERVO_AVAILABLE}")
        if SERVO_AVAILABLE:
            try:
                print("[DEBUG] Creating ServoController instance...")
                self.servo_controller = ServoController()
                print("[DEBUG] Calling initialize()...")
                init_result = self.servo_controller.initialize()
                print(f"[DEBUG] initialize() returned: {init_result}")
                if init_result:
                    print("[INFO] ✅ Servo controller initialized successfully")
                else:
                    print("[ERROR] ❌ Servo initialization returned False")
                    self.servo_controller = None
            except Exception as e:
                print(f"[ERROR] ❌ Exception during servo init: {e}")
                import traceback
                traceback.print_exc()
                self.servo_controller = None
        else:
            print("[ERROR] SERVO_AVAILABLE is False - import failed")

        # ArUco dog ID mapping - DICT_4X4_1000 (IDs 0-999)
        # 4cm x 4cm markers
        self.dog_names = {
            315: "Elsa",    # ArUco marker ID 315
            832: "Bezik"    # ArUco marker ID 832
        }
        self.expected_dog_ids = [315, 832]  # Only track these IDs

        # Dog tracking data
        self.dog_profiles = {}  # Store per-dog data
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

        # Camera control - servo controller uses -90 to +90 range with 0 as center
        self.pan_angle = 0  # Center position
        self.tilt_angle = 0  # Center position
        self.angle_step = 10  # Larger steps for better responsiveness

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
            # Only process known dog markers
            if marker_id not in self.expected_dog_ids:
                # Draw unknown markers in gray
                cv2.putText(frame, f"Unknown: {marker_id}",
                           (int(cx - 30), int(cy - 40)),
                           self.font, 0.5, (128, 128, 128), 1)
                continue

            # Get dog name
            dog_name = self.dog_names[marker_id]

            # Draw marker box (larger for known dogs)
            box_size = 80
            color = (0, 255, 0) if marker_id == 315 else (255, 0, 255)  # Green for Elsa, Magenta for Bezik
            cv2.rectangle(frame,
                         (int(cx - box_size), int(cy - box_size)),
                         (int(cx + box_size), int(cy + box_size)),
                         color, 3)

            # Draw dog name prominently
            label = f"{dog_name} (ID: {marker_id})"
            cv2.putText(frame, label,
                       (int(cx - box_size), int(cy - box_size - 10)),
                       self.font, 1.0, color, 3)

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

    def draw_detections(self, frame, detections, poses, behaviors):
        """Draw AI detection results on frame"""
        if not detections:
            return

        # Draw detection boxes
        for i, det in enumerate(detections):
            bbox = det.bbox if hasattr(det, 'bbox') else []
            if len(bbox) >= 4:
                # Draw bounding box
                x1, y1, x2, y2 = map(int, bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['detection_box'], 2)

                # Draw confidence
                conf = det.confidence if hasattr(det, 'confidence') else 0
                label = f"Dog {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           self.font, 0.6, self.colors['confidence_text'], 2)

        # Draw pose keypoints
        for pose in poses:
            if hasattr(pose, 'keypoints'):
                keypoints = pose.keypoints
                if keypoints is not None:
                    for kp in keypoints:
                        if len(kp) >= 2:
                            x, y = int(kp[0]), int(kp[1])
                            cv2.circle(frame, (x, y), 5, self.colors['pose_keypoints'], -1)

        # Draw behaviors
        for i, behavior in enumerate(behaviors):
            if hasattr(behavior, 'behavior') and behavior.behavior != 'unknown':
                text = f"BEHAVIOR: {behavior.behavior.upper()}"
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
        # Debug output
        if key in [ord('w'), ord('s'), ord('a'), ord('d'), ord('h')]:
            print(f"[DEBUG] Key pressed: {chr(key)}")

        if not self.servo_controller:
            print("[ERROR] Servo controller is None!")
            return

        moved = False
        if key == ord('w'):  # Tilt up (positive angle)
            self.tilt_angle = min(45, self.tilt_angle + self.angle_step)  # Max +45°
            moved = True
            print(f"[DEBUG] W pressed - tilt up to {self.tilt_angle}")
        elif key == ord('s'):  # Tilt down (negative angle)
            self.tilt_angle = max(-45, self.tilt_angle - self.angle_step)  # Min -45°
            moved = True
            print(f"[DEBUG] S pressed - tilt down to {self.tilt_angle}")
        elif key == ord('a'):  # Pan left (negative angle)
            self.pan_angle = max(-90, self.pan_angle - self.angle_step)  # Min -90°
            moved = True
            print(f"[DEBUG] A pressed - pan left to {self.pan_angle}")
        elif key == ord('d'):  # Pan right (positive angle)
            self.pan_angle = min(90, self.pan_angle + self.angle_step)  # Max +90°
            moved = True
            print(f"[DEBUG] D pressed - pan right to {self.pan_angle}")
        elif key == ord('h'):  # Home position (center)
            self.pan_angle = 0
            self.tilt_angle = 0
            moved = True
            print(f"[DEBUG] H pressed - home position")

        if moved:
            print(f"[DEBUG] Attempting to move servos...")
            try:
                # Use the correct method names from servo controller
                self.servo_controller.set_pan_angle(self.pan_angle, smooth=False)  # Fast response
                self.servo_controller.set_tilt_angle(self.tilt_angle, smooth=False)  # Fast response
                print(f"[SERVO MOVED] Pan: {self.pan_angle}°, Tilt: {self.tilt_angle}°")
            except Exception as e:
                print(f"[ERROR] Failed to move servos: {e}")
                import traceback
                traceback.print_exc()

    def save_frame(self, frame, prefix="detection"):
        """Save frame to file in headless mode"""
        output_dir = Path("detection_results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_dir / f"{prefix}_{timestamp}_{self.frame_count:06d}.jpg"
        cv2.imwrite(str(filename), frame)
        return filename

    def run(self):
        """Main GUI loop - LOCAL HDMI ONLY"""
        if not self.initialize_camera():
            print("[ERROR] Failed to initialize camera")
            return

        # Create window for local display
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
                    # Only print if we detect our dogs
                    for marker_id, cx, cy in aruco_markers:
                        if marker_id in self.expected_dog_ids:
                            dog_name = self.dog_names[marker_id]
                            print(f"[DOG DETECTED] {dog_name} (ID: {marker_id}) at position ({cx:.1f}, {cy:.1f})")

                # Run AI detection - returns tuple (detections, poses, behaviors)
                result = self.ai.process_frame(frame)
                if isinstance(result, tuple) and len(result) == 3:
                    detections, poses, behaviors = result
                else:
                    detections, poses, behaviors = [], [], []

                self.draw_detections(frame, detections, poses, behaviors)

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
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()


    def cleanup(self):
        """Clean up resources"""
        print("[INFO] Cleaning up...")
        if self.camera and PICAMERA2_AVAILABLE:
            self.camera.stop()
        if self.servo_controller:
            try:
                # Return to home position (center = 0°)
                print("[INFO] Centering servos...")
                self.servo_controller.set_pan_angle(0, smooth=False)
                self.servo_controller.set_tilt_angle(0, smooth=False)
                self.servo_controller.cleanup()
            except Exception as e:
                print(f"[WARNING] Error during servo cleanup: {e}")
        cv2.destroyAllWindows()
        print("[INFO] Cleanup complete")

if __name__ == "__main__":
    gui = WIMZDetectionGUI()
    gui.run()