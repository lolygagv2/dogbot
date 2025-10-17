#!/usr/bin/env python3
"""
Real-time GUI for TreatSensei DogBot - Live Detection Display
Shows camera feed with detection bounding boxes, poses, and behaviors on HDMI output
Includes pan/pitch camera controls
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

class LiveDetectionGUI:
    """Real-time GUI for dog detection with visual feedback"""

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
            'background': (0, 0, 0)             # Black
        }

        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.8
        self.font_thickness = 2

        # AI and camera
        self.ai = AI3StageControllerFixed()
        self.camera = None

        # Frame processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.running = False

        # Statistics
        self.stats = {
            'fps': 0,
            'total_frames': 0,
            'detections_count': 0,
            'poses_count': 0,
            'behaviors_count': 0,
            'inference_time': 0
        }

        # Frame timing
        self.last_frame_time = time.time()
        self.frame_times = []

        # Pan/tilt controls (placeholder for servo integration)
        self.pan_angle = 90   # Center position
        self.tilt_angle = 90  # Center position

    def initialize(self):
        """Initialize AI and camera systems"""
        print("🎬 Initializing Live Detection GUI")
        print("=" * 50)

        # Initialize AI
        if not self.ai.initialize():
            print("❌ AI initialization failed")
            return False
        print("✅ AI system ready")

        # Initialize camera
        if not self._init_camera():
            print("❌ Camera initialization failed")
            return False
        print("✅ Camera ready")

        # Setup display window for HDMI output
        cv2.namedWindow('TreatSensei DogBot - Live Detection', cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('TreatSensei DogBot - Live Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("✅ GUI initialized - displaying on HDMI")
        return True

    def _init_camera(self):
        """Initialize camera system"""
        if not PICAMERA2_AVAILABLE:
            print("❌ Picamera2 not available")
            return False

        try:
            self.camera = Picamera2()
            config = self.camera.create_still_configuration(
                main={"size": (self.camera_width, self.camera_height)},
                display="main"
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(1)  # Camera warm-up
            return True
        except Exception as e:
            print(f"❌ Camera error: {e}")
            return False

    def _capture_thread(self):
        """Background thread for continuous frame capture"""
        while self.running:
            try:
                if self.camera:
                    frame = self.camera.capture_array()
                    # Convert RGB to BGR for OpenCV
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Add to queue (non-blocking)
                    try:
                        self.frame_queue.put_nowait(frame)
                    except queue.Full:
                        # Skip frame if queue is full
                        pass

                time.sleep(0.01)  # Small delay to prevent overwhelming
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)

    def _ai_processing_thread(self):
        """Background thread for AI processing"""
        while self.running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=1.0)

                # Process with AI
                start_time = time.time()
                detections, poses, behaviors = self.ai.process_frame(frame)
                inference_time = (time.time() - start_time) * 1000

                # Package results
                result = {
                    'frame': frame,
                    'detections': detections,
                    'poses': poses,
                    'behaviors': behaviors,
                    'inference_time': inference_time,
                    'timestamp': time.time()
                }

                # Add to result queue (non-blocking)
                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    # Skip if display is too slow
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"AI processing error: {e}")
                time.sleep(0.1)

    def _draw_detection_box(self, frame, detection):
        """Draw detection bounding box and info"""
        x1, y1, x2, y2 = detection.x1, detection.y1, detection.x2, detection.y2
        confidence = detection.confidence

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors['detection_box'], 3)

        # Draw confidence
        conf_text = f"Dog: {confidence:.2f}"
        text_size = cv2.getTextSize(conf_text, self.font, self.font_scale, self.font_thickness)[0]

        # Background rectangle for text
        cv2.rectangle(frame, (x1, y1 - text_size[1] - 10),
                     (x1 + text_size[0] + 10, y1), self.colors['detection_box'], -1)

        # Text
        cv2.putText(frame, conf_text, (x1 + 5, y1 - 5),
                   self.font, self.font_scale, (0, 0, 0), self.font_thickness)

    def _draw_pose_keypoints(self, frame, pose):
        """Draw pose keypoints if available"""
        if pose and hasattr(pose, 'keypoints') and pose.keypoints is not None:
            keypoints = pose.keypoints

            # Draw each keypoint
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.3:  # Only draw confident keypoints
                    # Scale keypoints to detection box coordinates if needed
                    if hasattr(pose, 'detection'):
                        det = pose.detection
                        # Convert from crop coordinates to full frame coordinates
                        x_scaled = int(det.x1 + x * (det.x2 - det.x1) / 640)
                        y_scaled = int(det.y1 + y * (det.y2 - det.y1) / 640)
                    else:
                        x_scaled, y_scaled = int(x), int(y)

                    # Draw keypoint
                    cv2.circle(frame, (x_scaled, y_scaled), 4, self.colors['pose_keypoints'], -1)
                    cv2.circle(frame, (x_scaled, y_scaled), 6, (255, 255, 255), 1)

    def _draw_behavior_info(self, frame, behaviors, detection):
        """Draw behavior classification info"""
        if behaviors:
            behavior = behaviors[0]  # Show first behavior
            behavior_text = f"Behavior: {behavior.behavior} ({behavior.confidence:.2f})"

            # Position text below detection box
            x1, y2 = detection.x1, detection.y2
            text_y = y2 + 30

            # Background for text
            text_size = cv2.getTextSize(behavior_text, self.font, self.font_scale, self.font_thickness)[0]
            cv2.rectangle(frame, (x1, text_y - text_size[1] - 5),
                         (x1 + text_size[0] + 10, text_y + 5), self.colors['behavior_text'], -1)

            # Text
            cv2.putText(frame, behavior_text, (x1 + 5, text_y),
                       self.font, self.font_scale, (0, 0, 0), self.font_thickness)

    def _draw_stats_overlay(self, frame):
        """Draw statistics overlay"""
        # Top-left stats panel
        stats_y = 30
        line_height = 35

        stats_lines = [
            f"TreatSensei DogBot - Live Detection",
            f"FPS: {self.stats['fps']:.1f}",
            f"Detections: {self.stats['detections_count']}",
            f"Poses: {self.stats['poses_count']}",
            f"Behaviors: {self.stats['behaviors_count']}",
            f"Inference: {self.stats['inference_time']:.1f}ms",
            f"Pan: {self.pan_angle}° Tilt: {self.tilt_angle}°"
        ]

        # Background panel
        panel_height = len(stats_lines) * line_height + 20
        cv2.rectangle(frame, (10, 10), (450, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, panel_height), (255, 255, 255), 2)

        # Stats text
        for i, line in enumerate(stats_lines):
            y_pos = stats_y + (i * line_height)
            color = self.colors['confidence_text'] if i == 0 else self.colors['stats_text']
            weight = self.font_thickness if i == 0 else 1
            cv2.putText(frame, line, (20, y_pos), self.font, self.font_scale, color, weight)

    def _draw_controls_help(self, frame):
        """Draw control instructions"""
        help_lines = [
            "Controls:",
            "Q - Quit",
            "WASD - Pan/Tilt Camera",
            "R - Reset Camera",
            "SPACE - Screenshot"
        ]

        # Bottom-right help panel
        start_y = self.display_height - (len(help_lines) * 25) - 20
        line_height = 25

        # Background
        cv2.rectangle(frame, (self.display_width - 250, start_y - 10),
                     (self.display_width - 10, self.display_height - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (self.display_width - 250, start_y - 10),
                     (self.display_width - 10, self.display_height - 10), (255, 255, 255), 1)

        # Help text
        for i, line in enumerate(help_lines):
            y_pos = start_y + (i * line_height)
            cv2.putText(frame, line, (self.display_width - 240, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['stats_text'], 1)

    def _update_stats(self, result):
        """Update frame statistics"""
        current_time = time.time()

        # Update frame timing for FPS calculation
        self.frame_times.append(current_time)
        self.frame_times = [t for t in self.frame_times if current_time - t < 1.0]  # Keep last 1 second
        self.stats['fps'] = len(self.frame_times)

        # Update counts
        self.stats['total_frames'] += 1
        self.stats['detections_count'] = len(result['detections'])
        self.stats['poses_count'] = len(result['poses'])
        self.stats['behaviors_count'] = len(result['behaviors'])
        self.stats['inference_time'] = result['inference_time']

    def _handle_keyboard_input(self, key):
        """Handle keyboard controls for camera pan/tilt"""
        if key == ord('q') or key == ord('Q'):
            return False  # Quit
        elif key == ord('w') or key == ord('W'):
            self.tilt_angle = max(0, self.tilt_angle - 5)  # Tilt up
        elif key == ord('s') or key == ord('S'):
            self.tilt_angle = min(180, self.tilt_angle + 5)  # Tilt down
        elif key == ord('a') or key == ord('A'):
            self.pan_angle = max(0, self.pan_angle - 5)  # Pan left
        elif key == ord('d') or key == ord('D'):
            self.pan_angle = min(180, self.pan_angle + 5)  # Pan right
        elif key == ord('r') or key == ord('R'):
            self.pan_angle = 90   # Reset to center
            self.tilt_angle = 90
        elif key == ord(' '):  # Space for screenshot
            self._save_screenshot()

        # TODO: Send pan/tilt commands to servo hardware
        # self._update_camera_position(self.pan_angle, self.tilt_angle)

        return True  # Continue

    def _save_screenshot(self):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_dir = Path("screenshots")
        screenshot_dir.mkdir(exist_ok=True)

        filename = screenshot_dir / f"dogbot_screenshot_{timestamp}.jpg"

        # Get current display frame
        try:
            result = self.result_queue.queue[-1] if not self.result_queue.empty() else None
            if result:
                cv2.imwrite(str(filename), result['frame'])
                print(f"📸 Screenshot saved: {filename}")
        except:
            print("❌ Failed to save screenshot")

    def run(self):
        """Main GUI loop"""
        print("\n🎬 Starting Live Detection GUI")
        print("Press 'Q' to quit, WASD for camera control")

        self.running = True

        # Start background threads
        capture_thread = threading.Thread(target=self._capture_thread, daemon=True)
        ai_thread = threading.Thread(target=self._ai_processing_thread, daemon=True)

        capture_thread.start()
        ai_thread.start()

        try:
            while self.running:
                # Get latest AI result
                try:
                    result = self.result_queue.get(timeout=0.1)
                except queue.Empty:
                    # Show blank frame if no results yet
                    blank_frame = np.zeros((self.display_height, self.display_width, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "Initializing...", (self.display_width//2 - 100, self.display_height//2),
                               self.font, 1.0, self.colors['stats_text'], 2)
                    cv2.imshow('TreatSensei DogBot - Live Detection', blank_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        if not self._handle_keyboard_input(key):
                            break
                    continue

                # Update statistics
                self._update_stats(result)

                # Create display frame
                display_frame = result['frame'].copy()

                # Draw detections
                for i, detection in enumerate(result['detections']):
                    self._draw_detection_box(display_frame, detection)

                    # Draw corresponding pose if available
                    if i < len(result['poses']):
                        self._draw_pose_keypoints(display_frame, result['poses'][i])

                    # Draw corresponding behavior if available
                    if i < len(result['behaviors']):
                        self._draw_behavior_info(display_frame, [result['behaviors'][i]], detection)

                # Draw overlays
                self._draw_stats_overlay(display_frame)
                self._draw_controls_help(display_frame)

                # Display frame
                cv2.imshow('TreatSensei DogBot - Live Detection', display_frame)

                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if not self._handle_keyboard_input(key):
                        break

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up GUI...")

        self.running = False

        if self.camera:
            self.camera.stop()

        self.ai.cleanup()
        cv2.destroyAllWindows()

        print("✅ GUI cleanup complete")

def main():
    gui = LiveDetectionGUI()

    if not gui.initialize():
        print("❌ Failed to initialize GUI")
        return

    gui.run()

if __name__ == "__main__":
    main()