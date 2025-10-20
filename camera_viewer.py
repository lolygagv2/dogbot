#!/usr/bin/env python3
"""
Simple Camera Viewer for DogBot
Live camera view with detection overlay and manual controls
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import logging
from datetime import datetime
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.treat_dispenser_robot import TreatDispenserRobot, RobotMode
from core.hardware.led_controller import LEDMode

class CameraViewer:
    """Simple camera viewer with detection overlay and controls"""

    def __init__(self, robot: TreatDispenserRobot):
        self.robot = robot
        self.root = tk.Tk()
        self.root.title("🐕 DogBot Camera Viewer")
        self.root.geometry("1000x700")

        # Camera and detection
        self.current_frame = None
        self.detection_overlay = True
        self.latest_detection = None
        self.show_fps = True
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        # Training state
        self.dog_detected = False
        self.current_behavior = "idle"
        self.behavior_confidence = 0.0

        # UI elements
        self.video_label = None
        self.status_label = None
        self.detection_text = None
        self.command_history = []

        self.setup_gui()
        self.setup_robot_events()

        # Start camera feed
        self.start_camera_feed()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('CameraViewer')

    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Left side - Camera feed
        camera_frame = tk.Frame(main_frame, relief='ridge', bd=2)
        camera_frame.pack(side='left', fill='both', expand=True, padx=(0,10))

        # Camera title
        title_frame = tk.Frame(camera_frame)
        title_frame.pack(fill='x', pady=5)

        tk.Label(title_frame, text="🎥 Live Camera Feed",
                font=('Arial', 14, 'bold')).pack(side='left')

        # Status indicator
        self.camera_status = tk.Label(title_frame, text="🟢 LIVE",
                                     font=('Arial', 10, 'bold'), fg='green')
        self.camera_status.pack(side='right')

        # Video display
        self.video_label = tk.Label(camera_frame, bg='black', text="📷 Initializing Camera...",
                                   font=('Arial', 16), fg='white')
        self.video_label.pack(padx=10, pady=10)

        # Camera controls
        cam_controls = tk.Frame(camera_frame)
        cam_controls.pack(pady=5)

        tk.Button(cam_controls, text="📹 Center Camera",
                 command=self.center_camera, bg='lightblue').pack(side='left', padx=5)
        tk.Button(cam_controls, text="👀 Toggle Overlay",
                 command=self.toggle_overlay, bg='lightyellow').pack(side='left', padx=5)
        tk.Button(cam_controls, text="📊 Toggle FPS",
                 command=self.toggle_fps, bg='lightgreen').pack(side='left', padx=5)

        # Servo controls
        servo_frame = tk.LabelFrame(camera_frame, text="📡 Camera Controls", font=('Arial', 10, 'bold'))
        servo_frame.pack(pady=10, fill='x', padx=10)

        # Pan control (inverted direction, shifted left for better FOV)
        tk.Label(servo_frame, text="Pan (Left 30° ← 120° Center → 180° Right)").pack()
        self.pan_scale = tk.Scale(servo_frame, from_=30, to=180,
                                 orient='horizontal', command=self.update_pan,
                                 length=300)
        self.pan_scale.set(120)  # New center position (shifted left)
        self.pan_scale.pack(fill='x', padx=20)

        # Pitch control (extended range for ceiling view)
        tk.Label(servo_frame, text="Pitch (Down 30° ↓ 55° Center ↑ 150° Up)").pack()
        self.tilt_scale = tk.Scale(servo_frame, from_=30, to=150,
                                  orient='horizontal', command=self.update_tilt,
                                  length=300)
        self.tilt_scale.set(55)  # Center position from center_camera()
        self.tilt_scale.pack(fill='x', padx=20)

        # Right side - Detection and controls
        control_frame = tk.Frame(main_frame, relief='ridge', bd=2)
        control_frame.pack(side='right', fill='y', padx=(10,0))

        # Detection info
        detection_frame = tk.LabelFrame(control_frame, text="🔍 Detection Info", font=('Arial', 12, 'bold'))
        detection_frame.pack(fill='x', padx=10, pady=5)

        self.detection_text = tk.Text(detection_frame, height=8, width=35,
                                     font=('Courier', 9), bg='black', fg='green')
        self.detection_text.pack(fill='x', padx=5, pady=5)

        # Robot status
        status_frame = tk.LabelFrame(control_frame, text="🤖 Robot Status", font=('Arial', 12, 'bold'))
        status_frame.pack(fill='x', padx=10, pady=5)

        self.status_text = tk.Text(status_frame, height=6, width=35,
                                  font=('Courier', 9), bg='navy', fg='cyan')
        self.status_text.pack(fill='x', padx=5, pady=5)

        # Training controls
        training_frame = tk.LabelFrame(control_frame, text="🎯 Quick Training", font=('Arial', 12, 'bold'))
        training_frame.pack(fill='x', padx=10, pady=5)

        # Command buttons
        commands = [
            ('🗣️ Sit', 'SIT', 'lightblue'),
            ('🛑 Stay', 'STAY', 'orange'),
            ('⬇️ Lie Down', 'LIE_DOWN', 'lightcoral'),
            ('🎉 Good Dog!', 'GOOD_DOG', 'lightgreen'),
            ('🍖 Treat', 'treat', 'gold')
        ]

        for text, command, color in commands:
            btn = tk.Button(training_frame, text=text,
                           command=lambda c=command: self.quick_command(c),
                           bg=color, width=12, font=('Arial', 9, 'bold'))
            btn.pack(pady=2, fill='x', padx=5)

        # Robot mode controls
        mode_frame = tk.LabelFrame(control_frame, text="🔧 Robot Mode", font=('Arial', 12, 'bold'))
        mode_frame.pack(fill='x', padx=10, pady=5)

        mode_buttons = [
            ('👁️ Tracking', RobotMode.TRACKING, 'lightgreen'),
            ('🎓 Training', RobotMode.TRAINING, 'lightblue'),
            ('🎮 Manual', RobotMode.MANUAL, 'lightyellow')
        ]

        for text, mode, color in mode_buttons:
            btn = tk.Button(mode_frame, text=text,
                           command=lambda m=mode: self.set_robot_mode(m),
                           bg=color, width=12, font=('Arial', 9))
            btn.pack(pady=1, fill='x', padx=5)

        # Emergency stop
        emergency_frame = tk.Frame(control_frame)
        emergency_frame.pack(fill='x', padx=10, pady=10)

        tk.Button(emergency_frame, text="🚨 EMERGENCY STOP",
                 command=self.emergency_stop,
                 bg='red', fg='white', font=('Arial', 12, 'bold'),
                 width=15, height=2).pack()

    def setup_robot_events(self):
        """Setup event handlers for robot detection events"""
        if hasattr(self.robot, 'event_bus'):
            self.robot.event_bus.subscribe('dog_detected', self.on_dog_detected)
            self.robot.event_bus.subscribe('dog_lost', self.on_dog_lost)
            self.robot.event_bus.subscribe('behavior_detected', self.on_behavior_detected)
            self.robot.event_bus.subscribe('no_detections', self.on_no_detections)
            self.add_to_log("🔔 Event handlers registered")

    def start_camera_feed(self):
        """Start the camera feed thread"""
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

        # Start status update thread
        self.status_thread = threading.Thread(target=self.status_update_loop, daemon=True)
        self.status_thread.start()

    def camera_loop(self):
        """Main camera processing loop"""
        while True:
            try:
                if self.robot.vision and hasattr(self.robot.vision, 'get_latest_frame'):
                    frame = self.robot.vision.get_latest_frame()
                    if frame is not None:
                        self.current_frame = frame
                        self.update_video_display(frame)
                        self.fps_counter += 1

                        # Calculate FPS
                        current_time = time.time()
                        if current_time - self.last_fps_time >= 1.0:
                            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
                            self.fps_counter = 0
                            self.last_fps_time = current_time

                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                self.logger.error(f"Camera loop error: {e}")
                time.sleep(1)

    def update_video_display(self, frame):
        """Update the video display with detection overlays"""
        if frame is None:
            return

        display_frame = frame.copy()

        # Add detection overlay if enabled
        if self.detection_overlay:
            self.draw_detection_overlay(display_frame)

        # Add FPS counter
        if self.show_fps:
            cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add robot mode
        mode_text = f"Mode: {self.robot.current_mode.value.upper()}"
        cv2.putText(display_frame, mode_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Add detection status
        if self.dog_detected:
            status_text = f"🐕 Dog Detected - {self.current_behavior.upper()}"
            color = (0, 255, 0)  # Green
        else:
            status_text = "🔍 Searching for dogs..."
            color = (0, 255, 255)  # Yellow

        cv2.putText(display_frame, status_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert to RGB and resize
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        display_frame = cv2.resize(display_frame, (640, 480))

        # Convert to PhotoImage
        image = Image.fromarray(display_frame)
        photo = ImageTk.PhotoImage(image=image)

        # Update display (thread-safe)
        self.root.after(0, self._update_video_label, photo)

    def _update_video_label(self, photo):
        """Thread-safe video label update"""
        if self.video_label:
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo

    def draw_detection_overlay(self, frame):
        """Draw detection information on frame"""
        # Add crosshair in center
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Draw crosshair
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (255, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (255, 255, 255), 1)

        # Draw real detection data if available
        if self.dog_detected and self.latest_detection:
            bbox = self.latest_detection.get('bbox', [200, 150, 200, 180])
            confidence = self.latest_detection.get('confidence', 0.0)

            if len(bbox) >= 4:
                x, y, w, h = bbox
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Draw confidence and behavior labels
                label = f"Dog: {confidence:.1%}"
                if self.current_behavior != "idle":
                    label += f" | {self.current_behavior} ({self.behavior_confidence:.1%})"

                cv2.putText(frame, label, (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Add detection status overlay
        status_text = "🔍 Searching..." if not self.dog_detected else "🐕 Dog Found!"
        cv2.putText(frame, status_text, (10, h - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def status_update_loop(self):
        """Update status displays"""
        while True:
            try:
                self.update_detection_display()
                self.update_robot_status()
                time.sleep(0.5)  # Update every 500ms
            except Exception as e:
                self.logger.error(f"Status update error: {e}")
                time.sleep(1)

    def update_detection_display(self):
        """Update the detection information display"""
        timestamp = datetime.now().strftime('%H:%M:%S')

        info = f"[{timestamp}] Detection Status\n"
        info += f"{'='*30}\n"
        info += f"Dog Detected: {'YES' if self.dog_detected else 'NO'}\n"
        info += f"Current Behavior: {self.current_behavior}\n"
        info += f"Confidence: {self.behavior_confidence:.1%}\n"
        info += f"FPS: {self.current_fps:.1f}\n"
        info += f"Mode: {self.robot.current_mode.value}\n"

        if hasattr(self.robot, 'vision') and self.robot.vision:
            vision_status = self.robot.vision.get_status()
            info += f"Camera: {'Active' if vision_status.get('camera_active') else 'Inactive'}\n"
            info += f"Detector: {vision_status.get('active_detector', 'None')}\n"

        self.root.after(0, self._update_detection_text, info)

    def _update_detection_text(self, text):
        """Thread-safe detection text update"""
        if self.detection_text:
            self.detection_text.delete(1.0, tk.END)
            self.detection_text.insert(1.0, text)

    def update_robot_status(self):
        """Update robot status display"""
        status = self.robot.get_status()
        timestamp = datetime.now().strftime('%H:%M:%S')

        info = f"[{timestamp}] Robot Status\n"
        info += f"{'='*30}\n"
        info += f"Running: {'YES' if status['is_running'] else 'NO'}\n"
        info += f"Emergency: {'ACTIVE' if status['emergency_stop'] else 'Normal'}\n"

        # Hardware status
        info += f"\nHardware Status:\n"
        for component, hw_status in status['hardware'].items():
            if isinstance(hw_status, dict):
                initialized = hw_status.get('initialized', False)
                icon = '✅' if initialized else '❌'
                info += f"{icon} {component.capitalize()}\n"

        self.root.after(0, self._update_status_text, info)

    def _update_status_text(self, text):
        """Thread-safe status text update"""
        if self.status_text:
            self.status_text.delete(1.0, tk.END)
            self.status_text.insert(1.0, text)

    def quick_command(self, command):
        """Execute a quick training command"""
        try:
            if command == 'treat':
                # Manual treat dispense
                if hasattr(self.robot, 'reward_system'):
                    success = self.robot.reward_system.manual_dispense("manual_command")
                    if success:
                        self.add_to_log("🍖 Treat dispensed manually")
                    else:
                        self.add_to_log("❌ Treat dispense failed (cooldown?)")
            else:
                # Play audio command
                if self.robot.hardware.get('audio'):
                    self.robot.hardware['audio'].play_sound(command)
                    self.add_to_log(f"🔊 Played: {command}")

                # If it's GOOD_DOG, also give treat
                if command == 'GOOD_DOG':
                    time.sleep(1)  # Wait for audio
                    self.quick_command('treat')

        except Exception as e:
            self.logger.error(f"Quick command error: {e}")
            self.add_to_log(f"❌ Command failed: {e}")

    def set_robot_mode(self, mode):
        """Set robot operating mode"""
        try:
            self.robot.set_mode(mode)
            self.add_to_log(f"🔧 Mode changed to: {mode.value}")
        except Exception as e:
            self.logger.error(f"Mode change error: {e}")

    def center_camera(self):
        """Center the camera servos"""
        # Use the new center positions with inverted pan
        self.pan_scale.set(120)  # New center pan position (slider)
        self.tilt_scale.set(55)  # Center pitch position
        if self.robot.hardware.get('servos'):
            try:
                # Call servo controller center function
                self.robot.hardware['servos'].center_camera()
                self.add_to_log("📹 Camera centered (Pan: Center, Pitch: 55°)")
            except Exception as e:
                self.logger.error(f"Center camera error: {e}")

    def update_pan(self, value):
        """Update pan servo position (inverted direction)"""
        slider_value = float(value)
        # Invert the direction: 30->180, 120 center becomes 180->30, 120 center
        inverted_angle = 210 - slider_value  # Maps 30->180, 120->90, 180->30

        if self.robot.hardware.get('servos'):
            try:
                # Call servo controller directly with inverted angle
                self.robot.hardware['servos'].set_camera_pan(inverted_angle)
                direction = "Left" if slider_value < 120 else "Right" if slider_value > 120 else "Center"
                self.add_to_log(f"📡 Pan: {direction} (slider:{slider_value}° → servo:{inverted_angle}°)")
            except Exception as e:
                self.logger.error(f"Pan update error: {e}")

    def update_tilt(self, value):
        """Update tilt servo position (pitch)"""
        angle = float(value)
        if self.robot.hardware.get('servos'):
            try:
                # Call servo controller directly with correct method
                self.robot.hardware['servos'].set_camera_pitch(angle)
                self.add_to_log(f"📡 Pitch: {angle}°")
            except Exception as e:
                self.logger.error(f"Tilt update error: {e}")

    def toggle_overlay(self):
        """Toggle detection overlay on/off"""
        self.detection_overlay = not self.detection_overlay
        status = "ON" if self.detection_overlay else "OFF"
        self.add_to_log(f"👀 Detection overlay: {status}")

    def toggle_fps(self):
        """Toggle FPS display"""
        self.show_fps = not self.show_fps
        status = "ON" if self.show_fps else "OFF"
        self.add_to_log(f"📊 FPS display: {status}")

    def emergency_stop(self):
        """Emergency stop all robot systems"""
        self.robot.emergency_stop()
        self.add_to_log("🚨 EMERGENCY STOP ACTIVATED")

    def add_to_log(self, message):
        """Add message to activity log"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)  # Also print to console

    # Event handlers for robot events
    def on_dog_detected(self, event_data):
        """Handle dog detection event"""
        self.dog_detected = True
        detection = event_data.get('detection', {})
        confidence = detection.get('confidence', 0.0)
        self.latest_detection = detection
        self.add_to_log(f"🐕 Dog detected! Confidence: {confidence:.1%}")

    def on_dog_lost(self, event_data):
        """Handle dog lost event"""
        self.dog_detected = False
        self.latest_detection = None
        self.add_to_log("🔍 Dog lost - searching...")

    def on_behavior_detected(self, event_data):
        """Handle behavior detection event"""
        behavior = event_data.get('behavior', 'unknown')
        confidence = event_data.get('confidence', 0.0)

        self.current_behavior = behavior
        self.behavior_confidence = confidence

        self.add_to_log(f"🎯 Behavior: {behavior} ({confidence:.1%})")

    def on_no_detections(self, event_data):
        """Handle no detections event"""
        if self.dog_detected:  # Only log when losing detection
            self.dog_detected = False
            self.latest_detection = None

    def run(self):
        """Start the camera viewer"""
        try:
            # Ensure robot is initialized
            if not self.robot.initialization_successful:
                self.add_to_log("⚠️ Robot initialization incomplete - some features may not work")

            # Check if vision system is available
            if not self.robot.vision:
                self.add_to_log("❌ Vision system not available!")
            else:
                self.add_to_log("✅ Vision system ready")

            # Set robot to tracking mode for detection
            self.robot.set_mode(RobotMode.TRACKING)
            self.add_to_log("🎯 Started in tracking mode")

            # Force start detection if not already started
            if self.robot.vision and hasattr(self.robot.vision, 'start_detection'):
                if not self.robot.vision.camera_active:
                    success = self.robot.vision.start_detection()
                    if success:
                        self.add_to_log("📹 Detection system started")
                    else:
                        self.add_to_log("❌ Failed to start detection system")
                else:
                    self.add_to_log("📹 Detection system already active")

            # Start GUI
            self.root.mainloop()

        except Exception as e:
            self.add_to_log(f"❌ Startup error: {e}")
            self.root.mainloop()  # Still show GUI even if there are errors

    def cleanup(self):
        """Clean up resources"""
        pass

def main():
    """Main function to run camera viewer"""
    try:
        # Initialize robot
        print("🔧 Initializing robot systems...")
        robot = TreatDispenserRobot()

        if not robot.initialization_successful:
            print("❌ Robot initialization failed!")
            print("   Camera viewer will still work with available systems")

        print("🎥 Starting Camera Viewer...")

        # Create and run GUI
        viewer = CameraViewer(robot)
        viewer.run()

    except KeyboardInterrupt:
        print("\n⏹️ Camera viewer interrupted by user")
    except Exception as e:
        print(f"❌ Camera viewer error: {e}")
        logging.error(f"Camera viewer error: {e}")
    finally:
        if 'robot' in locals():
            robot.cleanup()
        if 'viewer' in locals():
            viewer.cleanup()

if __name__ == "__main__":
    main()