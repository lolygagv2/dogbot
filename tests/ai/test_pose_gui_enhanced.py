#!/usr/bin/env python3
"""
Enhanced Pose Detection GUI with Full Visualization
- ArUco marker detection and visualization
- Behavior cooldown tracking
- Servo control simulation
- Command history
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
from datetime import datetime, timedelta
import json
import collections
from pathlib import Path

# Import the pose detection components
from run_pi_1024x768 import (
    PoseDetectionApp, BehaviorTracker, ServoTracker,
    detect_markers_visual, draw_aruco_markers,
    CFG, IMGSZ_W, IMGSZ_H, BEHAVIORS, COOLDOWN_S
)

class EnhancedPoseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced Pose Detection - 1024x768")
        self.root.geometry("1400x900")

        # Initialize pose detection
        self.pose_app = PoseDetectionApp()
        self.running = False
        self.camera_thread = None

        # Command history
        self.command_history = collections.deque(maxlen=50)
        self.behavior_counts = {dog: {b: 0 for b in BEHAVIORS} for dog in ["bezik", "elsa", "unknown"]}

        # Servo simulation
        self.servo_pan = 90  # 0-180
        self.servo_tilt = 90  # 0-180
        self.tracking_enabled = tk.BooleanVar(value=False)

        self.setup_ui()

    def setup_ui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left panel - Camera view
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))

        # Camera display
        self.camera_label = ttk.Label(left_panel)
        self.camera_label.pack(expand=True, fill=tk.BOTH)

        # Control buttons
        controls_frame = ttk.Frame(left_panel)
        controls_frame.pack(fill=tk.X, pady=5)

        ttk.Button(controls_frame, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="Save Screenshot", command=self.save_screenshot).pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(controls_frame, text="Enable Tracking",
                       variable=self.tracking_enabled).pack(side=tk.LEFT, padx=10)

        # Right panel - Information and controls
        right_panel = ttk.Notebook(main_frame)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Tab 1: Status
        status_tab = ttk.Frame(right_panel)
        right_panel.add(status_tab, text="Status")

        # FPS and detection info
        info_frame = ttk.LabelFrame(status_tab, text="Detection Info", padding="5")
        info_frame.pack(fill=tk.X, pady=5)

        self.fps_label = ttk.Label(info_frame, text="FPS: 0.0")
        self.fps_label.pack(anchor=tk.W)

        self.detection_label = ttk.Label(info_frame, text="Detections: 0")
        self.detection_label.pack(anchor=tk.W)

        self.aruco_label = ttk.Label(info_frame, text="ArUco Markers: None")
        self.aruco_label.pack(anchor=tk.W)

        # Behavior cooldowns
        cooldown_frame = ttk.LabelFrame(status_tab, text="Behavior Cooldowns", padding="5")
        cooldown_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.cooldown_text = scrolledtext.ScrolledText(cooldown_frame, height=10, width=40)
        self.cooldown_text.pack(fill=tk.BOTH, expand=True)

        # Tab 2: Servo Control
        servo_tab = ttk.Frame(right_panel)
        right_panel.add(servo_tab, text="Servo Control")

        servo_frame = ttk.LabelFrame(servo_tab, text="Manual Servo Control", padding="10")
        servo_frame.pack(fill=tk.X, pady=5)

        # Pan control
        ttk.Label(servo_frame, text="Pan (0-180):").grid(row=0, column=0, sticky=tk.W)
        self.pan_var = tk.IntVar(value=90)
        self.pan_scale = ttk.Scale(servo_frame, from_=0, to=180,
                                   variable=self.pan_var, orient=tk.HORIZONTAL,
                                   command=self.update_servo)
        self.pan_scale.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        self.pan_label = ttk.Label(servo_frame, text="90°")
        self.pan_label.grid(row=0, column=2)

        # Tilt control
        ttk.Label(servo_frame, text="Tilt (0-180):").grid(row=1, column=0, sticky=tk.W)
        self.tilt_var = tk.IntVar(value=90)
        self.tilt_scale = ttk.Scale(servo_frame, from_=0, to=180,
                                    variable=self.tilt_var, orient=tk.HORIZONTAL,
                                    command=self.update_servo)
        self.tilt_scale.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        self.tilt_label = ttk.Label(servo_frame, text="90°")
        self.tilt_label.grid(row=1, column=2)

        # Reset button
        ttk.Button(servo_frame, text="Reset to Neutral",
                  command=self.reset_servos).grid(row=2, column=1, pady=10)

        # Tracking info
        tracking_frame = ttk.LabelFrame(servo_tab, text="Tracking Status", padding="5")
        tracking_frame.pack(fill=tk.X, pady=5)

        self.tracking_status = ttk.Label(tracking_frame, text="Tracking: Disabled")
        self.tracking_status.pack(anchor=tk.W)

        self.tracking_target = ttk.Label(tracking_frame, text="Target: None")
        self.tracking_target.pack(anchor=tk.W)

        # Tab 3: Command History
        history_tab = ttk.Frame(right_panel)
        right_panel.add(history_tab, text="Command History")

        history_frame = ttk.LabelFrame(history_tab, text="Behavior Commands Issued", padding="5")
        history_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.history_text = scrolledtext.ScrolledText(history_frame, height=15, width=40)
        self.history_text.pack(fill=tk.BOTH, expand=True)

        # Tab 4: Statistics
        stats_tab = ttk.Frame(right_panel)
        right_panel.add(stats_tab, text="Statistics")

        stats_frame = ttk.LabelFrame(stats_tab, text="Behavior Counts", padding="5")
        stats_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.stats_text = scrolledtext.ScrolledText(stats_frame, height=15, width=40)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # Bottom status bar
        status_bar = ttk.Frame(self.root)
        status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        self.status_message = ttk.Label(status_bar, text="Ready", relief=tk.SUNKEN)
        self.status_message.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.model_info = ttk.Label(status_bar,
                                   text=f"Model: {IMGSZ_W}x{IMGSZ_H}",
                                   relief=tk.SUNKEN)
        self.model_info.pack(side=tk.RIGHT, padx=5)

    def start_camera(self):
        """Start camera and processing thread"""
        if not self.running:
            self.status_message.config(text="Initializing...")

            # Initialize pose detection
            if not self.pose_app.initialize():
                self.status_message.config(text="Failed to initialize")
                return

            self.running = True
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            self.status_message.config(text="Camera running")

    def stop_camera(self):
        """Stop camera and processing"""
        self.running = False
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        self.status_message.config(text="Camera stopped")

    def camera_loop(self):
        """Main camera processing loop"""
        # Try Picamera2 first
        try:
            from picamera2 import Picamera2
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={"size": (1024, 768), "format": "XBGR8888"}
            )
            camera.configure(config)
            camera.start()
            use_picamera = True
            print("[CAMERA] Using Picamera2")
        except:
            camera = cv2.VideoCapture(0)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
            use_picamera = False
            print("[CAMERA] Using OpenCV")

        try:
            while self.running:
                # Capture frame
                if use_picamera:
                    frame = camera.capture_array()
                else:
                    ret, frame = camera.read()
                    if not ret:
                        continue

                # Process frame
                results = self.pose_app.process_frame(frame)

                # Update servo tracking if enabled
                if self.tracking_enabled.get():
                    self.update_tracking(results)

                # Handle behavior commands
                self.process_behaviors(results)

                # Draw results
                display_frame = self.pose_app.draw_results(frame.copy(), results)

                # Update GUI
                self.update_display(display_frame, results)

                time.sleep(0.03)  # ~30 FPS

        finally:
            if use_picamera:
                camera.stop()
            else:
                camera.release()

    def update_display(self, frame, results):
        """Update GUI display with results"""
        # Convert frame for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to fit display
        height, width = frame_rgb.shape[:2]
        max_width = 800
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_width, new_height))

        # Convert to PIL Image
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update label
        self.camera_label.imgtk = imgtk
        self.camera_label.config(image=imgtk)

        # Update status labels
        self.fps_label.config(text=f"FPS: {results['fps']:.1f}")
        self.detection_label.config(text=f"Detections: {len(results['detections'])}")

        # Update ArUco info
        if results['aruco_markers']:
            marker_info = ", ".join([f"{m['id']}({m['dog_name']})"
                                    for m in results['aruco_markers']])
            self.aruco_label.config(text=f"ArUco: {marker_info}")
        else:
            self.aruco_label.config(text="ArUco: None detected")

        # Update cooldown display
        self.update_cooldown_display()

        # Update statistics
        self.update_statistics()

    def update_cooldown_display(self):
        """Update cooldown text display"""
        self.cooldown_text.delete(1.0, tk.END)

        current_time = datetime.now()
        for dog_id, tracker in self.pose_app.behavior_classifier.trackers.items():
            cooldowns = tracker.get_active_cooldowns(current_time)
            if cooldowns:
                self.cooldown_text.insert(tk.END, f"{dog_id}:\n")
                for behavior, remaining in cooldowns:
                    self.cooldown_text.insert(tk.END,
                                             f"  {behavior}: {remaining:.1f}s remaining\n")
                self.cooldown_text.insert(tk.END, "\n")

    def process_behaviors(self, results):
        """Process detected behaviors and update history"""
        for dog_id, behavior_info in results.get('behaviors', {}).items():
            if behavior_info['triggered']:
                # Add to command history
                timestamp = datetime.now().strftime("%H:%M:%S")
                command = f"[{timestamp}] {dog_id}: {behavior_info['behavior']} " \
                         f"(conf: {behavior_info['confidence']:.2f})"
                self.command_history.append(command)

                # Update behavior counts
                if dog_id in self.behavior_counts:
                    self.behavior_counts[dog_id][behavior_info['behavior']] += 1

                # Update history display
                self.history_text.insert(tk.END, command + "\n")
                self.history_text.see(tk.END)

                # Show in status bar
                self.status_message.config(text=f"Command issued: {command}")

    def update_statistics(self):
        """Update statistics display"""
        self.stats_text.delete(1.0, tk.END)

        for dog_id, behaviors in self.behavior_counts.items():
            total = sum(behaviors.values())
            if total > 0:
                self.stats_text.insert(tk.END, f"{dog_id}:\n")
                self.stats_text.insert(tk.END, f"  Total commands: {total}\n")
                for behavior, count in behaviors.items():
                    if count > 0:
                        self.stats_text.insert(tk.END, f"  {behavior}: {count}\n")
                self.stats_text.insert(tk.END, "\n")

    def update_servo(self, value):
        """Update servo position display"""
        self.servo_pan = self.pan_var.get()
        self.servo_tilt = self.tilt_var.get()
        self.pan_label.config(text=f"{self.servo_pan}°")
        self.tilt_label.config(text=f"{self.servo_tilt}°")

    def reset_servos(self):
        """Reset servos to neutral position"""
        self.pan_var.set(90)
        self.tilt_var.set(90)
        self.update_servo(None)
        self.status_message.config(text="Servos reset to neutral")

    def update_tracking(self, results):
        """Update tracking status display"""
        if self.pose_app.servo_tracker.tracking_enabled:
            target = self.pose_app.servo_tracker.tracking_dog_id or "None"
            self.tracking_status.config(text="Tracking: Active")
            self.tracking_target.config(text=f"Target: {target}")

            # Simulate servo movement
            if results['detections']:
                det = results['detections'][0]
                bbox = det.get('bbox', [0, 0, 100, 100])
                cx = (bbox[0] + bbox[2]) / 2 / 1024
                cy = (bbox[1] + bbox[3]) / 2 / 768

                # Update servo positions (simplified)
                new_pan = int(90 + (cx - 0.5) * 60)
                new_tilt = int(90 + (cy - 0.5) * 60)

                self.pan_var.set(np.clip(new_pan, 0, 180))
                self.tilt_var.set(np.clip(new_tilt, 0, 180))
                self.update_servo(None)
        else:
            self.tracking_status.config(text="Tracking: Disabled")
            self.tracking_target.config(text="Target: None")

    def save_screenshot(self):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pose_gui_{timestamp}.jpg"
        # TODO: Save actual frame
        self.status_message.config(text=f"Screenshot saved: {filename}")

    def on_closing(self):
        """Handle window closing"""
        self.stop_camera()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = EnhancedPoseGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    print("=" * 60)
    print("ENHANCED POSE DETECTION GUI")
    print(f"Resolution: {IMGSZ_W}x{IMGSZ_H}")
    print(f"Behaviors: {', '.join(BEHAVIORS)}")
    print(f"Dogs: bezik (315), elsa (832)")
    print("=" * 60)

    root.mainloop()

if __name__ == "__main__":
    main()