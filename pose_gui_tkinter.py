#!/usr/bin/env python3
"""
Tkinter-based GUI for pose detection with live camera feed
This will display properly on HDMI monitor
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import sys
import threading
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

class PoseDetectionGUI:
    def __init__(self):
        """Initialize the GUI application"""
        self.root = tk.Tk()
        self.root.title("YOLOv11 Pose Detection - Live View")
        self.root.geometry("900x700")

        # Initialize pose detector
        from core.pose_detector import PoseDetector
        print("Initializing Pose Detector...")
        self.detector = PoseDetector()

        if not self.detector.initialize():
            print("Failed to initialize detector")
            sys.exit(1)

        # Camera setup
        self.camera = None
        self.setup_camera()

        # GUI state
        self.running = True
        self.show_keypoints = True
        self.show_skeleton = False
        self.frame_count = 0
        self.start_time = time.time()

        # Setup GUI elements
        self.setup_gui()

        # Start update loop
        self.update_frame()

    def setup_camera(self):
        """Initialize camera"""
        try:
            from picamera2 import Picamera2
            print("Using Picamera2")
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 480)}
            )
            self.camera.configure(config)
            self.camera.start()
            self.use_picamera = True
        except:
            print("Using OpenCV camera")
            self.camera = cv2.VideoCapture(0)
            self.use_picamera = False

    def setup_gui(self):
        """Create GUI elements"""
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Video frame
        video_frame = ttk.LabelFrame(main_frame, text="Live Camera Feed")
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Canvas for video
        self.canvas = tk.Canvas(video_frame, width=640, height=480, bg='black')
        self.canvas.pack(padx=5, pady=5)

        # Control frame
        control_frame = ttk.LabelFrame(main_frame, text="Controls", width=200)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Stats frame
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        # Stats labels
        self.fps_label = ttk.Label(stats_frame, text="FPS: 0.0")
        self.fps_label.pack(anchor=tk.W, padx=5, pady=2)

        self.detection_label = ttk.Label(stats_frame, text="Detections: 0")
        self.detection_label.pack(anchor=tk.W, padx=5, pady=2)

        self.inference_label = ttk.Label(stats_frame, text="Inference: 0ms")
        self.inference_label.pack(anchor=tk.W, padx=5, pady=2)

        # Options frame
        options_frame = ttk.LabelFrame(control_frame, text="Display Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        # Checkboxes
        self.keypoints_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Show Keypoints",
            variable=self.keypoints_var,
            command=self.toggle_keypoints
        ).pack(anchor=tk.W, padx=5, pady=2)

        self.skeleton_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Show Skeleton",
            variable=self.skeleton_var,
            command=self.toggle_skeleton
        ).pack(anchor=tk.W, padx=5, pady=2)

        self.bbox_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Show Bounding Box",
            variable=self.bbox_var
        ).pack(anchor=tk.W, padx=5, pady=2)

        # Behavior frame
        behavior_frame = ttk.LabelFrame(control_frame, text="Detected Behaviors")
        behavior_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.behavior_text = tk.Text(behavior_frame, height=8, width=25)
        self.behavior_text.pack(padx=5, pady=5)

        # Buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(
            button_frame,
            text="Save Screenshot",
            command=self.save_screenshot
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            button_frame,
            text="Exit",
            command=self.quit
        ).pack(side=tk.RIGHT, padx=2)

    def get_frame(self):
        """Get frame from camera"""
        if self.use_picamera:
            return self.camera.capture_array()
        else:
            ret, frame = self.camera.read()
            return frame if ret else None

    def draw_pose_on_frame(self, frame, detections, behaviors):
        """Draw pose detections on frame"""
        for det in detections:
            x1, y1, x2, y2 = det.bbox.astype(int)

            # Draw bounding box
            if self.bbox_var.get():
                color = (0, 255, 0) if det.dog_id else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw confidence
                conf_text = f"{det.confidence:.2f}"
                cv2.putText(frame, conf_text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw dog ID and behavior
                if det.dog_id and det.dog_id in behaviors:
                    beh_info = behaviors[det.dog_id]
                    label = f"{det.dog_id}: {beh_info['behavior']}"
                    cv2.putText(frame, label, (x1, y1 - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw keypoints
            if self.show_keypoints:
                for i in range(24):
                    x, y, conf = det.keypoints[i]
                    if conf > 0.3:
                        # Color based on confidence
                        if conf > 0.7:
                            kp_color = (0, 255, 0)  # Green
                        elif conf > 0.5:
                            kp_color = (0, 255, 255)  # Yellow
                        else:
                            kp_color = (0, 0, 255)  # Red

                        cv2.circle(frame, (int(x), int(y)), 4, kp_color, -1)

                        # Draw keypoint index
                        cv2.putText(frame, str(i), (int(x)+5, int(y)-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, kp_color, 1)

            # Draw skeleton connections (if enabled)
            if self.show_skeleton and self.skeleton_var.get():
                # Define skeleton connections for dog (simplified)
                connections = [
                    (0, 1), (1, 2), (2, 3),  # Head to tail
                    (3, 4), (4, 5),  # Front legs
                    (3, 6), (6, 7),  # Back legs
                ]

                for start, end in connections:
                    if start < 24 and end < 24:
                        x1, y1, c1 = det.keypoints[start]
                        x2, y2, c2 = det.keypoints[end]

                        if c1 > 0.3 and c2 > 0.3:
                            cv2.line(frame,
                                    (int(x1), int(y1)),
                                    (int(x2), int(y2)),
                                    (255, 255, 0), 2)

        return frame

    def update_frame(self):
        """Update video frame"""
        if not self.running:
            return

        # Get frame
        frame = self.get_frame()

        if frame is not None:
            # Process with pose detector
            result = self.detector.process_frame(frame)

            # Draw detections
            frame = self.draw_pose_on_frame(
                frame,
                result['detections'],
                result['behaviors']
            )

            # Update stats
            self.frame_count += 1
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0

            self.fps_label.config(text=f"FPS: {fps:.1f}")
            self.detection_label.config(text=f"Detections: {len(result['detections'])}")
            self.inference_label.config(text=f"Inference: {result.get('inference_time', 0):.1f}ms")

            # Update behavior text with debug info
            self.behavior_text.delete(1.0, tk.END)

            # Show detection info even without behaviors
            for det in result['detections']:
                text = f"Detection {det.confidence:.2f}\n"
                if det.dog_id:
                    text += f"Dog ID: {det.dog_id}\n"
                else:
                    text += "No Dog ID (no ArUco)\n"

                # Check keypoint quality
                valid_kpts = sum(1 for kp in det.keypoints if kp[2] > 0.3)
                text += f"Valid keypoints: {valid_kpts}/24\n"

                # Show behavior if available
                if det.dog_id and det.dog_id in result['behaviors']:
                    beh_info = result['behaviors'][det.dog_id]
                    text += f"Behavior: {beh_info['behavior']}\n"
                    text += f"Confidence: {beh_info['confidence']:.2f}\n"
                else:
                    text += "No behavior (need ArUco ID)\n"

                text += "-" * 20 + "\n"
                self.behavior_text.insert(tk.END, text)

            # Convert frame to RGB (OpenCV uses BGR)
            if len(frame.shape) == 3:
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                elif frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PIL Image and then to PhotoImage
            image = Image.fromarray(frame)
            photo = ImageTk.PhotoImage(image=image)

            # Update canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo  # Keep a reference

        # Schedule next update
        self.root.after(30, self.update_frame)  # ~33 FPS

    def toggle_keypoints(self):
        """Toggle keypoint display"""
        self.show_keypoints = self.keypoints_var.get()

    def toggle_skeleton(self):
        """Toggle skeleton display"""
        self.show_skeleton = self.skeleton_var.get()

    def save_screenshot(self):
        """Save current frame"""
        frame = self.get_frame()
        if frame is not None:
            filename = f"pose_screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved screenshot: {filename}")

    def quit(self):
        """Clean shutdown"""
        self.running = False

        # Cleanup
        if self.use_picamera:
            self.camera.stop()
        else:
            self.camera.release()

        self.detector.cleanup()
        self.root.quit()

    def run(self):
        """Start the GUI"""
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit()

if __name__ == "__main__":
    print("Starting Pose Detection GUI...")
    print("This should display on your HDMI monitor!")

    app = PoseDetectionGUI()
    app.run()