#!/usr/bin/env python3
"""
Adaptive Pose GUI with adjustable confidence for different dog types
Better handling of white/fluffy dogs and lying poses
"""

import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

class AdaptivePoseGUI:
    def __init__(self):
        """Initialize the adaptive GUI"""
        self.root = tk.Tk()
        self.root.title("Adaptive Pose Detection - Adjustable Thresholds")
        self.root.geometry("1000x750")

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

        # Adjustable parameters
        self.confidence_threshold = 0.25  # Start low
        self.keypoint_threshold = 0.2  # Lower for scattered keypoints

        # GUI state
        self.running = True
        self.show_all_keypoints = False  # Show even low confidence
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
        """Create GUI elements with threshold controls"""
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
        control_frame = ttk.LabelFrame(main_frame, text="Controls", width=300)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Threshold controls
        threshold_frame = ttk.LabelFrame(control_frame, text="Detection Thresholds")
        threshold_frame.pack(fill=tk.X, padx=5, pady=5)

        # Confidence threshold slider
        ttk.Label(threshold_frame, text="Detection Confidence:").pack(anchor=tk.W, padx=5)
        self.conf_var = tk.DoubleVar(value=0.25)
        self.conf_slider = ttk.Scale(
            threshold_frame,
            from_=0.1, to=0.9,
            variable=self.conf_var,
            orient=tk.HORIZONTAL,
            command=self.update_confidence
        )
        self.conf_slider.pack(fill=tk.X, padx=5, pady=2)
        self.conf_label = ttk.Label(threshold_frame, text="0.25")
        self.conf_label.pack(anchor=tk.W, padx=5)

        # Keypoint threshold slider
        ttk.Label(threshold_frame, text="Keypoint Visibility:").pack(anchor=tk.W, padx=5)
        self.kp_var = tk.DoubleVar(value=0.2)
        self.kp_slider = ttk.Scale(
            threshold_frame,
            from_=0.1, to=0.9,
            variable=self.kp_var,
            orient=tk.HORIZONTAL,
            command=self.update_keypoint_threshold
        )
        self.kp_slider.pack(fill=tk.X, padx=5, pady=2)
        self.kp_label = ttk.Label(threshold_frame, text="0.2")
        self.kp_label.pack(anchor=tk.W, padx=5)

        # Stats frame
        stats_frame = ttk.LabelFrame(control_frame, text="Statistics")
        stats_frame.pack(fill=tk.X, padx=5, pady=5)

        self.fps_label = ttk.Label(stats_frame, text="FPS: 0.0")
        self.fps_label.pack(anchor=tk.W, padx=5, pady=2)

        self.detection_label = ttk.Label(stats_frame, text="Detections: 0")
        self.detection_label.pack(anchor=tk.W, padx=5, pady=2)

        self.inference_label = ttk.Label(stats_frame, text="Inference: 0ms")
        self.inference_label.pack(anchor=tk.W, padx=5, pady=2)

        # Options frame
        options_frame = ttk.LabelFrame(control_frame, text="Display Options")
        options_frame.pack(fill=tk.X, padx=5, pady=5)

        self.show_all_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Show ALL Keypoints (even low conf)",
            variable=self.show_all_var,
            command=self.toggle_show_all
        ).pack(anchor=tk.W, padx=5, pady=2)

        self.bbox_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            options_frame,
            text="Show Bounding Box",
            variable=self.bbox_var
        ).pack(anchor=tk.W, padx=5, pady=2)

        # Detection info
        info_frame = ttk.LabelFrame(control_frame, text="Detection Info")
        info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.info_text = tk.Text(info_frame, height=12, width=35, font=('Courier', 9))
        self.info_text.pack(padx=5, pady=5)

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
            text="Reset Thresholds",
            command=self.reset_thresholds
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            button_frame,
            text="Exit",
            command=self.quit
        ).pack(side=tk.RIGHT, padx=2)

    def update_confidence(self, value):
        """Update detection confidence threshold"""
        self.confidence_threshold = float(value)
        self.conf_label.config(text=f"{self.confidence_threshold:.2f}")

    def update_keypoint_threshold(self, value):
        """Update keypoint visibility threshold"""
        self.keypoint_threshold = float(value)
        self.kp_label.config(text=f"{self.keypoint_threshold:.2f}")

    def toggle_show_all(self):
        """Toggle showing all keypoints"""
        self.show_all_keypoints = self.show_all_var.get()

    def reset_thresholds(self):
        """Reset to default thresholds"""
        self.conf_var.set(0.25)
        self.kp_var.set(0.2)

    def get_frame(self):
        """Get frame from camera"""
        if self.use_picamera:
            return self.camera.capture_array()
        else:
            ret, frame = self.camera.read()
            return frame if ret else None

    def draw_adaptive_pose(self, frame, detections, behaviors):
        """Draw pose with adaptive visualization"""
        for i, det in enumerate(detections):
            # Skip low confidence detections unless threshold is met
            if det.confidence < self.confidence_threshold:
                continue

            x1, y1, x2, y2 = det.bbox.astype(int)

            # Color based on detection confidence
            if det.confidence > 0.5:
                box_color = (0, 255, 0)  # Green - high conf
            elif det.confidence > 0.3:
                box_color = (0, 255, 255)  # Yellow - medium
            else:
                box_color = (0, 0, 255)  # Red - low conf

            # Draw bounding box
            if self.bbox_var.get():
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # Show confidence and detection number
                label = f"D{i} Conf: {det.confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Count valid keypoints
            valid_kpts = 0
            total_conf = 0

            # Draw keypoints with adaptive threshold
            for kp_idx in range(24):
                x, y, conf = det.keypoints[kp_idx]

                # Show keypoint based on threshold or show_all setting
                if self.show_all_keypoints or conf > self.keypoint_threshold:
                    if conf > self.keypoint_threshold:
                        valid_kpts += 1
                        total_conf += conf

                    # Color coding for keypoint confidence
                    if conf > 0.7:
                        kp_color = (0, 255, 0)  # Green
                        kp_size = 5
                    elif conf > 0.4:
                        kp_color = (0, 255, 255)  # Yellow
                        kp_size = 4
                    elif conf > 0.2:
                        kp_color = (0, 165, 255)  # Orange
                        kp_size = 3
                    else:
                        kp_color = (0, 0, 255)  # Red
                        kp_size = 2

                    cv2.circle(frame, (int(x), int(y)), kp_size, kp_color, -1)

                    # Show keypoint index if show_all is enabled
                    if self.show_all_keypoints:
                        cv2.putText(frame, str(kp_idx), (int(x)+6, int(y)-6),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.25, kp_color, 1)

            # Show keypoint quality
            avg_conf = total_conf / valid_kpts if valid_kpts > 0 else 0
            quality_text = f"KPs: {valid_kpts}/24 Avg: {avg_conf:.2f}"
            cv2.putText(frame, quality_text, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

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

            # Draw detections with adaptive visualization
            frame = self.draw_adaptive_pose(
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

            # Update detection info
            self.info_text.delete(1.0, tk.END)
            for i, det in enumerate(result['detections']):
                # Count keypoint quality
                high_conf = sum(1 for kp in det.keypoints if kp[2] > 0.7)
                med_conf = sum(1 for kp in det.keypoints if 0.4 < kp[2] <= 0.7)
                low_conf = sum(1 for kp in det.keypoints if 0.2 < kp[2] <= 0.4)
                very_low = sum(1 for kp in det.keypoints if kp[2] <= 0.2)

                info = f"Detection {i}:\n"
                info += f"  Confidence: {det.confidence:.3f}\n"
                info += f"  Box: {int(det.bbox[2]-det.bbox[0])}x{int(det.bbox[3]-det.bbox[1])}\n"
                info += f"  Keypoints Quality:\n"
                info += f"    High (>0.7): {high_conf}\n"
                info += f"    Med (0.4-0.7): {med_conf}\n"
                info += f"    Low (0.2-0.4): {low_conf}\n"
                info += f"    V.Low (<0.2): {very_low}\n"

                # Check pose based on bbox aspect ratio
                w = det.bbox[2] - det.bbox[0]
                h = det.bbox[3] - det.bbox[1]
                aspect = w / h if h > 0 else 1

                if aspect > 1.5:
                    pose_guess = "Lying?"
                elif aspect < 0.8:
                    pose_guess = "Standing?"
                else:
                    pose_guess = "Sitting?"

                info += f"  Pose (by box): {pose_guess}\n"
                info += f"  Aspect ratio: {aspect:.2f}\n"
                info += "-" * 30 + "\n"

                self.info_text.insert(tk.END, info)

            # Convert frame to RGB
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
        self.root.after(30, self.update_frame)

    def save_screenshot(self):
        """Save current frame with detections"""
        frame = self.get_frame()
        if frame is not None:
            result = self.detector.process_frame(frame)
            frame = self.draw_adaptive_pose(frame, result['detections'], result['behaviors'])
            filename = f"adaptive_pose_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved screenshot: {filename}")

    def quit(self):
        """Clean shutdown"""
        self.running = False
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
    print("Starting Adaptive Pose Detection GUI...")
    print("Adjust thresholds for better detection of white/fluffy dogs")
    print("Lower thresholds help detect lying poses")

    app = AdaptivePoseGUI()
    app.run()