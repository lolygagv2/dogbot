#!/usr/bin/env python3
"""
Camera Viewer with AI Detection for TreatSensei
Uses the new TreatSenseiCore and AIController with InferVStreams API
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
from typing import Optional, List, Dict
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import the new architecture components
from main import TreatSenseiCore
from core.hardware.led_controller import LEDMode
from core.hardware.motor_controller import MotorDirection

# Try to import Picamera2 for Raspberry Pi camera
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("⚠️ Picamera2 not available - will use OpenCV camera instead")

class CameraViewerAI:
    """Camera viewer with real AI detection using InferVStreams API"""

    def __init__(self):
        """Initialize camera viewer with AI detection"""
        self.root = tk.Tk()
        self.root.title("🐕 TreatSensei AI Camera Viewer")
        self.root.geometry("1100x750")

        # Initialize TreatSensei core system
        print("Initializing TreatSensei Core...")
        self.robot = TreatSenseiCore()

        # Camera setup
        self.camera = None
        self.camera_active = False
        self.current_frame = None
        self.frame_lock = threading.Lock()

        # Detection state
        self.detection_overlay = True
        self.latest_detections = []
        self.show_fps = True
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        # Detection statistics
        self.total_detections = 0
        self.detection_history = []

        # Behavior tracking
        self.current_behavior = "unknown"
        self.previous_behavior = "unknown"
        self.behavior_changes = []
        self.behavior_start_time = time.time()
        self.show_behavior_alerts = True

        # UI elements
        self.video_label = None
        self.status_label = None
        self.detection_text = None
        self.ai_status_text = None
        self.behavior_text = None
        self.pan_scale = None
        self.tilt_scale = None

        # Setup GUI
        self.setup_gui()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('CameraViewerAI')

        # Start camera and detection threads
        self.running = True
        self.start_camera()
        self.start_detection_thread()

        # Set up behavior change callback
        if self.robot.ai and self.robot.ai.is_initialized():
            self.robot.ai.set_behavior_change_callback(self.on_behavior_change)

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

        tk.Label(title_frame, text="🎥 Live Camera Feed with AI Detection",
                font=('Arial', 14, 'bold')).pack(side='left')

        # Status indicator
        self.camera_status = tk.Label(title_frame, text="⚫ INITIALIZING",
                                     font=('Arial', 10, 'bold'), fg='gray')
        self.camera_status.pack(side='right')

        # Video display
        self.video_label = tk.Label(camera_frame, bg='black', text="📷 Starting Camera...",
                                   font=('Arial', 16), fg='white')
        self.video_label.pack(padx=10, pady=10)

        # Camera controls
        cam_controls = tk.Frame(camera_frame)
        cam_controls.pack(pady=5)

        tk.Button(cam_controls, text="📹 Restart Camera",
                 command=self.restart_camera, bg='lightblue').pack(side='left', padx=5)
        tk.Button(cam_controls, text="👀 Toggle Overlay",
                 command=self.toggle_overlay, bg='lightyellow').pack(side='left', padx=5)
        tk.Button(cam_controls, text="📊 Toggle FPS",
                 command=self.toggle_fps, bg='lightgreen').pack(side='left', padx=5)
        tk.Button(cam_controls, text="💾 Save Frame",
                 command=self.save_frame, bg='lightcyan').pack(side='left', padx=5)

        # Servo controls with sliders (if available)
        if self.robot.servos and self.robot.servos.is_initialized():
            servo_frame = tk.LabelFrame(camera_frame, text="📡 Camera Controls", font=('Arial', 10, 'bold'))
            servo_frame.pack(pady=10, fill='x', padx=10)

            # Pan control slider (inverted direction for intuitive control)
            tk.Label(servo_frame, text="Pan (Left 30° ← 120° Center → 180° Right)").pack()
            self.pan_scale = tk.Scale(servo_frame, from_=30, to=180,
                                     orient='horizontal', command=self.update_pan,
                                     length=300)
            self.pan_scale.set(120)  # Center position
            self.pan_scale.pack(fill='x', padx=20)

            # Pitch control slider
            tk.Label(servo_frame, text="Pitch (Down 30° ↓ 55° Center ↑ 150° Up)").pack()
            self.tilt_scale = tk.Scale(servo_frame, from_=30, to=150,
                                      orient='horizontal', command=self.update_tilt,
                                      length=300)
            self.tilt_scale.set(55)  # Center position
            self.tilt_scale.pack(fill='x', padx=20)

            # Center button
            tk.Button(servo_frame, text="📹 Center Camera",
                     command=self.center_camera, bg='lightblue').pack(pady=5)

        # Right side - AI Detection and controls
        control_frame = tk.Frame(main_frame, relief='ridge', bd=2)
        control_frame.pack(side='right', fill='y', padx=(10,0))

        # AI Status
        ai_frame = tk.LabelFrame(control_frame, text="🤖 AI Status", font=('Arial', 12, 'bold'))
        ai_frame.pack(fill='x', padx=10, pady=5)

        self.ai_status_text = tk.Text(ai_frame, height=6, width=40,
                                      font=('Courier', 9), bg='black', fg='cyan')
        self.ai_status_text.pack(fill='x', padx=5, pady=5)

        # Behavior Detection Display
        behavior_frame = tk.LabelFrame(control_frame, text="🐕 Behavior Detection", font=('Arial', 12, 'bold'))
        behavior_frame.pack(fill='x', padx=10, pady=5)

        self.behavior_text = tk.Text(behavior_frame, height=8, width=40,
                                    font=('Courier', 9), bg='darkgreen', fg='white')
        self.behavior_text.pack(fill='x', padx=5, pady=5)

        # Detection info
        detection_frame = tk.LabelFrame(control_frame, text="🔍 Detection Results", font=('Arial', 12, 'bold'))
        detection_frame.pack(fill='x', padx=10, pady=5)

        self.detection_text = tk.Text(detection_frame, height=8, width=40,
                                     font=('Courier', 9), bg='navy', fg='lime')
        self.detection_text.pack(fill='x', padx=5, pady=5)

        # Robot controls
        robot_frame = tk.LabelFrame(control_frame, text="🎮 Robot Controls", font=('Arial', 12, 'bold'))
        robot_frame.pack(fill='x', padx=10, pady=5)

        # Behavior alert controls
        tk.Button(robot_frame, text="🔔 Toggle Behavior Alerts",
                 command=self.toggle_behavior_alerts, width=20, font=('Arial', 9)).pack(pady=2, fill='x', padx=5)

        # Quick action buttons
        actions = [
            ('🔊 Good Dog', lambda: self.robot_action('audio', 'good')),
            ('🔊 No', lambda: self.robot_action('audio', 'no')),
            ('💡 LED Search', lambda: self.robot_action('led', 'searching')),
            ('💡 LED Track', lambda: self.robot_action('led', 'tracking')),
            ('🎯 Test Sequence', self.run_test_sequence)
        ]

        for text, command in actions:
            btn = tk.Button(robot_frame, text=text, command=command,
                           width=15, font=('Arial', 9))
            btn.pack(pady=2, fill='x', padx=5)

        # Detection threshold controls
        threshold_frame = tk.LabelFrame(control_frame, text="⚙️ Detection Settings", font=('Arial', 12, 'bold'))
        threshold_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(threshold_frame, text="Confidence Threshold:").pack()
        self.conf_scale = tk.Scale(threshold_frame, from_=0.1, to=0.9,
                                  orient='horizontal', resolution=0.05)
        if self.robot.ai and self.robot.ai.is_initialized():
            self.conf_scale.set(self.robot.ai.current_conf_threshold)
        else:
            self.conf_scale.set(0.5)
        self.conf_scale.pack(fill='x', padx=10)

        # Emergency stop
        emergency_frame = tk.Frame(control_frame)
        emergency_frame.pack(fill='x', padx=10, pady=10)

        tk.Button(emergency_frame, text="🚨 EMERGENCY STOP",
                 command=self.emergency_stop,
                 bg='red', fg='white', font=('Arial', 12, 'bold'),
                 width=20, height=2).pack()

    def start_camera(self):
        """Initialize and start the camera"""
        try:
            if PICAMERA_AVAILABLE:
                # Use Picamera2 for Raspberry Pi
                print("📷 Starting Picamera2...")
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                self.camera_active = True
                self.camera_type = "Picamera2"
                print("✅ Picamera2 started successfully")
            else:
                # Fallback to OpenCV
                print("📷 Starting OpenCV camera...")
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    raise Exception("Failed to open camera")
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera_active = True
                self.camera_type = "OpenCV"
                print("✅ OpenCV camera started successfully")

            # Update status
            self.root.after(0, lambda: self.camera_status.config(text="🟢 LIVE", fg='green'))

            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()

        except Exception as e:
            self.logger.error(f"Camera initialization error: {e}")
            self.root.after(0, lambda: self.camera_status.config(text="🔴 ERROR", fg='red'))
            print(f"❌ Camera initialization failed: {e}")

    def camera_loop(self):
        """Main camera capture loop with error recovery"""
        consecutive_errors = 0
        max_errors = 10

        while self.running:
            try:
                # Check if camera is active
                if not self.camera_active:
                    time.sleep(0.5)
                    continue

                # Check if camera exists
                if not self.camera:
                    print("⚠️ Camera not initialized, attempting to restart...")
                    self.start_camera()
                    time.sleep(1)
                    continue

                # Capture frame based on camera type
                frame = None
                if PICAMERA_AVAILABLE and self.camera:
                    try:
                        # Capture from Picamera2
                        frame = self.camera.capture_array()
                        # Convert RGB to BGR for OpenCV compatibility
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    except Exception as e:
                        self.logger.error(f"Picamera2 capture error: {e}")
                        consecutive_errors += 1
                elif self.camera:
                    try:
                        # Capture from OpenCV
                        ret, frame = self.camera.read()
                        if not ret:
                            consecutive_errors += 1
                            if consecutive_errors > 5:
                                print("⚠️ Camera read failed multiple times")
                            time.sleep(0.1)
                            continue
                    except Exception as e:
                        self.logger.error(f"OpenCV capture error: {e}")
                        consecutive_errors += 1
                else:
                    time.sleep(0.1)
                    continue

                # If we got a frame, process it
                if frame is not None:
                    # Store frame for processing
                    with self.frame_lock:
                        self.current_frame = frame

                    # Update FPS counter
                    self.fps_counter += 1
                    current_time = time.time()
                    if current_time - self.last_fps_time >= 1.0:
                        self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
                        self.fps_counter = 0
                        self.last_fps_time = current_time

                    # Display frame
                    self.update_video_display(frame)

                    # Reset error counter on successful frame
                    consecutive_errors = 0

                # Control frame rate (~30 FPS)
                time.sleep(0.033)

                # Check for too many errors - but don't auto-restart to prevent loops
                if consecutive_errors >= max_errors:
                    print(f"❌ Too many camera errors ({consecutive_errors}) - manual restart required")
                    self.root.after(0, lambda: self.camera_status.config(text="🔴 ERROR", fg='red'))
                    # Don't auto-restart to prevent loops - user can manually restart
                    consecutive_errors = 0
                    time.sleep(5)  # Wait longer before retrying

            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Camera loop error #{consecutive_errors}: {e}")
                if consecutive_errors >= max_errors:
                    print("❌ Camera loop crashed, attempting recovery...")
                    time.sleep(2)
                    consecutive_errors = 0
                else:
                    time.sleep(0.5)

    def start_detection_thread(self):
        """Start the AI detection thread"""
        if self.robot.ai and self.robot.ai.is_initialized():
            self.detection_thread = threading.Thread(target=self.detection_loop, daemon=True)
            self.detection_thread.start()
            print("🤖 AI detection thread started")
        else:
            print("⚠️ AI not available - detection disabled")
            self.root.after(0, lambda: self.update_ai_status("AI Not Available", "red"))

    def detection_loop(self):
        """AI detection processing loop - continuous detection without locking"""
        consecutive_errors = 0

        while self.running:
            try:
                if not self.robot.ai or not self.robot.ai.is_initialized():
                    time.sleep(1)
                    continue

                # Get current frame
                with self.frame_lock:
                    if self.current_frame is None:
                        time.sleep(0.1)
                        continue
                    frame = self.current_frame.copy()

                # Run AI detection
                start_time = time.time()
                detections = self.robot.ai.detect_objects(frame)
                inference_time = time.time() - start_time

                # Always update detections (even if empty) to avoid locking on old detection
                self.latest_detections = detections if detections else []

                # Track detection statistics and log when dogs are found
                if detections and len(detections) > 0:
                    self.total_detections += len(detections)
                    self.detection_history.append({
                        'time': datetime.now(),
                        'count': len(detections),
                        'inference_ms': inference_time * 1000
                    })
                    self.detection_history = self.detection_history[-10:]

                    # Log detection with behavior info
                    for i, det in enumerate(detections):
                        behavior = det.get('behavior', 'unknown')
                        conf = det.get('confidence', 0)
                        print(f"🐕 Dog {i+1}: {behavior} (confidence: {conf:.2f})")
                elif len(detections) == 0:
                    # Clear behavior when no dogs detected
                    self.current_behavior = "no_dog"
                    print("👻 No dogs detected in frame")

                # Update AI status display
                self.update_ai_status_display(inference_time, detections)

                # Reset error counter on successful inference
                consecutive_errors = 0

                # Rate limit detection (aim for ~10 FPS)
                time.sleep(0.1)

            except Exception as e:
                consecutive_errors += 1
                self.logger.error(f"Detection loop error #{consecutive_errors}: {e}")

                # Clear detections on error to avoid stale data
                self.latest_detections = []

                # Back off on repeated errors
                if consecutive_errors > 5:
                    print("⚠️ Multiple detection errors, pausing detection for 5 seconds")
                    time.sleep(5)
                    consecutive_errors = 0
                else:
                    time.sleep(1)

    def update_video_display(self, frame):
        """Update the video display with detection overlays"""
        if frame is None:
            return

        display_frame = frame.copy()

        # Add detection overlay if enabled
        if self.detection_overlay and self.latest_detections:
            self.draw_detection_overlay(display_frame)

        # Add FPS counter
        if self.show_fps:
            cv2.putText(display_frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Add AI status
        ai_status = "AI: Active" if (self.robot.ai and self.robot.ai.is_initialized()) else "AI: Inactive"
        ai_color = (0, 255, 0) if self.robot.ai and self.robot.ai.is_initialized() else (0, 0, 255)
        cv2.putText(display_frame, ai_status, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ai_color, 2)

        # Add detection count and behavior
        detection_text = f"Dogs: {len(self.latest_detections)}"
        cv2.putText(display_frame, detection_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Add current behavior
        if self.latest_detections and len(self.latest_detections) > 0:
            behavior = self.latest_detections[0].get('behavior', 'unknown')
            behavior_text = f"Behavior: {behavior.replace('_', ' ').title()}"
            cv2.putText(display_frame, behavior_text, (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Convert to RGB and resize for display
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
        """Draw AI detection results with behavior information on frame"""
        for detection in self.latest_detections:
            bbox = detection.get('bbox', (0, 0, 0, 0))
            confidence = detection.get('confidence', 0.0)
            class_name = detection.get('class', 'unknown')
            behavior = detection.get('behavior', 'unknown')

            if len(bbox) >= 4:
                x, y, w, h = bbox

                # Choose color based on behavior
                behavior_colors = {
                    'sitting': (0, 255, 0),      # Green
                    'standing': (255, 255, 0),   # Yellow
                    'lying_down': (255, 0, 0),   # Red
                    'walking': (0, 255, 255),    # Cyan
                    'running': (255, 0, 255),    # Magenta
                    'staying_sit': (0, 200, 0),  # Dark green
                    'staying_stand': (200, 200, 0), # Dark yellow
                    'lying_to_sit': (128, 255, 0), # Lime (transitional)
                    'sit_to_stand': (255, 128, 0), # Orange (transitional)
                    'no_dog': (128, 128, 128),   # Gray (no detection)
                }
                color = behavior_colors.get(behavior, (255, 255, 255))  # White default

                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

                # Draw detection label
                detection_label = f"{class_name}: {confidence:.2f}"
                detection_size, _ = cv2.getTextSize(detection_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Draw behavior label
                behavior_label = f"Behavior: {behavior.replace('_', ' ').title()}"
                behavior_size, _ = cv2.getTextSize(behavior_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Draw label backgrounds
                cv2.rectangle(frame, (x, y - detection_size[1] - behavior_size[1] - 6),
                            (x + max(detection_size[0], behavior_size[0]), y), color, -1)

                # Draw label texts
                cv2.putText(frame, detection_label, (x, y - behavior_size[1] - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, behavior_label, (x, y - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def update_ai_status_display(self, inference_time, detections):
        """Update AI status display"""
        if not self.robot.ai:
            return

        try:
            status = self.robot.ai.get_status()
        except Exception as e:
            self.logger.error(f"AI status error: {e}")
            status = {'initialized': False, 'model_loaded': False, 'inference_count': 0}

        timestamp = datetime.now().strftime('%H:%M:%S')
        info = f"[{timestamp}] AI System Status\n"
        info += f"{'='*35}\n"
        info += f"Status: {'Active' if status.get('initialized', False) else 'Inactive'}\n"
        info += f"Model Loaded: {'Yes' if status.get('model_loaded', False) else 'No'}\n"
        info += f"Inference: {inference_time*1000:.1f}ms\n"

        # Handle different status formats safely
        avg_time = status.get('avg_inference_time', '0s')
        if isinstance(avg_time, str):
            info += f"Avg Time: {avg_time}\n"
        else:
            info += f"Avg Time: {avg_time*1000:.1f}ms\n"

        info += f"Total Inferences: {status.get('inference_count', 0)}\n"

        self.root.after(0, self._update_ai_status_text, info)

        # Update detection text
        det_info = f"[{timestamp}] Detections\n"
        det_info += f"{'='*35}\n"
        det_info += f"Current Frame: {len(detections)} dogs\n"
        det_info += f"Total Detections: {self.total_detections}\n\n"

        if detections:
            det_info += "Current Detections:\n"
            for i, det in enumerate(detections, 1):
                behavior = det.get('behavior', 'unknown')
                det_info += f"{i}. {det['class']} ({det['confidence']:.2f})\n"
                det_info += f"   Behavior: {behavior.replace('_', ' ').title()}\n"
                det_info += f"   Box: {det['bbox']}\n"
        else:
            det_info += "No dogs detected\n"

        if self.detection_history:
            det_info += f"\nRecent History ({len(self.detection_history)} frames):\n"
            for hist in self.detection_history[-3:]:
                det_info += f"  {hist['time'].strftime('%H:%M:%S')}: {hist['count']} dogs ({hist['inference_ms']:.1f}ms)\n"

        self.root.after(0, self._update_detection_text, det_info)

        # Update behavior display
        self.update_behavior_display(detections)

    def _update_ai_status_text(self, text):
        """Thread-safe AI status text update"""
        if self.ai_status_text:
            self.ai_status_text.delete(1.0, tk.END)
            self.ai_status_text.insert(1.0, text)

    def _update_detection_text(self, text):
        """Thread-safe detection text update"""
        if self.detection_text:
            self.detection_text.delete(1.0, tk.END)
            self.detection_text.insert(1.0, text)

    def _update_behavior_text(self, text):
        """Thread-safe behavior text update"""
        if self.behavior_text:
            self.behavior_text.delete(1.0, tk.END)
            self.behavior_text.insert(1.0, text)

    def update_behavior_display(self, detections):
        """Update behavior information display"""
        timestamp = datetime.now().strftime('%H:%M:%S')

        behavior_info = f"[{timestamp}] Behavior Analysis\n"
        behavior_info += f"{'='*35}\n"

        if detections and len(detections) > 0:
            detection = detections[0]
            behavior = detection.get('behavior', 'unknown')
            self.current_behavior = behavior

            # Calculate behavior duration
            duration = time.time() - self.behavior_start_time

            behavior_info += f"Current: {behavior.replace('_', ' ').title()}\n"
            behavior_info += f"Duration: {duration:.1f}s\n"
            behavior_info += f"Confidence: {detection.get('confidence', 0):.2f}\n\n"

            # Show recent behavior changes
            if self.behavior_changes:
                behavior_info += "Recent Changes:\n"
                for change in self.behavior_changes[-5:]:  # Last 5 changes
                    change_time = change['timestamp'].strftime('%H:%M:%S')
                    behavior_info += f"  {change_time}: {change['from']} → {change['to']}\n"
        else:
            behavior_info += "No dog detected\n"
            if self.current_behavior != "no_dog":
                self.current_behavior = "no_dog"

        self.root.after(0, self._update_behavior_text, behavior_info)

    def on_behavior_change(self, change_data):
        """Handle behavior change notification from AI controller"""
        previous = change_data['previous']
        current = change_data['current']
        timestamp = datetime.fromtimestamp(change_data['timestamp'])

        # Record the change
        self.behavior_changes.append({
            'from': previous,
            'to': current,
            'timestamp': timestamp
        })

        # Keep only last 20 changes
        self.behavior_changes = self.behavior_changes[-20:]

        # Update behavior start time
        self.behavior_start_time = time.time()

        # Print and optionally alert
        change_msg = f"🐕 Behavior Change: {previous} → {current}"
        print(change_msg)

        if self.show_behavior_alerts:
            # Flash LED if available
            if self.robot.leds and self.robot.leds.is_initialized():
                threading.Thread(target=self._flash_behavior_alert, daemon=True).start()

            # Play audio cue if available
            if self.robot.audio and self.robot.audio.is_initialized():
                # Play different sounds for different behaviors
                if current == 'sitting':
                    threading.Thread(target=lambda: self.robot.audio.play_sound("good_dog"), daemon=True).start()
                elif current == 'lying_down':
                    threading.Thread(target=lambda: self.robot.audio.play_sound("door_scan"), daemon=True).start()

        # Update status in GUI
        self.update_status_message(f"Behavior: {current.replace('_', ' ').title()}")

    def _flash_behavior_alert(self):
        """Flash LEDs to indicate behavior change"""
        try:
            original_mode = self.robot.leds.current_mode if hasattr(self.robot.leds, 'current_mode') else None

            # Flash 3 times
            for _ in range(3):
                self.robot.leds.set_mode(LEDMode.TRACKING)
                time.sleep(0.2)
                self.robot.leds.set_mode(LEDMode.IDLE)
                time.sleep(0.2)

            # Restore original mode
            if original_mode:
                self.robot.leds.set_mode(original_mode)

        except Exception as e:
            self.logger.error(f"LED flash error: {e}")

    def update_status_message(self, message):
        """Update status message in GUI"""
        self.root.after(0, lambda: self.camera_status.config(text=f"🟢 {message}", fg='green'))

    def toggle_behavior_alerts(self):
        """Toggle behavior change alerts on/off"""
        self.show_behavior_alerts = not self.show_behavior_alerts
        status = "ON" if self.show_behavior_alerts else "OFF"
        print(f"🔔 Behavior alerts: {status}")
        self.update_status_message(f"Alerts: {status}")

    def update_ai_status(self, message, color='green'):
        """Update AI status indicator"""
        status_colors = {'green': '🟢', 'yellow': '🟡', 'red': '🔴'}
        status_icon = status_colors.get(color, '⚫')
        self.root.after(0, lambda: self.camera_status.config(
            text=f"{status_icon} {message}", fg=color))

    def robot_action(self, system, command):
        """Execute robot action"""
        try:
            if system == 'audio' and self.robot.audio:
                if command == 'good':
                    self.robot.audio.play_sound("good_dog")
                elif command == 'no':
                    self.robot.audio.play_sound("no")
                print(f"🔊 Played: {command}")
            elif system == 'led' and self.robot.leds:
                if command == 'searching':
                    self.robot.leds.set_mode(LEDMode.SEARCHING)
                elif command == 'tracking':
                    self.robot.leds.set_mode(LEDMode.TRACKING)
                print(f"💡 LED mode: {command}")
        except Exception as e:
            self.logger.error(f"Robot action error: {e}")

    def run_test_sequence(self):
        """Run robot test sequence"""
        if self.robot.initialization_successful:
            threading.Thread(target=self.robot.run_basic_test_sequence, daemon=True).start()
            print("🎯 Running test sequence...")

    def update_pan(self, value):
        """Update pan servo position with inverted direction for intuitive control"""
        if not self.robot.servos or not self.robot.servos.is_initialized():
            return

        try:
            slider_value = float(value)
            # Invert the direction: slider left (30) = servo right (180)
            inverted_angle = 210 - slider_value  # Maps 30->180, 120->90, 180->30

            self.robot.servos.set_camera_pan(inverted_angle)
            direction = "Left" if slider_value < 120 else "Right" if slider_value > 120 else "Center"
            # Don't print every update to avoid spam
            if abs(slider_value - 120) < 2 or abs(slider_value - 30) < 2 or abs(slider_value - 180) < 2:
                print(f"📡 Pan: {direction} (slider:{slider_value:.0f}° → servo:{inverted_angle:.0f}°)")
        except Exception as e:
            self.logger.error(f"Pan update error: {e}")

    def update_tilt(self, value):
        """Update tilt/pitch servo position"""
        if not self.robot.servos or not self.robot.servos.is_initialized():
            return

        try:
            angle = float(value)
            self.robot.servos.set_camera_pitch(angle)
            # Don't print every update to avoid spam
            if abs(angle - 55) < 2 or abs(angle - 30) < 2 or abs(angle - 150) < 2:
                print(f"📡 Pitch: {angle:.0f}°")
        except Exception as e:
            self.logger.error(f"Tilt update error: {e}")

    def center_camera(self):
        """Center the camera"""
        if self.robot.servos and self.robot.servos.is_initialized():
            if self.pan_scale:
                self.pan_scale.set(120)  # Center pan position on slider
            if self.tilt_scale:
                self.tilt_scale.set(55)  # Center pitch position
            self.robot.servos.center_camera()
            print("📡 Camera centered")

    def toggle_overlay(self):
        """Toggle detection overlay"""
        self.detection_overlay = not self.detection_overlay
        print(f"👀 Overlay: {'ON' if self.detection_overlay else 'OFF'}")

    def toggle_fps(self):
        """Toggle FPS display"""
        self.show_fps = not self.show_fps
        print(f"📊 FPS: {'ON' if self.show_fps else 'OFF'}")

    def save_frame(self):
        """Save current frame with detections"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"capture_{timestamp}.jpg"

            # Draw detections on frame if any
            save_frame = self.current_frame.copy()
            if self.latest_detections:
                self.draw_detection_overlay(save_frame)

            cv2.imwrite(filename, save_frame)
            print(f"💾 Saved: {filename}")

    def update_confidence_threshold(self, value):
        """Update AI confidence threshold"""
        if self.robot.ai and self.robot.ai.is_initialized():
            self.robot.ai.conf_threshold = float(value)
            print(f"⚙️ Confidence threshold: {value}")

    def restart_camera(self):
        """Restart the camera safely without loops"""
        try:
            print("🔄 Manual camera restart requested...")

            # Update status
            self.root.after(0, lambda: self.camera_status.config(text="🔄 RESTARTING...", fg='orange'))

            # Stop camera capture loop
            old_camera_active = self.camera_active
            self.camera_active = False
            time.sleep(1.0)  # Give more time for threads to stop

            # Clean up old camera thoroughly
            if self.camera:
                try:
                    if PICAMERA_AVAILABLE:
                        self.camera.stop()
                        if hasattr(self.camera, 'close'):
                            self.camera.close()  # Also close the camera
                    else:
                        self.camera.release()
                except Exception as cleanup_error:
                    print(f"⚠️ Camera cleanup warning: {cleanup_error}")

            self.camera = None
            time.sleep(1.5)  # Longer wait before restart

            # Only restart if it was active and system is still running
            if old_camera_active and self.running:
                print("Attempting to reinitialize camera...")
                self.start_camera()
            else:
                print("⚠️ Camera restart skipped")
                self.root.after(0, lambda: self.camera_status.config(text="🔴 STOPPED", fg='red'))

        except Exception as e:
            self.logger.error(f"Camera restart error: {e}")
            print(f"❌ Camera restart failed: {e}")
            self.root.after(0, lambda: self.camera_status.config(text="❌ ERROR", fg='red'))

    def emergency_stop(self):
        """Emergency stop all systems"""
        self.robot.emergency_stop()
        print("🚨 EMERGENCY STOP ACTIVATED")

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        self.camera_active = False

        # Stop camera
        if PICAMERA_AVAILABLE and self.camera:
            self.camera.stop()
        elif self.camera:
            self.camera.release()

        # Cleanup robot
        if self.robot:
            self.robot.shutdown()

        print("🧹 Cleanup complete")

    def run(self):
        """Start the camera viewer application"""
        try:
            print("=" * 50)
            print("🐕 TreatSensei AI Camera Viewer")
            print("=" * 50)

            if self.robot.initialization_successful:
                print("✅ Robot systems initialized")
            else:
                print("⚠️ Some robot systems unavailable")

            if self.robot.ai and self.robot.ai.is_initialized():
                print("✅ AI detection system ready")
            else:
                print("⚠️ AI detection not available")

            print("=" * 50)

            # Set up cleanup on window close
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

            # Start GUI
            self.root.mainloop()

        except Exception as e:
            print(f"❌ Application error: {e}")
            self.cleanup()

    def on_closing(self):
        """Handle window closing"""
        print("\n👋 Closing camera viewer...")
        self.cleanup()
        self.root.destroy()

def main():
    """Main entry point"""
    try:
        viewer = CameraViewerAI()
        viewer.run()
    except KeyboardInterrupt:
        print("\n⏹️ Camera viewer interrupted")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        logging.error(f"Fatal error: {e}", exc_info=True)

if __name__ == "__main__":
    main()