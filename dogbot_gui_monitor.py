#!/usr/bin/env python3
"""
DogBot Visual GUI Monitor & Control System
Real-time visualization of detection, tracking, and control
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, Scale, Canvas, Label, Button, Frame
from PIL import Image, ImageTk
import threading
import queue
import time
import logging
from datetime import datetime
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import yaml

logger = logging.getLogger(__name__)

class DogBotGUI:
    """Visual GUI for DogBot monitoring and control"""
    
    def __init__(self, dogbot_instance=None):
        """Initialize GUI"""
        self.dogbot = dogbot_instance
        self.root = tk.Tk()
        self.root.title("DogBot AI Vision & Control Center")
        self.root.geometry("1400x900")
        
        # Video feed variables
        self.video_frame = None
        self.video_label = None
        self.frame_queue = queue.Queue(maxsize=10)
        
        # Detection visualization
        self.detection_overlay = True
        self.keypoint_overlay = True
        self.tracking_trail = []
        self.max_trail_points = 30
        
        # Stats tracking
        self.detection_history = []
        self.behavior_counts = {}
        self.fps_history = []
        
        # Control variables
        self.camera_params = {}
        self.servo_positions = {'pan': 0, 'tilt': 0}
        
        # Colors for visualization
        self.behavior_colors = {
            'idle': (128, 128, 128),
            'sitting': (0, 255, 0),
            'lying': (0, 0, 255),
            'standing': (255, 255, 0),
            'playing': (255, 0, 255),
            'spinning': (0, 255, 255),
            'jumping': (255, 128, 0),
            'approaching': (128, 255, 0),
            'leaving': (255, 0, 128)
        }
        
        self.setup_gui()
        self.running = True
        
    def setup_gui(self):
        """Setup GUI components"""
        # Main container with tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Vision Monitor
        self.vision_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.vision_frame, text='Vision Monitor')
        self.setup_vision_tab()
        
        # Tab 2: Camera Controls
        self.camera_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.camera_frame, text='Camera Settings')
        self.setup_camera_tab()
        
        # Tab 3: Servo Controls
        self.servo_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.servo_frame, text='Servo Control')
        self.setup_servo_tab()
        
        # Tab 4: Behavior Analytics
        self.analytics_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analytics_frame, text='Analytics')
        self.setup_analytics_tab()
        
        # Tab 5: Manual Control
        self.control_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.control_frame, text='Manual Control')
        self.setup_control_tab()
        
    def setup_vision_tab(self):
        """Setup vision monitoring tab"""
        # Left: Video feed
        video_container = Frame(self.vision_frame)
        video_container.pack(side='left', padx=10, pady=10)
        
        # Video display
        self.video_label = Label(video_container)
        self.video_label.pack()
        
        # Overlay controls
        overlay_frame = Frame(video_container)
        overlay_frame.pack(pady=5)
        
        self.detection_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(overlay_frame, text="Show Detection", 
                       variable=self.detection_var).pack(side='left', padx=5)
        
        self.keypoint_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(overlay_frame, text="Show Keypoints", 
                       variable=self.keypoint_var).pack(side='left', padx=5)
        
        self.trail_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(overlay_frame, text="Show Trail", 
                       variable=self.trail_var).pack(side='left', padx=5)
        
        # Right: Detection info
        info_container = Frame(self.vision_frame)
        info_container.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Current detection info
        Label(info_container, text="Current Detection", font=('Arial', 14, 'bold')).pack()
        
        self.detection_info = tk.Text(info_container, height=8, width=40)
        self.detection_info.pack(pady=5)
        
        # Behavior status
        Label(info_container, text="Detected Behavior", font=('Arial', 14, 'bold')).pack()
        
        self.behavior_label = Label(info_container, text="IDLE", 
                                   font=('Arial', 20, 'bold'), fg='gray')
        self.behavior_label.pack(pady=5)
        
        # Confidence meter
        Label(info_container, text="Confidence", font=('Arial', 12)).pack()
        
        self.confidence_bar = ttk.Progressbar(info_container, length=300, mode='determinate')
        self.confidence_bar.pack(pady=5)
        
        # FPS counter
        self.fps_label = Label(info_container, text="FPS: 0", font=('Arial', 12))
        self.fps_label.pack(pady=5)
        
        # Treat status
        self.treat_status = Label(info_container, text="Last Treat: Never", 
                                font=('Arial', 12))
        self.treat_status.pack(pady=5)
        
    def setup_camera_tab(self):
        """Setup camera controls tab"""
        # Create sliders for camera parameters
        params = [
            ('Brightness', 'brightness', -1.0, 1.0, 0.0),
            ('Contrast', 'contrast', 0.0, 2.0, 1.0),
            ('Saturation', 'saturation', 0.0, 2.0, 1.0),
            ('Sharpness', 'sharpness', 0.0, 16.0, 1.0),
            ('Gain', 'analogue_gain', 1.0, 16.0, 1.0)
        ]
        
        for i, (label, param, min_val, max_val, default) in enumerate(params):
            frame = Frame(self.camera_frame)
            frame.pack(fill='x', padx=20, pady=5)
            
            Label(frame, text=label, width=15).pack(side='left')
            
            slider = Scale(frame, from_=min_val, to=max_val, 
                          orient='horizontal', resolution=0.1,
                          command=lambda v, p=param: self.update_camera_param(p, float(v)))
            slider.set(default)
            slider.pack(side='left', fill='x', expand=True)
            
            value_label = Label(frame, text=f"{default:.1f}", width=5)
            value_label.pack(side='right')
            
            # Store references
            setattr(self, f"{param}_slider", slider)
            setattr(self, f"{param}_label", value_label)
            
        # White balance controls
        wb_frame = Frame(self.camera_frame)
        wb_frame.pack(fill='x', padx=20, pady=10)
        
        Label(wb_frame, text="White Balance Mode:").pack(side='left')
        
        self.wb_var = tk.StringVar(value='auto')
        wb_menu = ttk.Combobox(wb_frame, textvariable=self.wb_var,
                              values=['auto', 'daylight', 'cloudy', 'tungsten', 'fluorescent'])
        wb_menu.pack(side='left', padx=10)
        wb_menu.bind('<<ComboboxSelected>>', self.update_white_balance)
        
        # Auto adjust button
        Button(self.camera_frame, text="Auto Adjust All", 
               command=self.auto_adjust_camera,
               bg='green', fg='white').pack(pady=20)
        
        # Snapshot button
        Button(self.camera_frame, text="Take Snapshot", 
               command=self.take_snapshot).pack(pady=5)
        
    def setup_servo_tab(self):
        """Setup servo control tab"""
        # Pan control
        pan_frame = Frame(self.servo_frame)
        pan_frame.pack(fill='x', padx=20, pady=20)
        
        Label(pan_frame, text="Pan (Left/Right)", font=('Arial', 12, 'bold')).pack()
        
        self.pan_slider = Scale(pan_frame, from_=-90, to=90, 
                               orient='horizontal', length=400,
                               command=self.update_pan)
        self.pan_slider.set(0)
        self.pan_slider.pack()
        
        self.pan_value = Label(pan_frame, text="0°")
        self.pan_value.pack()
        
        # Tilt control
        tilt_frame = Frame(self.servo_frame)
        tilt_frame.pack(fill='x', padx=20, pady=20)
        
        Label(tilt_frame, text="Tilt (Up/Down)", font=('Arial', 12, 'bold')).pack()
        
        self.tilt_slider = Scale(tilt_frame, from_=-45, to=45, 
                                orient='horizontal', length=400,
                                command=self.update_tilt)
        self.tilt_slider.set(0)
        self.tilt_slider.pack()
        
        self.tilt_value = Label(tilt_frame, text="0°")
        self.tilt_value.pack()
        
        # Center button
        Button(self.servo_frame, text="Center All Servos", 
               command=self.center_servos,
               bg='blue', fg='white').pack(pady=20)
        
        # Tracking toggle
        self.tracking_var = tk.BooleanVar()
        ttk.Checkbutton(self.servo_frame, text="Enable Auto Tracking", 
                       variable=self.tracking_var,
                       command=self.toggle_tracking).pack(pady=10)
        
        # Calibration button
        Button(self.servo_frame, text="Calibrate Servos", 
               command=self.calibrate_servos).pack(pady=10)
        
    def setup_analytics_tab(self):
        """Setup behavior analytics tab"""
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6))
        
        # Behavior pie chart
        self.ax1 = self.fig.add_subplot(221)
        self.ax1.set_title('Behavior Distribution')
        
        # Detection timeline
        self.ax2 = self.fig.add_subplot(222)
        self.ax2.set_title('Detection Confidence Over Time')
        
        # FPS graph
        self.ax3 = self.fig.add_subplot(223)
        self.ax3.set_title('FPS Performance')
        
        # Treat history
        self.ax4 = self.fig.add_subplot(224)
        self.ax4.set_title('Treat Dispensing History')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.analytics_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Update button
        Button(self.analytics_frame, text="Update Charts", 
               command=self.update_analytics).pack(pady=10)
        
    def setup_control_tab(self):
        """Setup manual control tab"""
        # Movement controls
        move_frame = Frame(self.control_frame)
        move_frame.pack(pady=20)
        
        Label(move_frame, text="Movement Control", 
              font=('Arial', 14, 'bold')).pack(pady=10)
        
        # Direction pad
        Button(move_frame, text="↑", command=lambda: self.move_robot('forward'),
               width=5, height=2).grid(row=0, column=1)
        Button(move_frame, text="←", command=lambda: self.move_robot('left'),
               width=5, height=2).grid(row=1, column=0)
        Button(move_frame, text="→", command=lambda: self.move_robot('right'),
               width=5, height=2).grid(row=1, column=2)
        Button(move_frame, text="↓", command=lambda: self.move_robot('backward'),
               width=5, height=2).grid(row=2, column=1)
        Button(move_frame, text="⬤", command=lambda: self.move_robot('stop'),
               width=5, height=2, bg='red', fg='white').grid(row=1, column=1)
        
        # Treat controls
        treat_frame = Frame(self.control_frame)
        treat_frame.pack(pady=20)
        
        Label(treat_frame, text="Treat Dispenser", 
              font=('Arial', 14, 'bold')).pack(pady=10)
        
        Button(treat_frame, text="Dispense Treat", 
               command=self.dispense_treat,
               bg='green', fg='white', width=20, height=3).pack(pady=5)
        
        Button(treat_frame, text="Rotate Carousel", 
               command=self.rotate_carousel).pack(pady=5)
        
        # Audio controls
        audio_frame = Frame(self.control_frame)
        audio_frame.pack(pady=20)
        
        Label(audio_frame, text="Audio", 
              font=('Arial', 14, 'bold')).pack(pady=10)
        
        sounds = ['good_dog', 'come_here', 'sit', 'stay', 'play']
        for sound in sounds:
            Button(audio_frame, text=f"Play '{sound}'",
                   command=lambda s=sound: self.play_sound(s)).pack(side='left', padx=5)
                   
        # LED patterns
        led_frame = Frame(self.control_frame)
        led_frame.pack(pady=20)
        
        Label(led_frame, text="LED Patterns", 
              font=('Arial', 14, 'bold')).pack(pady=10)
        
        patterns = ['breathing', 'rainbow', 'celebration', 'searching', 'off']
        for pattern in patterns:
            Button(led_frame, text=pattern.capitalize(),
                   command=lambda p=pattern: self.set_led_pattern(p)).pack(side='left', padx=5)
                   
    def update_video_feed(self, frame: np.ndarray):
        """Update video display with overlays"""
        if frame is None:
            return
            
        display_frame = frame.copy()
        
        # Draw detection overlay
        if self.detection_var.get() and self.dogbot and self.dogbot.current_detection:
            det = self.dogbot.current_detection
            x, y, w, h = det.bbox
            
            # Draw bounding box
            color = self.behavior_colors.get(det.behavior.value, (255, 255, 255))
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw behavior label
            label = f"{det.behavior.value} ({det.confidence:.2f})"
            cv2.putText(display_frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                       
        # Draw keypoints
        if self.keypoint_var.get() and self.dogbot and self.dogbot.current_detection:
            if self.dogbot.current_detection.keypoints is not None:
                for kp in self.dogbot.current_detection.keypoints:
                    cv2.circle(display_frame, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)
                    
        # Draw tracking trail
        if self.trail_var.get() and len(self.tracking_trail) > 1:
            for i in range(1, len(self.tracking_trail)):
                cv2.line(display_frame, self.tracking_trail[i-1], 
                        self.tracking_trail[i], (0, 255, 255), 2)
                        
        # Convert to RGB and resize for display
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        display_frame = cv2.resize(display_frame, (640, 480))
        
        # Convert to PIL Image
        image = Image.fromarray(display_frame)
        photo = ImageTk.PhotoImage(image=image)
        
        # Update label
        self.video_label.configure(image=photo)
        self.video_label.image = photo
        
    def update_detection_info(self):
        """Update detection information display"""
        if self.dogbot and self.dogbot.current_detection:
            det = self.dogbot.current_detection
            
            # Update info text
            info = f"Position: ({det.bbox[0]}, {det.bbox[1]})\n"
            info += f"Size: {det.bbox[2]}x{det.bbox[3]}\n"
            info += f"Confidence: {det.confidence:.3f}\n"
            info += f"Behavior: {det.behavior.value}\n"
            info += f"Time: {datetime.fromtimestamp(det.timestamp).strftime('%H:%M:%S')}"
            
            self.detection_info.delete(1.0, tk.END)
            self.detection_info.insert(1.0, info)
            
            # Update behavior label
            self.behavior_label.config(text=det.behavior.value.upper(),
                                      fg=self._rgb_to_hex(self.behavior_colors.get(det.behavior.value)))
                                      
            # Update confidence bar
            self.confidence_bar['value'] = det.confidence * 100
            
            # Add to tracking trail
            if det.bbox:
                center = (det.bbox[0] + det.bbox[2]//2, 
                         det.bbox[1] + det.bbox[3]//2)
                self.tracking_trail.append(center)
                if len(self.tracking_trail) > self.max_trail_points:
                    self.tracking_trail.pop(0)
                    
    def update_camera_param(self, param: str, value: float):
        """Update camera parameter"""
        if self.dogbot and hasattr(self.dogbot, 'camera'):
            self.dogbot.camera.set_parameter(param, value)
            
        # Update label
        if hasattr(self, f"{param}_label"):
            getattr(self, f"{param}_label").config(text=f"{value:.1f}")
            
    def update_white_balance(self, event=None):
        """Update white balance mode"""
        if self.dogbot and hasattr(self.dogbot, 'camera'):
            self.dogbot.camera.set_parameter('awb_mode', self.wb_var.get())
            
    def auto_adjust_camera(self):
        """Auto adjust camera settings"""
        if self.dogbot and hasattr(self.dogbot, 'camera'):
            self.dogbot.camera.auto_adjust()
            
            # Update sliders to reflect new values
            params = self.dogbot.camera.get_parameters()
            for param, value in params.items():
                if hasattr(self, f"{param}_slider"):
                    getattr(self, f"{param}_slider").set(value)
                    
    def take_snapshot(self):
        """Take camera snapshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/home/morgan/dogbot/snapshots/snapshot_{timestamp}.jpg"
        
        if self.dogbot and hasattr(self.dogbot, 'camera'):
            if self.dogbot.camera.save_snapshot(filename):
                print(f"Snapshot saved to {filename}")
                
    def update_pan(self, value):
        """Update pan servo"""
        angle = float(value)
        self.pan_value.config(text=f"{angle:.0f}°")
        
        if self.dogbot and hasattr(self.dogbot, 'servo_controller'):
            self.dogbot.servo_controller.set_pan_angle(angle)
            
    def update_tilt(self, value):
        """Update tilt servo"""
        angle = float(value)
        self.tilt_value.config(text=f"{angle:.0f}°")
        
        if self.dogbot and hasattr(self.dogbot, 'servo_controller'):
            self.dogbot.servo_controller.set_tilt_angle(angle)
            
    def center_servos(self):
        """Center all servos"""
        self.pan_slider.set(0)
        self.tilt_slider.set(0)
        
        if self.dogbot and hasattr(self.dogbot, 'servo_controller'):
            self.dogbot.servo_controller.center_all()
            
    def toggle_tracking(self):
        """Toggle auto tracking"""
        if self.dogbot:
            self.dogbot.is_tracking = self.tracking_var.get()
            
    def calibrate_servos(self):
        """Open servo calibration"""
        # This would open a separate calibration window
        pass
        
    def update_analytics(self):
        """Update analytics charts"""
        # Clear axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        
        # Update behavior pie chart
        if self.behavior_counts:
            self.ax1.pie(self.behavior_counts.values(), 
                        labels=self.behavior_counts.keys(),
                        autopct='%1.1f%%')
                        
        # Update other charts...
        
        self.canvas.draw()
        
    def move_robot(self, direction: str):
        """Send movement command to robot"""
        if self.dogbot:
            commands = {
                'forward': {'type': 'move_forward', 'duration': 1.0},
                'backward': {'type': 'move_backward', 'duration': 1.0},
                'left': {'type': 'turn_left', 'angle': 45},
                'right': {'type': 'turn_right', 'angle': 45},
                'stop': {'type': 'stop'}
            }
            if direction in commands:
                self.dogbot.add_command(commands[direction])
                
    def dispense_treat(self):
        """Dispense treat command"""
        if self.dogbot:
            self.dogbot.add_command({'type': 'dispense_treat'})
            self.treat_status.config(text=f"Last Treat: {datetime.now().strftime('%H:%M:%S')}")
            
    def rotate_carousel(self):
        """Rotate treat carousel"""
        if self.dogbot:
            self.dogbot.servo_controller.rotate_carousel(1)
            
    def play_sound(self, sound: str):
        """Play sound effect"""
        if self.dogbot:
            self.dogbot.add_command({'type': 'play_sound', 'sound': sound})
            
    def set_led_pattern(self, pattern: str):
        """Set LED pattern"""
        if self.dogbot and hasattr(self.dogbot, 'led_controller'):
            self.dogbot.led_controller.set_pattern(pattern)
            
    def _rgb_to_hex(self, rgb: Tuple[int, int, int]) -> str:
        """Convert RGB to hex color"""
        return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
        
    def update_loop(self):
        """Main update loop for GUI"""
        while self.running:
            try:
                # Update video feed
                if self.dogbot and hasattr(self.dogbot, 'camera'):
                    frame = self.dogbot.camera.capture_frame()
                    if frame is not None:
                        self.update_video_feed(frame)
                        
                # Update detection info
                self.update_detection_info()
                
                # Update FPS
                # Calculate actual FPS here
                
                # Update behavior counts
                if self.dogbot and self.dogbot.current_detection:
                    behavior = self.dogbot.current_detection.behavior.value
                    if behavior not in self.behavior_counts:
                        self.behavior_counts[behavior] = 0
                    self.behavior_counts[behavior] += 1
                    
                time.sleep(0.033)  # ~30 FPS update
                
            except Exception as e:
                logger.error(f"GUI update error: {e}")
                
    def run(self):
        """Start GUI"""
        # Start update thread
        update_thread = threading.Thread(target=self.update_loop)
        update_thread.daemon = True
        update_thread.start()
        
        # Start tkinter mainloop
        self.root.mainloop()
        
    def close(self):
        """Close GUI"""
        self.running = False
        self.root.quit()