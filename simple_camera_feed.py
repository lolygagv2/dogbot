#!/usr/bin/env python3
"""
Simple Camera Feed Viewer
Just shows live camera feed - no detection complexity
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

class SimpleCameraFeed:
    """Simple live camera feed viewer"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("📹 Simple Camera Feed")
        self.root.geometry("800x600")

        # Camera setup
        self.camera = None
        self.camera_active = False
        self.current_frame = None

        # FPS tracking
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0

        self.setup_gui()
        self.initialize_camera()

    def setup_gui(self):
        """Setup simple GUI"""
        # Title
        title = tk.Label(self.root, text="📹 Live Camera Feed",
                        font=('Arial', 16, 'bold'))
        title.pack(pady=10)

        # Video display
        self.video_label = tk.Label(self.root, bg='black',
                                   text="📷 Initializing Camera...",
                                   font=('Arial', 14), fg='white')
        self.video_label.pack(padx=20, pady=20)

        # Status
        self.status_label = tk.Label(self.root, text="Status: Initializing...",
                                    font=('Arial', 12))
        self.status_label.pack(pady=5)

        # Controls
        controls = tk.Frame(self.root)
        controls.pack(pady=10)

        tk.Button(controls, text="📷 Start Camera",
                 command=self.start_camera, bg='green', fg='white',
                 font=('Arial', 12)).pack(side='left', padx=5)

        tk.Button(controls, text="⏹️ Stop Camera",
                 command=self.stop_camera, bg='red', fg='white',
                 font=('Arial', 12)).pack(side='left', padx=5)

        tk.Button(controls, text="📸 Snapshot",
                 command=self.take_snapshot, bg='blue', fg='white',
                 font=('Arial', 12)).pack(side='left', padx=5)

    def initialize_camera(self):
        """Initialize camera system"""
        try:
            if PICAMERA_AVAILABLE:
                print("🔧 Initializing Picamera2...")
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.camera.configure(config)
                self.status_label.config(text="Status: Picamera2 Ready", fg='green')
                print("✅ Picamera2 ready")
            else:
                print("🔧 Initializing OpenCV camera...")
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    raise Exception("Failed to open camera")
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.camera.set(cv2.CAP_PROP_FPS, 30)
                self.status_label.config(text="Status: OpenCV Camera Ready", fg='green')
                print("✅ OpenCV camera ready")

        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            self.status_label.config(text=f"Status: Camera Error - {e}", fg='red')
            self.camera = None

    def start_camera(self):
        """Start camera feed"""
        if not self.camera:
            self.status_label.config(text="Status: No camera available", fg='red')
            return

        try:
            if PICAMERA_AVAILABLE and hasattr(self.camera, 'start'):
                self.camera.start()

            self.camera_active = True
            self.status_label.config(text="Status: Camera Active", fg='green')

            # Start capture thread
            self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
            self.capture_thread.start()

            print("✅ Camera feed started")

        except Exception as e:
            print(f"❌ Failed to start camera: {e}")
            self.status_label.config(text=f"Status: Start Error - {e}", fg='red')

    def stop_camera(self):
        """Stop camera feed"""
        self.camera_active = False

        try:
            if PICAMERA_AVAILABLE and hasattr(self.camera, 'stop'):
                self.camera.stop()

            self.status_label.config(text="Status: Camera Stopped", fg='orange')
            self.video_label.config(image='', text="📷 Camera Stopped")
            print("⏹️ Camera feed stopped")

        except Exception as e:
            print(f"⚠️ Stop camera error: {e}")

    def capture_loop(self):
        """Main capture loop"""
        print("🎥 Starting capture loop...")

        while self.camera_active:
            try:
                # Capture frame
                if PICAMERA_AVAILABLE and hasattr(self.camera, 'capture_array'):
                    frame = self.camera.capture_array()
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        print("❌ Failed to capture frame")
                        continue
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Update display
                self.current_frame = frame
                self.update_display(frame)

                # FPS calculation
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.last_fps_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
                    self.fps_counter = 0
                    self.last_fps_time = current_time

                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                print(f"❌ Capture error: {e}")
                time.sleep(1)

        print("⏹️ Capture loop ended")

    def update_display(self, frame):
        """Update video display"""
        try:
            # Add FPS counter
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Add timestamp
            timestamp = time.strftime("%H:%M:%S")
            cv2.putText(frame, timestamp, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Convert to PhotoImage
            frame_resized = cv2.resize(frame, (640, 480))
            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image=image)

            # Update display (thread-safe)
            self.root.after(0, self._update_label, photo)

        except Exception as e:
            print(f"❌ Display update error: {e}")

    def _update_label(self, photo):
        """Thread-safe label update"""
        if self.video_label:
            self.video_label.configure(image=photo, text="")
            self.video_label.image = photo

    def take_snapshot(self):
        """Take a snapshot"""
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"

            # Convert RGB to BGR for saving
            frame_bgr = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filename, frame_bgr)

            print(f"📸 Snapshot saved: {filename}")
            self.status_label.config(text=f"Status: Snapshot saved - {filename}", fg='blue')
        else:
            print("❌ No frame to save")

    def run(self):
        """Start the application"""
        print("🚀 Starting Simple Camera Feed...")
        self.root.mainloop()

    def cleanup(self):
        """Cleanup resources"""
        self.camera_active = False
        if self.camera:
            try:
                if PICAMERA_AVAILABLE and hasattr(self.camera, 'close'):
                    self.camera.close()
                elif hasattr(self.camera, 'release'):
                    self.camera.release()
            except Exception as e:
                print(f"⚠️ Cleanup error: {e}")

def main():
    """Main function"""
    try:
        app = SimpleCameraFeed()
        app.run()
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"❌ Application error: {e}")
    finally:
        if 'app' in locals():
            app.cleanup()

if __name__ == "__main__":
    main()