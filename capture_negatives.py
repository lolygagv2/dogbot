#!/usr/bin/env python3
"""
Camera capture tool for collecting negative (no-dog) training images.
Automatically captures frames at regular intervals for building training dataset.
Now with servo control for automated pan/tilt sweeping to capture all angles.
"""

import cv2
import numpy as np
import os
import time
import json
from datetime import datetime
from pathlib import Path
import argparse
import sys

# Add path for servo controller
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core', 'hardware'))

try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Warning: Picamera2 not available, using OpenCV camera")

try:
    from core.hardware.servo_controller import ServoController
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False
    print("Warning: Servo controller not available, manual camera positioning only")

# Check if OpenCV has GUI support
CV2_GUI_AVAILABLE = True
try:
    # Try to create a test window to check GUI support
    test_window = "test"
    cv2.namedWindow(test_window)
    cv2.destroyWindow(test_window)
except cv2.error:
    CV2_GUI_AVAILABLE = False
    print("Warning: OpenCV GUI not available, running in headless mode")


class NegativeImageCapture:
    def __init__(self, output_dir="training_data/negatives",
                 capture_interval=0.5,  # Reduced for servo sweep mode
                 total_captures=500,
                 resolution=(1024, 768),
                 show_preview=True,
                 use_servos=True):
        """
        Initialize the negative image capture system.

        Args:
            output_dir: Directory to save captured images
            capture_interval: Time between captures in seconds
            total_captures: Total number of images to capture
            resolution: Camera resolution (width, height)
            show_preview: Show live preview window
            use_servos: Enable servo control for automatic pan/tilt sweeping
        """
        self.output_dir = Path(output_dir)
        self.capture_interval = capture_interval
        self.total_captures = total_captures
        self.resolution = resolution
        self.show_preview = show_preview and CV2_GUI_AVAILABLE
        if show_preview and not CV2_GUI_AVAILABLE:
            print("Note: Preview requested but GUI not available, running headless")
        self.use_servos = use_servos and SERVO_AVAILABLE
        self.camera = None
        self.servo_controller = None
        self.captures_done = 0

        # Servo sweep parameters
        self.pan_min = 10    # From servo_controller.py scan ranges
        self.pan_max = 200   # From servo_controller.py scan ranges
        self.pitch_min = 20  # Look down angle
        self.pitch_max = 150 # Look up angle
        self.pan_step = 10   # Degrees per pan step
        self.pitch_step = 20 # Degrees per pitch step

        # Current servo positions for sweep
        self.current_pan = self.pan_min
        self.current_pitch = self.pitch_min

        # Create output directory structure
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_dir = self.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir.mkdir(exist_ok=True)

        # Metadata tracking
        self.metadata = {
            "session_start": datetime.now().isoformat(),
            "resolution": list(resolution),
            "capture_interval": capture_interval,
            "total_planned": total_captures,
            "servo_enabled": self.use_servos,
            "captures": []
        }

        # Initialize camera
        self._init_camera()

        # Initialize servo controller if available
        if self.use_servos:
            self._init_servos()

    def _init_camera(self):
        """Initialize camera (Picamera2 or OpenCV fallback)."""
        if PICAMERA_AVAILABLE:
            print("Initializing Picamera2...")
            self.camera = Picamera2()

            # Configure camera for consistent capture
            config = self.camera.create_still_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                buffer_count=2
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(2)  # Let camera stabilize
            print(f"Picamera2 initialized at {self.resolution[0]}x{self.resolution[1]}")
        else:
            print("Initializing OpenCV camera...")
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            print(f"OpenCV camera initialized at {self.resolution[0]}x{self.resolution[1]}")

    def _init_servos(self):
        """Initialize servo controller for pan/tilt control."""
        try:
            self.servo_controller = ServoController()
            if self.servo_controller.is_initialized():
                print("Servo controller initialized successfully")
                # Move to starting position
                print(f"Moving to start position: Pan={self.pan_min}°, Pitch={self.pitch_min}°")
                self.servo_controller.set_camera_pan(self.pan_min, smooth=True)
                self.servo_controller.set_camera_pitch(self.pitch_min, smooth=True)
                time.sleep(2)  # Let servos settle
            else:
                print("Servo controller failed to initialize, continuing without servos")
                self.use_servos = False
                self.servo_controller = None
        except Exception as e:
            print(f"Servo initialization error: {e}")
            self.use_servos = False
            self.servo_controller = None

    def capture_frame(self):
        """Capture a single frame from the camera."""
        if PICAMERA_AVAILABLE:
            # Capture from Picamera2
            frame = self.camera.capture_array()
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame_bgr
        else:
            # Capture from OpenCV camera
            ret, frame = self.camera.read()
            if ret:
                return frame
            return None

    def letterbox_preview(self, frame):
        """
        Apply letterbox transformation to show what model will see.
        This helps verify the preprocessing matches training.
        """
        h, w = frame.shape[:2]
        new_shape = (1024, 1024)
        color = (114, 114, 114)

        # Calculate scaling ratio
        r = min(new_shape[0]/h, new_shape[1]/w)
        nh, nw = int(round(h*r)), int(round(w*r))

        # Resize image
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # Create padded canvas
        canvas = np.full((new_shape[0], new_shape[1], 3), color, dtype=resized.dtype)
        top = (new_shape[0] - nh) // 2
        left = (new_shape[1] - nw) // 2
        canvas[top:top+nh, left:left+nw] = resized

        return canvas

    def save_frame(self, frame, frame_num):
        """Save captured frame with metadata including servo positions."""
        timestamp = datetime.now()
        filename = f"negative_{frame_num:05d}_{timestamp.strftime('%H%M%S')}.jpg"
        filepath = self.session_dir / filename

        # Save the original frame
        cv2.imwrite(str(filepath), frame)

        # Also save letterboxed version for verification
        letterboxed = self.letterbox_preview(frame)
        letterbox_path = self.session_dir / f"letterbox_{filename}"
        cv2.imwrite(str(letterbox_path), letterboxed)

        # Update metadata with servo positions if available
        capture_data = {
            "frame_num": frame_num,
            "timestamp": timestamp.isoformat(),
            "filename": filename,
            "letterbox_filename": f"letterbox_{filename}",
            "shape": list(frame.shape)
        }

        if self.use_servos:
            capture_data["servo_pan"] = self.current_pan
            capture_data["servo_pitch"] = self.current_pitch

        self.metadata["captures"].append(capture_data)

        return filepath

    def move_to_next_position(self):
        """Move servos to next position in the sweep pattern."""
        if not self.use_servos or not self.servo_controller:
            return False

        # Increment pan position
        self.current_pan += self.pan_step

        # If we've reached the end of a pan sweep
        if self.current_pan > self.pan_max:
            # Reset pan and increment pitch
            self.current_pan = self.pan_min
            self.current_pitch += self.pitch_step

            # If we've completed all angles, reset to beginning
            if self.current_pitch > self.pitch_max:
                self.current_pitch = self.pitch_min
                print("\n🔄 Completed full sweep, restarting from beginning")

            # Move to new pitch position first (smooth)
            print(f"\n⬆️ Moving to new pitch level: {self.current_pitch}°")
            self.servo_controller.set_camera_pitch(self.current_pitch, smooth=True)
            time.sleep(0.5)  # Let pitch movement complete

        # Move to pan position (quick, not smooth for efficiency)
        self.servo_controller.set_camera_pan(self.current_pan, smooth=False)

        return True

    def run_capture_session(self):
        """Run the automated capture session with servo sweep."""
        print(f"\n🎥 Starting capture session")
        print(f"📁 Output directory: {self.session_dir}")
        print(f"📷 Target captures: {self.total_captures}")
        print(f"⏱️  Interval: {self.capture_interval} seconds")
        if self.use_servos:
            print(f"🎯 Servo sweep enabled: Pan {self.pan_min}°-{self.pan_max}°, Pitch {self.pitch_min}°-{self.pitch_max}°")
            print(f"📐 Steps: Pan={self.pan_step}°, Pitch={self.pitch_step}°")
        if CV2_GUI_AVAILABLE and self.show_preview:
            print(f"\nPress 'q' to quit early, 'p' to pause/resume, 's' to skip current interval")
            if self.use_servos:
                print(f"Press 'm' to toggle manual/auto servo mode")
        else:
            print(f"\n🖥️ Running in headless mode (no GUI)")
            print(f"Press Ctrl+C to stop the capture session")
        print("-" * 50)

        last_capture_time = 0
        paused = False
        servo_auto_mode = self.use_servos  # Track if we're in auto servo mode

        try:
            while self.captures_done < self.total_captures:
                frame = self.capture_frame()
                if frame is None:
                    print("Error: Failed to capture frame")
                    continue

                current_time = time.time()
                time_until_capture = self.capture_interval - (current_time - last_capture_time)

                # Display preview if enabled and GUI is available
                if self.show_preview and CV2_GUI_AVAILABLE:
                    display_frame = frame.copy()

                    # Add letterbox preview (picture-in-picture)
                    letterboxed = self.letterbox_preview(frame)
                    letterbox_small = cv2.resize(letterboxed, (256, 256))
                    display_frame[10:266, 10:266] = letterbox_small

                    # Add status text
                    status_text = f"Captures: {self.captures_done}/{self.total_captures}"
                    if self.use_servos:
                        status_text += f" | Pan: {self.current_pan}° Pitch: {self.current_pitch}°"
                    if paused:
                        status_text += " [PAUSED]"
                    elif time_until_capture > 0:
                        status_text += f" | Next in: {time_until_capture:.1f}s"
                    else:
                        status_text += " | [CAPTURING]"

                    cv2.putText(display_frame, status_text, (280, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Add instructions
                    instructions = "q:quit p:pause s:skip"
                    if self.use_servos:
                        instructions += " m:manual/auto"
                    cv2.putText(display_frame, instructions, (280, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    # Add servo mode indicator if servos are available
                    if self.use_servos:
                        servo_mode = "AUTO" if servo_auto_mode else "MANUAL"
                        cv2.putText(display_frame, f"Servo: {servo_mode}", (280, 95),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 1)

                    # Add grid to help with room coverage
                    h, w = display_frame.shape[:2]
                    for i in range(1, 3):
                        cv2.line(display_frame, (w//3 * i, 0), (w//3 * i, h), (100, 100, 100), 1)
                        cv2.line(display_frame, (0, h//3 * i), (w, h//3 * i), (100, 100, 100), 1)

                    cv2.imshow("Negative Image Capture", display_frame)

                # Handle keyboard input
                if CV2_GUI_AVAILABLE and self.show_preview:
                    key = cv2.waitKey(1) & 0xFF
                else:
                    # In headless mode, just check for basic timing
                    time.sleep(0.001)
                    key = 0xFF  # No key pressed
                if CV2_GUI_AVAILABLE and self.show_preview:
                    if key == ord('q'):
                        print("\n⚠️  Capture session stopped by user")
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print(f"{'⏸️  PAUSED' if paused else '▶️  RESUMED'}")
                    elif key == ord('s'):
                        last_capture_time = 0  # Force immediate capture
                        print("⏩ Skipping to next capture")
                    elif key == ord('m') and self.use_servos:
                        servo_auto_mode = not servo_auto_mode
                        print(f"🎯 Servo mode: {'AUTO SWEEP' if servo_auto_mode else 'MANUAL'}")

                # Perform capture if not paused and interval elapsed
                if not paused and current_time - last_capture_time >= self.capture_interval:
                    filepath = self.save_frame(frame, self.captures_done)
                    self.captures_done += 1
                    last_capture_time = current_time

                    capture_info = f"📸 Captured {self.captures_done}/{self.total_captures}: {filepath.name}"
                    if self.use_servos:
                        capture_info += f" (Pan:{self.current_pan}° Pitch:{self.current_pitch}°)"
                    print(capture_info)

                    # Move to next servo position if in auto mode
                    if servo_auto_mode and self.use_servos:
                        self.move_to_next_position()
                        # Small delay to let servo settle before next capture
                        time.sleep(0.1)

                    # Progress bar
                    progress = self.captures_done / self.total_captures
                    bar_length = 40
                    filled = int(bar_length * progress)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"   [{bar}] {progress*100:.1f}%")

        except KeyboardInterrupt:
            print("\n⚠️  Capture session interrupted")

        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up camera and servo resources, save metadata."""
        print("\n🔧 Cleaning up...")

        # Return servos to center position if available
        if self.use_servos and self.servo_controller:
            print("Centering camera...")
            self.servo_controller.center_camera(smooth=True)
            time.sleep(1)
            self.servo_controller.cleanup()

        # Save metadata
        self.metadata["session_end"] = datetime.now().isoformat()
        self.metadata["total_captured"] = self.captures_done

        metadata_path = self.session_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        print(f"📊 Metadata saved to {metadata_path}")

        # Generate empty annotation file for YOLO training
        annotations_path = self.session_dir / "annotations.txt"
        with open(annotations_path, 'w') as f:
            for capture in self.metadata["captures"]:
                # Empty annotation (no objects) for negative images
                f.write(f"{capture['filename']}\n")
        print(f"📝 Empty annotations file created: {annotations_path}")

        # Cleanup camera
        if PICAMERA_AVAILABLE and self.camera:
            self.camera.stop()
        elif self.camera:
            self.camera.release()

        if CV2_GUI_AVAILABLE:
            cv2.destroyAllWindows()

        print(f"\n✅ Session complete!")
        print(f"📸 Total images captured: {self.captures_done}")
        print(f"📁 Images saved to: {self.session_dir}")
        print(f"\n💡 Next steps:")
        print(f"   1. Review images in {self.session_dir}")
        print(f"   2. Remove any with dogs/people accidentally in frame")
        print(f"   3. Use these as negative samples in YOLO training")


def main():
    parser = argparse.ArgumentParser(description="Capture negative training images (no dogs) with automatic servo sweep")
    parser.add_argument("--output-dir", default="training_data/negatives",
                       help="Output directory for captured images")
    parser.add_argument("--interval", type=float, default=0.5,
                       help="Seconds between captures (default: 0.5 for servo sweep)")
    parser.add_argument("--count", type=int, default=500,
                       help="Total number of images to capture (default: 500)")
    parser.add_argument("--width", type=int, default=1024,
                       help="Camera width (default: 1024)")
    parser.add_argument("--height", type=int, default=768,
                       help="Camera height (default: 768)")
    parser.add_argument("--no-preview", action="store_true",
                       help="Disable preview window (auto-disabled if GUI not available)")
    parser.add_argument("--no-servos", action="store_true",
                       help="Disable servo control (manual camera positioning)")
    parser.add_argument("--room-mode", action="store_true",
                       help="Guided mode for systematic room coverage")
    parser.add_argument("--pan-step", type=int, default=10,
                       help="Degrees to move pan servo per step (default: 10)")
    parser.add_argument("--pitch-step", type=int, default=20,
                       help="Degrees to move pitch servo per step (default: 20)")

    args = parser.parse_args()

    # Auto-disable preview if GUI not available
    if not CV2_GUI_AVAILABLE:
        args.no_preview = True
        print("📝 Auto-disabled preview due to headless environment")

    if args.room_mode:
        print("\n🏠 ROOM COVERAGE MODE")
        if not args.no_servos and SERVO_AVAILABLE:
            print("🎯 AUTOMATIC SERVO SWEEP ENABLED")
            print("The camera will automatically sweep through all angles:")
            print(f"• Pan range: 10° to 200° in {args.pan_step}° steps")
            print(f"• Pitch range: 20° to 150° in {args.pitch_step}° steps")
            print("• Each position will be captured automatically")
            print("\nThis will systematically cover:")
            print("• All corners and walls")
            print("• Floor to ceiling views")
            print("• Complete room coverage")
        else:
            print("📹 MANUAL CAMERA MODE")
            print("Systematically capture your environment:")
            print("1. Start in one corner of the room")
            print("2. Slowly pan camera across the room")
            print("3. Capture ceiling, floor, furniture, walls")
            print("4. Move to different rooms and repeat")
            print("5. Vary lighting conditions if possible")
        print("\n⚠️ Starting in 5 seconds...")
        print("Press Ctrl+C to abort")
        try:
            for i in range(5, 0, -1):
                print(f"  {i}...")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nAborted by user")
            sys.exit(0)

    capturer = NegativeImageCapture(
        output_dir=args.output_dir,
        capture_interval=args.interval,
        total_captures=args.count,
        resolution=(args.width, args.height),
        show_preview=not args.no_preview,
        use_servos=not args.no_servos
    )

    # Override step sizes if provided
    if not args.no_servos and capturer.use_servos:
        capturer.pan_step = args.pan_step
        capturer.pitch_step = args.pitch_step

    capturer.run_capture_session()


if __name__ == "__main__":
    main()