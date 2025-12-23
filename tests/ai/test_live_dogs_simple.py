#!/usr/bin/env python3
"""
Simplified Live Testing Script for 3-Stage AI Pipeline
Works around PipeWire camera conflicts using libcamera directly
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("⚠️ Picamera2 not available, falling back to OpenCV")

import cv2
from core.ai_controller_3stage_fixed import AI3StageControllerFixed

class SimpleLiveTester:
    """Simplified live testing with PipeWire workaround"""

    def __init__(self, duration_minutes=3):
        self.duration_minutes = duration_minutes

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"simple_test_results_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize AI controller
        self.ai = AI3StageControllerFixed()

        # Camera
        self.camera = None
        self.use_picamera2 = False

        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_detections': 0,
            'total_detections': 0,
            'total_poses': 0,
            'behaviors_detected': {},
            'inference_times': []
        }

        # Video recording
        self.video_writer = None

    def initialize(self):
        """Initialize AI and camera"""
        print("🐕 Initializing Simple Live Dog Testing")
        print("=" * 50)

        # Initialize AI controller
        if not self.ai.initialize():
            print("❌ Failed to initialize AI controller")
            return False

        print("✅ AI Controller initialized")

        # Try to initialize camera
        print("\n📹 Setting up camera...")

        if PICAMERA2_AVAILABLE:
            if self._init_picamera2():
                self.use_picamera2 = True
                print("✅ Using Picamera2 (native)")
                return True

        if self._init_opencv_camera():
            self.use_picamera2 = False
            print("✅ Using OpenCV camera")
            return True

        print("❌ No camera available")
        return False

    def _init_picamera2(self):
        """Try to initialize with Picamera2"""
        try:
            print("   Trying Picamera2...")
            self.camera = Picamera2()

            # Configure for still capture
            config = self.camera.create_still_configuration(
                main={"size": (1920, 1080)},
                lores={"size": (640, 480)},
                display="lores"
            )
            self.camera.configure(config)
            self.camera.start()

            # Test capture
            time.sleep(0.5)  # Let camera stabilize
            test_frame = self.camera.capture_array()

            if test_frame is not None and test_frame.size > 0:
                print(f"   Picamera2 test frame: {test_frame.shape}")
                return True
            else:
                self.camera.stop()
                self.camera = None
                return False

        except Exception as e:
            print(f"   Picamera2 failed: {e}")
            if self.camera:
                try:
                    self.camera.stop()
                except:
                    pass
                self.camera = None
            return False

    def _init_opencv_camera(self):
        """Try to initialize with OpenCV (bypass PipeWire)"""
        try:
            print("   Trying OpenCV with exclusive access...")

            # Try to stop PipeWire temporarily (requires sudo)
            try:
                print("   Attempting to stop PipeWire...")
                os.system("sudo systemctl stop pipewire pipewire-session-manager wireplumber > /dev/null 2>&1")
                time.sleep(2)
            except:
                pass

            # Try different camera indices
            for i in range(8):
                try:
                    print(f"   Trying /dev/video{i}...")
                    cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"   OpenCV camera working on /dev/video{i}")
                            self.camera = cap
                            return True
                        else:
                            cap.release()
                except:
                    if 'cap' in locals():
                        cap.release()
                    continue

            return False

        except Exception as e:
            print(f"   OpenCV camera failed: {e}")
            return False

    def run_test(self):
        """Run the simplified test"""
        print("\n" + "=" * 50)
        print("🎬 Starting Live Dog Detection Test")
        print("=" * 50)
        print(f"Duration: {self.duration_minutes} minutes")
        print("Press Ctrl+C to quit early")

        start_time = time.time()
        end_time = start_time + (self.duration_minutes * 60)

        # Setup video recording with H264 codec
        video_path = self.output_dir / "live_test_video.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # More compatible codec
        self.video_writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (1920, 1080))

        if not self.video_writer.isOpened():
            print("⚠️ Video writer failed to initialize, trying MJPEG...")
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_path = self.output_dir / "live_test_video.avi"
            self.video_writer = cv2.VideoWriter(str(video_path), fourcc, 10.0, (1920, 1080))

        if self.video_writer.isOpened():
            print(f"📹 Recording video to: {video_path}")
        else:
            print("❌ Video recording disabled - writer failed to initialize")
            self.video_writer = None

        try:
            while time.time() < end_time:
                # Capture frame
                frame = self._capture_frame()
                if frame is None:
                    print("❌ Failed to capture frame")
                    time.sleep(0.1)
                    continue

                # Save frame to video regardless of detections
                if self.video_writer:
                    self.video_writer.write(frame)

                # Run AI pipeline
                inference_start = time.time()
                detections, poses, behaviors = self.ai.process_frame(frame)
                inference_time = time.time() - inference_start

                # Update statistics
                self._update_stats(detections, poses, behaviors, inference_time)

                # Print results
                self.stats['total_frames'] += 1
                if self.stats['total_frames'] % 30 == 0:  # Every 30 frames
                    elapsed = time.time() - start_time
                    remaining = (end_time - time.time()) / 60
                    print(f"[{elapsed:.0f}s] Frame {self.stats['total_frames']} | "
                          f"Dogs: {len(detections)} | Poses: {len(poses)} | "
                          f"Behaviors: {len(behaviors)} | {inference_time*1000:.1f}ms | "
                          f"Remaining: {remaining:.1f}m")

                # Save frames periodically (every 60 frames, regardless of detections)
                if self.stats['total_frames'] % 60 == 0:
                    self._save_frame(frame, detections, poses, behaviors)

                time.sleep(0.033)  # ~30 FPS

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")

        finally:
            self._cleanup()

        # Generate report
        self._generate_report(time.time() - start_time)

    def _capture_frame(self):
        """Capture a frame from camera"""
        try:
            if self.use_picamera2:
                frame = self.camera.capture_array()
                # Convert RGB to BGR for OpenCV compatibility
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            else:
                ret, frame = self.camera.read()
                return frame if ret else None
        except Exception as e:
            print(f"Capture error: {e}")
            return None

    def _update_stats(self, detections, poses, behaviors, inference_time):
        """Update statistics"""
        self.stats['inference_times'].append(inference_time)

        if detections:
            self.stats['frames_with_detections'] += 1
            self.stats['total_detections'] += len(detections)

        if poses:
            self.stats['total_poses'] += len(poses)

        for behavior in behaviors:
            name = behavior.behavior
            if name not in self.stats['behaviors_detected']:
                self.stats['behaviors_detected'][name] = 0
            self.stats['behaviors_detected'][name] += 1

    def _save_frame(self, frame, detections, poses, behaviors):
        """Save annotated frame"""
        try:
            annotated = frame.copy()

            # Add timestamp and stats overlay
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(annotated, f"Time: {timestamp}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f"Frame: {self.stats['total_frames']}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f"Detections: {len(detections)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Draw detections if any
            for i, det in enumerate(detections):
                cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 3)
                cv2.putText(annotated, f"Dog {i+1}: {det.confidence:.2f}",
                           (det.x1, det.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Save frame (even without detections)
            filename = f"frame_{self.stats['total_frames']:05d}_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(self.output_dir / filename), annotated)

        except Exception as e:
            print(f"Save frame error: {e}")

    def _cleanup(self):
        """Clean up resources"""
        try:
            # Close video writer
            if self.video_writer:
                self.video_writer.release()
                print("✅ Video saved")

            if self.use_picamera2 and self.camera:
                self.camera.stop()
            elif self.camera:
                self.camera.release()

            self.ai.cleanup()

            # Restart PipeWire if we stopped it
            try:
                os.system("sudo systemctl start pipewire pipewire-session-manager wireplumber > /dev/null 2>&1")
            except:
                pass

        except Exception as e:
            print(f"Cleanup error: {e}")

    def _generate_report(self, duration):
        """Generate final report"""
        print("\n" + "=" * 50)
        print("📊 LIVE DOG TESTING REPORT")
        print("=" * 50)

        avg_inference = np.mean(self.stats['inference_times']) * 1000 if self.stats['inference_times'] else 0
        detection_rate = (self.stats['frames_with_detections'] / max(1, self.stats['total_frames'])) * 100
        fps = self.stats['total_frames'] / max(1, duration)

        print(f"Duration: {duration:.1f} seconds")
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Average FPS: {fps:.1f}")
        print(f"Average Inference: {avg_inference:.1f}ms")
        print(f"")
        print(f"Detection Results:")
        print(f"  Frames with Dogs: {self.stats['frames_with_detections']} ({detection_rate:.1f}%)")
        print(f"  Total Detections: {self.stats['total_detections']}")
        print(f"  Total Poses: {self.stats['total_poses']}")

        if self.stats['behaviors_detected']:
            print(f"  Behaviors: {self.stats['behaviors_detected']}")
        else:
            print(f"  No behaviors detected")

        print(f"\n📁 Results saved to: {self.output_dir}")

def main():
    print("🐕 TreatSensei Simple Live Dog Testing")
    print("Working around PipeWire camera conflicts")
    print("=" * 50)

    tester = SimpleLiveTester(duration_minutes=2)  # Start with 2 minutes

    if not tester.initialize():
        print("❌ Failed to initialize testing system")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure you're running as user (not sudo)")
        print("   2. Try: sudo pkill -f python")
        print("   3. Try: sudo systemctl restart pipewire")
        return

    print("\n⚠️  NOTES:")
    print("   - This bypasses PipeWire for camera access")
    print("   - Dogs should be visible and moving for best results")
    print("   - Press Ctrl+C to quit early")
    print("   - Video will be saved regardless of detections")

    print("\nStarting test in 3 seconds...")
    time.sleep(3)
    tester.run_test()

    print("\n✅ Testing completed!")

if __name__ == "__main__":
    main()