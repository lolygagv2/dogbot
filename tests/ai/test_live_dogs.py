#!/usr/bin/env python3
"""
Live Testing Script for 3-Stage AI Pipeline with Real Dogs
Tests the complete dogdetector_14.hef + dogpose_14.hef + behavior analysis pipeline
Saves video and screenshots for accuracy verification
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ai_controller_3stage_fixed import AI3StageControllerFixed

class LiveDogTester:
    """Live testing with video recording and screenshot capture"""

    def __init__(self, duration_minutes=5, record_video=True, save_screenshots=True):
        self.duration_minutes = duration_minutes
        self.record_video = record_video
        self.save_screenshots = save_screenshots

        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"live_test_results_{timestamp}")
        self.output_dir.mkdir(exist_ok=True)

        # Initialize AI controller
        self.ai = AI3StageControllerFixed()

        # Video writer
        self.video_writer = None
        self.video_fps = 10  # Record at 10 FPS to keep file size manageable

        # Statistics
        self.stats = {
            'total_frames': 0,
            'frames_with_detections': 0,
            'frames_with_poses': 0,
            'frames_with_behaviors': 0,
            'total_detections': 0,
            'total_poses': 0,
            'behaviors_detected': {},
            'avg_inference_time': 0,
            'inference_times': []
        }

    def initialize(self):
        """Initialize camera and AI models"""
        print("🐕 Initializing Live Dog Testing System")
        print("=" * 60)

        # Initialize AI controller
        if not self.ai.initialize():
            print("❌ Failed to initialize AI controller")
            return False

        status = self.ai.get_status()
        print("✅ AI Controller Status:")
        for key, value in status.items():
            print(f"   {key}: {value}")

        # Test camera
        print("\n📹 Setting up camera...")
        self.cap = None

        # Try more camera indices and different backends
        camera_indices = [0, 1, 2, 3, 4, 5, 6, 7]
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

        for backend in backends:
            for camera_index in camera_indices:
                try:
                    print(f"   Trying camera {camera_index} with backend {backend}...")
                    cap = cv2.VideoCapture(camera_index, backend)

                    if cap.isOpened():
                        # Test if we can actually read a frame
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None:
                            print(f"✅ Camera found at index {camera_index} (backend {backend})")
                            self.cap = cap
                            break
                        else:
                            print(f"   Camera {camera_index} opened but can't read frames")
                            cap.release()
                    else:
                        cap.release()
                except Exception as e:
                    print(f"   Camera {camera_index} failed: {e}")
                    continue

            if self.cap is not None:
                break

        if self.cap is None:
            print("❌ No working camera found")
            print("   Make sure camera is not being used by another process")
            print("   Try: sudo pkill -f python")
            return False

        # Set camera properties
        print("📷 Setting camera resolution...")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Also try setting buffer size to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_size = (actual_width, actual_height)

        print(f"📷 Camera resolution: {actual_width}x{actual_height}")

        # Test reading a few frames to make sure it's working
        print("🧪 Testing camera capture...")
        for i in range(3):
            ret, frame = self.cap.read()
            if not ret:
                print(f"❌ Failed to read test frame {i+1}")
                return False
            print(f"✅ Test frame {i+1}: {frame.shape}")

        print("✅ Camera working properly")

        # Initialize video writer if requested
        if self.record_video:
            video_path = self.output_dir / "live_test_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(video_path), fourcc, self.video_fps, self.frame_size)
            print(f"📹 Recording video to: {video_path}")

        print(f"\n🕐 Test duration: {self.duration_minutes} minutes")
        print(f"📁 Results will be saved to: {self.output_dir}")

        return True

    def run_live_test(self):
        """Run the live testing session"""
        print("\n" + "=" * 60)
        print("🎬 Starting Live Dog Testing Session")
        print("=" * 60)
        print("Press 'q' to quit early, 's' to save screenshot")

        start_time = time.time()
        end_time = start_time + (self.duration_minutes * 60)
        last_video_frame_time = 0

        frame_count = 0
        screenshot_count = 0
        consecutive_failures = 0

        try:
            while time.time() < end_time:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    print(f"❌ Failed to read camera frame (attempt {consecutive_failures})")

                    if consecutive_failures >= 5:
                        print("❌ Too many consecutive camera failures, stopping")
                        break

                    time.sleep(0.1)  # Brief pause before retry
                    continue

                consecutive_failures = 0  # Reset on successful read

                frame_count += 1
                current_time = time.time()

                # Run AI pipeline
                inference_start = time.time()
                detections, poses, behaviors = self.ai.process_frame(frame)
                inference_time = time.time() - inference_start

                # Update statistics
                self._update_stats(detections, poses, behaviors, inference_time)

                # Create annotated frame
                annotated_frame = self._annotate_frame(frame, detections, poses, behaviors, inference_time)

                # Add timestamp and stats overlay
                self._add_overlay(annotated_frame, frame_count, current_time - start_time)

                # Save to video (at reduced frame rate)
                if self.video_writer and (current_time - last_video_frame_time) >= (1.0 / self.video_fps):
                    self.video_writer.write(annotated_frame)
                    last_video_frame_time = current_time

                # Check for key presses (if display available)
                try:
                    cv2.imshow('Live Dog Detection', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        print("\n👋 User requested quit")
                        break
                    elif key == ord('s'):
                        screenshot_count += 1
                        screenshot_path = self.output_dir / f"screenshot_{screenshot_count:03d}.jpg"
                        cv2.imwrite(str(screenshot_path), annotated_frame)
                        print(f"📸 Screenshot saved: {screenshot_path}")

                except cv2.error:
                    # No display available, continue without display
                    pass

                # Auto-save interesting frames
                if detections or behaviors:
                    if self.save_screenshots and frame_count % 30 == 0:  # Every 30th interesting frame
                        auto_screenshot_path = self.output_dir / f"auto_frame_{frame_count:05d}.jpg"
                        cv2.imwrite(str(auto_screenshot_path), annotated_frame)

                # Print periodic status
                if frame_count % 60 == 0:  # Every 60 frames
                    elapsed = current_time - start_time
                    remaining = (end_time - current_time) / 60
                    print(f"[{elapsed:.0f}s] Frame {frame_count} | "
                          f"Det: {len(detections)} | Poses: {len(poses)} | Behaviors: {len(behaviors)} | "
                          f"Remaining: {remaining:.1f}m")

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")

        finally:
            self._cleanup()

        # Generate final report
        self._generate_report(time.time() - start_time)

    def _update_stats(self, detections, poses, behaviors, inference_time):
        """Update running statistics"""
        self.stats['total_frames'] += 1
        self.stats['inference_times'].append(inference_time)

        if detections:
            self.stats['frames_with_detections'] += 1
            self.stats['total_detections'] += len(detections)

        if poses:
            self.stats['frames_with_poses'] += 1
            self.stats['total_poses'] += len(poses)

        if behaviors:
            self.stats['frames_with_behaviors'] += 1
            for behavior in behaviors:
                behavior_name = behavior.behavior
                if behavior_name not in self.stats['behaviors_detected']:
                    self.stats['behaviors_detected'][behavior_name] = 0
                self.stats['behaviors_detected'][behavior_name] += 1

    def _annotate_frame(self, frame, detections, poses, behaviors, inference_time):
        """Add visual annotations to frame"""
        annotated = frame.copy()

        # Draw detections
        for i, det in enumerate(detections):
            cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 3)
            cv2.putText(annotated, f"Dog {i+1}: {det.confidence:.2f}",
                       (det.x1, det.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw pose keypoints
        for pose_idx, pose in enumerate(poses):
            keypoints = pose.keypoints
            det = pose.detection

            # Scale keypoints to detection box
            scale_x = det.width / 640
            scale_y = det.height / 640

            for kpt_idx, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:  # Only confident keypoints
                    x_px = int(det.x1 + x * scale_x)
                    y_px = int(det.y1 + y * scale_y)

                    # Color keypoints by confidence
                    color = (0, int(255 * conf), int(255 * (1-conf)))
                    cv2.circle(annotated, (x_px, y_px), 4, color, -1)

        # Draw behaviors
        behavior_y = 30
        for behavior in behaviors:
            text = f"{behavior.behavior.upper()}: {behavior.confidence:.2f}"
            cv2.putText(annotated, text, (10, behavior_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            behavior_y += 35

        return annotated

    def _add_overlay(self, frame, frame_count, elapsed_time):
        """Add timestamp and statistics overlay"""
        # Semi-transparent overlay area
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame.shape[1]-300, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Text overlay
        texts = [
            f"Frame: {frame_count}",
            f"Time: {elapsed_time:.0f}s",
            f"Detections: {self.stats['total_detections']}",
            f"Poses: {self.stats['total_poses']}"
        ]

        for i, text in enumerate(texts):
            cv2.putText(frame, text, (frame.shape[1]-290, 25 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def _cleanup(self):
        """Clean up resources"""
        try:
            if self.cap:
                self.cap.release()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            self.ai.cleanup()
        except:
            pass

    def _generate_report(self, total_duration):
        """Generate final test report"""
        print("\n" + "=" * 60)
        print("📊 LIVE DOG TESTING REPORT")
        print("=" * 60)

        # Calculate averages
        if self.stats['inference_times']:
            avg_inference = np.mean(self.stats['inference_times']) * 1000  # ms
            max_inference = np.max(self.stats['inference_times']) * 1000  # ms
        else:
            avg_inference = max_inference = 0

        detection_rate = (self.stats['frames_with_detections'] / max(1, self.stats['total_frames'])) * 100
        pose_rate = (self.stats['frames_with_poses'] / max(1, self.stats['total_frames'])) * 100
        behavior_rate = (self.stats['frames_with_behaviors'] / max(1, self.stats['total_frames'])) * 100

        fps = self.stats['total_frames'] / max(1, total_duration)

        print(f"Duration: {total_duration:.1f} seconds")
        print(f"Total Frames: {self.stats['total_frames']}")
        print(f"Average FPS: {fps:.1f}")
        print(f"")
        print(f"Performance:")
        print(f"  Avg Inference Time: {avg_inference:.1f}ms")
        print(f"  Max Inference Time: {max_inference:.1f}ms")
        print(f"")
        print(f"Detection Results:")
        print(f"  Frames with Dogs: {self.stats['frames_with_detections']} ({detection_rate:.1f}%)")
        print(f"  Total Detections: {self.stats['total_detections']}")
        print(f"")
        print(f"Pose Results:")
        print(f"  Frames with Poses: {self.stats['frames_with_poses']} ({pose_rate:.1f}%)")
        print(f"  Total Poses: {self.stats['total_poses']}")
        print(f"")
        print(f"Behavior Results:")
        print(f"  Frames with Behaviors: {self.stats['frames_with_behaviors']} ({behavior_rate:.1f}%)")

        if self.stats['behaviors_detected']:
            print(f"  Detected Behaviors:")
            for behavior, count in self.stats['behaviors_detected'].items():
                print(f"    {behavior}: {count} times")
        else:
            print(f"  No behaviors detected")

        print(f"")
        print(f"📁 Results saved to: {self.output_dir}")

        # Save report to file
        report_path = self.output_dir / "test_report.txt"
        with open(report_path, 'w') as f:
            f.write("Live Dog Testing Report\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Duration: {total_duration:.1f} seconds\n")
            f.write(f"Total Frames: {self.stats['total_frames']}\n")
            f.write(f"Average FPS: {fps:.1f}\n")
            f.write(f"Detection Rate: {detection_rate:.1f}%\n")
            f.write(f"Pose Rate: {pose_rate:.1f}%\n")
            f.write(f"Behavior Rate: {behavior_rate:.1f}%\n")
            f.write(f"Total Detections: {self.stats['total_detections']}\n")
            f.write(f"Total Poses: {self.stats['total_poses']}\n")
            f.write(f"Avg Inference Time: {avg_inference:.1f}ms\n")

        print(f"📄 Report saved to: {report_path}")

def main():
    print("🐕 TreatSensei Live Dog Testing")
    print("Using 3-Stage AI Pipeline: Detection -> Pose -> Behavior")
    print("=" * 70)

    # Configuration
    duration_minutes = 3  # Start with 3 minutes for initial testing
    record_video = True
    save_screenshots = True

    print(f"⚙️  Configuration:")
    print(f"   Duration: {duration_minutes} minutes")
    print(f"   Record Video: {record_video}")
    print(f"   Save Screenshots: {save_screenshots}")
    print()

    # Initialize tester
    tester = LiveDogTester(duration_minutes, record_video, save_screenshots)

    if not tester.initialize():
        print("❌ Failed to initialize testing system")
        return

    print("\n⚠️  IMPORTANT:")
    print("   - Make sure dogs are visible in camera view")
    print("   - Good lighting helps detection accuracy")
    print("   - Dogs should be moving for behavior analysis")
    print("   - Press 'q' to quit early, 's' for manual screenshots")

    try:
        input("\nPress Enter to start live testing...")
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Cancelled by user")
        return

    # Run the test
    tester.run_live_test()

    print("\n✅ Live testing completed!")
    print("📋 Review the results for accuracy verification")

if __name__ == "__main__":
    main()