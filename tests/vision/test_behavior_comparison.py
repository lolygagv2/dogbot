#!/usr/bin/env python3
"""
Comparison test between heuristic and temporal behavior models
Runs both AI controllers in parallel to compare their behavior detection
"""

import time
import cv2
import numpy as np
import logging
from typing import List, Tuple
from datetime import datetime

# Import both AI controllers
from ...core.ai_controller_3stage_fixed import AI3StageControllerFixed
from ...core.ai_controller_3stage_temporal import AI3StageControllerTemporal

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BehaviorComparison:
    """Compare heuristic vs temporal behavior detection"""

    def __init__(self):
        # Initialize both controllers
        self.heuristic_controller = AI3StageControllerFixed()
        self.temporal_controller = AI3StageControllerTemporal()

        # Statistics
        self.stats = {
            'heuristic': {
                'detections': 0,
                'poses': 0,
                'behaviors': {},
                'inference_times': []
            },
            'temporal': {
                'detections': 0,
                'poses': 0,
                'behaviors': {},
                'inference_times': []
            }
        }

        # Camera
        self.camera = None
        self.frame_count = 0

    def initialize(self) -> bool:
        """Initialize both controllers and camera"""
        print("🚀 Initializing behavior comparison test...")

        # Initialize heuristic controller
        print("\n📊 Initializing HEURISTIC controller (Y-position/limb spread)...")
        if not self.heuristic_controller.initialize():
            print("❌ Failed to initialize heuristic controller")
            return False
        print("✅ Heuristic controller ready")

        # Initialize temporal controller
        print("\n📊 Initializing TEMPORAL controller (behavior_14.ts)...")
        if not self.temporal_controller.initialize():
            print("❌ Failed to initialize temporal controller")
            return False
        print("✅ Temporal controller ready")

        # Initialize camera
        print("\n📷 Initializing camera...")
        try:
            from picamera2 import Picamera2
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"size": (640, 640), "format": "RGB888"}
            )
            self.camera.configure(config)
            self.camera.start()
            print("✅ Camera initialized")
        except Exception as e:
            print(f"⚠️  Camera initialization failed: {e}")
            print("   Will use test frames instead")

        return True

    def capture_frame(self) -> np.ndarray:
        """Capture a frame from camera or generate test frame"""
        if self.camera:
            frame = self.camera.capture_array()
            # Convert RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            # Generate test frame with some variation
            frame = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)

        return frame

    def run_comparison(self, duration: int = 60):
        """Run comparison test for specified duration"""
        print(f"\n🏁 Starting {duration}-second comparison test...")
        print("=" * 60)

        start_time = time.time()
        last_report_time = start_time

        while time.time() - start_time < duration:
            self.frame_count += 1
            frame = self.capture_frame()

            # Process with heuristic controller
            h_start = time.time()
            h_detections, h_poses, h_behaviors = self.heuristic_controller.process_frame(frame.copy())
            h_time = (time.time() - h_start) * 1000  # ms

            # Process with temporal controller
            t_start = time.time()
            t_detections, t_poses, t_behaviors = self.temporal_controller.process_frame(frame.copy())
            t_time = (time.time() - t_start) * 1000  # ms

            # Update statistics
            self._update_stats('heuristic', h_detections, h_poses, h_behaviors, h_time)
            self._update_stats('temporal', t_detections, t_poses, t_behaviors, t_time)

            # Display real-time comparison
            self._display_frame_comparison(
                frame,
                h_detections, h_poses, h_behaviors,
                t_detections, t_poses, t_behaviors,
                h_time, t_time
            )

            # Report every 5 seconds
            if time.time() - last_report_time >= 5:
                self._print_comparison_report()
                last_report_time = time.time()

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n⏹️  Test stopped by user")
                break

        # Final report
        self._print_final_report()

    def _update_stats(self, model_name: str, detections, poses, behaviors, inference_time):
        """Update statistics for a model"""
        stats = self.stats[model_name]

        stats['detections'] += len(detections)
        stats['poses'] += len(poses)
        stats['inference_times'].append(inference_time)

        for behavior in behaviors:
            behavior_name = behavior.behavior
            if behavior_name not in stats['behaviors']:
                stats['behaviors'][behavior_name] = 0
            stats['behaviors'][behavior_name] += 1

    def _display_frame_comparison(self, frame, h_det, h_pose, h_beh, t_det, t_pose, t_beh, h_time, t_time):
        """Display side-by-side comparison"""
        display = np.zeros((640, 1280, 3), dtype=np.uint8)

        # Left side - Heuristic
        left_frame = frame.copy()
        cv2.putText(left_frame, "HEURISTIC (Y-pos/Limb)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(left_frame, f"Time: {h_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw detections
        for det in h_det:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(left_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show behaviors
        y_offset = 90
        for beh in h_beh:
            text = f"{beh.behavior}: {beh.confidence:.2f}"
            cv2.putText(left_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            y_offset += 30

        # Right side - Temporal
        right_frame = frame.copy()
        cv2.putText(right_frame, "TEMPORAL (behavior_14.ts)", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(right_frame, f"Time: {t_time:.1f}ms", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw detections
        for det in t_det:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(right_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show behaviors
        y_offset = 90
        for beh in t_beh:
            text = f"{beh.behavior}: {beh.confidence:.2f}"
            cv2.putText(right_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            y_offset += 30

        # Combine
        display[:, :640] = left_frame
        display[:, 640:] = right_frame

        # Show comparison
        cv2.imshow("Behavior Model Comparison", display)

    def _print_comparison_report(self):
        """Print comparison statistics"""
        print(f"\n📊 Frame {self.frame_count} Comparison:")
        print("-" * 60)

        for model_name in ['heuristic', 'temporal']:
            stats = self.stats[model_name]
            avg_time = np.mean(stats['inference_times'][-100:]) if stats['inference_times'] else 0

            print(f"{model_name.upper()}:")
            print(f"  Detections: {stats['detections']}")
            print(f"  Poses: {stats['poses']}")
            print(f"  Behaviors: {stats['behaviors']}")
            print(f"  Avg inference: {avg_time:.1f}ms")

    def _print_final_report(self):
        """Print final comparison report"""
        print("\n" + "=" * 60)
        print("📊 FINAL COMPARISON REPORT")
        print("=" * 60)

        # Calculate averages
        for model_name in ['heuristic', 'temporal']:
            stats = self.stats[model_name]
            avg_time = np.mean(stats['inference_times']) if stats['inference_times'] else 0
            total_behaviors = sum(stats['behaviors'].values())

            print(f"\n{model_name.upper()} MODEL:")
            print(f"  Total frames: {self.frame_count}")
            print(f"  Total detections: {stats['detections']}")
            print(f"  Total poses: {stats['poses']}")
            print(f"  Total behaviors: {total_behaviors}")
            print(f"  Average inference: {avg_time:.1f}ms")

            if stats['behaviors']:
                print(f"  Behavior breakdown:")
                for behavior, count in sorted(stats['behaviors'].items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total_behaviors * 100) if total_behaviors > 0 else 0
                    print(f"    - {behavior}: {count} ({percentage:.1f}%)")

        # Compare accuracy (if ground truth available)
        print("\n🎯 KEY DIFFERENCES:")
        h_behaviors = set(self.stats['heuristic']['behaviors'].keys())
        t_behaviors = set(self.stats['temporal']['behaviors'].keys())

        unique_to_heuristic = h_behaviors - t_behaviors
        unique_to_temporal = t_behaviors - h_behaviors
        common = h_behaviors & t_behaviors

        if unique_to_heuristic:
            print(f"  Behaviors only detected by HEURISTIC: {unique_to_heuristic}")
        if unique_to_temporal:
            print(f"  Behaviors only detected by TEMPORAL: {unique_to_temporal}")
        if common:
            print(f"  Behaviors detected by BOTH: {common}")

        # Performance comparison
        h_avg = np.mean(self.stats['heuristic']['inference_times']) if self.stats['heuristic']['inference_times'] else 0
        t_avg = np.mean(self.stats['temporal']['inference_times']) if self.stats['temporal']['inference_times'] else 0

        if h_avg > 0 and t_avg > 0:
            if h_avg < t_avg:
                print(f"\n⚡ HEURISTIC is {t_avg/h_avg:.1f}x faster")
            else:
                print(f"\n⚡ TEMPORAL is {h_avg/t_avg:.1f}x faster")

    def cleanup(self):
        """Cleanup resources"""
        cv2.destroyAllWindows()

        if self.camera:
            self.camera.stop()

        self.heuristic_controller.cleanup()
        self.temporal_controller.cleanup()

        print("\n✅ Cleanup complete")


def main():
    """Run the comparison test"""
    comparison = BehaviorComparison()

    if not comparison.initialize():
        print("❌ Failed to initialize comparison test")
        return

    try:
        # Run comparison for 10 seconds (quick test)
        comparison.run_comparison(duration=10)
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
    finally:
        comparison.cleanup()


if __name__ == "__main__":
    main()