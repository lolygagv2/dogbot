#!/usr/bin/env python3
"""
Debug script to verify camera input and detection output
Saves raw camera frames and detection results for inspection
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

import cv2
from core.ai_controller_3stage_fixed import AI3StageControllerFixed

class CameraDetectionDebugger:
    """Debug camera input and detection output"""

    def __init__(self):
        self.output_dir = Path(f"debug_camera_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(exist_ok=True)

        self.ai = AI3StageControllerFixed()
        self.camera = None
        self.use_picamera2 = False

    def initialize(self):
        """Initialize everything"""
        print("🔍 Camera Detection Debugger")
        print("=" * 40)

        # Initialize AI
        if not self.ai.initialize():
            print("❌ AI Controller failed")
            return False
        print("✅ AI Controller ready")

        # Initialize camera
        if PICAMERA2_AVAILABLE:
            if self._init_picamera2():
                print("✅ Picamera2 ready")
                return True

        print("❌ Camera failed")
        return False

    def _init_picamera2(self):
        """Initialize Picamera2"""
        try:
            self.camera = Picamera2()
            config = self.camera.create_still_configuration(
                main={"size": (1920, 1080)},
                lores={"size": (640, 480)},
                display="lores"
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(1)  # Let camera stabilize
            self.use_picamera2 = True
            return True
        except Exception as e:
            print(f"Picamera2 error: {e}")
            return False

    def run_debug(self):
        """Run debug session"""
        print("\n🎬 Debug Session Started")
        print("Taking 10 frames for analysis...")

        for i in range(10):
            print(f"\n--- Frame {i+1}/10 ---")

            # Capture frame
            frame = self._capture_frame()
            if frame is None:
                print("❌ Failed to capture frame")
                continue

            print(f"📷 Captured: {frame.shape}, dtype: {frame.dtype}")
            print(f"   Min/Max values: {frame.min()}/{frame.max()}")

            # Save raw frame
            raw_filename = f"raw_frame_{i+1:02d}.jpg"
            cv2.imwrite(str(self.output_dir / raw_filename), frame)
            print(f"💾 Saved raw: {raw_filename}")

            # Run detection
            start_time = time.time()
            detections, poses, behaviors = self.ai.process_frame(frame)
            inference_time = (time.time() - start_time) * 1000

            print(f"🧠 Inference: {inference_time:.1f}ms")
            print(f"🐕 Detections: {len(detections)}")
            print(f"🦴 Poses: {len(poses)}")
            print(f"🎭 Behaviors: {len(behaviors)}")

            # Print detection details
            for j, det in enumerate(detections):
                print(f"   Det {j+1}: ({det.x1},{det.y1}) to ({det.x2},{det.y2}) conf={det.confidence:.3f}")

            # Create annotated version
            annotated_frame = self._create_annotated_frame(frame, detections, poses, behaviors)

            # Save annotated frame
            annotated_filename = f"annotated_frame_{i+1:02d}.jpg"
            cv2.imwrite(str(self.output_dir / annotated_filename), annotated_frame)
            print(f"💾 Saved annotated: {annotated_filename}")

            # Test with different confidence thresholds
            self._test_detection_thresholds(frame, i+1)

            time.sleep(1)

        self._cleanup()
        print(f"\n✅ Debug complete! Check: {self.output_dir}")

    def _capture_frame(self):
        """Capture frame"""
        try:
            if self.use_picamera2:
                frame = self.camera.capture_array()
                # Convert RGB to BGR for OpenCV
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame
            return None
        except Exception as e:
            print(f"Capture error: {e}")
            return None

    def _create_annotated_frame(self, frame, detections, poses, behaviors):
        """Create annotated frame"""
        annotated = frame.copy()

        # Draw detections
        for i, det in enumerate(detections):
            cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 3)
            cv2.putText(annotated, f"Dog {i+1}: {det.confidence:.3f}",
                       (det.x1, det.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw poses
        for pose in poses:
            keypoints = pose.keypoints
            det = pose.detection
            scale_x = det.width / 640
            scale_y = det.height / 640

            for kpt_idx, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:
                    x_px = int(det.x1 + x * scale_x)
                    y_px = int(det.y1 + y * scale_y)
                    cv2.circle(annotated, (x_px, y_px), 3, (0, 0, 255), -1)

        # Add text overlay
        cv2.putText(annotated, f"Detections: {len(detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated, f"Poses: {len(poses)}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated, f"Behaviors: {len(behaviors)}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        return annotated

    def _test_detection_thresholds(self, frame, frame_num):
        """Test different detection confidence thresholds"""
        print("   🔧 Testing detection thresholds...")

        # Temporarily modify the detection parser to use lower thresholds
        # This is a hack to test if detections exist at lower confidence
        original_method = self.ai._parse_detection_output

        def test_threshold(conf_thresh):
            def modified_parser(output_data, orig_w, orig_h):
                detections = []
                try:
                    for output_name, output_tensor in output_data.items():
                        if isinstance(output_tensor, list):
                            if len(output_tensor) > 0 and isinstance(output_tensor[0], np.ndarray):
                                output_tensor = output_tensor[0]
                            else:
                                output_tensor = np.array(output_tensor)

                        if not isinstance(output_tensor, np.ndarray):
                            continue

                        if len(output_tensor.shape) == 3 and output_tensor.shape[1] == 5:
                            detections_data = output_tensor[0].T
                        elif len(output_tensor.shape) == 2 and output_tensor.shape[1] == 5:
                            detections_data = output_tensor
                        else:
                            continue

                        scale_x = orig_w / 640
                        scale_y = orig_h / 640

                        for detection in detections_data:
                            x1, y1, x2, y2, conf = detection
                            if conf > conf_thresh:  # Use test threshold
                                x1_scaled = int(x1 * scale_x)
                                y1_scaled = int(y1 * scale_y)
                                x2_scaled = int(x2 * scale_x)
                                y2_scaled = int(y2 * scale_y)

                                from core.ai_controller_3stage_fixed import Detection
                                detections.append(Detection(x1_scaled, y1_scaled, x2_scaled, y2_scaled, float(conf)))
                        break
                except:
                    pass
                return detections
            return modified_parser

        # Test different thresholds
        for thresh in [0.1, 0.2, 0.3, 0.4, 0.5]:
            self.ai._parse_detection_output = test_threshold(thresh)
            detections, _, _ = self.ai.process_frame(frame)
            print(f"      Threshold {thresh}: {len(detections)} detections")

            if detections:
                for det in detections:
                    print(f"        conf={det.confidence:.3f}")

        # Restore original method
        self.ai._parse_detection_output = original_method

    def _cleanup(self):
        """Cleanup"""
        try:
            if self.camera and self.use_picamera2:
                self.camera.stop()
            self.ai.cleanup()
        except:
            pass

def main():
    debugger = CameraDetectionDebugger()

    if not debugger.initialize():
        print("❌ Failed to initialize")
        return

    print("\n📋 This will:")
    print("   1. Take 10 camera frames")
    print("   2. Run AI detection on each")
    print("   3. Save raw and annotated images")
    print("   4. Test different confidence thresholds")
    print("   5. Show exactly what the camera sees")

    print("\nStarting debug session...")
    debugger.run_debug()

if __name__ == "__main__":
    main()