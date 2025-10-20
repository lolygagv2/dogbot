#!/usr/bin/env python3
"""
Test script for 3-Stage AI Pipeline
Tests dogdetector_14.hef -> dogpose_14.hef -> behavior analysis
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ai_controller_3stage import AI3StageController

def test_with_camera():
    """Test with live camera feed"""
    print("🐕 Testing 3-Stage AI Pipeline with Camera")
    print("=" * 50)

    # Initialize AI controller
    ai = AI3StageController()

    if not ai.initialize():
        print("❌ Failed to initialize AI controller")
        return False

    print(f"✅ AI Controller initialized: {ai.get_status()}")

    try:
        # Initialize camera (try multiple approaches)
        cap = None
        camera_found = False

        # Try different camera indices
        for camera_index in [0, 1, 2]:
            try:
                cap = cv2.VideoCapture(camera_index)
                if cap.isOpened():
                    camera_found = True
                    print(f"✅ Camera found at index {camera_index}")
                    break
                else:
                    cap.release()
            except:
                continue

        if not camera_found:
            print("❌ No camera found")
            return False

        # Set camera to highest resolution possible
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)  # Try for high res
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"📷 Camera resolution: {actual_width}x{actual_height} @ {fps:.1f}fps")

        frame_count = 0
        total_time = 0
        detection_results = []

        print("\n🎬 Starting pipeline test (Press 'q' to quit)")
        print("-" * 50)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break

            start_time = time.time()

            # Run 3-stage pipeline
            detections, poses, behaviors = ai.process_frame(frame)

            end_time = time.time()
            process_time = end_time - start_time
            total_time += process_time
            frame_count += 1

            # Log results
            print(f"[Frame {frame_count:04d}] "
                  f"Detections: {len(detections)} | "
                  f"Poses: {len(poses)} | "
                  f"Behaviors: {len(behaviors)} | "
                  f"Time: {process_time*1000:.1f}ms")

            # Store results for analysis
            detection_results.append({
                'frame': frame_count,
                'detections': len(detections),
                'poses': len(poses),
                'behaviors': len(behaviors),
                'process_time': process_time
            })

            # Detailed detection info
            if detections:
                for i, det in enumerate(detections):
                    print(f"  Det {i}: conf={det.confidence:.3f}, "
                          f"box=({det.x1},{det.y1},{det.x2},{det.y2}), "
                          f"size={det.width}x{det.height}")

            # Detailed pose info
            if poses:
                for i, pose in enumerate(poses):
                    keypoint_count = np.sum(pose.keypoints[:, 2] > 0.3)  # Count confident keypoints
                    print(f"  Pose {i}: {keypoint_count}/24 confident keypoints")

            # Detailed behavior info
            if behaviors:
                for i, behavior in enumerate(behaviors):
                    print(f"  Behavior {i}: {behavior.behavior} (conf={behavior.confidence:.3f})")

            # Save detection screenshots occasionally
            if frame_count % 30 == 0 or detections:  # Every 30 frames or when detection found
                save_debug_frame(frame, detections, poses, f"debug_frame_{frame_count:04d}.jpg")

            # Check for quit
            if frame_count >= 100:  # Test 100 frames
                print(f"\n📊 Test completed after {frame_count} frames")
                break

            time.sleep(0.1)  # Small delay to avoid overloading

    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")

    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if cap:
            cap.release()
        ai.cleanup()

        # Print summary statistics
        if detection_results:
            print_test_summary(detection_results)

    return True

def save_debug_frame(frame, detections, poses, filename):
    """Save frame with detection/pose overlays for debugging"""
    try:
        debug_frame = frame.copy()

        # Draw detections
        for det in detections:
            cv2.rectangle(debug_frame, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Dog {det.confidence:.2f}",
                       (det.x1, det.y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw pose keypoints
        for pose in poses:
            keypoints = pose.keypoints
            for i, (x, y, conf) in enumerate(keypoints):
                if conf > 0.3:  # Only draw confident keypoints
                    x_px = int(x * frame.shape[1] / 640)  # Scale from 640 to actual frame size
                    y_px = int(y * frame.shape[0] / 640)
                    cv2.circle(debug_frame, (x_px, y_px), 3, (0, 0, 255), -1)

        # Save to debug directory
        debug_dir = Path("debug_frames")
        debug_dir.mkdir(exist_ok=True)

        cv2.imwrite(str(debug_dir / filename), debug_frame)
        print(f"💾 Saved debug frame: {filename}")

    except Exception as e:
        print(f"Warning: Could not save debug frame: {e}")

def print_test_summary(results):
    """Print summary statistics"""
    total_frames = len(results)
    total_detections = sum(r['detections'] for r in results)
    frames_with_detections = sum(1 for r in results if r['detections'] > 0)
    avg_process_time = sum(r['process_time'] for r in results) / total_frames
    avg_fps = 1.0 / avg_process_time if avg_process_time > 0 else 0

    print("\n📈 Test Summary")
    print("=" * 30)
    print(f"Total frames processed: {total_frames}")
    print(f"Total detections: {total_detections}")
    print(f"Frames with detections: {frames_with_detections} ({frames_with_detections/total_frames*100:.1f}%)")
    print(f"Average processing time: {avg_process_time*1000:.1f}ms")
    print(f"Average FPS: {avg_fps:.1f}")

    if frames_with_detections == 0:
        print("⚠️  No dogs detected - this could indicate:")
        print("   - No dogs in camera view (expected)")
        print("   - Model not working correctly")
        print("   - Camera/preprocessing issues")
    elif frames_with_detections > total_frames * 0.8:
        print("⚠️  Very high detection rate - check for:")
        print("   - False positives")
        print("   - Detection threshold too low")

def test_with_static_image():
    """Test with a static image if available"""
    print("🖼️  Testing with static image...")

    # Look for test images
    test_image_paths = [
        "test_images/dog.jpg",
        "test_photo_of_elsa.jpg",
        "media/test_dog.jpg"
    ]

    image_path = None
    for path in test_image_paths:
        if Path(path).exists():
            image_path = path
            break

    if not image_path:
        print("No test image found, skipping static test")
        return

    # Initialize AI controller
    ai = AI3StageController()
    if not ai.initialize():
        print("❌ Failed to initialize AI controller")
        return

    try:
        # Load and process image
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Could not load image: {image_path}")
            return

        print(f"📷 Processing image: {image_path} ({frame.shape[1]}x{frame.shape[0]})")

        start_time = time.time()
        detections, poses, behaviors = ai.process_frame(frame)
        process_time = time.time() - start_time

        print(f"✅ Processing completed in {process_time*1000:.1f}ms")
        print(f"   Detections: {len(detections)}")
        print(f"   Poses: {len(poses)}")
        print(f"   Behaviors: {len(behaviors)}")

        # Save result
        save_debug_frame(frame, detections, poses, "static_test_result.jpg")

    except Exception as e:
        print(f"❌ Error processing static image: {e}")
    finally:
        ai.cleanup()

if __name__ == "__main__":
    print("🤖 TreatSensei 3-Stage AI Pipeline Test")
    print("=" * 50)

    # Test with static image first if available
    test_with_static_image()

    print()

    # Test with camera
    success = test_with_camera()

    if success:
        print("\n✅ Pipeline test completed!")
        print("Check debug_frames/ directory for saved detection images")
    else:
        print("\n❌ Pipeline test failed!")

    print("\nFor troubleshooting:")
    print("- Check hailort.log for Hailo-specific errors")
    print("- Verify models exist in ai/models/")
    print("- Check camera permissions and connection")