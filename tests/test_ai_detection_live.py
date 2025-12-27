#!/usr/bin/env python3
"""
Quick test: Verify AI detection produces vision events with pose confidence
Run this to see what the AI sees dogs doing in real-time
"""

import sys
import os
import time
import signal

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_controller_3stage_fixed import AI3StageControllerFixed
import cv2

# Camera setup
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    print("Picamera2 not available")


def main():
    """Test AI detection and display pose confidence"""

    if not PICAMERA_AVAILABLE:
        print("ERROR: Picamera2 required")
        return 1

    print("=" * 60)
    print("AI DETECTION LIVE TEST")
    print("Testing: Dog detection + Pose + Behavior confidence")
    print("=" * 60)

    # Initialize AI controller
    print("\n[1/3] Initializing AI controller...")
    ai = AI3StageControllerFixed()
    if not ai.initialize():
        print("ERROR: AI controller failed to initialize")
        return 1
    print("✅ AI controller ready")

    # Initialize camera
    print("\n[2/3] Initializing camera...")
    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": (640, 640), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    time.sleep(1.0)  # Let camera warm up
    print("✅ Camera ready")

    # Detection loop
    print("\n[3/3] Starting detection loop...")
    print("-" * 60)
    print("BEHAVIORS: stand, sit, lie, cross, spin")
    print("Press Ctrl+C to stop")
    print("-" * 60)

    running = True
    def signal_handler(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, signal_handler)

    frame_count = 0
    last_fps_time = time.time()
    fps = 0.0

    # Track best confidence per behavior
    best_detections = {
        "stand": 0.0,
        "sit": 0.0,
        "lie": 0.0,
        "cross": 0.0,
        "spin": 0.0
    }

    try:
        while running:
            # Capture frame
            frame = camera.capture_array()
            # Rotate if needed
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Run detection
            detections, poses, behaviors = ai.process_frame(frame)

            # Update FPS
            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                frame_count = 0
                last_fps_time = now

            # Display results
            if detections:
                for i, det in enumerate(detections):
                    dog_conf = det.confidence
                    print(f"\n🐕 DOG {i+1} detected (conf: {dog_conf:.2f}) at ({det.center[0]}, {det.center[1]})")

                    # Show pose keypoints summary if available
                    if i < len(poses):
                        pose = poses[i]
                        kpts = pose.keypoints
                        valid_kpts = sum(1 for k in kpts if k[2] > 0.3)
                        print(f"   Pose: {valid_kpts}/24 keypoints visible")

                    # Show behaviors if any
                    if behaviors:
                        for bhv in behaviors:
                            behavior_name = bhv.behavior
                            conf = bhv.confidence

                            # Update best
                            if behavior_name in best_detections:
                                if conf > best_detections[behavior_name]:
                                    best_detections[behavior_name] = conf

                            print(f"   🎯 BEHAVIOR: {behavior_name.upper()} (conf: {conf:.2f})")
                    else:
                        print(f"   ⏳ Waiting for temporal behavior analysis...")

            # Show status line
            print(f"\rFPS: {fps:.1f} | Dogs: {len(detections)} | Best: sit={best_detections['sit']:.2f} lie={best_detections['lie']:.2f} stand={best_detections['stand']:.2f}", end="", flush=True)

            time.sleep(0.033)  # ~30 FPS

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\n\n" + "=" * 60)
        print("BEST CONFIDENCE SCORES:")
        for behavior, conf in best_detections.items():
            bar = "█" * int(conf * 20)
            print(f"  {behavior:6}: {conf:.2f} {bar}")
        print("=" * 60)

        # Cleanup
        camera.stop()
        camera.close()
        ai.cleanup()
        print("✅ Cleanup complete")

    return 0


if __name__ == "__main__":
    exit(main())
