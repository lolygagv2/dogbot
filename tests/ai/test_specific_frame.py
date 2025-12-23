#!/usr/bin/env python3
"""
Test detection on the specific frame that worked before
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ai_controller_3stage_fixed import AI3StageControllerFixed
import cv2

def test_specific_frame():
    # Test with user-verified frames
    test_cases = [
        ("simple_test_results_20251016_174201/frame_00480_174306.jpg", "TWO DOGS", 2),
        ("simple_test_results_20251016_174201/frame_00300_174245.jpg", "ONE DOG", 1),
        ("simple_test_results_20251016_174201/frame_00900_174359.jpg", "NO DOGS (blurry)", 0),
    ]

    # Initialize AI once
    ai = AI3StageControllerFixed()
    if not ai.initialize():
        print("❌ AI init failed")
        return

    print("✅ AI initialized")
    print("\n" + "="*60)
    print("🧪 Testing AI Controller with User-Verified Frames")
    print("="*60)

    for frame_path, description, expected_count in test_cases:
        print(f"\n📋 Test: {description}")
        print(f"📁 File: {frame_path}")
        print(f"🎯 Expected: {expected_count} detection(s)")

        if not os.path.exists(frame_path):
            print(f"❌ Frame not found: {frame_path}")
            continue

        # Load frame
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"❌ Failed to load frame")
            continue

        print(f"📷 Frame loaded: {frame.shape}")

        # Process frame
        detections, poses, behaviors = ai.process_frame(frame)

        # Results
        actual_count = len(detections)
        status = "✅ PASS" if actual_count == expected_count else "❌ FAIL"

        print(f"📊 Results: {status}")
        print(f"   Detections: {actual_count} (expected {expected_count})")

        for i, det in enumerate(detections):
            print(f"     Dog {i+1}: confidence={det.confidence:.3f}, box=({det.x1},{det.y1},{det.x2},{det.y2})")

        print(f"   Poses: {len(poses)}")
        print(f"   Behaviors: {len(behaviors)}")

        print("-" * 40)

    ai.cleanup()
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    test_specific_frame()