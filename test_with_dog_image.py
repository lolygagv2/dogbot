#!/usr/bin/env python3
"""
Test detection with a sample dog image to verify the model works
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ai_controller_3stage_fixed import AI3StageControllerFixed

def create_test_dog_image():
    """Create a simple test image with dog-like features"""
    # Create a 640x640 image
    img = np.zeros((640, 640, 3), dtype=np.uint8)

    # Add some background
    img[:] = [100, 120, 110]  # Greenish background

    # Draw a simple dog-like shape
    # Body (ellipse)
    cv2.ellipse(img, (320, 400), (120, 80), 0, 0, 360, (139, 69, 19), -1)  # Brown body

    # Head (circle)
    cv2.circle(img, (320, 280), 60, (160, 82, 45), -1)  # Brown head

    # Ears
    cv2.ellipse(img, (280, 240), (25, 40), -30, 0, 360, (101, 67, 33), -1)  # Left ear
    cv2.ellipse(img, (360, 240), (25, 40), 30, 0, 360, (101, 67, 33), -1)   # Right ear

    # Eyes
    cv2.circle(img, (300, 270), 8, (0, 0, 0), -1)  # Left eye
    cv2.circle(img, (340, 270), 8, (0, 0, 0), -1)  # Right eye

    # Nose
    cv2.ellipse(img, (320, 290), (8, 6), 0, 0, 360, (0, 0, 0), -1)

    # Legs
    cv2.rectangle(img, (280, 460), (300, 520), (139, 69, 19), -1)  # Front left leg
    cv2.rectangle(img, (340, 460), (360, 520), (139, 69, 19), -1)  # Front right leg
    cv2.rectangle(img, (290, 470), (310, 530), (139, 69, 19), -1)  # Back left leg
    cv2.rectangle(img, (330, 470), (350, 530), (139, 69, 19), -1)  # Back right leg

    # Tail
    cv2.ellipse(img, (420, 380), (30, 15), 45, 0, 360, (139, 69, 19), -1)

    return img

def test_model_with_sample_images():
    """Test the model with various sample images"""
    print("🧪 Testing Detection Model with Sample Images")
    print("=" * 50)

    # Initialize AI
    ai = AI3StageControllerFixed()
    if not ai.initialize():
        print("❌ AI initialization failed")
        return

    print("✅ AI initialized")

    # Create output directory
    output_dir = Path("test_detection_samples")
    output_dir.mkdir(exist_ok=True)

    # Test 1: Simple dog drawing
    print("\n🐕 Test 1: Simple dog drawing")
    dog_img = create_test_dog_image()
    cv2.imwrite(str(output_dir / "test_dog_drawing.jpg"), dog_img)

    detections, poses, behaviors = ai.process_frame(dog_img)
    print(f"   Detections: {len(detections)}")
    for i, det in enumerate(detections):
        print(f"     Dog {i+1}: conf={det.confidence:.3f}, box=({det.x1},{det.y1},{det.x2},{det.y2})")

    # Test 2: Solid color test (should detect nothing)
    print("\n🟦 Test 2: Solid blue image")
    blue_img = np.full((640, 640, 3), [255, 100, 100], dtype=np.uint8)
    detections, poses, behaviors = ai.process_frame(blue_img)
    print(f"   Detections: {len(detections)} (should be 0)")

    # Test 3: Random noise (should detect nothing)
    print("\n🎲 Test 3: Random noise")
    noise_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    detections, poses, behaviors = ai.process_frame(noise_img)
    print(f"   Detections: {len(detections)} (should be 0)")

    # Test 4: Test with different input preprocessing
    print("\n🔄 Test 4: Testing input format variations")

    # Original dog image but test different preprocessing
    test_variations = [
        ("BGR format", dog_img),
        ("RGB format", cv2.cvtColor(dog_img, cv2.COLOR_BGR2RGB)),
        ("Normalized 0-1", (dog_img.astype(np.float32) / 255.0)),
        ("Normalized -1 to 1", (dog_img.astype(np.float32) / 127.5) - 1.0),
    ]

    for name, test_img in test_variations:
        try:
            # Ensure uint8 format for the model
            if test_img.dtype != np.uint8:
                if test_img.max() <= 1.0:
                    test_img = (test_img * 255).astype(np.uint8)
                else:
                    test_img = ((test_img + 1.0) * 127.5).astype(np.uint8)

            detections, poses, behaviors = ai.process_frame(test_img)
            print(f"   {name}: {len(detections)} detections")

        except Exception as e:
            print(f"   {name}: Error - {e}")

    # Test 5: Check if model expects different resolution
    print("\n📏 Test 5: Testing different resolutions")

    resolutions = [(416, 416), (512, 512), (640, 640), (1024, 1024)]

    for w, h in resolutions:
        try:
            resized_dog = cv2.resize(dog_img, (w, h))
            detections, poses, behaviors = ai.process_frame(resized_dog)
            print(f"   {w}x{h}: {len(detections)} detections")
        except Exception as e:
            print(f"   {w}x{h}: Error - {e}")

    ai.cleanup()
    print(f"\n✅ Test complete! Check {output_dir} for test images")

def main():
    test_model_with_sample_images()

if __name__ == "__main__":
    main()