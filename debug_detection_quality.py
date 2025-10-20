#!/usr/bin/env python3
"""
Debug Detection Quality
Analyze pose detection issues - keypoint clustering, bbox accuracy
"""

import cv2
import numpy as np
import json
from pathlib import Path

def analyze_detection_image(image_path):
    """Analyze a saved detection image to understand quality issues"""

    print(f"\n=== ANALYZING: {image_path} ===")

    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return

    h, w = img.shape[:2]
    print(f"Image dimensions: {w}x{h}")

    # Look for red dots (keypoints) - they should be BGR(0,0,255)
    red_mask = cv2.inRange(img, (0, 0, 200), (50, 50, 255))
    red_points = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    print(f"Found {len(red_points)} red keypoint clusters")

    if len(red_points) > 0:
        # Analyze keypoint distribution
        centers = []
        for contour in red_points:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centers.append((cx, cy))

        if len(centers) > 1:
            # Calculate spread of keypoints
            centers_np = np.array(centers)
            mean_x, mean_y = np.mean(centers_np, axis=0)
            std_x, std_y = np.std(centers_np, axis=0)

            print(f"Keypoint center: ({mean_x:.1f}, {mean_y:.1f})")
            print(f"Keypoint spread: X±{std_x:.1f}, Y±{std_y:.1f}")

            # Check if keypoints are clustered (low spread)
            if std_x < 50 and std_y < 50:
                print("⚠️  ISSUE: Keypoints are tightly clustered - may be detecting ArUco marker instead of dog body")

            # Check relative position
            collar_region = (mean_y < h * 0.4)  # Top 40% of image
            if collar_region:
                print("⚠️  ISSUE: Keypoints concentrated in collar/head region")

    # Look for bounding boxes (should be green rectangles)
    green_mask = cv2.inRange(img, (0, 200, 0), (100, 255, 100))
    green_contours = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    print(f"Found {len(green_contours)} bounding box elements")

    # Analyze bounding box quality
    for i, contour in enumerate(green_contours):
        x, y, w_box, h_box = cv2.boundingRect(contour)
        aspect_ratio = w_box / h_box if h_box > 0 else 0
        area_ratio = (w_box * h_box) / (w * h)

        print(f"  Box {i+1}: {w_box}x{h_box} at ({x},{y})")
        print(f"    Aspect ratio: {aspect_ratio:.2f} (dog should be ~1.2-2.0)")
        print(f"    Area ratio: {area_ratio:.3f} (should be 0.1-0.8 for dog)")

        if aspect_ratio < 0.8 or aspect_ratio > 3.0:
            print(f"    ⚠️  ISSUE: Unusual aspect ratio for dog")

        if area_ratio < 0.05:
            print(f"    ⚠️  ISSUE: Bounding box too small")
        elif area_ratio > 0.9:
            print(f"    ⚠️  ISSUE: Bounding box too large")

def main():
    """Analyze recent detection images"""

    pose_output_dir = Path("pose_output")
    if not pose_output_dir.exists():
        print("No pose_output directory found")
        return

    # Get recent images
    image_files = sorted(pose_output_dir.glob("*.jpg"))[-5:]  # Last 5 images

    if not image_files:
        print("No detection images found")
        return

    print("=== POSE DETECTION QUALITY ANALYSIS ===")
    print(f"Analyzing {len(image_files)} recent images...")

    for image_path in image_files:
        analyze_detection_image(image_path)

    print("\n=== RECOMMENDATIONS ===")
    print("1. Check if model is trained for dogs vs humans")
    print("2. Verify confidence threshold (currently 0.25)")
    print("3. Check coordinate transformation/scaling")
    print("4. Consider different pose detection model")
    print("5. Test with dog in different positions")

if __name__ == "__main__":
    main()