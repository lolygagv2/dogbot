# Pose Validation Filter - Integration Guide

## The Problem

Your YOLOv8 + TorchScript pipeline is producing false "CROSS" classifications because:

1. **Harness tags** are being detected as keypoints (the `[ELSA]` / `[BEZIK]` tags)
2. **Motion blur** causes garbage skeleton estimations
3. **Empty/no-dog frames** still get classified with 100% confidence
4. **Background objects** sometimes get detected as dogs

## The Solution

The `pose_validator.py` module adds a filtering layer between pose estimation and classification.

## Quick Start

```python
from pose_validator import PoseValidator
from ultralytics import YOLO
import torch
import cv2

# Load your models
pose_model = YOLO('yolov8n-pose.pt')  # or your custom pose model
classifier = torch.jit.load('your_cross_classifier.torchscript')

# Initialize validator
validator = PoseValidator(
    min_keypoint_confidence=0.4,   # Reject low-confidence keypoints
    min_visible_keypoints=6,        # Need at least 6 good keypoints
    blur_threshold=100.0,           # Laplacian variance threshold
    min_bbox_area=5000,             # Minimum detection size in pixels
    temporal_window=5,              # Frames for temporal smoothing
    temporal_threshold=0.6,         # 60% of frames must agree
)

# Process video
cap = cv2.VideoCapture('your_video.mp4')
validator.reset_temporal()  # Reset at start of each video/dog

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLOv8 pose
    results = pose_model(frame, verbose=False)
    
    if len(results) == 0 or results[0].keypoints is None:
        continue  # No detection
    
    # Extract pose data
    keypoints = results[0].keypoints.xy[0].cpu().numpy()
    confidences = results[0].keypoints.conf[0].cpu().numpy()
    bbox = results[0].boxes.xyxy[0].cpu().numpy()
    
    # VALIDATE BEFORE CLASSIFYING
    validation = validator.validate_frame(frame, keypoints, confidences, bbox)
    
    if not validation.is_valid:
        print(f"Frame rejected: {validation.reason}")
        continue
    
    # Only now run your classifier
    # ... your classification code here ...
    raw_prediction = "CROSS"  # or "NO_CROSS"
    raw_confidence = 0.95
    
    # Apply temporal smoothing
    smooth_pred, smooth_conf, is_stable = validator.temporal_vote(
        raw_prediction, raw_confidence
    )
    
    if is_stable:
        print(f"Prediction: {smooth_pred} ({smooth_conf:.1%})")
    else:
        print(f"Unstable: {smooth_pred} - waiting for more frames")

cap.release()
```

## What Each Filter Catches

| Filter | What It Catches | Your Video Examples |
|--------|-----------------|---------------------|
| Detection Region Quality | Empty frames, motion blur in detection area | `elsanocrosswtf2.mp4` frames 102, 153 |
| Edge Density | No actual object structure | `crosswtf.mp4` frames 24, 48 |
| Keypoint Confidence | Uncertain pose estimations | Most low-quality frames |
| Skeleton Geometry | Impossible poses, skeleton on wrong object | `reallybad.mp4` |
| Temporal Voting | Flickering predictions, noise | Prevents single-frame false positives |

## Tuning the Thresholds

Based on your videos, I recommend starting with:

```python
validator = PoseValidator(
    min_keypoint_confidence=0.4,  # Increase to 0.5 if too many false positives
    min_visible_keypoints=6,      # Decrease to 4 for partially visible dogs
    blur_threshold=100.0,         # Increase to 150 for stricter blur filtering
    min_bbox_area=5000,           # Adjust based on your camera resolution
    temporal_window=5,            # Increase for more stability, decrease for faster response
    temporal_threshold=0.6,       # Increase for stricter temporal consistency
)
```

## Files Included

- `pose_validator.py` - The main validation module
- `test_validator.py` - Test script that analyzes your problem videos
- `INTEGRATION_GUIDE.md` - This file

## What This Doesn't Fix

The validator **prevents** bad predictions, but doesn't fix the underlying pose estimation issues:

- YOLOv8 pose still struggles with fluffy dogs
- The harness tag will still confuse keypoint detection
- You may want to consider training a custom pose model specifically for Pomeranians

## Next Steps

1. Copy `pose_validator.py` to your project
2. Integrate as shown above
3. Run on your problem videos and tune thresholds
4. Consider adding harness tag masking if tag-on-keypoint issue persists
