#!/usr/bin/env python3
# test_model.py

from ultralytics import YOLO
import cv2

# Load YOUR trained model
model = YOLO('runs/detect/train/weights/best.pt')

# Test on new image
results = model('test_photo_of_elsa.jpg')

# See what it detected
for r in results:
    for box in r.boxes:
        class_id = int(box.cls)
        confidence = float(box.conf)
        class_name = model.names[class_id]
        print(f"Detected: {class_name} with {confidence:.2f} confidence")