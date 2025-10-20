#!/usr/bin/env python3

from ultralytics import YOLO
import yaml

# Create dataset config
dataset_config = {
    'path': '/home/morgan/dogbot/dog_dataset',
    'train': 'images',
    'val': 'images',  # You'd split this properly
    
    'names': {
        0: 'elsa_sitting',
        1: 'elsa_lying', 
        2: 'elsa_standing',
        3: 'elsa_spinning',
        4: 'bezik_sitting',
        5: 'bezik_lying',
        6: 'bezik_standing', 
        7: 'bezik_spinning'
    }
}

with open('dog_dataset.yaml', 'w') as f:
    yaml.dump(dataset_config, f)

# Train the model
model = YOLO('yolov8n.pt')  # Start with small model
model.train(
    data='dog_dataset.yaml',
    epochs=100,
    imgsz=640,
    device='cpu'  # Use 'cuda' if you have GPU
)

print("Training complete!")
print(f"Model saved to: runs/detect/train/weights/best.pt")