#!/usr/bin/env python3
"""
Training data preparation tool for YOLO model retraining.
Processes captured images and ensures proper format for training.
"""

import cv2
import numpy as np
import os
import shutil
import json
from pathlib import Path
import argparse
from datetime import datetime
import yaml


def letterbox(im, new_shape=(1024, 1024), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """
    Resize and pad image to a specific shape while preserving aspect ratio.
    This matches YOLO's preprocessing exactly.

    Args:
        im: Input image
        new_shape: Target shape (height, width)
        color: Padding color
        auto: Minimum rectangle padding
        scaleFill: Stretch to fill
        scaleup: Allow scaling up
        stride: Stride multiple for auto padding

    Returns:
        Letterboxed image, scale ratio, padding (width, height)
    """
    shape = im.shape[:2]  # Current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # Only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # padding

    if auto:  # Minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:  # Stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # Divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # Resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, ratio, (dw, dh)


class TrainingDataPreparer:
    def __init__(self, input_dirs, output_dir="dataset",
                 train_split=0.8, val_split=0.15, test_split=0.05):
        """
        Initialize the training data preparation tool.

        Args:
            input_dirs: List of directories containing images
            output_dir: Output directory for organized dataset
            train_split: Training set percentage
            val_split: Validation set percentage
            test_split: Test set percentage
        """
        self.input_dirs = [Path(d) for d in input_dirs]
        self.output_dir = Path(output_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        # Create output directory structure
        self.setup_directories()

        # Track statistics
        self.stats = {
            "total_images": 0,
            "negative_images": 0,
            "positive_images": 0,
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
            "processed_files": []
        }

    def setup_directories(self):
        """Create YOLO-compatible directory structure."""
        # Main directories
        for split in ['train', 'val', 'test']:
            (self.output_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.output_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

        # Preprocessed samples for calibration
        (self.output_dir / 'calibration').mkdir(exist_ok=True)

        print(f"📁 Created dataset structure in {self.output_dir}")

    def process_image(self, image_path, output_path, create_letterbox_sample=False):
        """
        Process an image to ensure correct format.

        Args:
            image_path: Path to input image
            output_path: Path to save processed image
            create_letterbox_sample: Whether to create a letterboxed sample
        """
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"⚠️  Failed to read {image_path}")
            return False

        # Convert BGR to RGB (critical for Picamera images)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create letterboxed version for calibration
        if create_letterbox_sample:
            letterboxed, ratio, pad = letterbox(img_rgb, new_shape=(1024, 1024))

            # Save letterboxed version for calibration dataset
            calib_path = self.output_dir / 'calibration' / f"calib_{image_path.name}"
            cv2.imwrite(str(calib_path), cv2.cvtColor(letterboxed, cv2.COLOR_RGB2BGR))

            # Also create normalized version
            normalized = letterboxed.astype(np.float32) / 255.0
            norm_path = self.output_dir / 'calibration' / f"norm_{image_path.stem}.npy"
            np.save(str(norm_path), normalized)

        # Save original (YOLO will handle letterboxing during training)
        cv2.imwrite(str(output_path), img)
        return True

    def create_empty_label(self, label_path):
        """Create empty label file for negative samples."""
        with open(label_path, 'w') as f:
            # Empty file for no detections
            pass

    def organize_dataset(self):
        """Organize images into train/val/test splits."""
        print("\n📊 Organizing dataset...")

        all_images = []

        # Collect all images
        for input_dir in self.input_dirs:
            if not input_dir.exists():
                print(f"⚠️  Directory not found: {input_dir}")
                continue

            # Find all images
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images = list(input_dir.glob(ext))
                all_images.extend(images)

                # Skip letterbox preview images
                all_images = [img for img in all_images if 'letterbox' not in img.name]

        print(f"📷 Found {len(all_images)} images")

        # Shuffle for random split
        import random
        random.shuffle(all_images)

        # Calculate split indices
        n_total = len(all_images)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)

        # Split images
        train_images = all_images[:n_train]
        val_images = all_images[n_train:n_train + n_val]
        test_images = all_images[n_train + n_val:]

        # Process each split
        for split_name, split_images in [('train', train_images),
                                         ('val', val_images),
                                         ('test', test_images)]:
            print(f"\n🔄 Processing {split_name} set ({len(split_images)} images)...")

            for i, img_path in enumerate(split_images):
                # Output paths
                img_output = self.output_dir / 'images' / split_name / img_path.name
                label_output = self.output_dir / 'labels' / split_name / f"{img_path.stem}.txt"

                # Process image
                create_calib = (split_name == 'train' and i < 500)  # First 500 train images for calibration
                if self.process_image(img_path, img_output, create_calib):
                    # Create empty label (negative sample)
                    self.create_empty_label(label_output)

                    # Update stats
                    self.stats["total_images"] += 1
                    self.stats["negative_images"] += 1
                    self.stats[f"{split_name}_count"] += 1
                    self.stats["processed_files"].append(str(img_path))

                # Progress indicator
                if (i + 1) % 100 == 0:
                    print(f"   Processed {i + 1}/{len(split_images)}")

        print(f"\n✅ Dataset organized successfully!")
        self.print_statistics()

    def create_yaml_config(self):
        """Create YOLO training configuration file."""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,  # Number of classes (just 'dog')
            'names': ['dog'],

            # Keypoint configuration for pose model
            'kpt_shape': [24, 3],  # 24 keypoints with x,y,visibility
        }

        yaml_path = self.output_dir / 'dataset.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"📝 Created YOLO config: {yaml_path}")
        return yaml_path

    def create_training_script(self):
        """Generate training script with recommended parameters."""
        script = '''#!/bin/bash
# YOLO training script with recommended parameters

# Training with proper square input
yolo pose train \\
    model=yolo11s-pose.pt \\
    data=dataset/dataset.yaml \\
    imgsz=1024 \\
    rect=False \\
    epochs=50 \\
    batch=16 \\
    lr0=0.001 \\
    mosaic=0.1 \\
    mixup=0.0 \\
    flipud=0.0 \\
    degrees=0.0 \\
    device=0 \\
    workers=8 \\
    patience=20 \\
    save=True \\
    cache=True \\
    project=runs/pose \\
    name=dogpose_1024_clean

# Export to ONNX after training
yolo export \\
    model=runs/pose/dogpose_1024_clean/weights/best.pt \\
    format=onnx \\
    imgsz=1024 \\
    simplify=True \\
    opset=11

echo "Training complete! Model saved to runs/pose/dogpose_1024_clean/"
echo "Next steps:"
echo "1. Review metrics in runs/pose/dogpose_1024_clean/"
echo "2. Test model on live camera feed"
echo "3. Compile ONNX to HEF using Hailo Dataflow Compiler"
'''

        script_path = self.output_dir / 'train_model.sh'
        with open(script_path, 'w') as f:
            f.write(script)

        # Make executable
        os.chmod(script_path, 0o755)
        print(f"📜 Created training script: {script_path}")

    def print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "="*50)
        print("📊 DATASET STATISTICS")
        print("="*50)
        print(f"Total images:     {self.stats['total_images']}")
        print(f"Negative samples: {self.stats['negative_images']}")
        print(f"Positive samples: {self.stats['positive_images']}")
        print("-"*50)
        print(f"Training set:     {self.stats['train_count']} ({self.stats['train_count']/max(1,self.stats['total_images'])*100:.1f}%)")
        print(f"Validation set:   {self.stats['val_count']} ({self.stats['val_count']/max(1,self.stats['total_images'])*100:.1f}%)")
        print(f"Test set:         {self.stats['test_count']} ({self.stats['test_count']/max(1,self.stats['total_images'])*100:.1f}%)")
        print("-"*50)
        print(f"Calibration samples: {len(list((self.output_dir / 'calibration').glob('*.jpg')))}")
        print("="*50)

    def save_metadata(self):
        """Save processing metadata."""
        metadata = {
            "processing_date": datetime.now().isoformat(),
            "statistics": self.stats,
            "splits": {
                "train": self.train_split,
                "val": self.val_split,
                "test": self.test_split
            },
            "preprocessing": {
                "letterbox_size": [1024, 1024],
                "padding_color": [114, 114, 114],
                "normalization": "0-255 to 0-1"
            }
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"💾 Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data for YOLO model")
    parser.add_argument("--input", nargs='+', required=True,
                       help="Input directories containing images")
    parser.add_argument("--output", default="dataset",
                       help="Output directory for organized dataset")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Training set percentage (default: 0.8)")
    parser.add_argument("--val-split", type=float, default=0.15,
                       help="Validation set percentage (default: 0.15)")
    parser.add_argument("--test-split", type=float, default=0.05,
                       help="Test set percentage (default: 0.05)")

    args = parser.parse_args()

    # Validate splits
    total = args.train_split + args.val_split + args.test_split
    if abs(total - 1.0) > 0.001:
        print(f"⚠️  Warning: Splits sum to {total:.2f}, normalizing...")
        args.train_split /= total
        args.val_split /= total
        args.test_split /= total

    preparer = TrainingDataPreparer(
        input_dirs=args.input,
        output_dir=args.output,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split
    )

    # Process dataset
    preparer.organize_dataset()
    preparer.create_yaml_config()
    preparer.create_training_script()
    preparer.save_metadata()

    print("\n✅ Dataset preparation complete!")
    print(f"📁 Dataset ready in: {preparer.output_dir}")
    print("\n🚀 Next steps:")
    print("1. Add positive samples (images with dogs) if available")
    print("2. Review and run: dataset/train_model.sh")
    print("3. Monitor training metrics in TensorBoard")
    print("4. Compile best model to HEF for deployment")


if __name__ == "__main__":
    main()