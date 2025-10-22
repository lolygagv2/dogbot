#!/usr/bin/env python3
"""
Show which AI models are available and being used
This helps identify why you got 21,504 fake detections
"""

import json
from pathlib import Path

def main():
    print("🔍 AI Model Investigation")
    print("=" * 50)

    # Check AI3StageControllerFixed models
    print("✅ CORRECT AI System (AI3StageControllerFixed):")

    config_file = Path("ai/models/config.json")
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)

        detect_model = config.get("detect_path", "dogdetector_14.hef")
        pose_model = config.get("hef_path", "dogpose_14.hef")

        print(f"  Detection model: {detect_model}")
        print(f"  Pose model: {pose_model}")

        # Check if files exist
        detect_path = Path("ai/models") / detect_model
        pose_path = Path("ai/models") / pose_model

        print(f"  Detection file exists: {detect_path.exists()}")
        print(f"  Pose file exists: {pose_path.exists()}")
    else:
        print("  Config file not found - using defaults")
        print("  Detection model: dogdetector_14.hef")
        print("  Pose model: dogpose_14.hef")

    print()

    # Check broken system models
    print("❌ BROKEN System (run_pi_1024_fixed.py):")

    main_config = Path("config/config.json")
    if main_config.exists():
        with open(main_config, 'r') as f:
            main_cfg = json.load(f)

        broken_model = main_cfg.get("hef_path", "ai/models/yolo_pose.hef")
        prob_th = main_cfg.get("prob_th", 0.6)
        rotation = main_cfg.get("camera_rotation_deg", 90)

        print(f"  Model: {broken_model}")
        print(f"  Threshold: {prob_th}")
        print(f"  Rotation: {rotation}°")
        print(f"  ⚠️  This gives 21,504 fake detections!")

    print()
    print("🎯 SOLUTION:")
    print("  Run: python test_mission_with_controls.py")
    print("  NOT: python run_pi_1024_fixed.py")
    print()
    print("  The correct system uses:")
    print("  - AI3StageControllerFixed")
    print("  - dogdetector_14.hef + dogpose_14.hef")
    print("  - Proper NMS post-processing")
    print("  - 90° rotation built-in")

if __name__ == "__main__":
    main()