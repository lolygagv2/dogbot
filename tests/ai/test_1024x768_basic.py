#!/usr/bin/env python3
"""
Basic test script for 1024x768 pose detection
Minimal dependencies, maximum debugging
"""

import json
import numpy as np
import cv2
from pathlib import Path

print("=" * 60)
print("1024x768 POSE DETECTION TEST")
print("=" * 60)

# Load config
config_path = Path("config/config.json")
if config_path.exists():
    with open(config_path) as f:
        config = json.load(f)
    print(f"✅ Config loaded from {config_path}")
else:
    print(f"❌ Config not found at {config_path}")
    exit(1)

# Parse resolution
imgsz = config.get("imgsz", [768, 1024])
if isinstance(imgsz, list):
    height, width = imgsz
    print(f"✅ Resolution: {width}x{height} (WxH)")
else:
    height = width = imgsz
    print(f"✅ Resolution: {width}x{height} (square)")

# Check model files
hef_path = Path(config.get("hef_path", "ai/models/yolo_pose_1024x768.hef"))
behavior_path = Path(config.get("behavior_head_ts", "ai/models/behavior_head.ts"))

print(f"\n📁 Model Files:")
print(f"  HEF: {hef_path}")
print(f"    Exists: {'✅' if hef_path.exists() else '❌'}")
if hef_path.exists():
    print(f"    Size: {hef_path.stat().st_size / 1024 / 1024:.1f} MB")

print(f"  Behavior: {behavior_path}")
print(f"    Exists: {'✅' if behavior_path.exists() else '❌'}")
if behavior_path.exists():
    print(f"    Size: {behavior_path.stat().st_size / 1024:.1f} KB")

# Check dogs and markers
print(f"\n🐕 Configured Dogs:")
for dog in config.get("dogs", []):
    print(f"  {dog['id']}: Marker ID {dog['marker_id']}")

# Check behaviors and cooldowns
print(f"\n🎯 Behaviors:")
behaviors = config.get("behaviors", [])
cooldowns = config.get("cooldown_s", {})
for behavior in behaviors:
    cooldown = cooldowns.get(behavior, 0)
    print(f"  {behavior}: {cooldown}s cooldown")

# Test ArUco detection
print(f"\n🏷️ ArUco Configuration:")
aruco_dict = config.get("aruco_dict", "DICT_4X4_1000")
print(f"  Dictionary: {aruco_dict}")

try:
    dict_const = getattr(cv2.aruco, aruco_dict)
    aruco = cv2.aruco.getPredefinedDictionary(dict_const)
    print(f"  ✅ ArUco dictionary loaded")
except Exception as e:
    print(f"  ❌ Failed to load ArUco: {e}")

# Test Hailo availability
print(f"\n🔧 Hailo Platform:")
try:
    import hailo_platform as hpf
    print(f"  ✅ HailoRT available")

    # Try to get version
    try:
        from hailo_platform import __version__
        print(f"  Version: {__version__}")
    except:
        print(f"  Version: Unknown")

    # Try to create VDevice
    try:
        vdevice = hpf.VDevice()
        print(f"  ✅ VDevice created successfully")
        vdevice = None  # Clean up
    except Exception as e:
        print(f"  ❌ VDevice creation failed: {e}")

except ImportError:
    print(f"  ❌ HailoRT not available")

# Test PyTorch for behavior model
print(f"\n🧠 PyTorch (for behavior model):")
try:
    import torch
    print(f"  ✅ PyTorch available")
    print(f"  Version: {torch.__version__}")

    # Try to load behavior model
    if behavior_path.exists():
        try:
            model = torch.jit.load(str(behavior_path), map_location="cpu")
            print(f"  ✅ Behavior model loaded")
            model = None  # Clean up
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
except ImportError:
    print(f"  ❌ PyTorch not available")

# Test camera
print(f"\n📷 Camera Test:")
try:
    # Try Picamera2 first
    from picamera2 import Picamera2
    camera = Picamera2()
    config_cam = camera.create_preview_configuration(
        main={"size": (1024, 768), "format": "XBGR8888"}
    )
    camera.configure(config_cam)
    camera.start()

    # Capture test frame
    frame = camera.capture_array()
    print(f"  ✅ Picamera2 working")
    print(f"  Frame shape: {frame.shape}")
    print(f"  Frame dtype: {frame.dtype}")

    camera.stop()

except Exception as e:
    print(f"  ❌ Picamera2 failed: {e}")
    print(f"  Trying OpenCV...")

    try:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

        ret, frame = cap.read()
        if ret:
            print(f"  ✅ OpenCV camera working")
            print(f"  Frame shape: {frame.shape}")
        else:
            print(f"  ❌ OpenCV camera read failed")

        cap.release()
    except Exception as e:
        print(f"  ❌ OpenCV failed: {e}")

# Test letterbox function
print(f"\n🖼️ Letterbox Test:")
def letterbox_test(img_shape, target_w, target_h):
    """Test letterbox calculation"""
    h, w = img_shape[:2]
    scale = min(target_w/w, target_h/h)
    new_w, new_h = int(w * scale), int(h * scale)
    dy, dx = (target_h - new_h) // 2, (target_w - new_w) // 2
    return f"  {w}x{h} -> {new_w}x{new_h} @ ({dx},{dy})"

print(letterbox_test((768, 1024), 1024, 768))  # Already correct size
print(letterbox_test((480, 640), 1024, 768))   # Smaller image
print(letterbox_test((1080, 1920), 1024, 768)) # Larger image

# Summary
print(f"\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

checks = {
    "Config loaded": config_path.exists(),
    "HEF model exists": hef_path.exists(),
    "Behavior model exists": behavior_path.exists(),
    "HailoRT available": 'hpf' in locals(),
    "PyTorch available": 'torch' in locals(),
    "Camera available": 'frame' in locals()
}

all_good = all(checks.values())

for check, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"{status} {check}")

print(f"\n{'✅ READY TO RUN' if all_good else '⚠️ SOME ISSUES FOUND'}")

if all_good:
    print("\nNext steps:")
    print("1. Run: python run_pi_1024x768.py")
    print("2. Or:  python test_pose_gui_enhanced.py")
else:
    print("\nPlease fix the issues above before running the main scripts.")