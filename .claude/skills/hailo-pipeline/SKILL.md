---
name: hailo-pipeline
description: Work with the WIM-Z 3-stage AI detection pipeline (Hailo-8 + TorchScript). Use when modifying AI detection, pose estimation, behavior classification, model files, or the vision pipeline.
---

# WIM-Z AI Pipeline (3-Stage Architecture)

## Pipeline Overview
```
Stage 1: YOLOv8 Detection (Hailo-8 NPU) → Dog bounding boxes
Stage 2: YOLOv8-Pose Estimation (Hailo-8 NPU) → Keypoint extraction
Stage 3: Behavior Classification (TorchScript on CPU) → sit/lie/stand/walk/spin
```

## Model Files (PROTECTED — never overwrite without versioned backup)
- **Detection:** `ai/models/dogdetector_14.hef` — YOLOv8 compiled for Hailo-8
- **Pose:** `ai/models/dogpose_14.hef` — YOLOv8-pose compiled for Hailo-8
- **Behavior:** `ai/models/behavior_14.ts` — TorchScript temporal classifier (runs on CPU)

## Core Implementation
- **Pipeline controller:** `core/ai_controller_3stage_fixed.py`
- **Detection service:** `services/perception/detector.py`
- **Inference speed:** 30+ FPS @ 640x640 on Hailo-8

## HEF Model Compilation (requires separate Docker environment)
```bash
# HEF compilation uses Hailo Dataflow Compiler in Docker container
# Do NOT attempt compilation on the Pi itself
# Source models: trained YOLOv8 → export ONNX → compile HEF via Dataflow Compiler
```

## Behavior Detection Outputs
- `sitting` — dog in sit position (front legs extended, rear tucked)
- `lying` — dog lying down
- `standing` — dog upright on all fours
- `walking` — dog in motion
- `spinning` — dog rotating (trick)

## Dog Identification
- **Method:** ArUco markers on collar tags
- **Per-dog profiles:** Stored in app, synced to robot
- **Dogs:** Elsa and Bezik (Pomeranians)

## Rules for AI Code Changes
1. Never modify `.hef` files directly — they are compiled binaries
2. The behavior classifier (`behavior_14.ts`) runs on CPU, not Hailo — different optimization path
3. Detection confidence thresholds are tuned per-mode (Coach vs Silent Guardian)
4. Bark detection is a SEPARATE system (`services/perception/bark_detector.py`) using TFLite, not Hailo
5. Camera input flows through `core/vision/camera_manager.py` — unified interface for all modes
6. Test pipeline changes with `test_3stage_fixed.py` before deploying

## Common Issues
- "HEF format not compatible" → wrong Hailo version (must be Hailo-8, NOT Hailo-8L)
- Slow inference → check if camera is in high-res photo mode vs detection mode
- Missing detections → verify `camera_mode_controller.py` has switched to detection resolution
