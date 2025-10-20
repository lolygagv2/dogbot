# DogBot Project State

## Project Overview
AI-powered robotic dog trainer with computer vision for pose detection and behavior analysis.

## 🔴 CRITICAL CURRENT ISSUE (October 11, 2025)
**System is non-functional: 21,504 false detections bug persists**

### Test Results from User:
- **Test 1**: Dogs brought into view → NO detections
- **Test 2**: Detection script → Still showing 21,504 false positives
- **Inference Time**: Improved to ~175ms (good)
- **Fundamental Problem**: Quantization scheme mismatch

## System Architecture

### Hardware
- **Compute**: Raspberry Pi 5 with Hailo-8 AI accelerator (26 TOPS)
- **Camera**: Pi Camera Module (IMX500)
- **Motors**: Tank steering control system
- **Audio**: DFPlayer + Pi audio dual system
- **LEDs**: Status indication system
- **Servos**: Camera pan/tilt control

### Software Stack

#### Core Models (Two-Stage Pipeline)
1. **Pose Detection**: `yolo_pose.hef`
   - YOLOv11s-pose compiled for Hailo-8 (26 TOPS)
   - Input: 1024×1024 RGB images
   - Outputs: 9 quantized tensors (uint8/uint16)
   - Structure: 3 scales × [objectness, boxes, keypoints]
   - 24 keypoints per dog
   - Compiled with YOLOv8 pipeline (proprietary quantization)

2. **Behavior Classification**: `behavior_head.ts`
   - TorchScript LSTM model
   - Input: 48 features (24 keypoints × 2 coords)
   - Output: 5 behaviors (stand, sit, lie, cross, spin)

## Quantized Model Analysis

### Raw Output Inspection Results:
```
Objectness (uint8): Range 164-231, Mean ~200, Zero-point likely ~200
Boxes (uint8): Range 79-215, Mean ~155
Keypoints (uint16): Range 4344-26578, Mean varies by scale
```

### The 21,504 Problem:
- Model outputs for ALL grid cells: 128×128 + 64×64 + 32×32 = 21,504
- Without proper dequantization, all cells appear as detections
- Objectness clustered around 200-218 suggests wrong zero-point assumption

## Critical Technical Decisions

### ✅ Confirmed Correct:
1. **RGB Input**: Model expects RGB (Picamera2 provides RGB888 directly)
2. **No Camera Rotation**: Hardware already correctly oriented (0°)
3. **Two-Stage Pipeline**: YOLO pose → behavior classification
4. **Mixed Quantization**: uint8 for objectness/boxes, uint16 for keypoints

### ❌ Issues to Fix:
1. **Quantization Zero-Point**: Currently assuming 128, data suggests ~200
2. **Scale Factors**: Complete guesses, need actual values from compilation
3. **Threshold After Dequantization**: Need to determine proper confidence threshold

## Files Created This Session

### Detection Scripts:
- `run_pi_1024_fixed_v2.py` - First fix attempt (failed)
- `run_dog_detection_final.py` - Latest with quantization handling
- `test_detection_quick.py` - Quick validation script

### Analysis Tools:
- `check_model_outputs.py` - Raw output inspection tool

## Code Issues Identified

### Current (Wrong) Dequantization:
```python
OBJECTNESS_ZP = 128  # Should be ~200?
OBJECTNESS_SCALE = 32.0  # Complete guess

obj_score = (raw - 128) / 32.0  # Wrong formula?
```

### Should Investigate:
```python
# Based on observed data:
OBJECTNESS_ZP = 200  # Match observed center
obj_score = (raw - 200) / scale
```

## User's Position
> "if the models fundamentally aren't correct, we will redo them to align, however i suspect your code is still shit"

User is willing to retrain but believes the parsing code is the core issue.

## Next Immediate Actions

1. **Test with Correct Zero-Point**:
   - Try zero-point = 200 based on observed data
   - Adjust scale factor based on results

2. **Capture Test Data**:
   - Image WITH dog present
   - Image WITHOUT dog
   - Compare raw outputs to identify actual changes

3. **Verify Model Compilation**:
   - Check calibration dataset used
   - Confirm quantization parameters
   - Verify if NMS is built-in

## Performance Metrics

### Current Status:
- Inference Time: ~175ms ✅
- Detection Accuracy: 0% (completely broken)
- False Positive Rate: 100% (21,504 per frame)

### Target:
- Inference Time: <200ms
- Detection Accuracy: >90%
- False Positive Rate: <5%

## Development Principles (From User Feedback)

1. **Don't Overthink**: Original solutions often correct
2. **Complete Integration**: Include all features, don't strip functionality
3. **Direct Fixes**: Action over explanation
4. **Test with Real Data**: Dogs must actually be detected

## Project File Structure
```
/home/morgan/dogbot/
├── config/config.json       # Main configuration
├── ai/models/
│   ├── yolo_pose.hef       # Quantized YOLOv11
│   └── behavior_head.ts    # Behavior classifier
├── .claude_context/
│   ├── resume_chat.md       # Session continuity
│   └── state/
│       └── project_state.md # This file
└── [detection scripts...]
```

## Critical Information Needed

1. **Exact quantization parameters from HEF compilation**
2. **Calibration dataset used during compilation**
3. **Expected output format documentation**
4. **Whether NMS is built into the model**

## Session Summary

Started with 21,504 false detections, tried multiple fixes:
- Fixed color space (RGB confirmed)
- Removed unnecessary rotation (was correct at 0°)
- Improved inference speed dramatically
- Identified quantization as root cause
- Still unable to detect actual dogs

**Bottom Line**: System completely non-functional for dog detection. Quantization handling is primary suspect. Need correct dequantization parameters or model recompilation.

## Hardware Capability Assessment

### Hailo-8 (26 TOPS) Performance:
- **Significantly more powerful** than Hailo-8L
- Can handle larger models and higher resolutions
- Should easily support 1024×1024 inference
- Previous 640×640 failures were parsing issues, not hardware limits

### Resolution Recommendations:
- **1024×1024**: ✅ **Optimal** - Fully utilizes your 26 TOPS capability
- **640×640**: Underutilizes the hardware but safer for debugging
- **Higher resolutions**: Possible but test inference times

### Recompilation Strategy for Hailo-8:
1. **Target 1024×1024** - matches your current model and takes advantage of hardware
2. **Use standard Hailo compilation** (avoid proprietary quantization)
3. **Test both resolutions** if 1024×1024 compilation fails
4. **Known quantization parameters** for easier debugging

---
**Last Updated**: October 11, 2025 - Current Session
**Hardware**: Hailo-8 (26 TOPS) - **CORRECTED**
**Priority**: FIX THE 21,504 DETECTION BUG OR RECOMPILE