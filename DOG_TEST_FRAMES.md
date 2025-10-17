# Dog Detection Test Frames

## Known Test Cases for Detection Validation

**User-verified frames from simple_test_results_20251016_174201/**

### Two Dogs Present
- **File**: `frame_00480_174306.jpg`
- **Expected**: 2 dog detections
- **Description**: TWO DOGS clearly visible

### One Dog Present
- **File**: `frame_00300_174245.jpg`
- **Expected**: 1 dog detection
- **Description**: 1 DOG clearly visible

### No Dogs (Negative Test)
- **File**: `frame_00900_174359.jpg`
- **Expected**: 0 detections
- **Description**: Basically "no dogs" - dog is blurry white blur walking away

## Testing Notes
- Full dog needs to be detected (not just parts like tail/butt)
- These frames provide ground truth for debugging detection pipeline
- Use these specific frames to validate detection fixes before live testing