# DogBot File Guide - What to Run for Testing

## 🎯 **RECOMMENDED FOR TESTING**

### **Main Testing File** (Start Here)
```bash
python3 test_pose_headless.py --duration 30
```
- **Purpose**: Complete pose detection test with image saving
- **Output**: Images saved to `pose_output/` directory
- **Features**: 1024x768 detection, ArUco markers, behavior analysis
- **Status**: ✅ WORKING (detections confirmed)

### **Quick System Check**
```bash
python3 test_1024x768_basic.py
```
- **Purpose**: Verify all components are ready
- **Output**: System status report
- **Use**: Run first to check everything is working

### **Low Threshold Testing** (If no detections)
```bash
python3 test_low_threshold.py
```
- **Purpose**: Test with very low confidence threshold
- **Output**: Shows actual detection capabilities
- **Use**: Debug detection issues

## 📁 **All Created Files Explained**

### **Core System Files**
1. **`run_pi_1024x768.py`** - Main 1024x768 pose detection system
2. **`config/config.json`** - Configuration with correct model paths

### **Testing Files**
3. **`test_pose_headless.py`** - 🎯 **MAIN TESTING SCRIPT**
4. **`test_1024x768_basic.py`** - System diagnostics
5. **`test_low_threshold.py`** - Debug very low confidence
6. **`debug_confidence.py`** - Analyze confidence values

### **GUI Files** (For when you have HDMI display)
7. **`test_pose_gui_enhanced.py`** - Full GUI with tabs and controls
8. **`test_pose_quick.py`** - Simple camera test (has OpenCV display issues)

### **Documentation**
9. **`docs/feature_roadmap_and_commands.md`** - Complete feature specifications
10. **`docs/file_guide.md`** - This file

## 🚀 **Quick Testing Workflow**

### **Step 1: System Check**
```bash
python3 test_1024x768_basic.py
```
Should show all ✅ checkmarks.

### **Step 2: Run Main Test**
```bash
python3 test_pose_headless.py --duration 30
```
Will save images and show detection results.

### **Step 3: Check Results**
```bash
ls -la pose_output/
```
View saved images to see what camera captured.

## 🔧 **Current Issues & Status**

### ✅ **Working Components**
- 1024x768 pose detection
- Hailo-8 hardware acceleration
- Camera capture (Picamera2)
- Model loading and inference
- Image preprocessing and saving
- ArUco marker detection
- Servo controller integration

### 🔧 **Needs Tuning**
- **Confidence threshold**: Currently too high (0.25), needs lowering to ~0.0001
- **Behavior classification**: Input format mismatch (expects 48D, gets 2D)

### 🆕 **TODO Features**
- **High-res photo mode**: 4096x3008 for professional shots
- **Video recording**: High-quality training documentation
- **Dual camera modes**: Switch between AI detection and photo modes

## 📷 **Camera Mode Architecture**

### **Current Implementation**
```python
# AI Detection Mode (1024x768)
config = camera.create_preview_configuration(
    main={"size": (1024, 768), "format": "XBGR8888"}
)
```

### **Planned High-Res Mode**
```python
# Photo Mode (4096x3008) - TODO
config = camera.create_still_configuration(
    main={"size": (4096, 3008), "format": "RGB888"}
)
```

## 🎯 **Next Steps Priority**

1. **Fix confidence threshold** (5 minutes)
2. **Test with dogs in view** (immediate)
3. **Fix behavior classification** (15 minutes)
4. **Add high-res photo mode** (30 minutes)
5. **Integrate into main robot** (30 minutes)

## 💡 **Pro Tips**

- **For testing**: Use `test_pose_headless.py`
- **For debugging**: Use `test_low_threshold.py`
- **For system check**: Use `test_1024x768_basic.py`
- **Check images**: Look in `pose_output/` directory
- **Monitor performance**: Watch FPS values in output

---

**The system is operational! Main testing file: `test_pose_headless.py`**