#!/usr/bin/env python3
"""
Test script for camera and AI integration
Run this to verify all components are working
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import modules (adjust paths as needed)
from vision.camera_interface import CameraInterface
from ai.dog_detection import DogDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_camera_basic():
    """Test basic camera functionality"""
    print("\n=== Testing Camera Interface ===")
    
    cam = CameraInterface(resolution=(640, 480))
    
    # Initialize
    if not cam.initialize():
        print("❌ Camera initialization failed")
        return False
    
    print("✅ Camera initialized")
    
    # Start capture
    if not cam.start():
        print("❌ Camera start failed")
        return False
    
    print("✅ Camera started")
    
    # Wait for frames
    time.sleep(2)
    
    # Get a frame
    frame = cam.get_frame()
    if frame is None:
        print("❌ No frame captured")
        cam.stop()
        return False
    
    print(f"✅ Frame captured: {frame.shape}")
    
    # Capture test image
    if cam.capture_image("test_image.jpg"):
        print("✅ Test image saved")
    else:
        print("⚠️ Could not save test image")
    
    # Stop camera
    cam.stop()
    print("✅ Camera stopped successfully")
    
    return True

def test_ai_detection():
    """Test AI detection pipeline"""
    print("\n=== Testing AI Detection ===")
    
    # Initialize components
    cam = CameraInterface(resolution=(640, 480))
    detector = DogDetector(confidence_threshold=0.6)
    
    if not cam.initialize():
        print("❌ Camera initialization failed")
        return False
    
    print("✅ Detector initialized")
    
    # Process callback
    def process_frame(frame):
        # Run detection
        results = detector.process_frame(frame)
        
        # Log results
        if results["dog_detected"]:
            logger.info(f"Dog detected! Behavior: {results['behavior']}, "
                       f"Confidence: {results['confidence']:.2f}")
    
    # Start with callback
    cam.start(frame_callback=process_frame)
    print("✅ Processing pipeline started")
    
    # Run for 10 seconds
    print("Running detection for 10 seconds...")
    for i in range(10):
        time.sleep(1)
        print(f"  {i+1}/10 seconds...")
        
        # Check if dog is present
        if detector.is_dog_present():
            print("  🐕 Dog detected in frame!")
    
    # Stop
    cam.stop()
    print("✅ Detection test complete")
    
    return True

def test_reward_logic():
    """Test reward triggering logic"""
    print("\n=== Testing Reward Logic ===")
    
    detector = DogDetector()
    
    # Simulate detection history
    print("Simulating dog sitting for 3 seconds...")
    for _ in range(90):  # 3 seconds at 30fps
        detector.pose_history.append("sit")
        detector.detection_history.append(True)
    
    # Check reward criteria
    should_reward = detector.should_reward("sit", min_duration=2.0)
    if should_reward:
        print("✅ Reward criteria met for 'sit'")
    else:
        print("❌ Reward criteria not met")
    
    # Test with insufficient duration
    detector.pose_history = ["sit"] * 30  # Only 1 second
    detector.detection_history = [True] * 30
    
    should_reward = detector.should_reward("sit", min_duration=2.0)
    if not should_reward:
        print("✅ Correctly rejected short duration")
    else:
        print("❌ Incorrectly triggered reward for short duration")
    
    return True

def verify_dependencies():
    """Verify all required dependencies are installed"""
    print("\n=== Verifying Dependencies ===")
    
    dependencies = {
        "numpy": "✅ NumPy",
        "cv2": "✅ OpenCV",
        "mediapipe": "✅ MediaPipe",
        "picamera2": "✅ Picamera2",
        "serial": "✅ PySerial",
        "adafruit_servokit": "✅ Adafruit ServoKit",
        "yaml": "✅ PyYAML",
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(name)
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install", ' '.join(missing))
        return False
    
    print("\n✅ All dependencies verified")
    return True

def main():
    """Run all tests"""
    print("=" * 50)
    print("TreatBot Camera & AI Integration Test")
    print("=" * 50)
    
    # Verify dependencies first
    if not verify_dependencies():
        print("\n⚠️ Please install missing dependencies first")
        return 1
    
    # Run tests
    tests = [
        ("Camera Basic", test_camera_basic),
        ("AI Detection", test_ai_detection),
        ("Reward Logic", test_reward_logic),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            print(f"\nRunning: {name}")
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{name}: {status}")
    
    # Overall result
    all_passed = all(success for _, success in results)
    if all_passed:
        print("\n🎉 All tests passed! Ready to integrate.")
    else:
        print("\n⚠️ Some tests failed. Check logs above.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
