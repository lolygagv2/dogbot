#!/usr/bin/env python3
"""
Test script for camera mode controller
Demonstrates all 4 camera modes and transitions
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from core.camera_mode_controller import CameraModeController, CameraMode
from core.ai_controller_3stage_fixed import AI3StageControllerFixed

def test_photography_mode(controller):
    """Test photography mode - max resolution, no AI"""
    print("\n" + "="*60)
    print("📸 PHOTOGRAPHY MODE TEST")
    print("="*60)
    
    controller.set_mode(CameraMode.PHOTOGRAPHY, force=True)
    print(f"Status: {controller.get_status()}")
    
    # Capture high-res photo
    frame = controller.capture_frame()
    if frame is not None:
        print(f"Captured frame: {frame.shape}")
        # Save photo
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"photo_{timestamp}.jpg", frame)
        print(f"Saved: photo_{timestamp}.jpg")
    else:
        print("Failed to capture frame")
    
    time.sleep(2)

def test_ai_detection_mode(controller):
    """Test AI detection mode - single 640x640 frame"""
    print("\n" + "="*60)
    print("🤖 AI DETECTION MODE TEST")
    print("="*60)
    
    controller.set_mode(CameraMode.AI_DETECTION, force=True)
    print(f"Status: {controller.get_status()}")
    
    # Run for 5 seconds
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 5:
        frame = controller.capture_frame()
        if frame is not None:
            frame_count += 1
            # AI inference would happen here
            if controller.ai_controller:
                detections = controller.ai_controller.detect_dogs(frame)
                print(f"Frame {frame_count}: {len(detections)} dogs detected")
            else:
                print(f"Frame {frame_count}: {frame.shape}")
        
        time.sleep(0.033)  # ~30 FPS
    
    fps = frame_count / 5
    print(f"Average FPS: {fps:.1f}")

def test_vigilant_mode(controller):
    """Test vigilant mode - full frame tiling"""
    print("\n" + "="*60)
    print("👁️ VIGILANT MODE TEST")
    print("="*60)
    
    controller.set_mode(CameraMode.VIGILANT, force=True)
    print(f"Status: {controller.get_status()}")
    
    # Capture and process one frame
    frame = controller.capture_frame()
    if frame is not None:
        print(f"Captured frame: {frame.shape}")
        
        # Generate tiles
        height, width = frame.shape[:2]
        tiles = controller._generate_tiles(width, height)
        print(f"Generated {len(tiles)} tiles for {width}x{height} frame")
        
        # Process with tiling
        if controller.ai_controller:
            detections = controller.process_vigilant_mode(frame)
            print(f"Total detections across all tiles: {len(detections)}")
            
            # Group by tile
            tiles_with_detections = set(d['tile_id'] for d in detections)
            print(f"Tiles with detections: {tiles_with_detections}")
        else:
            print("No AI controller - skipping inference")
    
    time.sleep(2)

def test_auto_switching(controller):
    """Test automatic mode switching based on vehicle state"""
    print("\n" + "="*60)
    print("🔄 AUTO-SWITCH MODE TEST")
    print("="*60)
    
    # Simulate vehicle starting to move
    print("\n📍 Vehicle starting to move...")
    controller.set_vehicle_state(is_moving=True)
    time.sleep(1)
    print(f"Mode: {controller.current_mode.value}")
    print(f"Status: {controller.get_status()['resolution']}")
    
    # Simulate vehicle stopping
    print("\n📍 Vehicle stopped...")
    controller.set_vehicle_state(is_moving=False)
    time.sleep(1)
    print(f"Mode: {controller.current_mode.value}")
    print(f"Status: {controller.get_status()['resolution']}")
    
    # Test manual override
    print("\n📍 Enabling manual override...")
    controller.manual_override = True
    controller.set_mode(CameraMode.PHOTOGRAPHY, force=True)
    print(f"Mode: {controller.current_mode.value}")
    
    # Try auto-switch with override active
    print("\n📍 Vehicle moving (with override)...")
    controller.set_vehicle_state(is_moving=True)
    time.sleep(1)
    print(f"Mode still: {controller.current_mode.value} (override active)")

def main():
    print("\n" + "="*60)
    print("🎥 CAMERA MODE CONTROLLER TEST")
    print("="*60)
    
    # Initialize AI controller (optional)
    ai_controller = None
    try:
        print("\nInitializing AI controller...")
        ai_controller = AI3StageControllerFixed()
        if ai_controller.initialize():
            print("✅ AI controller ready")
        else:
            print("⚠️ AI controller initialization failed - continuing without AI")
            ai_controller = None
    except Exception as e:
        print(f"⚠️ AI controller not available: {e}")
        ai_controller = None
    
    # Initialize camera controller
    print("\nInitializing camera controller...")
    controller = CameraModeController(ai_controller=ai_controller)
    
    try:
        # Test each mode
        test_photography_mode(controller)
        test_ai_detection_mode(controller)
        test_vigilant_mode(controller)
        test_auto_switching(controller)
        
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETE")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        controller.cleanup()
        if ai_controller:
            ai_controller.cleanup()
        print("Done.")

if __name__ == "__main__":
    main()