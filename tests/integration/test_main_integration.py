#!/usr/bin/env python3
"""
Test TreatSensei main integration without hardware dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Mock the hardware modules to test AI integration without actual hardware
class MockController:
    def __init__(self):
        self.initialized = True

    def is_initialized(self):
        return self.initialized

    def get_status(self):
        return {'status': 'mock_ok'}

    def cleanup(self):
        pass

    def emergency_stop(self):
        pass

# Replace hardware modules with mocks for testing
import core.motor_controller
import core.audio_controller
import core.led_controller
import core.servo_controller

core.motor_controller.MotorController = MockController
core.audio_controller.AudioController = MockController
core.led_controller.LEDController = MockController
core.servo_controller.ServoController = MockController

# Now test the main system
from main import TreatSenseiCore

def test_main_integration():
    """Test main system integration with AI"""
    print("🚀 Testing TreatSensei integration with AI...")

    try:
        # Initialize robot
        robot = TreatSenseiCore()

        if robot.initialization_successful:
            print("✅ Robot initialization successful!")

            # Test AI status
            if robot.ai and robot.ai.is_initialized():
                print("✅ AI system integrated successfully!")

                # Get full status
                status = robot.get_system_status()
                print("\n📊 System Status:")
                for subsystem, info in status['subsystems'].items():
                    print(f"  {subsystem}: {info}")

                # Test AI detection
                print("\n🤖 Testing AI detection...")
                import numpy as np
                test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                detections = robot.ai.detect_objects(test_frame)
                print(f"AI detection test: {len(detections)} detections found")

                print("\n🎉 INTEGRATION TEST SUCCESSFUL!")
                print("TreatSensei with Hailo AI is ready!")

            else:
                print("❌ AI system not properly integrated")

        else:
            print("❌ Robot initialization failed")

        # Cleanup
        robot.shutdown()

    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_main_integration()