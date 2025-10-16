#!/usr/bin/env python3
"""
Simple test of AI controller integration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ai_integration():
    """Test just the AI controller integration"""
    print("🤖 Testing AI Controller Integration...")
    print("=" * 50)

    try:
        # Test AI controller import and initialization
        from core.ai_controller import AIController

        print("✅ AI Controller imported successfully")

        # Initialize AI
        ai = AIController()
        success = ai.initialize()

        if success:
            print("✅ AI Controller initialized successfully!")

            # Get status
            status = ai.get_status()
            print("\n📊 AI Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")

            # Test detection
            print("\n🔍 Testing detection...")
            import numpy as np
            test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            detections = ai.detect_objects(test_frame)

            print(f"✅ Detection test complete: {len(detections)} objects detected")

            # Cleanup
            ai.cleanup()

            print("\n🎉 AI INTEGRATION TEST SUCCESSFUL!")
            print("Ready to integrate into full TreatSensei system!")

        else:
            print("❌ AI Controller initialization failed")

    except Exception as e:
        print(f"❌ AI integration test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ai_integration()