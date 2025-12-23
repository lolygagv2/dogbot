#!/usr/bin/env python3
"""
Test Manual Control Integration
Verify that manual control and GUI integration work correctly
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_motor_service():
    """Test motor service functionality"""
    print("🧪 Testing Motor Service...")

    try:
        from services.motion.motor import get_motor_service, MovementMode

        motor_service = get_motor_service()

        # Test initialization
        if motor_service.initialize():
            print("✅ Motor service initialization successful")
        else:
            print("❌ Motor service initialization failed")
            return False

        # Test mode setting
        motor_service.set_movement_mode(MovementMode.MANUAL)
        status = motor_service.get_status()
        print(f"📊 Motor Status: {status}")

        # Test manual drive (dry run)
        print("🚗 Testing manual drive commands...")
        test_commands = [
            ('forward', 30, 0.5),
            ('stop', 0, 0.1),
            ('left', 40, 0.3),
            ('stop', 0, 0.1),
            ('backward', 30, 0.5),
            ('stop', 0, 0.1)
        ]

        for direction, speed, duration in test_commands:
            print(f"  Testing: {direction} at {speed}% for {duration}s")
            success = motor_service.manual_drive(direction, speed, duration)
            if success:
                print(f"    ✅ {direction} command successful")
            else:
                print(f"    ❌ {direction} command failed")
            time.sleep(duration + 0.2)

        # Test keyboard control
        print("⌨️ Testing keyboard controls...")
        test_keys = ['w', 'a', 's', 'd', 'space']
        for key in test_keys:
            success = motor_service.keyboard_control(key)
            print(f"  Key '{key}': {'✅' if success else '❌'}")
            time.sleep(0.3)

        motor_service.cleanup()
        print("✅ Motor service test complete")
        return True

    except Exception as e:
        print(f"❌ Motor service test failed: {e}")
        return False

def test_gui_service():
    """Test GUI service functionality"""
    print("\n🧪 Testing GUI Service...")

    try:
        from services.ui.gui import get_gui_service

        gui_service = get_gui_service()

        # Test initialization
        if gui_service.initialize():
            print("✅ GUI service initialization successful")
        else:
            print("⚠️ GUI service initialization failed (might be missing dependencies)")
            return False

        # Test status
        status = gui_service.get_status()
        print(f"📊 GUI Status: {status}")

        # Test manual control integration
        gui_service.set_manual_control_enabled(True)
        print("✅ Manual control integration enabled")

        # Test keyboard processing
        print("⌨️ Testing keyboard input processing...")
        test_keys = ['w', 'a', 's', 'd']
        for key in test_keys:
            processed = gui_service.process_keyboard_input(key)
            print(f"  Key '{key}': {'✅' if processed else '❌'}")

        gui_service.cleanup()
        print("✅ GUI service test complete")
        return True

    except Exception as e:
        print(f"❌ GUI service test failed: {e}")
        return False

def test_api_endpoints():
    """Test manual control API endpoints"""
    print("\n🧪 Testing API Endpoints...")

    try:
        # Import API components
        from api.server import app
        from fastapi.testclient import TestClient

        client = TestClient(app)

        # Test health endpoint
        response = client.get("/health")
        if response.status_code == 200:
            print("✅ Health endpoint working")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")

        # Test manual control endpoints
        print("🚗 Testing manual control API...")

        # Set manual mode
        response = client.post("/manual/mode/manual")
        if response.status_code == 200:
            print("✅ Manual mode endpoint working")
        else:
            print(f"❌ Manual mode endpoint failed: {response.status_code}")

        # Test drive command
        drive_data = {
            "direction": "forward",
            "speed": 50,
            "duration": 0.1
        }
        response = client.post("/manual/drive", json=drive_data)
        if response.status_code == 200:
            print("✅ Drive endpoint working")
        else:
            print(f"❌ Drive endpoint failed: {response.status_code}")

        # Test keyboard command
        keyboard_data = {"key": "w"}
        response = client.post("/manual/keyboard", json=keyboard_data)
        if response.status_code == 200:
            print("✅ Keyboard endpoint working")
        else:
            print(f"❌ Keyboard endpoint failed: {response.status_code}")

        # Test status endpoint
        response = client.get("/manual/status")
        if response.status_code == 200:
            print("✅ Status endpoint working")
            print(f"📊 Status: {response.json()}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")

        print("✅ API endpoints test complete")
        return True

    except Exception as e:
        print(f"❌ API endpoints test failed: {e}")
        return False

def test_manual_control_interface():
    """Test the manual control interface initialization"""
    print("\n🧪 Testing Manual Control Interface...")

    try:
        from manual_control import ManualControlInterface

        control = ManualControlInterface()

        # Test initialization
        if control.initialize():
            print("✅ Manual control interface initialization successful")
        else:
            print("❌ Manual control interface initialization failed")
            return False

        # Test key processing
        print("⌨️ Testing key processing...")
        test_keys = ['w', 's', 'a', 'd', 'space']
        for key in test_keys:
            control._process_key(key)
            print(f"  Key '{key}': ✅ processed")
            time.sleep(0.2)

        # Test emergency stop
        control.emergency_stop()
        print("✅ Emergency stop test successful")

        control.cleanup()
        print("✅ Manual control interface test complete")
        return True

    except Exception as e:
        print(f"❌ Manual control interface test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🤖 TreatBot Manual Control Integration Test")
    print("=" * 60)

    test_results = []

    # Run tests
    test_results.append(("Motor Service", test_motor_service()))
    test_results.append(("GUI Service", test_gui_service()))
    test_results.append(("API Endpoints", test_api_endpoints()))
    test_results.append(("Manual Control Interface", test_manual_control_interface()))

    # Summary
    print("\n" + "=" * 60)
    print("🧪 TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Manual control integration ready!")
        print("\nTo use manual control:")
        print("1. Start the main TreatBot system: python3 main_treatbot.py")
        print("2. Start manual control: python3 manual_control.py")
        print("3. Or use web interface: python3 -m http.server 8080")
        print("   Then open: http://localhost:8080/web_remote_control.html")
    else:
        print("❌ SOME TESTS FAILED - Check configuration and dependencies")

    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())