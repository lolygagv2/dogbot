#!/usr/bin/env python3
"""
Test script to verify Xbox controller fixes
Run this to test the key issues that were fixed
"""

import time
import requests

API_BASE = "http://localhost:8000"

def test_api_connection():
    """Test if API server is responding"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=2)
        if response.status_code == 200:
            print("✅ API server is responding")
            return True
        else:
            print(f"⚠️ API server responded with status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API server not responding: {e}")
        return False

def test_led_endpoints():
    """Test LED control endpoints"""
    print("\n=== Testing LED Controls ===")

    # Test blue LED on
    try:
        response = requests.post(f"{API_BASE}/leds/blue/on", timeout=5)
        if response.status_code == 200:
            print("✅ Blue LED ON endpoint working")
        else:
            print(f"⚠️ Blue LED ON failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Blue LED ON error: {e}")

    time.sleep(1)

    # Test blue LED off
    try:
        response = requests.post(f"{API_BASE}/leds/blue/off", timeout=5)
        if response.status_code == 200:
            print("✅ Blue LED OFF endpoint working")
        else:
            print(f"⚠️ Blue LED OFF failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Blue LED OFF error: {e}")

    # Test NeoPixel mode cycling
    modes = ["idle", "searching", "dog_detected", "manual_rc", "off"]
    for mode in modes:
        try:
            response = requests.post(f"{API_BASE}/leds/mode",
                                   json={"mode": mode}, timeout=5)
            if response.status_code == 200:
                print(f"✅ NeoPixel mode '{mode}' working")
            else:
                print(f"⚠️ NeoPixel mode '{mode}' failed: {response.status_code}")
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ NeoPixel mode '{mode}' error: {e}")

def test_system_status():
    """Check current system status"""
    print("\n=== System Status ===")

    try:
        response = requests.get(f"{API_BASE}/status", timeout=5)
        if response.status_code == 200:
            status = response.json()
            print(f"✅ System running: {status.get('running', 'unknown')}")
            print(f"✅ Current mode: {status.get('state', {}).get('mode', 'unknown')}")
            print(f"✅ Xbox service: {status.get('services', {}).get('xbox_controller', 'unknown')}")
        else:
            print(f"⚠️ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status check error: {e}")

def main():
    print("🤖 Xbox Controller Fixes Test")
    print("=" * 40)

    if not test_api_connection():
        print("\n❌ Cannot continue - API server not available")
        print("Make sure the robot is running: python3 main_treatbot.py")
        return

    test_system_status()
    test_led_endpoints()

    print("\n" + "=" * 40)
    print("🎮 Test Xbox Controller Now:")
    print("- X button should control blue LED")
    print("- Left Trigger should cycle NeoPixel modes")
    print("- Camera should stop cycling when controller is used")
    print("- Movement should work normally")

if __name__ == "__main__":
    main()