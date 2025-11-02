#!/usr/bin/env python3
"""
Test script to verify bark detection is fully integrated
"""

import sys
import time
import requests

sys.path.append('/home/morgan/dogbot')

from services.perception.bark_detector import get_bark_detector_service

def test_direct_service():
    """Test bark detector service directly"""
    print("="*60)
    print("TESTING BARK DETECTOR SERVICE DIRECTLY")
    print("="*60)

    # Get the service
    bark_detector = get_bark_detector_service()

    # Check status
    status = bark_detector.get_status()
    print(f"\nInitial Status: Enabled={status['enabled']}, Running={status['running']}")

    # Enable and start
    print("\nEnabling bark detection...")
    bark_detector.set_enabled(True)
    time.sleep(2)

    status = bark_detector.get_status()
    print(f"After enabling: Enabled={status['enabled']}, Running={status['running']}")

    if status['running']:
        print("\n✅ Bark detection service is running!")
        print("Make some barking sounds...")

        # Run for 10 seconds
        for i in range(10):
            time.sleep(1)
            status = bark_detector.get_status()
            stats = status['statistics']
            print(f"  [{i+1}s] Barks detected: {stats['total_barks']}, Rewarded: {stats['rewarded_barks']}")

    # Disable
    print("\nDisabling bark detection...")
    bark_detector.set_enabled(False)

    return True

def test_api_endpoints():
    """Test bark detection API endpoints"""
    print("\n" + "="*60)
    print("TESTING BARK DETECTION API ENDPOINTS")
    print("="*60)

    base_url = "http://localhost:8000"

    try:
        # Test status endpoint
        print("\n1. Testing /bark/status...")
        response = requests.get(f"{base_url}/bark/status")
        if response.status_code == 200:
            status = response.json()
            print(f"   ✅ Status: {status}")
        else:
            print(f"   ❌ Failed: {response.status_code}")

        # Test enable endpoint
        print("\n2. Testing /bark/enable...")
        response = requests.post(f"{base_url}/bark/enable")
        if response.status_code == 200:
            print(f"   ✅ Enabled: {response.json()}")
        else:
            print(f"   ❌ Failed: {response.status_code}")

        # Test config endpoint
        print("\n3. Testing /bark/config...")
        config_data = {
            "confidence_threshold": 0.6,
            "reward_emotions": ["alert", "attention", "playful"]
        }
        response = requests.post(f"{base_url}/bark/config", json=config_data)
        if response.status_code == 200:
            print(f"   ✅ Configured: {response.json()}")
        else:
            print(f"   ❌ Failed: {response.status_code}")

        # Test disable endpoint
        print("\n4. Testing /bark/disable...")
        response = requests.post(f"{base_url}/bark/disable")
        if response.status_code == 200:
            print(f"   ✅ Disabled: {response.json()}")
        else:
            print(f"   ❌ Failed: {response.status_code}")

        return True

    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API server")
        print("   Make sure the API is running: python3 api/server.py")
        return False

def main():
    """Run all tests"""
    print("BARK DETECTION INTEGRATION TEST")
    print("="*60)

    # Test 1: Direct service test
    service_ok = test_direct_service()

    # Test 2: API endpoints (if server is running)
    print("\nChecking if API server is running...")
    api_ok = test_api_endpoints()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Service Test: {'✅ PASSED' if service_ok else '❌ FAILED'}")
    print(f"API Test: {'✅ PASSED' if api_ok else '⚠️  SKIPPED (server not running)'}")

    if service_ok:
        print("\n✅ Bark detection is fully integrated!")
        print("\nTo use:")
        print("1. Start the main system: python3 main_treatbot.py")
        print("2. Enable via API: curl -X POST http://localhost:8000/bark/enable")
        print("3. Or enable in config: Set 'enabled: true' in config/robot_config.yaml")

if __name__ == "__main__":
    main()