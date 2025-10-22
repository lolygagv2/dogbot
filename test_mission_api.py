#!/usr/bin/env python3
"""
Test the Unified Mission API
"""

import time
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from missions import MissionController, simple_sit_mission

def test_basic_mission():
    """Test basic mission functionality"""
    print("🎯 Testing Basic Mission API")
    print("=" * 50)

    # Create mission
    mission = MissionController("test_mission")

    # Start mission
    mission_id = mission.start()
    print(f"✅ Mission started with ID: {mission_id}")

    # Simulate some events
    mission.log_event("dog_detected", {"confidence": 0.85})
    time.sleep(1)

    mission.log_event("pose_detected", {"pose": "sit", "confidence": 0.75})
    time.sleep(1)

    # Simulate pose detection
    mission.set_current_pose("sit", 0.8)

    # Test reward system
    print("🎁 Testing reward system...")
    success = mission.reward(treat=True, audio="good_dog.mp3", lights="celebration")
    print(f"Reward result: {'✅ Success' if success else '❌ Failed'}")

    # Test condition waiting (will timeout quickly for demo)
    print("⏳ Testing pose condition (2s timeout)...")
    condition_met = mission.wait_for_condition("sit", duration=1.0, timeout=2.0)
    print(f"Condition result: {'✅ Met' if condition_met else '❌ Timeout'}")

    # End mission
    summary = mission.end(success=True)
    print(f"✅ Mission ended. Total events: {summary['total_events']}")

    return summary

def test_sit_mission():
    """Test pre-configured sit mission"""
    print("\n🎯 Testing Sit Training Mission")
    print("=" * 50)

    # Use convenience function
    mission = simple_sit_mission(duration=2.0)

    mission_id = mission.start()
    print(f"✅ Sit mission started: {mission_id}")

    # Simulate detection and reward
    mission.set_current_pose("sit", 0.9)
    mission.reward(treat=True, audio="good_dog.mp3")

    # Get status
    status = mission.get_status()
    print(f"📊 Mission status: {status['mission_name']} - {status['total_events']} events")

    summary = mission.end(success=True)
    print(f"✅ Sit mission completed in {summary['duration_seconds']:.1f}s")

    return summary

def test_yaml_config():
    """Test YAML configuration loading"""
    print("\n🎯 Testing YAML Configuration")
    print("=" * 50)

    # Load from YAML config
    mission = MissionController("sit_training")  # Loads sit_training.yaml

    print(f"📝 Loaded config: {mission.config}")
    print(f"🎯 Target pose: {mission.config.get('pose_detection', {}).get('target_pose', 'unknown')}")
    print(f"⏱️ Duration required: {mission.config.get('pose_detection', {}).get('duration_required', 0)}s")
    print(f"🎁 Reward includes treat: {mission.config.get('reward', {}).get('treat', False)}")

    return mission.config

def main():
    """Run all tests"""
    print("🚀 Mission API Test Suite")
    print("=" * 60)

    # Create directories
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("missions/configs").mkdir(parents=True, exist_ok=True)

    try:
        # Test 1: Basic functionality
        basic_summary = test_basic_mission()

        # Test 2: Convenience functions
        sit_summary = test_sit_mission()

        # Test 3: YAML configuration
        yaml_config = test_yaml_config()

        print("\n🎉 All Tests Completed!")
        print("=" * 60)
        print(f"✅ Basic mission: {basic_summary['total_events']} events")
        print(f"✅ Sit mission: {sit_summary['total_events']} events")
        print(f"✅ YAML config loaded: {len(yaml_config)} settings")

        print("\n📁 Check these files:")
        print(f"   - Database: data/missions.db")
        print(f"   - Logs: logs/mission_*.log")
        print(f"   - Configs: missions/configs/*.yaml")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()