#!/usr/bin/env python3
"""
Test Phase 1 core infrastructure
Tests event bus, state manager, store, and safety monitor
"""

import time
import threading
import json

# Test imports
from core.bus import get_bus, publish_vision_event, publish_system_event
from core.state import get_state, SystemMode, MissionState
from core.store import get_store
from core.safety import get_safety_monitor

def test_event_bus():
    """Test event bus functionality"""
    print("🧪 Testing Event Bus...")

    bus = get_bus()
    received_events = []

    def test_handler(event):
        received_events.append(event)
        print(f"  📨 Received: {event.type.value}.{event.subtype}")

    # Subscribe to vision events
    bus.subscribe('vision', test_handler)

    # Publish test events
    publish_vision_event('dog_detected', {'confidence': 0.95, 'bbox': [100, 100, 200, 200]})
    publish_vision_event('pose', {'behavior': 'sitting', 'keypoints': 24})

    # Wait for async delivery
    time.sleep(0.2)

    # Check results
    assert len(received_events) == 2
    assert received_events[0].subtype == 'dog_detected'
    assert received_events[1].subtype == 'pose'

    print("  ✅ Event bus working")
    return True

def test_state_manager():
    """Test state manager functionality"""
    print("🧪 Testing State Manager...")

    state = get_state()
    mode_changes = []

    def mode_change_handler(data):
        mode_changes.append(data)
        print(f"  🔄 Mode changed: {data['previous_mode']} -> {data['new_mode']}")

    # Subscribe to mode changes
    state.subscribe('mode_change', mode_change_handler)

    # Test mode changes
    assert state.set_mode(SystemMode.DETECTION, "Starting detection")
    assert state.get_mode() == SystemMode.DETECTION

    # Test hardware updates
    state.update_hardware(battery_voltage=13.8, temperature=45.2)
    assert state.hardware.battery_voltage == 13.8

    # Test mission updates
    state.update_mission(name="test_mission", state=MissionState.ACTIVE, rewards_given=2)
    assert state.mission.name == "test_mission"
    assert state.mission.rewards_given == 2

    # Test detection updates
    state.update_detection(dogs_detected=1, current_behavior="sitting")
    assert state.detection.dogs_detected == 1

    # Wait for async notifications
    time.sleep(0.1)

    # Check mode change was notified
    assert len(mode_changes) > 0

    print("  ✅ State manager working")
    return True

def test_store():
    """Test database store functionality"""
    print("🧪 Testing Store...")

    store = get_store()

    # Test event logging
    event_id = store.log_event('vision', 'dog_detected', 'test', {
        'confidence': 0.95,
        'bbox': [100, 100, 200, 200]
    })
    assert event_id > 0

    # Test dog registration
    success = store.register_dog('test_dog_001', 'TestBuddy', {
        'breed': 'Test Retriever',
        'age': 3
    })
    assert success

    # Test reward logging
    reward_id = store.log_reward('test_dog_001', 'sit', 0.87, True, 1, 'test_mission')
    assert reward_id > 0

    # Test telemetry
    success = store.log_telemetry(battery_voltage=13.2, temperature=42.5, mode='detection')
    assert success

    # Test retrieval
    events = store.get_recent_events(5)
    assert len(events) > 0

    dogs = store.get_dog_stats()
    assert len(dogs) > 0
    assert dogs[0]['name'] == 'TestBuddy'

    rewards = store.get_reward_history()
    assert len(rewards) > 0

    # Test stats
    stats = store.get_database_stats()
    assert stats['events_count'] > 0
    assert stats['dogs_count'] > 0
    assert stats['rewards_count'] > 0

    print("  ✅ Store working")
    return True

def test_safety_monitor():
    """Test safety monitoring"""
    print("🧪 Testing Safety Monitor...")

    safety = get_safety_monitor()
    emergency_triggered = []

    def emergency_handler(reason, data):
        emergency_triggered.append((reason, data))
        print(f"  🚨 Emergency: {reason}")

    # Add emergency callback
    safety.add_emergency_callback(emergency_handler)

    # Start monitoring briefly
    safety.start_monitoring(interval=0.5)

    # Send heartbeats
    for i in range(3):
        safety.heartbeat()
        time.sleep(0.1)

    # Test status
    status = safety.get_status()
    assert status['monitoring'] == True
    assert 'measurements' in status

    # Test safety check
    assert safety.is_safe_to_operate() == True

    # Stop monitoring
    safety.stop_monitoring()

    print("  ✅ Safety monitor working")
    return True

def test_integration():
    """Test integration between components"""
    print("🧪 Testing Integration...")

    bus = get_bus()
    state = get_state()
    store = get_store()

    integration_events = []

    def integration_handler(event):
        integration_events.append(event)

        # When vision event occurs, update state
        if event.type.value == 'vision' and event.subtype == 'dog_detected':
            state.update_detection(dogs_detected=1, last_detection_time=time.time())

            # Log to store
            store.log_event(event.type.value, event.subtype, event.source, event.data)

    # Subscribe to all events
    bus.subscribe('vision', integration_handler)
    bus.subscribe('system', integration_handler)

    # Trigger a cascade
    publish_vision_event('dog_detected', {'confidence': 0.92, 'dog_id': 'integration_test'})

    # Wait for processing
    time.sleep(0.2)

    # Check state was updated
    assert state.detection.dogs_detected == 1

    # Check event was stored
    recent_events = store.get_recent_events(1)
    assert len(recent_events) > 0
    assert recent_events[0]['subtype'] == 'dog_detected'

    print("  ✅ Integration working")
    return True

def main():
    """Run all Phase 1 tests"""
    print("🚀 Testing Phase 1 Core Infrastructure")
    print("=" * 50)

    tests = [
        test_event_bus,
        test_state_manager,
        test_store,
        test_safety_monitor,
        test_integration
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"  ❌ {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"  ❌ {test_func.__name__} failed: {e}")

    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("✅ Phase 1 Core Infrastructure: ALL TESTS PASSED")
        print("Ready to proceed to Phase 2!")
    else:
        print("❌ Phase 1 Core Infrastructure: SOME TESTS FAILED")
        print("Fix issues before proceeding to Phase 2")

    return failed == 0

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)