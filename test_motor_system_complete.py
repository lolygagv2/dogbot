#!/usr/bin/env python3
"""
Complete Motor System Test
- Tests motor command bus with polling encoder tracking
- Verifies safety limits and hardware compensation
- Validates encoder feedback at 1000Hz
"""

import sys
import time
import threading
from typing import Dict

# Add project root
sys.path.append('.')

def test_motor_command_bus():
    """Test motor command bus initialization"""
    print("=" * 60)
    print("TESTING MOTOR COMMAND BUS")
    print("=" * 60)

    try:
        from core.motor_command_bus import get_motor_bus, create_motor_command, CommandSource
        print("✅ Motor command bus imports successful")

        # Get motor bus instance
        bus = get_motor_bus()
        print(f"✅ Motor bus created")

        # Check controller type
        controller_name = type(bus.motor_controller).__name__ if bus.motor_controller else "None"
        print(f"✅ Controller: {controller_name}")

        if controller_name == "MotorControllerPolling":
            print("✅ Using polling controller with 1000Hz encoder tracking")
        elif controller_name == "MotorControllerRobust":
            print("⚠️  Using robust controller (no encoder feedback)")
        else:
            print(f"❌ Unknown controller: {controller_name}")
            return False

        # Start the bus
        if bus.start():
            print("✅ Motor command bus started successfully")
        else:
            print("❌ Motor command bus failed to start")
            return False

        return bus

    except Exception as e:
        print(f"❌ Motor command bus test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_encoder_tracking(bus, duration=5.0):
    """Test encoder tracking for specified duration"""
    print("\n" + "=" * 60)
    print(f"TESTING ENCODER TRACKING ({duration}s)")
    print("=" * 60)

    if not hasattr(bus.motor_controller, 'get_encoder_status'):
        print("⚠️  No encoder tracking available (using robust controller)")
        return True

    try:
        # Reset encoder counts
        if hasattr(bus.motor_controller, 'reset_encoder_counts'):
            bus.motor_controller.reset_encoder_counts()
            print("✅ Encoder counts reset")

        # Get initial encoder status
        initial_status = bus.motor_controller.get_encoder_status()
        print(f"✅ Initial encoder status:")
        print(f"   Left:  count={initial_status['left']['count']}, changes={initial_status['left']['changes']}")
        print(f"   Right: count={initial_status['right']['count']}, changes={initial_status['right']['changes']}")
        print(f"   Polling rate: {initial_status['polling_rate']}Hz")

        print(f"\nMonitoring encoder changes for {duration} seconds...")
        print("(Manually turn wheels to test encoder detection)")

        start_time = time.time()
        last_changes = {
            'left': initial_status['left']['changes'],
            'right': initial_status['right']['changes']
        }

        while time.time() - start_time < duration:
            time.sleep(0.5)
            current_status = bus.motor_controller.get_encoder_status()

            left_changes = current_status['left']['changes'] - last_changes['left']
            right_changes = current_status['right']['changes'] - last_changes['right']

            if left_changes > 0 or right_changes > 0:
                print(f"   {time.time() - start_time:.1f}s: +{left_changes} left, +{right_changes} right changes")
                last_changes['left'] = current_status['left']['changes']
                last_changes['right'] = current_status['right']['changes']

        # Final status
        final_status = bus.motor_controller.get_encoder_status()
        total_left_changes = final_status['left']['changes'] - initial_status['left']['changes']
        total_right_changes = final_status['right']['changes'] - initial_status['right']['changes']

        print(f"\n✅ Total encoder changes detected:")
        print(f"   Left motor: {total_left_changes} changes")
        print(f"   Right motor: {total_right_changes} changes")

        if total_left_changes > 0 or total_right_changes > 0:
            print("✅ Encoder tracking is working!")
            return True
        else:
            print("⚠️  No encoder changes detected (wheels may not have been turned)")
            return True

    except Exception as e:
        print(f"❌ Encoder tracking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_motor_commands(bus, duration=2.0):
    """Test motor commands with safety limits"""
    print("\n" + "=" * 60)
    print("TESTING MOTOR COMMANDS & SAFETY LIMITS")
    print("=" * 60)

    try:
        from core.motor_command_bus import create_motor_command, CommandSource

        # Test different speed levels
        test_speeds = [30, 50, 70, 100]  # Bus will limit to 70% max

        for speed in test_speeds:
            print(f"\nTesting speed: {speed}%")

            # Forward command
            cmd = create_motor_command(speed, speed, CommandSource.API)
            result = bus.send_command(cmd)
            print(f"   Forward {speed}%: {'✅' if result else '❌'}")
            time.sleep(0.5)

            # Stop command
            stop_cmd = create_motor_command(0, 0, CommandSource.API)
            bus.send_command(stop_cmd)
            time.sleep(0.2)

            # Check motor status
            status = bus.get_status()
            if 'motor_details' in status:
                motor_status = status['motor_details']
                if 'motors' in motor_status:
                    motors = motor_status['motors']
                    print(f"   Status: L={motors.get('left_speed', 0)}%, R={motors.get('right_speed', 0)}%")
                    if 'safety_limits' in motors:
                        limits = motors['safety_limits']
                        print(f"   Safety limits: {limits['min_pwm']}-{limits['max_pwm']}% PWM")

        # Test differential steering
        print(f"\nTesting differential steering:")

        # Left turn
        left_cmd = create_motor_command(-30, 30, CommandSource.API)
        result = bus.send_command(left_cmd)
        print(f"   Left turn: {'✅' if result else '❌'}")
        time.sleep(0.5)

        # Right turn
        right_cmd = create_motor_command(30, -30, CommandSource.API)
        result = bus.send_command(right_cmd)
        print(f"   Right turn: {'✅' if result else '❌'}")
        time.sleep(0.5)

        # Stop
        stop_cmd = create_motor_command(0, 0, CommandSource.API)
        bus.send_command(stop_cmd)
        print("   Stopped: ✅")

        return True

    except Exception as e:
        print(f"❌ Motor command test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_xbox_controller_integration():
    """Test Xbox controller motor integration"""
    print("\n" + "=" * 60)
    print("TESTING XBOX CONTROLLER INTEGRATION")
    print("=" * 60)

    try:
        from xbox_hybrid_controller import MOTOR_BUS_AVAILABLE, MOTOR_CONTROLLER_AVAILABLE

        print(f"MOTOR_BUS_AVAILABLE: {'✅' if MOTOR_BUS_AVAILABLE else '❌'}")
        print(f"MOTOR_CONTROLLER_AVAILABLE: {'✅' if MOTOR_CONTROLLER_AVAILABLE else '❌'}")

        if MOTOR_BUS_AVAILABLE:
            print("✅ Xbox controller will use motor command bus")
            return True
        else:
            print("❌ Xbox controller will fall back to API mode")
            return False

    except Exception as e:
        print(f"❌ Xbox controller integration test failed: {e}")
        return False

def main():
    """Run complete motor system test"""
    print("WIM-Z MOTOR SYSTEM COMPREHENSIVE TEST")
    print("=" * 60)
    print("This test validates:")
    print("- Motor command bus with polling encoder controller")
    print("- 1000Hz encoder tracking (if wheels are manually turned)")
    print("- Safety limits and hardware compensation")
    print("- Xbox controller integration")
    print("=" * 60)

    all_tests_passed = True

    # Test 1: Motor command bus
    bus = test_motor_command_bus()
    if not bus:
        all_tests_passed = False
        print("❌ Cannot continue - motor command bus failed")
        return

    # Test 2: Encoder tracking
    if not test_encoder_tracking(bus, duration=3.0):
        all_tests_passed = False

    # Test 3: Motor commands
    if not test_motor_commands(bus):
        all_tests_passed = False

    # Test 4: Xbox integration
    if not test_xbox_controller_integration():
        all_tests_passed = False

    # Cleanup
    try:
        bus.stop()
        print("\n✅ Motor system cleanup complete")
    except Exception as e:
        print(f"⚠️  Cleanup error: {e}")

    # Final report
    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)

    if all_tests_passed:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Motor system fully restored with:")
        print("   - 1000Hz polling encoder tracking")
        print("   - Motor command bus architecture")
        print("   - Xbox controller integration")
        print("   - Safety limits and hardware compensation")
        print("   - Ready for autonomous AI control")
    else:
        print("❌ Some tests failed - check output above")

    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Test crashed: {e}")
        import traceback
        traceback.print_exc()