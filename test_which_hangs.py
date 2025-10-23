#!/usr/bin/env python3
"""
Debug which test is hanging
"""

import sys
import os
import time
import signal

# Add timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Test timed out!")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_motor_init():
    """Just test motor service initialization"""
    print("🧪 Testing Motor Service Init...")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout

        from services.motion.motor import get_motor_service
        motor_service = get_motor_service()
        result = motor_service.initialize()

        signal.alarm(0)  # Cancel alarm
        print(f"✅ Motor init: {result}")
        return True
    except TimeoutError:
        print("❌ Motor init TIMEOUT!")
        return False
    except Exception as e:
        print(f"❌ Motor init error: {e}")
        return False

def test_gui_init():
    """Just test GUI service initialization"""
    print("\n🧪 Testing GUI Service Init...")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout

        from services.ui.gui import get_gui_service
        gui_service = get_gui_service()
        result = gui_service.initialize()

        signal.alarm(0)  # Cancel alarm
        print(f"✅ GUI init: {result}")
        return True
    except TimeoutError:
        print("❌ GUI init TIMEOUT!")
        return False
    except Exception as e:
        print(f"❌ GUI init error: {e}")
        return False

def test_manual_control_init():
    """Just test manual control initialization"""
    print("\n🧪 Testing Manual Control Init...")
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout

        from manual_control import ManualControlInterface
        control = ManualControlInterface()
        result = control.initialize()

        signal.alarm(0)  # Cancel alarm
        print(f"✅ Manual control init: {result}")
        return True
    except TimeoutError:
        print("❌ Manual control init TIMEOUT!")
        return False
    except Exception as e:
        print(f"❌ Manual control init error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 FINDING WHICH SERVICE HANGS")
    print("=" * 60)

    # Test each service separately
    motor_ok = test_motor_init()
    gui_ok = test_gui_init()
    manual_ok = test_manual_control_init()

    print("\n" + "=" * 60)
    print("📊 RESULTS:")
    print(f"Motor Service: {'✅' if motor_ok else '❌ HANGS'}")
    print(f"GUI Service: {'✅' if gui_ok else '❌ HANGS'}")
    print(f"Manual Control: {'✅' if manual_ok else '❌ HANGS'}")
    print("=" * 60)