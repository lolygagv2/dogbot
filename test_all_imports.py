#!/usr/bin/env python3
"""
Quick test to verify all imports are working
"""

import sys
import os
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test all critical imports"""
    tests = []
    failures = []

    # Core modules
    core_imports = [
        ("core.bus", ["get_bus", "MotionEvent"]),
        ("core.state", ["get_state", "SystemMode"]),
        ("core.store", ["get_store"]),
        ("core.safety", ["get_safety_monitor"]),
        ("core.hardware.motor_controller", ["MotorController"]),
        ("core.hardware.servo_controller", ["ServoController"]),
        ("core.ai_controller_3stage_fixed", ["AI3StageControllerFixed"]),
    ]

    # Service modules
    service_imports = [
        ("services.motion.motor", ["get_motor_service", "MovementMode"]),
        ("services.perception.detector", ["get_detector_service"]),
        ("services.ui.gui", ["get_gui_service"]),
        ("services.media.led", ["get_led_service"]),
        ("services.media.sfx", ["get_sfx_service"]),
        ("services.reward.dispenser", ["get_dispenser_service"]),
        ("services.motion.pan_tilt", ["get_pantilt_service"]),
    ]

    # API modules
    api_imports = [
        ("api.server", ["create_app"]),
    ]

    # Orchestrator modules
    orchestrator_imports = [
        ("orchestrators.sequence_engine", ["get_sequence_engine"]),
        ("orchestrators.reward_logic", ["get_reward_logic"]),
        ("orchestrators.mode_fsm", ["get_mode_fsm"]),
    ]

    all_imports = [
        ("Core Modules", core_imports),
        ("Service Modules", service_imports),
        ("API Modules", api_imports),
        ("Orchestrator Modules", orchestrator_imports)
    ]

    print("=" * 60)
    print("🧪 TESTING ALL IMPORTS")
    print("=" * 60)

    for category, imports_list in all_imports:
        print(f"\n📦 {category}:")
        for module_path, items in imports_list:
            try:
                # Dynamic import
                module = __import__(module_path, fromlist=items)

                # Verify each item exists
                missing = []
                for item in items:
                    if not hasattr(module, item):
                        missing.append(item)

                if missing:
                    print(f"  ❌ {module_path}: Missing {', '.join(missing)}")
                    failures.append(f"{module_path}: Missing {', '.join(missing)}")
                else:
                    print(f"  ✅ {module_path}")
                    tests.append(module_path)

            except ImportError as e:
                print(f"  ❌ {module_path}: {str(e)}")
                failures.append(f"{module_path}: {str(e)}")
            except Exception as e:
                print(f"  ❌ {module_path}: {type(e).__name__}: {str(e)}")
                failures.append(f"{module_path}: {str(e)}")

    print("\n" + "=" * 60)
    print("📊 IMPORT TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Successful imports: {len(tests)}")
    print(f"❌ Failed imports: {len(failures)}")

    if failures:
        print("\n⚠️ Failed imports:")
        for failure in failures:
            print(f"  - {failure}")
        print("\nPlease fix these import errors before proceeding.")
        return False
    else:
        print("\n✅ ALL IMPORTS SUCCESSFUL!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)