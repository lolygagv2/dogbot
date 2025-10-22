#!/usr/bin/env python3
"""
GUI Service - Unified interface integrating existing working GUI
Wraps live_gui_detection_with_modes.py into the unified architecture
"""

import sys
import os
import threading
import time
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import existing working GUI
try:
    from live_gui_detection_with_modes import LiveDetectionGUIWithModes
    GUI_AVAILABLE = True
except ImportError as e:
    GUI_AVAILABLE = False
    print(f"[WARNING] GUI not available: {e}")

from core.bus import get_bus
from core.state import get_state

class GUIService:
    """
    GUI Service - Unified wrapper for existing working GUI
    Provides integration with the unified architecture while preserving working functionality
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()

        # GUI instance
        self.gui = None
        self.gui_thread = None
        self.running = False

        # GUI status
        self.initialized = False
        self.display_active = False

        # Manual control integration
        self.manual_control_enabled = True

    def initialize(self) -> bool:
        """Initialize GUI service"""
        try:
            if not GUI_AVAILABLE:
                print("❌ GUI service not available - missing dependencies")
                return False

            # Create GUI instance
            self.gui = LiveDetectionGUIWithModes()

            # For testing, we'll defer actual camera initialization
            # to avoid blocking on camera.capture_array()
            self.initialized = True
            print("✅ GUI service initialized (deferred camera init)")
            return True

        except Exception as e:
            print(f"❌ GUI service initialization failed: {e}")
            return False

    def start_gui(self) -> bool:
        """Start the GUI in a separate thread"""
        if not self.initialized:
            print("❌ GUI not initialized")
            return False

        if self.running:
            print("⚠️ GUI already running")
            return True

        try:
            self.running = True

            # Start GUI in separate thread
            self.gui_thread = threading.Thread(
                target=self._gui_thread_main,
                daemon=True,
                name="GUIService"
            )
            self.gui_thread.start()

            # Subscribe to events for GUI integration
            self.bus.subscribe('vehicle.*', self._handle_vehicle_events)
            self.bus.subscribe('system.*', self._handle_system_events)

            self.display_active = True
            print("✅ GUI service started")
            return True

        except Exception as e:
            print(f"❌ Failed to start GUI service: {e}")
            self.running = False
            return False

    def _gui_thread_main(self):
        """Main GUI thread function"""
        try:
            print("🎬 Starting GUI thread...")

            # Initialize the GUI's camera and AI here, when actually running
            if hasattr(self.gui, 'initialize') and not hasattr(self.gui, '_initialized_camera'):
                self.gui.initialize()
                self.gui._initialized_camera = True

            # Run the GUI (this blocks until GUI exits)
            self.gui.run()

        except Exception as e:
            print(f"❌ GUI thread error: {e}")
        finally:
            self.running = False
            self.display_active = False
            print("🎬 GUI thread stopped")

    def _handle_vehicle_events(self, event_type: str, data: Dict[str, Any]):
        """Handle vehicle events for GUI updates"""
        try:
            if 'movement' in event_type and self.gui:
                # Update vehicle motion state in GUI
                direction = data.get('direction', 'unknown')
                manual = data.get('manual', False)

                # Update GUI vehicle state for auto-mode switching
                if hasattr(self.gui, 'simulated_vehicle_motion'):
                    self.gui.simulated_vehicle_motion = (direction != 'stop' and manual)

        except Exception as e:
            print(f"❌ GUI vehicle event error: {e}")

    def _handle_system_events(self, event_type: str, data: Dict[str, Any]):
        """Handle system events for GUI updates"""
        try:
            # Update GUI based on system state changes
            pass
        except Exception as e:
            print(f"❌ GUI system event error: {e}")

    def process_keyboard_input(self, key: str) -> bool:
        """
        Process keyboard input for both GUI and manual control

        Args:
            key: Keyboard key pressed

        Returns:
            True if key was processed
        """
        if not self.running or not self.gui:
            return False

        try:
            # Let GUI handle the input first
            gui_handled = False
            if hasattr(self.gui, '_handle_keyboard_input'):
                # Convert key to GUI format
                key_code = ord(key) if len(key) == 1 else 255
                gui_handled = self.gui._handle_keyboard_input(key_code)

            # Also send to manual control if it's movement keys
            if self.manual_control_enabled and key.lower() in ['w', 'a', 's', 'd', ' ']:
                try:
                    from services.motion.motor import get_motor_service
                    motor_service = get_motor_service()
                    motor_service.keyboard_control(key)
                except:
                    pass  # Motor service might not be available

            return gui_handled

        except Exception as e:
            print(f"❌ Keyboard input error: {e}")
            return False

    def set_manual_control_enabled(self, enabled: bool):
        """Enable/disable manual control integration"""
        self.manual_control_enabled = enabled
        print(f"🎮 Manual control integration: {'enabled' if enabled else 'disabled'}")

    def take_screenshot(self) -> bool:
        """Take a screenshot using GUI functionality"""
        try:
            if self.gui and hasattr(self.gui, '_save_screenshot'):
                self.gui._save_screenshot()
                return True
        except Exception as e:
            print(f"❌ Screenshot error: {e}")
        return False

    def set_camera_mode(self, mode: str) -> bool:
        """Set camera mode through GUI"""
        try:
            if not self.gui or not hasattr(self.gui, 'camera_controller'):
                return False

            mode_map = {
                'photography': 1,
                'detection': 2,
                'vigilant': 3,
                'idle': 4
            }

            if mode in mode_map:
                key_code = ord(str(mode_map[mode]))
                return self.gui._handle_keyboard_input(key_code)

        except Exception as e:
            print(f"❌ Camera mode error: {e}")
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get GUI service status"""
        return {
            'initialized': self.initialized,
            'running': self.running,
            'display_active': self.display_active,
            'gui_available': GUI_AVAILABLE,
            'manual_control_enabled': self.manual_control_enabled,
            'gui_thread_alive': self.gui_thread.is_alive() if self.gui_thread else False,
            'gui_stats': self.gui.stats if self.gui and hasattr(self.gui, 'stats') else None
        }

    def stop_gui(self):
        """Stop the GUI service"""
        print("🛑 Stopping GUI service...")

        try:
            self.running = False

            # Stop GUI if running
            if self.gui and hasattr(self.gui, 'running'):
                self.gui.running = False

            # Wait for GUI thread to complete
            if self.gui_thread and self.gui_thread.is_alive():
                self.gui_thread.join(timeout=5.0)

            self.display_active = False
            print("✅ GUI service stopped")

        except Exception as e:
            print(f"❌ GUI stop error: {e}")

    def cleanup(self):
        """Clean up GUI service"""
        print("🧹 Cleaning up GUI service...")

        try:
            # Stop GUI
            self.stop_gui()

            # Cleanup GUI instance
            if self.gui and hasattr(self.gui, 'cleanup'):
                self.gui.cleanup()

            self.initialized = False
            print("✅ GUI service cleanup complete")

        except Exception as e:
            print(f"❌ GUI cleanup error: {e}")

# Global service instance
_gui_service = None

def get_gui_service() -> GUIService:
    """Get global GUI service instance"""
    global _gui_service
    if _gui_service is None:
        _gui_service = GUIService()
    return _gui_service

# Test function
def test_gui_service():
    """Test GUI service functionality"""
    print("🧪 Testing GUI Service...")

    service = get_gui_service()

    if not service.initialize():
        print("❌ GUI service initialization failed")
        return

    print("✅ GUI service initialized")

    # Start GUI
    if service.start_gui():
        print("✅ GUI started")

        # Let it run for a bit
        time.sleep(5)

        # Test some functions
        print("📸 Testing screenshot...")
        service.take_screenshot()

        print("🎮 Testing manual control...")
        service.set_manual_control_enabled(True)

        # Test keyboard inputs
        test_keys = ['w', 's', 'a', 'd', ' ']
        for key in test_keys:
            print(f"Testing key: {key}")
            service.process_keyboard_input(key)
            time.sleep(0.5)

        print("Status:", service.get_status())

    # Cleanup
    service.cleanup()
    print("✅ GUI service test complete")

if __name__ == "__main__":
    test_gui_service()