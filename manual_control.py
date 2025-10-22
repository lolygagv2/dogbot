#!/usr/bin/env python3
"""
Manual Control Interface - RC Car Style Remote Control
Simple keyboard interface for manual vehicle control with live camera view
"""

import sys
import os
import time
import threading
import signal
from typing import Optional

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Unified architecture imports
from services.motion.motor import get_motor_service, MovementMode
from services.ui.gui import get_gui_service

class ManualControlInterface:
    """
    Manual control interface providing RC car style remote control
    Integrates motor control with live camera GUI
    """

    def __init__(self):
        self.motor_service = None
        self.gui_service = None
        self.running = False

        # Control settings
        self.default_speed = 60
        self.default_duration = 0.3  # Short bursts for responsive control

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def initialize(self) -> bool:
        """Initialize manual control system"""
        print("🎮 Initializing Manual Control Interface...")
        print("=" * 50)

        try:
            # Initialize motor service
            self.motor_service = get_motor_service()
            if not self.motor_service.initialize():
                print("❌ Motor service initialization failed")
                return False
            print("✅ Motor service ready")

            # Set to manual mode
            self.motor_service.set_movement_mode(MovementMode.MANUAL)
            print("✅ Manual control mode activated")

            # Initialize GUI service (optional)
            try:
                self.gui_service = get_gui_service()
                if self.gui_service.initialize():
                    print("✅ GUI service ready")
                    # Enable manual control integration
                    self.gui_service.set_manual_control_enabled(True)
                else:
                    print("⚠️ GUI service failed - running without display")
                    self.gui_service = None
            except Exception as e:
                print(f"⚠️ GUI service not available: {e}")
                self.gui_service = None

            return True

        except Exception as e:
            print(f"❌ Manual control initialization failed: {e}")
            return False

    def start(self):
        """Start manual control interface"""
        if not self.motor_service:
            print("❌ Motor service not initialized")
            return

        print("\n🚀 Manual Control Interface Started!")
        print("=" * 50)
        print("CONTROLS:")
        print("  W/↑  - Forward")
        print("  S/↓  - Backward")
        print("  A/←  - Turn Left")
        print("  D/→  - Turn Right")
        print("  SPACE - Emergency Stop")
        print("  Q     - Quit")
        print("")
        print("Camera Controls (if GUI active):")
        print("  1-4   - Camera modes")
        print("  V     - Toggle vehicle simulation")
        print("  R     - Reset camera position")
        print("")
        print("Press Ctrl+C to exit")
        print("=" * 50)

        self.running = True

        # Start GUI if available
        if self.gui_service:
            gui_thread = threading.Thread(
                target=self._start_gui,
                daemon=True,
                name="GUIThread"
            )
            gui_thread.start()
            time.sleep(2)  # Give GUI time to start

        # Start control loop
        try:
            if self.gui_service and self.gui_service.running:
                # GUI is running - it handles keyboard input
                print("🎬 GUI active - use GUI window for controls")
                self._run_with_gui()
            else:
                # No GUI - use terminal keyboard input
                print("⌨️ Terminal mode - use keyboard for controls")
                self._run_terminal_mode()

        except KeyboardInterrupt:
            print("\n🛑 Manual control interrupted by user")
        finally:
            self.cleanup()

    def _start_gui(self):
        """Start GUI in separate thread"""
        try:
            if self.gui_service:
                self.gui_service.start_gui()
        except Exception as e:
            print(f"❌ GUI start error: {e}")

    def _run_with_gui(self):
        """Run manual control with GUI active"""
        # GUI handles keyboard input, we just monitor status
        while self.running and self.gui_service.running:
            try:
                time.sleep(0.1)

                # Print status updates periodically
                if int(time.time()) % 10 == 0:
                    motor_status = self.motor_service.get_status()
                    print(f"📊 Status: Speed={motor_status.get('current_speed', 0)}% "
                          f"Direction={motor_status.get('current_direction', 'stop')} "
                          f"Moving={motor_status.get('is_moving', False)}")

            except Exception as e:
                print(f"❌ Control loop error: {e}")
                break

    def _run_terminal_mode(self):
        """Run manual control in terminal mode"""
        print("⚠️ Terminal mode - limited functionality")
        print("Use W/A/S/D keys followed by Enter...")

        while self.running:
            try:
                # Get user input
                user_input = input("Enter command (w/a/s/d/space/q): ").strip().lower()

                if user_input == 'q':
                    break
                elif user_input in ['w', 'a', 's', 'd', 'space', ' ']:
                    self._process_key(user_input)
                else:
                    print("Invalid command. Use w/a/s/d/space/q")

            except (EOFError, KeyboardInterrupt):
                break
            except Exception as e:
                print(f"❌ Input error: {e}")

    def _process_key(self, key: str):
        """Process keyboard input"""
        try:
            # Map keys to directions
            key_map = {
                'w': 'forward',
                'a': 'left',
                's': 'backward',
                'd': 'right',
                'space': 'stop',
                ' ': 'stop'
            }

            if key in key_map:
                direction = key_map[key]
                speed = self.default_speed if direction != 'stop' else 0
                duration = self.default_duration if direction != 'stop' else None

                success = self.motor_service.manual_drive(direction, speed, duration)

                if success:
                    print(f"🚗 {direction.capitalize()} at {speed}%")
                else:
                    print(f"❌ Failed to execute: {direction}")

        except Exception as e:
            print(f"❌ Key processing error: {e}")

    def emergency_stop(self):
        """Emergency stop all movement"""
        try:
            if self.motor_service:
                self.motor_service.emergency_stop()
            print("🛑 EMERGENCY STOP ACTIVATED")
        except Exception as e:
            print(f"❌ Emergency stop error: {e}")

    def _signal_handler(self, signum: int, frame):
        """Handle system signals"""
        print(f"\n📡 Received signal {signum} - shutting down...")
        self.running = False

    def cleanup(self):
        """Clean up manual control interface"""
        print("\n🧹 Cleaning up manual control interface...")

        try:
            self.running = False

            # Emergency stop
            if self.motor_service:
                self.motor_service.emergency_stop()

            # Cleanup services
            if self.gui_service:
                self.gui_service.cleanup()

            if self.motor_service:
                self.motor_service.cleanup()

            print("✅ Manual control cleanup complete")

        except Exception as e:
            print(f"❌ Cleanup error: {e}")

def main():
    """Main entry point"""
    print("🤖 TreatBot - Manual Control Interface")
    print("RC Car Style Remote Control with Live Camera")
    print("=" * 50)

    # Create and run manual control interface
    control = ManualControlInterface()

    try:
        if not control.initialize():
            print("❌ Initialization failed")
            return 1

        control.start()

    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1

    print("👋 Manual control shutdown complete")
    return 0

if __name__ == "__main__":
    exit(main())