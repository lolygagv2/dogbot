#!/usr/bin/env python3
"""
Simple Bluetooth Gamepad Control for TreatBot
Connect a gamepad and drive the robot around!
"""

import time
import signal
import sys

# Force gpioset motor controller
import services.motion.motor as motor_mod
motor_mod.CORE_MOTOR_AVAILABLE = False
motor_mod.ALT_MOTOR_AVAILABLE = False

from services.input.gamepad import get_gamepad_service
from services.motion.motor import get_motor_service

class GamepadController:
    """Simple gamepad to motor integration"""

    def __init__(self):
        self.gamepad = get_gamepad_service()
        self.motor = get_motor_service()
        self.running = False

    def movement_handler(self, direction: str, speed: int):
        """Handle movement commands from gamepad"""
        if direction == 'stop':
            self.motor.emergency_stop()
        else:
            # Send short pulses for continuous movement
            self.motor.manual_drive(direction, speed, 0.1)

    def button_handler(self, button: str):
        """Handle button presses from gamepad"""
        if button == 'treat':
            print("🦴 Treat button pressed!")
            # TODO: Add treat dispenser call
        elif button == 'emergency_stop':
            print("🚨 Emergency stop button!")
            self.motor.emergency_stop()

    def start(self):
        """Start gamepad control"""
        print("🎮 TreatBot Gamepad Control")
        print("=" * 40)

        # Initialize motor service
        if not self.motor.initialize():
            print("❌ Motor service failed to initialize")
            return False

        print(f"✅ Motor service ready (controller: {self.motor.controller_type})")

        # Scan for controllers
        controllers = self.gamepad.scan_controllers()
        if not controllers:
            print("❌ No Bluetooth controllers found")
            print("\nTo connect a controller:")
            print("1. Put your controller in pairing mode")
            print("2. Run: sudo bluetoothctl")
            print("3. In bluetoothctl: scan on")
            print("4. In bluetoothctl: pair <controller_address>")
            print("5. In bluetoothctl: connect <controller_address>")
            return False

        print(f"🎮 Found {len(controllers)} controller(s):")
        for ctrl in controllers:
            print(f"   {ctrl['id']}: {ctrl['name']}")

        # Connect to first controller
        if not self.gamepad.connect_controller(0):
            print("❌ Failed to connect to controller")
            return False

        # Set up callbacks
        self.gamepad.set_movement_callback(self.movement_handler)
        self.gamepad.set_button_callback(self.button_handler)

        # Start gamepad service
        if not self.gamepad.start():
            print("❌ Failed to start gamepad service")
            return False

        print("\n🚀 Gamepad control active!")
        print("📋 Controls:")
        print("   Left stick / D-pad: Move robot")
        print("   A button: Dispense treat")
        print("   B button: Emergency stop")
        print("   Ctrl+C: Quit")
        print("")

        self.running = True
        return True

    def stop(self):
        """Stop gamepad control"""
        print("\n🛑 Stopping gamepad control...")
        self.running = False
        self.motor.emergency_stop()
        self.gamepad.cleanup()
        print("✅ Gamepad control stopped")

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n⚠️ Interrupt received")
    controller.stop()
    sys.exit(0)

def main():
    global controller
    controller = GamepadController()

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Start the controller
    if controller.start():
        # Keep running until interrupted
        try:
            while controller.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            signal_handler(None, None)
    else:
        print("❌ Failed to start gamepad control")

if __name__ == "__main__":
    main()