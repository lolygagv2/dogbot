#!/usr/bin/env python3
"""
Direct Xbox Controller to Motor Control
Based on working test_simple_joystick_motor.py
"""

import struct
import time
import os
import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.hardware.motor_controller_polling import MotorControllerPolling

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(message)s')
logger = logging.getLogger('XboxDirect')

class XboxDirectController:
    def __init__(self):
        self.device_path = '/dev/input/js0'
        self.motor_controller = None
        self.running = False

    def initialize(self):
        """Initialize motor controller and Xbox device"""
        try:
            # Initialize motor controller
            self.motor_controller = MotorControllerPolling()
            logger.info("✅ Motor controller initialized")

            # Open Xbox controller
            self.device = open(self.device_path, 'rb')
            logger.info(f"✅ Xbox controller connected at {self.device_path}")
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def run(self):
        """Main control loop - direct Xbox to motor mapping"""
        if not self.initialize():
            return

        logger.info("🎮 XBOX DIRECT MOTOR CONTROL")
        logger.info("Move LEFT STICK for movement")
        logger.info("Hold RT for speed boost")
        logger.info("Press Ctrl+C to stop")

        self.running = True
        left_x, left_y = 0.0, 0.0
        right_trigger = 0.0

        try:
            while self.running:
                # Read Xbox controller events
                event_data = self.device.read(8)
                if len(event_data) != 8:
                    continue

                timestamp, value, event_type, number = struct.unpack('IhBB', event_data)

                if event_type == 2:  # Axis events
                    normalized = value / 32767.0

                    if number == 0:  # Left stick X
                        left_x = normalized
                    elif number == 1:  # Left stick Y
                        left_y = -normalized  # Invert Y for forward
                    elif number == 5:  # Right trigger
                        right_trigger = (normalized + 1.0) / 2.0  # Convert from -1..1 to 0..1

                elif event_type == 1 and number == 1 and value == 1:  # B button
                    logger.info("🚨 B BUTTON - EMERGENCY STOP!")
                    self.motor_controller.set_motor_speed('left', 0, 'stop')
                    self.motor_controller.set_motor_speed('right', 0, 'stop')
                    continue

                # Calculate motor speeds (same logic as working test)
                if right_trigger > 0.1:
                    # Speed boost with RT trigger
                    forward = left_y * 50  # 50% max with boost
                    turn = -left_x * 30    # Fixed turn direction
                else:
                    # Normal speed
                    forward = left_y * 30  # 30% max normal
                    turn = -left_x * 20    # Fixed turn direction

                # Tank steering
                left_speed = int(forward - turn)
                right_speed = int(forward + turn)

                # Apply limits
                left_speed = max(-70, min(70, left_speed))
                right_speed = max(-70, min(70, right_speed))

                # Send motor commands with deadzone
                if abs(left_speed) > 2:
                    if left_speed > 0:
                        self.motor_controller.set_motor_speed('left', abs(left_speed), 'forward')
                    else:
                        self.motor_controller.set_motor_speed('left', abs(left_speed), 'backward')
                else:
                    self.motor_controller.set_motor_speed('left', 0, 'stop')

                if abs(right_speed) > 2:
                    if right_speed > 0:
                        self.motor_controller.set_motor_speed('right', abs(right_speed), 'forward')
                    else:
                        self.motor_controller.set_motor_speed('right', abs(right_speed), 'backward')
                else:
                    self.motor_controller.set_motor_speed('right', 0, 'stop')

                # Show active commands only
                if abs(left_speed) > 2 or abs(right_speed) > 2:
                    boost = "🚀" if right_trigger > 0.1 else ""
                    logger.info(f"🎮{boost} L={left_speed:+3d}, R={right_speed:+3d}")

        except KeyboardInterrupt:
            logger.info("Stopping...")
        except Exception as e:
            logger.error(f"Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.motor_controller:
            self.motor_controller.set_motor_speed('left', 0, 'stop')
            self.motor_controller.set_motor_speed('right', 0, 'stop')
            self.motor_controller.cleanup()
        if hasattr(self, 'device'):
            self.device.close()
        logger.info("✅ Cleanup complete")

if __name__ == "__main__":
    controller = XboxDirectController()
    controller.run()