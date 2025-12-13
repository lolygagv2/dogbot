#!/usr/bin/env python3
"""
Motor Command Bus - Centralized motor control with polling encoders
Replaces broken lgpio interrupt system with reliable polling-based encoder tracking
"""

import time
import threading
import logging
from typing import Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger('MotorBus')

class CommandSource(Enum):
    XBOX_CONTROLLER = "xbox"
    API = "api"
    AUTONOMOUS = "auto"
    EMERGENCY = "emergency"

@dataclass
class MotorCommand:
    left_speed: int
    right_speed: int
    source: CommandSource
    timestamp: float

def create_motor_command(left: int, right: int, source: CommandSource) -> MotorCommand:
    """Create a motor command with current timestamp"""
    return MotorCommand(left, right, source, time.time())

class MotorCommandBus:
    """
    Centralized motor control with polling-based encoder tracking
    """

    def __init__(self):
        self.logger = logging.getLogger('MotorBus')
        self.motor_controller = None
        self.running = False
        self.last_command = None
        self.lock = threading.Lock()

        # Try to import motor controller - prioritize polling controller
        try:
            from core.hardware.motor_controller_polling import MotorControllerPolling
            self.motor_controller = MotorControllerPolling()
            self.logger.info("Polling motor controller initialized (with 1000Hz encoders)")
        except ImportError:
            try:
                from core.hardware.motor_controller_robust import MotorControllerRobust
                self.motor_controller = MotorControllerRobust()
                self.logger.info("Robust motor controller initialized (no encoders)")
            except ImportError:
                try:
                    from core.hardware.motor_controller_dfrobot_encoder import DFRobotEncoderMotorController
                    self.motor_controller = DFRobotEncoderMotorController()
                    self.logger.info("DFRobot encoder motor controller initialized")
                except ImportError:
                    self.logger.error("No motor controller available")
                    self.motor_controller = None

    def start(self) -> bool:
        """Start the motor command bus"""
        if self.motor_controller is None:
            self.logger.error("Cannot start - no motor controller available")
            return False

        try:
            if hasattr(self.motor_controller, 'initialize'):
                if not self.motor_controller.initialize():
                    self.logger.error("Motor controller initialization failed")
                    return False

            self.running = True
            self.logger.info("Motor command bus started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start motor command bus: {e}")
            return False

    def stop(self):
        """Stop the motor command bus"""
        self.running = False
        if self.motor_controller:
            try:
                # Emergency stop
                self.motor_controller.set_motor_speed('left', 0, 'forward')
                self.motor_controller.set_motor_speed('right', 0, 'forward')
                if hasattr(self.motor_controller, 'cleanup'):
                    self.motor_controller.cleanup()
            except Exception as e:
                self.logger.error(f"Error stopping motor controller: {e}")
        self.logger.info("Motor command bus stopped")

    def send_command(self, command: MotorCommand) -> bool:
        """Send motor command to the hardware"""
        if not self.running or self.motor_controller is None:
            return False

        try:
            with self.lock:
                # Store last command
                self.last_command = command

                # Apply motor speeds with safety limits
                left_speed = max(-70, min(70, command.left_speed))  # 70% max for safety
                right_speed = max(-70, min(70, command.right_speed))

                # Convert to motor controller format
                if left_speed == 0:
                    self.motor_controller.set_motor_speed('left', 0, 'forward')
                else:
                    left_dir = 'forward' if left_speed > 0 else 'backward'
                    self.motor_controller.set_motor_speed('left', abs(left_speed), left_dir)

                if right_speed == 0:
                    self.motor_controller.set_motor_speed('right', 0, 'forward')
                else:
                    right_dir = 'forward' if right_speed > 0 else 'backward'
                    self.motor_controller.set_motor_speed('right', abs(right_speed), right_dir)

                self.logger.debug(f"Motor command sent: L={left_speed}, R={right_speed}")
                return True

        except Exception as e:
            self.logger.error(f"Failed to send motor command: {e}")
            return False

    def get_status(self) -> dict:
        """Get motor bus status including encoder data if available"""
        status = {
            'running': self.running,
            'controller': type(self.motor_controller).__name__ if self.motor_controller else None,
            'last_command': None
        }

        if self.last_command:
            status['last_command'] = {
                'left': self.last_command.left_speed,
                'right': self.last_command.right_speed,
                'source': self.last_command.source.value,
                'timestamp': self.last_command.timestamp
            }

        # Add encoder status if available (polling controller)
        if self.motor_controller and hasattr(self.motor_controller, 'get_encoder_status'):
            try:
                status['encoders'] = self.motor_controller.get_encoder_status()
            except Exception as e:
                self.logger.error(f"Failed to get encoder status: {e}")

        # Add motor status if available
        if self.motor_controller and hasattr(self.motor_controller, 'get_status'):
            try:
                motor_status = self.motor_controller.get_status()
                if isinstance(motor_status, dict):
                    status['motor_details'] = motor_status
            except Exception as e:
                self.logger.error(f"Failed to get motor status: {e}")

        return status

# Global motor bus instance
_motor_bus = None

def get_motor_bus() -> MotorCommandBus:
    """Get singleton motor command bus instance"""
    global _motor_bus
    if _motor_bus is None:
        _motor_bus = MotorCommandBus()
    return _motor_bus