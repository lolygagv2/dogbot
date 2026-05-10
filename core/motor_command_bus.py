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
    WEBRTC = "webrtc"
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
        self.use_pid = True  # default, overridden by config

        # Load PID config from robot profile
        try:
            from config.config_loader import get_config
            self.use_pid = get_config().controller.use_pid_control
            self.logger.info(f"Motor bus: use_pid={self.use_pid} (from robot config)")
        except Exception:
            self.logger.warning("Motor bus: could not load config, defaulting use_pid=True")

        # Cytron MDD10A first when profile selects it (treatbot3/4/5 — 9V brushed,
        # no encoders, no PID). Gated by controller.driver=="cytron" so treatbot1/2
        # are unaffected and continue using ProperPIDMotorController below.
        try:
            from config.config_loader import get_config as _get_cfg
            _raw = _get_cfg().raw.get('controller', {})
            if _raw.get('driver') == "cytron":
                from core.hardware.motor_controller_cytron import MotorControllerCytron
                cy = _raw.get('cytron', {})
                cal = _raw.get('motor_calibration', {})
                self.motor_controller = MotorControllerCytron(
                    left_dir_pin=cy.get('left_dir_pin', 17),
                    left_pwm_pin=cy.get('left_pwm_pin', 13),
                    right_dir_pin=cy.get('right_dir_pin', 27),
                    right_pwm_pin=cy.get('right_pwm_pin', 19),
                    left_invert=cy.get('left_invert', False),
                    right_invert=cy.get('right_invert', False),
                    left_multiplier=cal.get('left_multiplier', 1.0),
                    right_multiplier=cal.get('right_multiplier', 1.0),
                    max_pwm_pct=cy.get('max_pwm_pct', 100),
                    pwm_freq_hz=cy.get('pwm_freq_hz', 1000),
                )
                self.logger.info("Cytron MDD10A motor controller initialized (open-loop, no encoders)")
        except Exception as e:
            self.logger.warning(f"Cytron dispatch skipped: {e}")
            self.motor_controller = None

        # Try to import motor controller - prioritize proper PID controller
        if self.motor_controller is None:
            try:
                from core.hardware.proper_pid_motor_controller import ProperPIDMotorController
                self.motor_controller = ProperPIDMotorController()
                self.logger.info("Proper PID motor controller initialized (closed-loop control)")
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
            # Start PID controller (ProperPIDMotorController uses start() method)
            if hasattr(self.motor_controller, 'start'):
                if not self.motor_controller.start():
                    self.logger.error("PID motor controller start failed")
                    return False
            elif hasattr(self.motor_controller, 'initialize'):
                if not self.motor_controller.initialize():
                    self.logger.error("Motor controller initialization failed")
                    return False

            self.running = True
            self.logger.info("Motor command bus started with PID control")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start motor command bus: {e}")
            return False

    def stop(self):
        """Stop the motor command bus"""
        self.running = False
        if self.motor_controller:
            try:
                # Use controller's native stop method if available (ProperPIDMotorController)
                if hasattr(self.motor_controller, 'stop'):
                    self.motor_controller.stop()
                # Fallback to set_motor_rpm for zero speed
                elif hasattr(self.motor_controller, 'set_motor_rpm'):
                    self.motor_controller.set_motor_rpm(0, 0)
                # Last resort - legacy interface
                elif hasattr(self.motor_controller, 'set_motor_speed'):
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

                # Use direct PWM when PID is disabled (broken encoders etc)
                if not self.use_pid and hasattr(self.motor_controller, 'set_motor_pwm_direct'):
                    self.motor_controller.set_motor_pwm_direct(float(left_speed), float(right_speed))
                    self.logger.debug(f"Motor PWM direct: L={left_speed}, R={right_speed}")
                elif hasattr(self.motor_controller, 'set_motor_rpm'):
                    # Convert speed percentages to RPM targets for PID control
                    max_rpm = 120  # Maximum safe RPM for DFRobot motors
                    left_rpm = (left_speed / 100.0) * max_rpm
                    right_rpm = (right_speed / 100.0) * max_rpm

                    self.motor_controller.set_motor_rpm(left_rpm, right_rpm)
                    self.logger.debug(f"Motor RPM command: L={left_rpm:.1f}, R={right_rpm:.1f} RPM")
                else:
                    # Direct PWM control
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
                    self.logger.debug(f"Motor PWM command: L={left_speed}, R={right_speed}")

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