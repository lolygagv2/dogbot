#!/usr/bin/env python3
"""
DFRobot Motor Controller with Polling-based Encoder Tracking
- 1000Hz polling rate for real-time encoder feedback
- Replaces broken lgpio.callback interrupts with reliable polling
- Hardware mapping: Motor A = Left (inverted), Motor B = Right
- Safety: 50% max PWM = 6.3V for 6V motors on 14V system
"""

import subprocess
import time
import threading
import signal
import atexit
from enum import Enum
from typing import Optional, Tuple, Dict
import logging

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.pins import TreatBotPins
from config.settings import SystemSettings

logger = logging.getLogger(__name__)

class MotorDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"

class EncoderState:
    """Track encoder state for quadrature decoding"""
    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.last_a = 0
        self.last_b = 0
        self.changes = 0
        self.last_change_time = time.time()

class MotorControllerPolling:
    """Motor controller with 1000Hz polling-based encoder tracking"""

    def __init__(self):
        self.pins = TreatBotPins()
        self.settings = SystemSettings()

        # Motor state tracking
        self.left_speed = 0
        self.right_speed = 0
        self.is_moving = False
        self.last_command_time = time.time()

        # Thread safety
        self.motor_lock = threading.Lock()
        self.encoder_lock = threading.Lock()

        # Encoder state
        self.encoders = {
            'left': EncoderState('Left'),
            'right': EncoderState('Right')
        }

        # PWM emulation via rapid gpioset
        self.pwm_threads = {}
        self.pwm_running = {}

        # Polling control
        self.polling_active = False
        self.polling_thread = None
        self.polling_rate = 1000  # 1000Hz polling

        # Safety limits from hardware specs
        self.MIN_SAFE_PWM = 20  # Minimum for motor movement
        self.MAX_SAFE_PWM = 50  # Maximum = 6.3V for 6V motors

        logger.info("Polling motor controller initialized")
        self._start_encoder_polling()

    def _start_encoder_polling(self):
        """Start 1000Hz encoder polling thread"""
        self.polling_active = True
        self.polling_thread = threading.Thread(target=self._encoder_polling_loop, daemon=False)
        self.polling_thread.start()
        logger.info("Started 1000Hz encoder polling thread")

    def _encoder_polling_loop(self):
        """Main encoder polling loop at 1000Hz"""
        poll_interval = 1.0 / self.polling_rate  # 0.001 seconds = 1ms

        while self.polling_active:
            try:
                start_time = time.time()

                # Poll left motor encoder (Motor A)
                self._poll_encoder('left', self.pins.ENCODER_A1, self.pins.ENCODER_B1)

                # Poll right motor encoder (Motor B)
                self._poll_encoder('right', self.pins.ENCODER_A2, self.pins.ENCODER_B2)

                # Maintain precise timing
                elapsed = time.time() - start_time
                sleep_time = max(0, poll_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Encoder polling error: {e}")
                time.sleep(0.001)  # Brief pause on error

    def _poll_encoder(self, motor: str, pin_a: int, pin_b: int):
        """Poll individual encoder and decode quadrature signals"""
        try:
            # Read current pin states using gpioget
            result_a = subprocess.run(['gpioget', 'gpiochip0', str(pin_a)],
                                    capture_output=True, text=True, timeout=0.001)
            result_b = subprocess.run(['gpioget', 'gpiochip0', str(pin_b)],
                                    capture_output=True, text=True, timeout=0.001)

            if result_a.returncode != 0 or result_b.returncode != 0:
                return  # Skip this poll cycle on GPIO read failure

            current_a = int(result_a.stdout.strip())
            current_b = int(result_b.stdout.strip())

            with self.encoder_lock:
                encoder = self.encoders[motor]

                # Quadrature decoding - detect state changes
                if current_a != encoder.last_a or current_b != encoder.last_b:
                    # State changed - decode direction
                    if encoder.last_a == 0 and current_a == 1:
                        # Rising edge on A
                        if current_b == 0:
                            encoder.count += 1  # Forward
                        else:
                            encoder.count -= 1  # Backward
                    elif encoder.last_a == 1 and current_a == 0:
                        # Falling edge on A
                        if current_b == 1:
                            encoder.count += 1  # Forward
                        else:
                            encoder.count -= 1  # Backward

                    encoder.changes += 1
                    encoder.last_change_time = time.time()

                # Update last known states
                encoder.last_a = current_a
                encoder.last_b = current_b

        except Exception as e:
            # Don't log every polling error to avoid spam
            pass

    def _run_gpio_command(self, pin: int, value: int) -> bool:
        """Execute gpioset command safely"""
        try:
            cmd = ['gpioset', 'gpiochip0', f'{pin}={value}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=0.1)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"GPIO command error on pin {pin}: {e}")
            return False

    def _emulate_pwm(self, pin: int, duty_cycle: int, motor_name: str):
        """Emulate PWM using rapid gpioset commands"""
        if duty_cycle == 0:
            self._run_gpio_command(pin, 0)
            return

        if duty_cycle >= 100:
            self._run_gpio_command(pin, 1)
            return

        # Calculate on/off times for 100Hz PWM emulation
        frequency = 100  # Lower frequency for stability
        period = 1.0 / frequency
        on_time = period * (duty_cycle / 100.0)
        off_time = period - on_time

        # Stop any existing PWM thread for this pin
        self._stop_pwm_thread(pin)

        # Create new PWM thread
        self.pwm_running[pin] = True

        def pwm_loop():
            while self.pwm_running.get(pin, False):
                try:
                    # On phase
                    self._run_gpio_command(pin, 1)
                    time.sleep(on_time)

                    # Off phase
                    if self.pwm_running.get(pin, False):
                        self._run_gpio_command(pin, 0)
                        time.sleep(off_time)
                except Exception as e:
                    logger.error(f"PWM emulation error on pin {pin}: {e}")
                    break

            # Ensure pin is off when stopping
            self._run_gpio_command(pin, 0)

        thread = threading.Thread(target=pwm_loop, daemon=False)
        thread.start()
        self.pwm_threads[pin] = thread

    def _stop_pwm_thread(self, pin: int):
        """Stop PWM emulation thread"""
        if pin in self.pwm_running:
            self.pwm_running[pin] = False

        if pin in self.pwm_threads:
            thread = self.pwm_threads[pin]
            thread.join(timeout=0.5)
            if thread.is_alive():
                logger.error(f"PWM thread for pin {pin} still alive after timeout!")
                self._run_gpio_command(pin, 0)
            del self.pwm_threads[pin]

    def set_motor_speed(self, motor: str, speed: int, direction: str) -> bool:
        """Control individual motor with hardware compensation and safety limits"""
        with self.motor_lock:
            try:
                # Apply safety limits
                speed = max(0, min(100, speed))

                # Convert to safe duty cycle (20-50% range)
                if speed > 0:
                    safe_duty = int(self.MIN_SAFE_PWM +
                                   (speed * (self.MAX_SAFE_PWM - self.MIN_SAFE_PWM) / 100))
                else:
                    safe_duty = 0

                safe_duty = max(0, min(self.MAX_SAFE_PWM, safe_duty))

                # Hardware mapping with direction compensation
                if motor in ['A', 'left']:
                    in1, in2, ena = self.pins.MOTOR_IN1, self.pins.MOTOR_IN2, self.pins.MOTOR_ENA
                    motor_name = "Left"
                    # Motor A direction inversion due to wiring
                    if direction == 'forward':
                        direction = 'backward'
                    elif direction == 'backward':
                        direction = 'forward'
                elif motor in ['B', 'right']:
                    in1, in2, ena = self.pins.MOTOR_IN3, self.pins.MOTOR_IN4, self.pins.MOTOR_ENB
                    motor_name = "Right"
                else:
                    logger.error(f"Invalid motor: {motor}")
                    return False

                # Set direction and PWM
                if direction == 'stop' or speed == 0:
                    self._run_gpio_command(in1, 0)
                    self._run_gpio_command(in2, 0)
                    self._stop_pwm_thread(ena)
                    if motor in ['A', 'left']:
                        self.left_speed = 0
                    else:
                        self.right_speed = 0

                elif direction == 'forward':
                    self._run_gpio_command(in1, 1)
                    self._run_gpio_command(in2, 0)
                    self._emulate_pwm(ena, safe_duty, motor_name)
                    if motor in ['A', 'left']:
                        self.left_speed = speed
                    else:
                        self.right_speed = speed

                elif direction == 'backward':
                    self._run_gpio_command(in1, 0)
                    self._run_gpio_command(in2, 1)
                    self._emulate_pwm(ena, safe_duty, motor_name)
                    if motor in ['A', 'left']:
                        self.left_speed = -speed
                    else:
                        self.right_speed = -speed

                self.last_command_time = time.time()
                effective_voltage = 12.6 * safe_duty / 100
                logger.debug(f"Motor {motor_name}: {direction} at {speed}% "
                           f"(PWM: {safe_duty}%, ~{effective_voltage:.1f}V)")
                return True

            except Exception as e:
                logger.error(f"Motor control error: {e}")
                return False

    def tank_steering(self, direction: MotorDirection, speed: int = 50, duration: Optional[float] = None) -> bool:
        """Tank-style differential steering"""
        try:
            if direction == MotorDirection.FORWARD:
                self.set_motor_speed('left', speed, 'forward')
                self.set_motor_speed('right', speed, 'forward')

            elif direction == MotorDirection.BACKWARD:
                self.set_motor_speed('left', speed, 'backward')
                self.set_motor_speed('right', speed, 'backward')

            elif direction == MotorDirection.LEFT:
                # Turn left: left motor backward, right motor forward
                self.set_motor_speed('left', speed, 'backward')
                self.set_motor_speed('right', speed, 'forward')

            elif direction == MotorDirection.RIGHT:
                # Turn right: left motor forward, right motor backward
                self.set_motor_speed('left', speed, 'forward')
                self.set_motor_speed('right', speed, 'backward')

            elif direction == MotorDirection.STOP:
                self.set_motor_speed('left', 0, 'stop')
                self.set_motor_speed('right', 0, 'stop')

            self.is_moving = (direction != MotorDirection.STOP)

            # Auto-stop after duration
            if duration and direction != MotorDirection.STOP:
                threading.Timer(duration, self.emergency_stop).start()

            return True

        except Exception as e:
            logger.error(f"Tank steering error: {e}")
            return False

    def emergency_stop(self):
        """Emergency stop all motors"""
        with self.motor_lock:
            logger.info("Emergency stop - all motors halted")

            # Stop all PWM threads
            for pin in list(self.pwm_running.keys()):
                self._stop_pwm_thread(pin)

            # Set all motor pins to 0
            motor_pins = [
                self.pins.MOTOR_IN1, self.pins.MOTOR_IN2,
                self.pins.MOTOR_IN3, self.pins.MOTOR_IN4,
                self.pins.MOTOR_ENA, self.pins.MOTOR_ENB
            ]

            for pin in motor_pins:
                self._run_gpio_command(pin, 0)

            self.left_speed = 0
            self.right_speed = 0
            self.is_moving = False

    def get_encoder_status(self) -> Dict:
        """Get current encoder readings"""
        with self.encoder_lock:
            return {
                'left': {
                    'count': self.encoders['left'].count,
                    'changes': self.encoders['left'].changes,
                    'last_change': time.time() - self.encoders['left'].last_change_time,
                    'pin_a': self.encoders['left'].last_a,
                    'pin_b': self.encoders['left'].last_b
                },
                'right': {
                    'count': self.encoders['right'].count,
                    'changes': self.encoders['right'].changes,
                    'last_change': time.time() - self.encoders['right'].last_change_time,
                    'pin_a': self.encoders['right'].last_a,
                    'pin_b': self.encoders['right'].last_b
                },
                'polling_rate': self.polling_rate,
                'polling_active': self.polling_active
            }

    def reset_encoder_counts(self):
        """Reset encoder counts to zero"""
        with self.encoder_lock:
            for encoder in self.encoders.values():
                encoder.count = 0
                encoder.changes = 0
            logger.info("Encoder counts reset")

    def is_initialized(self) -> bool:
        """Check if controller is properly initialized"""
        return self.polling_active and (self.polling_thread is not None)

    def get_status(self) -> Dict:
        """Get comprehensive motor and encoder status"""
        encoder_status = self.get_encoder_status()
        return {
            'motors': {
                'left_speed': self.left_speed,
                'right_speed': self.right_speed,
                'is_moving': self.is_moving,
                'last_command': time.time() - self.last_command_time,
                'safety_limits': {
                    'min_pwm': self.MIN_SAFE_PWM,
                    'max_pwm': self.MAX_SAFE_PWM
                }
            },
            'encoders': encoder_status,
            'initialized': self.is_initialized()
        }

    def cleanup(self):
        """Cleanup - stop polling and motors"""
        logger.info("Motor controller cleanup starting")

        # Stop encoder polling
        self.polling_active = False
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=2.0)
            if self.polling_thread.is_alive():
                logger.error("Polling thread did not stop cleanly")

        # Emergency stop motors
        self.emergency_stop()

        logger.info("Motor controller cleanup complete")

# Global cleanup function for safety
def global_motor_emergency_stop():
    """Global emergency stop function callable from anywhere"""
    try:
        motor_pins = [17, 18, 27, 22, 13, 19]  # All motor control pins
        for pin in motor_pins:
            subprocess.run(['gpioset', 'gpiochip0', f'{pin}=0'],
                          capture_output=True, timeout=0.1)
        subprocess.run(['killall', 'gpioset'], capture_output=True, timeout=0.1)
    except Exception:
        pass

# Register emergency stop
atexit.register(global_motor_emergency_stop)

# Signal handlers for clean shutdown
def signal_handler(sig, frame):
    global_motor_emergency_stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)