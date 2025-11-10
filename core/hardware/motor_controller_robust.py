#!/usr/bin/env python3
"""
Robust motor controller that can recover from errors
"""

import subprocess
import time
import threading
import signal
import atexit
from enum import Enum
from typing import Optional
import logging

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.pins import TreatBotPins
from config.settings import SystemSettings

logger = logging.getLogger(__name__)

# CRITICAL SAFETY: Global emergency stop
def motor_emergency_stop():
    """Emergency stop all motors via direct GPIO commands"""
    logger.critical("MOTOR EMERGENCY STOP")
    motor_pins = [17, 27, 22, 23, 24, 25]  # All motor pins
    for pin in motor_pins:
        try:
            subprocess.run(['gpioset', 'gpiochip0', f'{pin}=0'],
                          capture_output=True, timeout=0.1)
        except:
            pass
    # Kill any gpioset processes
    try:
        subprocess.run(['killall', 'gpioset'], capture_output=True, timeout=0.1)
    except:
        pass

# Register emergency stop on exit
atexit.register(motor_emergency_stop)

class MotorDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"

class MotorControllerRobust:
    """Motor controller using gpioset commands - more robust than PWM"""

    def __init__(self):
        self.pins = TreatBotPins()
        self.settings = SystemSettings()

        # Motor state tracking
        self.left_speed = 0
        self.right_speed = 0
        self.is_moving = False

        # Thread safety
        self.motor_lock = threading.Lock()
        self.last_command_time = time.time()

        # PWM emulation via rapid gpioset
        self.pwm_threads = {}
        self.pwm_running = {}

        logger.info("Robust motor controller initialized")

    def _run_gpio_command(self, pin: int, value: int) -> bool:
        """Execute gpioset command safely"""
        try:
            cmd = ['gpioset', 'gpiochip0', f'{pin}={value}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=0.1)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"GPIO command error on pin {pin}: {e}")
            return False

    def _emulate_pwm(self, pin: int, frequency: int, duty_cycle: int, motor_name: str):
        """Emulate PWM using rapid gpioset commands"""
        if duty_cycle == 0:
            self._run_gpio_command(pin, 0)
            return

        if duty_cycle >= 100:
            self._run_gpio_command(pin, 1)
            return

        # Calculate on/off times for PWM emulation
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

        # CRITICAL SAFETY: Use non-daemon thread so it MUST be stopped
        thread = threading.Thread(target=pwm_loop, daemon=False)
        thread.start()
        self.pwm_threads[pin] = thread

    def _stop_pwm_thread(self, pin: int):
        """Stop PWM emulation thread - CRITICAL for safety"""
        if pin in self.pwm_running:
            self.pwm_running[pin] = False

        if pin in self.pwm_threads:
            thread = self.pwm_threads[pin]
            thread.join(timeout=0.5)  # Increased timeout for safety
            if thread.is_alive():
                logger.error(f"WARNING: PWM thread for pin {pin} still alive after timeout!")
                # Force GPIO to 0 anyway
                self._run_gpio_command(pin, 0)
            del self.pwm_threads[pin]

    def set_motor_speed(self, motor: str, speed: int, direction: str):
        """Control individual motor with gpioset commands - SAFE FOR 6V MOTORS"""
        with self.motor_lock:
            try:
                # CRITICAL: Limit PWM for 6V motors on 14V system
                # Use dynamic motor profile for speed control
                try:
                    from config.motor_profiles import get_profile_manager
                    profile_mgr = get_profile_manager()
                    MAX_SAFE_DUTY = profile_mgr.get_max_pwm()
                except ImportError:
                    # Fallback to static config
                    try:
                        from config.motor_tuning import MAX_MOTOR_PWM
                        MAX_SAFE_DUTY = MAX_MOTOR_PWM
                    except ImportError:
                        MAX_SAFE_DUTY = 50  # Ultimate fallback

                # Scale input speed (0-100) to safe duty cycle (0-50)
                safe_speed = int(speed * MAX_SAFE_DUTY / 100)
                safe_speed = max(0, min(MAX_SAFE_DUTY, safe_speed))

                if motor in ['A', 'left']:
                    in1, in2, ena = self.pins.MOTOR_IN1, self.pins.MOTOR_IN2, self.pins.MOTOR_ENA
                    motor_name = "Left"
                    # Wiring correction for left motor
                    if direction == 'forward':
                        direction = 'backward'
                    elif direction == 'backward':
                        direction = 'forward'
                elif motor in ['B', 'right']:
                    in1, in2, ena = self.pins.MOTOR_IN3, self.pins.MOTOR_IN4, self.pins.MOTOR_ENB
                    motor_name = "Right"
                else:
                    return False

                # Set direction pins
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
                    self._emulate_pwm(ena, 100, safe_speed, motor_name)  # Use safe_speed!
                    if motor in ['A', 'left']:
                        self.left_speed = speed  # Store original for status
                    else:
                        self.right_speed = speed

                elif direction == 'backward':
                    self._run_gpio_command(in1, 0)
                    self._run_gpio_command(in2, 1)
                    self._emulate_pwm(ena, 100, safe_speed, motor_name)  # Use safe_speed!
                    if motor in ['A', 'left']:
                        self.left_speed = -speed  # Store original for status
                    else:
                        self.right_speed = -speed

                self.last_command_time = time.time()
                logger.debug(f"Motor {motor_name}: {direction} at {speed}% (PWM: {safe_speed}%)")
                logger.info(f"Motor voltage: ~{12.6 * safe_speed / 100:.1f}V (safe for 6V motor)")
                return True

            except Exception as e:
                logger.error(f"Motor control error: {e}")
                # Don't cleanup - just log error and continue
                return False

    def emergency_stop(self):
        """Emergency stop all motors"""
        with self.motor_lock:
            logger.info("EMERGENCY STOP")

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

    def is_initialized(self):
        """Always return True since we use subprocess"""
        return True

    def cleanup(self):
        """Cleanup - CRITICAL to stop all PWM threads"""
        logger.info("Motor controller cleanup - stopping all PWM threads")
        self.emergency_stop()

        # Wait for all threads to actually stop
        for pin in list(self.pwm_threads.keys()):
            thread = self.pwm_threads.get(pin)
            if thread and thread.is_alive():
                logger.warning(f"Waiting for PWM thread on pin {pin} to stop...")
                thread.join(timeout=1.0)
                if thread.is_alive():
                    logger.error(f"PWM thread on pin {pin} won't stop!")

        # Don't set anything to None - stay ready for reuse

    def get_status(self):
        """Get current motor status"""
        return {
            'left_speed': self.left_speed,
            'right_speed': self.right_speed,
            'is_moving': self.is_moving,
            'last_command': time.time() - self.last_command_time,
            'initialized': True
        }