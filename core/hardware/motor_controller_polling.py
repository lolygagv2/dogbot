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

# Import GPIO control
from gpiozero import OutputDevice, PWMOutputDevice

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.pins import TreatBotPins
from config.settings import SystemSettings

# Import PID control system
from core.control.pid_controller import MotorPIDController
from core.calibration.motor_calibration import MotorCalibrationSystem

logger = logging.getLogger(__name__)

class MotorDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"

class EncoderState:
    """Track encoder state for quadrature decoding with RPM calculation"""
    def __init__(self, name: str):
        self.name = name
        self.count = 0
        self.last_a = 0
        self.last_b = 0
        self.changes = 0
        self.last_change_time = time.time()

        # RPM calculation
        self.ticks_per_revolution = 660  # DFRobot motor spec
        self.rpm_window_size = 20  # Moving average window
        self.rpm_history = []  # Store recent RPM measurements
        self.last_rpm_calc_time = time.time()
        self.last_rpm_count = 0
        self.current_rpm = 0.0

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

        # GPIO control using gpiozero (replaces broken subprocess PWM)
        self.left_in1 = OutputDevice(self.pins.MOTOR_IN1)
        self.left_in2 = OutputDevice(self.pins.MOTOR_IN2)
        self.left_ena = PWMOutputDevice(self.pins.MOTOR_ENA)
        self.right_in3 = OutputDevice(self.pins.MOTOR_IN3)
        self.right_in4 = OutputDevice(self.pins.MOTOR_IN4)
        self.right_enb = PWMOutputDevice(self.pins.MOTOR_ENB)

        # Polling control
        self.polling_active = False
        self.polling_thread = None
        self.polling_rate = 1000  # 1000Hz polling

        # Motor power settings - INCREASED for proper performance
        self.MIN_SAFE_PWM = 20  # Lower minimum for smoother control
        self.MAX_SAFE_PWM = 100  # Full power for PID system
        self.RIGHT_MOTOR_BOOST = 1.2  # Right motor needs extra power

        # RPM calculation settings
        self.rpm_calc_interval = 0.1  # Calculate RPM every 100ms
        self.last_rpm_calc_time = time.time()

        # PID Control System - Conservative gains for initial testing
        self.pid_controller = MotorPIDController(
            left_gains=(0.8, 0.1, 0.01),   # Much more conservative gains
            right_gains=(0.8, 0.1, 0.01),  # Reduce oscillation and clicking
            pwm_min=self.MIN_SAFE_PWM,
            pwm_max=self.MAX_SAFE_PWM
        )

        # Motor Calibration System
        calibration_file = os.path.join(os.path.dirname(__file__), '../../config/motor_calibration.json')
        self.calibration = MotorCalibrationSystem(self, calibration_file)

        # Control mode: 'direct' (old PWM) or 'pid' (closed-loop RPM)
        self.control_mode = 'pid'  # Default to closed-loop control

        # Target velocities for PID control
        self.target_left_rpm = 0.0
        self.target_right_rpm = 0.0

        # PID control loop
        self.pid_active = False
        self.pid_thread = None
        self.pid_rate = 50  # 50Hz PID control loop

        # PID gains (conservative for stability)
        self.pid_kp = 0.8  # Proportional gain
        self.pid_ki = 0.1  # Integral gain
        self.pid_kd = 0.01  # Derivative gain

        # PID state variables
        self.left_integral = 0.0
        self.left_last_error = 0.0
        self.right_integral = 0.0
        self.right_last_error = 0.0
        self.last_pid_time = time.time()

        logger.info("Polling motor controller initialized")
        self._start_encoder_polling()
        self._start_pid_control_loop()

    def initialize(self) -> bool:
        """Initialize method for Xbox controller compatibility"""
        # Already initialized in __init__, just return success
        return True

    def _start_encoder_polling(self):
        """Start 1000Hz encoder polling thread"""
        self.polling_active = True
        self.polling_thread = threading.Thread(target=self._encoder_polling_loop, daemon=False)
        self.polling_thread.start()
        logger.info("Started 1000Hz encoder polling thread")

    def _start_pid_control_loop(self):
        """Start PID control loop thread"""
        if self.control_mode == 'pid':
            self.pid_active = True
            self.pid_thread = threading.Thread(target=self._pid_control_loop, daemon=False)
            self.pid_thread.start()
            logger.info(f"Started {self.pid_rate}Hz PID control loop")

    def _pid_control_loop(self):
        """PID control loop running at 50Hz"""
        pid_interval = 1.0 / self.pid_rate  # 20ms for 50Hz

        while self.pid_active:
            try:
                start_time = time.time()

                # Get current motor RPMs
                rpms = self.get_motor_rpm()
                current_left = rpms['left']
                current_right = rpms['right']

                # Calculate PID outputs
                if abs(self.target_left_rpm) > 5 or abs(self.target_right_rpm) > 5:
                    # Calculate time delta
                    current_time = time.time()
                    dt = current_time - self.last_pid_time
                    self.last_pid_time = current_time

                    # Left motor PID
                    left_error = self.target_left_rpm - current_left
                    self.left_integral += left_error * dt
                    self.left_integral = max(-50, min(50, self.left_integral))  # Anti-windup
                    left_derivative = (left_error - self.left_last_error) / dt if dt > 0 else 0
                    left_pwm = (self.pid_kp * left_error +
                               self.pid_ki * self.left_integral +
                               self.pid_kd * left_derivative)
                    self.left_last_error = left_error

                    # Right motor PID
                    right_error = self.target_right_rpm - current_right
                    self.right_integral += right_error * dt
                    self.right_integral = max(-50, min(50, self.right_integral))  # Anti-windup
                    right_derivative = (right_error - self.right_last_error) / dt if dt > 0 else 0
                    right_pwm = (self.pid_kp * right_error +
                                self.pid_ki * self.right_integral +
                                self.pid_kd * right_derivative)
                    self.right_last_error = right_error

                    # Clamp PWM to safe range
                    left_pwm = max(self.MIN_SAFE_PWM, min(self.MAX_SAFE_PWM, abs(left_pwm)))
                    right_pwm = max(self.MIN_SAFE_PWM, min(self.MAX_SAFE_PWM, abs(right_pwm)))

                    # Apply PWM with proper direction
                    left_dir = 'forward' if self.target_left_rpm >= 0 else 'backward'
                    right_dir = 'forward' if self.target_right_rpm >= 0 else 'backward'

                    # Use existing motor control with higher PWM limits
                    self.set_motor_speed('left', int(left_pwm), left_dir)
                    self.set_motor_speed('right', int(right_pwm), right_dir)

                    # Debug output every few cycles
                    debug_counter = getattr(self, '_debug_counter', 0) + 1
                    if debug_counter % 25 == 0:  # Every 0.5 seconds
                        logger.info(f"PID: Target L={self.target_left_rpm:.1f}R={self.target_right_rpm:.1f} | "
                                  f"Actual L={current_left:.1f}R={current_right:.1f} | "
                                  f"PWM L={left_pwm:.1f}R={right_pwm:.1f}")
                    self._debug_counter = debug_counter

                else:
                    # Stop motors when target is near zero
                    self.set_motor_speed('left', 0, 'stop')
                    self.set_motor_speed('right', 0, 'stop')

                # Maintain precise timing
                elapsed = time.time() - start_time
                sleep_time = max(0, pid_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"PID control loop error: {e}")
                time.sleep(pid_interval)

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

                # Calculate RPM periodically (every 100ms)
                if start_time - self.last_rpm_calc_time >= self.rpm_calc_interval:
                    self._calculate_rpm()
                    self.last_rpm_calc_time = start_time

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

    def _calculate_rpm(self):
        """Calculate RPM for both motors using moving average"""
        current_time = time.time()

        with self.encoder_lock:
            for motor_name, encoder in self.encoders.items():
                # Calculate ticks since last RPM calculation
                ticks_delta = encoder.count - encoder.last_rpm_count
                time_delta = current_time - encoder.last_rpm_calc_time

                if time_delta > 0:
                    # Calculate instantaneous RPM
                    revolutions = ticks_delta / encoder.ticks_per_revolution
                    minutes = time_delta / 60.0
                    instant_rpm = abs(revolutions / minutes) if minutes > 0 else 0.0

                    # Add to moving average window
                    encoder.rpm_history.append(instant_rpm)
                    if len(encoder.rpm_history) > encoder.rpm_window_size:
                        encoder.rpm_history.pop(0)

                    # Calculate smooth RPM using moving average
                    encoder.current_rpm = sum(encoder.rpm_history) / len(encoder.rpm_history)

                    # Update for next calculation
                    encoder.last_rpm_count = encoder.count
                    encoder.last_rpm_calc_time = current_time

    def _run_gpio_command(self, pin: int, value: int) -> bool:
        """Execute gpioset command safely"""
        try:
            cmd = ['gpioset', 'gpiochip0', f'{pin}={value}']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=0.1)
            if result.returncode != 0:
                logger.warning(f"GPIO{pin}={value} failed: {result.stderr.strip()}")
                return False
            return True
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

    def _set_motor_pwm_direct(self, motor: str, pwm_percent: float, direction: str):
        """Set motor PWM directly (used by PID controller)"""
        # Convert percentage to 0.0-1.0 range
        pwm_value = max(0.0, min(1.0, pwm_percent / 100.0))

        if motor == 'left':
            if direction == 'stop' or pwm_percent == 0:
                self.left_in1.off()
                self.left_in2.off()
                self.left_ena.value = 0
            elif direction == 'forward':
                self.left_in1.off()  # IN1=low
                self.left_in2.on()   # IN2=high
                self.left_ena.value = pwm_value
            elif direction == 'backward':
                self.left_in1.on()   # IN1=high
                self.left_in2.off()  # IN2=low
                self.left_ena.value = pwm_value

        elif motor == 'right':
            if direction == 'stop' or pwm_percent == 0:
                self.right_in3.off()
                self.right_in4.off()
                self.right_enb.value = 0
            elif direction == 'forward':
                self.right_in3.on()   # IN3=high
                self.right_in4.off()  # IN4=low
                self.right_enb.value = pwm_value
            elif direction == 'backward':
                self.right_in3.off()  # IN3=low
                self.right_in4.on()   # IN4=high
                self.right_enb.value = pwm_value

    def set_motor_rpm(self, left_rpm: float, right_rpm: float):
        """Set target motor RPMs (for PID control)"""
        self.target_left_rpm = left_rpm
        self.target_right_rpm = right_rpm
        logger.debug(f"Motor RPM targets: Left={left_rpm:.1f}, Right={right_rpm:.1f}")

    def set_motor_speed(self, motor: str, speed: int, direction: str) -> bool:
        """Set individual motor speed and direction using gpiozero with corrected directions"""
        with self.motor_lock:
            try:
                # Apply safety limits
                speed = max(0, min(100, speed))

                # Convert to PWM duty cycle (0.0 to 1.0)
                if speed > 0:
                    pwm_value = (self.MIN_SAFE_PWM +
                                (speed * (self.MAX_SAFE_PWM - self.MIN_SAFE_PWM) / 100)) / 100.0
                else:
                    pwm_value = 0.0

                pwm_value = max(0.0, min(self.MAX_SAFE_PWM / 100.0, pwm_value))

                # Apply right motor boost for power balance
                if motor in ['B', 'right']:
                    pwm_value = min(1.0, pwm_value * self.RIGHT_MOTOR_BOOST)

                if motor in ['A', 'left']:
                    motor_name = "Left"
                    # LEFT MOTOR DIRECTIONS (from test results):
                    # forward: IN1=low, IN2=high
                    # backward: IN1=high, IN2=low
                    if direction == 'stop' or speed == 0:
                        self.left_in1.off()
                        self.left_in2.off()
                        self.left_ena.value = 0
                        self.left_speed = 0
                    elif direction == 'forward':
                        self.left_in1.off()  # IN1=low
                        self.left_in2.on()   # IN2=high
                        self.left_ena.value = pwm_value
                        self.left_speed = speed
                    elif direction == 'backward':
                        self.left_in1.on()   # IN1=high
                        self.left_in2.off()  # IN2=low
                        self.left_ena.value = pwm_value
                        self.left_speed = -speed

                elif motor in ['B', 'right']:
                    motor_name = "Right"
                    # RIGHT MOTOR DIRECTIONS (from test results):
                    # forward: IN3=high, IN4=low
                    # backward: IN3=low, IN4=high
                    if direction == 'stop' or speed == 0:
                        self.right_in3.off()
                        self.right_in4.off()
                        self.right_enb.value = 0
                        self.right_speed = 0
                    elif direction == 'forward':
                        self.right_in3.on()   # IN3=high
                        self.right_in4.off()  # IN4=low
                        self.right_enb.value = pwm_value
                        self.right_speed = speed
                    elif direction == 'backward':
                        self.right_in3.off()  # IN3=low
                        self.right_in4.on()   # IN4=high
                        self.right_enb.value = pwm_value
                        self.right_speed = -speed
                else:
                    logger.error(f"Invalid motor: {motor}")
                    return False

                self.last_command_time = time.time()
                effective_voltage = 12.6 * pwm_value
                logger.debug(f"Motor {motor_name}: {direction} at {speed}% "
                           f"(PWM: {pwm_value:.2f}, ~{effective_voltage:.1f}V)")
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
        """Emergency stop all motors using gpiozero"""
        with self.motor_lock:
            logger.info("Emergency stop - all motors halted")

            try:
                # Stop all motor direction and enable pins using gpiozero
                self.left_in1.off()
                self.left_in2.off()
                self.left_ena.value = 0

                self.right_in3.off()
                self.right_in4.off()
                self.right_enb.value = 0

                self.left_speed = 0
                self.right_speed = 0
                self.is_moving = False

                # Reset PID targets
                self.target_left_rpm = 0.0
                self.target_right_rpm = 0.0

                logger.info("All motors stopped via gpiozero")
            except Exception as e:
                logger.error(f"Emergency stop error: {e}")

    def get_encoder_status(self) -> Dict:
        """Get current encoder readings"""
        with self.encoder_lock:
            return {
                'left': {
                    'count': self.encoders['left'].count,
                    'changes': self.encoders['left'].changes,
                    'last_change': time.time() - self.encoders['left'].last_change_time,
                    'pin_a': self.encoders['left'].last_a,
                    'pin_b': self.encoders['left'].last_b,
                    'rpm': self.encoders['left'].current_rpm,
                    'rpm_samples': len(self.encoders['left'].rpm_history)
                },
                'right': {
                    'count': self.encoders['right'].count,
                    'changes': self.encoders['right'].changes,
                    'last_change': time.time() - self.encoders['right'].last_change_time,
                    'pin_a': self.encoders['right'].last_a,
                    'pin_b': self.encoders['right'].last_b,
                    'rpm': self.encoders['right'].current_rpm,
                    'rpm_samples': len(self.encoders['right'].rpm_history)
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

    def get_motor_rpm(self) -> Dict[str, float]:
        """Get current RPM readings for both motors"""
        with self.encoder_lock:
            return {
                'left': self.encoders['left'].current_rpm,
                'right': self.encoders['right'].current_rpm
            }

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

    def set_control_mode(self, mode: str):
        """Switch between 'direct' PWM and 'pid' closed-loop control"""
        if mode not in ['direct', 'pid']:
            logger.error(f"Invalid control mode: {mode}")
            return False

        old_mode = self.control_mode
        self.control_mode = mode

        if mode == 'pid' and old_mode == 'direct':
            # Starting PID control
            self._start_pid_control_loop()
            logger.info("Switched to PID closed-loop control")
        elif mode == 'direct' and old_mode == 'pid':
            # Stopping PID control
            self.pid_active = False
            if self.pid_thread:
                self.pid_thread.join(timeout=1.0)
            logger.info("Switched to direct PWM control")

        return True

    def get_control_status(self) -> Dict:
        """Get current control system status"""
        return {
            'control_mode': self.control_mode,
            'pid_active': self.pid_active,
            'target_rpm': {
                'left': self.target_left_rpm,
                'right': self.target_right_rpm
            },
            'current_rpm': self.get_motor_rpm(),
            'pid_stats': self.pid_controller.get_stats() if self.control_mode == 'pid' else None,
            'calibration': self.calibration.get_calibration_status()
        }

    def cleanup(self):
        """Cleanup - stop polling and motors, close gpiozero devices"""
        logger.info("Motor controller cleanup starting")

        # Stop PID control loop
        self.pid_active = False
        if self.pid_thread and self.pid_thread.is_alive():
            self.pid_thread.join(timeout=2.0)
            if self.pid_thread.is_alive():
                logger.error("PID control thread did not stop cleanly")

        # Stop encoder polling
        self.polling_active = False
        if self.polling_thread and self.polling_thread.is_alive():
            self.polling_thread.join(timeout=2.0)
            if self.polling_thread.is_alive():
                logger.error("Polling thread did not stop cleanly")

        # Stop calibration if running
        if self.calibration.is_calibrating:
            self.calibration.stop_calibration()

        # Emergency stop motors
        self.emergency_stop()

        # Close gpiozero devices
        try:
            self.left_in1.close()
            self.left_in2.close()
            self.left_ena.close()
            self.right_in3.close()
            self.right_in4.close()
            self.right_enb.close()
            logger.info("GPIO devices closed")
        except Exception as e:
            logger.error(f"Error closing GPIO devices: {e}")

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