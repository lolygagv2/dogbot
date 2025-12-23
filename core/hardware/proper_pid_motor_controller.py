#!/usr/bin/env python3
"""
PROPER CLOSED-LOOP PID MOTOR CONTROLLER
Implementation based on PIDref.md reference for TreatBot

DFRobot FIT0521 Motor Specs:
- 6V rated, 210 RPM no-load
- 34:1 gearbox
- 11 PPR encoder = 341 PPR output shaft
- Running on 14V system with PWM limiting

Control Architecture:
Xbox Joystick → Target RPM → PID Controller → PWM → Motors
                                   ↑
                             Encoder Feedback
"""

import lgpio
import time
import threading
import logging
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional
from gpiozero import OutputDevice, PWMOutputDevice

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.pins import TreatBotPins

logger = logging.getLogger(__name__)

@dataclass
class PIDGains:
    """Conservative PID controller gains"""
    kp: float = 0.3    # Proportional - very conservative
    ki: float = 0.02   # Integral - low to prevent windup
    kd: float = 0.005  # Derivative - minimal to reduce noise

@dataclass
class MotorState:
    """Track complete state for one motor"""
    name: str

    # Encoder state
    encoder_count: int = 0
    last_encoder_count: int = 0
    last_a: int = 0
    last_b: int = 0

    # RPM calculation with moving average
    current_rpm: float = 0.0
    rpm_history: deque = None
    last_rpm_time: float = 0.0

    # PID control state
    target_rpm: float = 0.0
    integral: float = 0.0
    last_error: float = 0.0
    last_pid_time: float = 0.0

    # PWM output
    pwm_output: float = 0.0
    direction: str = 'forward'

    # Safety tracking
    stall_counter: int = 0

    def __post_init__(self):
        self.rpm_history = deque(maxlen=10)  # 10-sample moving average
        self.last_rpm_time = time.time()
        self.last_pid_time = time.time()

class ProperPIDMotorController:
    """
    Proper closed-loop PID motor controller with encoder feedback.

    Features:
    - 2000Hz encoder polling for high resolution
    - 50Hz PID control loop
    - Conservative gains to prevent oscillation
    - Anti-windup protection
    - Target ramping for smooth acceleration
    - Safety watchdog and stall detection
    """

    # DFRobot FIT0521 Hardware Constants
    ENCODER_PPR = 341      # 11 PPR * 34:1 gearbox
    MAX_RPM = 210          # No-load RPM at 6V

    # PWM Safety Limits (6V motors on 14V system)
    PWM_MIN = 30           # Minimum to overcome static friction (lower = whining)
    PWM_MAX = 75           # 75% of 14V ≈ 10.5V (normal mode uses 60% of this = ~6.3V)

    # Control Loop Timing
    ENCODER_POLL_RATE = 2000  # Hz - faster than 1190Hz encoder frequency
    PID_UPDATE_RATE = 50      # Hz - standard control loop frequency
    RPM_CALC_INTERVAL = 0.05  # Calculate RPM every 50ms

    def __init__(self):
        self.pins = TreatBotPins()

        # GPIO handles - initialized in start()
        self.gpio_handle = None
        self.left_in1 = None
        self.left_in2 = None
        self.left_ena = None
        self.right_in3 = None
        self.right_in4 = None
        self.right_enb = None
        self._gpio_initialized = False

        # PID gains - very conservative for initial testing
        self.left_gains = PIDGains(kp=0.3, ki=0.02, kd=0.005)
        self.right_gains = PIDGains(kp=0.3, ki=0.02, kd=0.005)

        # Motor states
        self.left = MotorState(name='left')
        self.right = MotorState(name='right')

        # Threading control
        self.running = False
        self.encoder_thread = None
        self.pid_thread = None
        self.lock = threading.Lock()

        # Safety systems - RE-ENABLED to prevent stuck motors
        self.watchdog_timeout = 2.0  # 2 second timeout - stop motors if no commands received
        self.last_command_time = time.time()
        self.emergency_stopped = False

        # Additional safety: track if motors should be running
        self.motors_should_be_stopped = True  # Start in stopped state
        self.last_nonzero_command_time = 0  # Track when we last got a non-zero command

        # Target ramping for smooth acceleration
        self.ramp_rate = 300  # RPM per second max change
        self.ramped_left_target = 0.0
        self.ramped_right_target = 0.0

        logger.info("🎯 Proper PID Motor Controller Initialized (GPIO deferred to start)")
        logger.info(f"   Encoder PPR: {self.ENCODER_PPR}")
        logger.info(f"   PWM range: {self.PWM_MIN}-{self.PWM_MAX}%")
        logger.info(f"   PID gains: Kp={self.left_gains.kp}, Ki={self.left_gains.ki}, Kd={self.left_gains.kd}")

    def start(self) -> bool:
        """Start encoder polling and PID control loops"""
        if self.running:
            logger.warning("Controller already running")
            return True

        # Initialize GPIO hardware if not done yet
        if not self._gpio_initialized:
            if not self._initialize_gpio():
                logger.error("Failed to initialize GPIO - cannot start controller")
                return False

        self.running = True
        self.last_command_time = time.time()

        # Start encoder polling thread (high frequency)
        self.encoder_thread = threading.Thread(target=self._encoder_loop, daemon=False)
        self.encoder_thread.start()

        # Start PID control thread (medium frequency)
        self.pid_thread = threading.Thread(target=self._pid_loop, daemon=False)
        self.pid_thread.start()

        logger.info("✅ PID Motor Controller Started")
        logger.info(f"   Encoder polling: {self.ENCODER_POLL_RATE}Hz")
        logger.info(f"   PID control: {self.PID_UPDATE_RATE}Hz")
        return True

    def _initialize_gpio(self) -> bool:
        """Initialize GPIO handles for motor control and encoders"""
        try:
            logger.info("🔧 Initializing GPIO for motor control...")

            # Open GPIO chip for encoder reading
            self.gpio_handle = lgpio.gpiochip_open(0)

            # Motor direction pins (using gpiozero for compatibility)
            self.left_in1 = OutputDevice(self.pins.MOTOR_IN1)
            self.left_in2 = OutputDevice(self.pins.MOTOR_IN2)
            self.right_in3 = OutputDevice(self.pins.MOTOR_IN3)
            self.right_in4 = OutputDevice(self.pins.MOTOR_IN4)

            # Motor PWM pins (using gpiozero for compatibility)
            self.left_ena = PWMOutputDevice(self.pins.MOTOR_ENA, frequency=1000)
            self.right_enb = PWMOutputDevice(self.pins.MOTOR_ENB, frequency=1000)

            # Ensure motors start stopped
            self.left_in1.off()
            self.left_in2.off()
            self.right_in3.off()
            self.right_in4.off()
            self.left_ena.value = 0
            self.right_enb.value = 0

            # Initialize encoder states
            self._init_encoder_states()

            self._gpio_initialized = True
            logger.info("✅ GPIO initialized successfully")
            return True

        except Exception as e:
            logger.error(f"❌ GPIO initialization failed: {e}")
            # Cleanup any partially initialized resources
            self._cleanup_gpio()
            return False

    def _cleanup_gpio(self):
        """Cleanup GPIO resources"""
        try:
            if self.left_in1:
                self.left_in1.close()
            if self.left_in2:
                self.left_in2.close()
            if self.left_ena:
                self.left_ena.close()
            if self.right_in3:
                self.right_in3.close()
            if self.right_in4:
                self.right_in4.close()
            if self.right_enb:
                self.right_enb.close()
            if self.gpio_handle:
                lgpio.gpiochip_close(self.gpio_handle)
        except Exception as e:
            logger.debug(f"GPIO cleanup error (may be expected): {e}")
        finally:
            self.gpio_handle = None
            self.left_in1 = None
            self.left_in2 = None
            self.left_ena = None
            self.right_in3 = None
            self.right_in4 = None
            self.right_enb = None
            self._gpio_initialized = False

    def stop(self):
        """Stop all motors and control loops"""
        self.running = False

        # Emergency stop motors
        self._emergency_stop()

        # Wait for threads to finish
        if self.encoder_thread:
            self.encoder_thread.join(timeout=2.0)
        if self.pid_thread:
            self.pid_thread.join(timeout=2.0)

        logger.info("🛑 PID Motor Controller Stopped")

    def set_motor_rpm(self, left_rpm: float, right_rpm: float):
        """
        Set target RPM for both motors - MAIN INTERFACE

        This is the method called by Xbox controller and motor command bus
        """
        # Clamp to safe RPM range
        left_rpm = max(-self.MAX_RPM, min(self.MAX_RPM, left_rpm))
        right_rpm = max(-self.MAX_RPM, min(self.MAX_RPM, right_rpm))

        current_time = time.time()

        with self.lock:
            self.left.target_rpm = left_rpm
            self.right.target_rpm = right_rpm
            self.last_command_time = current_time
            self.emergency_stopped = False

            # Track if we're commanding non-zero movement
            if abs(left_rpm) > 1 or abs(right_rpm) > 1:
                self.motors_should_be_stopped = False
                self.last_nonzero_command_time = current_time
            else:
                self.motors_should_be_stopped = True

        logger.debug(f"🎯 Target RPM: L={left_rpm:.1f}, R={right_rpm:.1f}")

    def get_motor_rpm(self) -> dict:
        """Get current actual RPM for both motors"""
        with self.lock:
            return {
                'left': self.left.current_rpm,
                'right': self.right.current_rpm
            }

    def get_encoder_counts(self) -> Tuple[int, int]:
        """Get current encoder counts (for debugging)"""
        with self.lock:
            return self.left.encoder_count, self.right.encoder_count

    def get_status(self) -> dict:
        """Get comprehensive motor status"""
        with self.lock:
            return {
                'targets': {
                    'left_rpm': self.left.target_rpm,
                    'right_rpm': self.right.target_rpm,
                    'ramped_left': self.ramped_left_target,
                    'ramped_right': self.ramped_right_target
                },
                'actual': {
                    'left_rpm': self.left.current_rpm,
                    'right_rpm': self.right.current_rpm
                },
                'pwm': {
                    'left': self.left.pwm_output,
                    'right': self.right.pwm_output
                },
                'encoders': {
                    'left_count': self.left.encoder_count,
                    'right_count': self.right.encoder_count
                },
                'directions': {
                    'left': self.left.direction,
                    'right': self.right.direction
                },
                'safety': {
                    'emergency_stopped': self.emergency_stopped,
                    'time_since_command': time.time() - self.last_command_time
                }
            }

    def _init_encoder_states(self):
        """Initialize encoder pins and read initial states"""
        # Configure encoder pins as inputs with pull-up
        lgpio.gpio_claim_input(self.gpio_handle, self.pins.ENCODER_A1, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.gpio_handle, self.pins.ENCODER_B1, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.gpio_handle, self.pins.ENCODER_A2, lgpio.SET_PULL_UP)
        lgpio.gpio_claim_input(self.gpio_handle, self.pins.ENCODER_B2, lgpio.SET_PULL_UP)

        # Read initial pin states
        time.sleep(0.01)
        self.left.last_a = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_A1)
        self.left.last_b = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_B1)
        self.right.last_a = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_A2)
        self.right.last_b = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_B2)

        logger.info(f"🔧 Encoder pins initialized on GPIO{self.pins.ENCODER_A1}/{self.pins.ENCODER_B1} (left), GPIO{self.pins.ENCODER_A2}/{self.pins.ENCODER_B2} (right)")
        logger.info(f"   Initial states: Left A={self.left.last_a},B={self.left.last_b} | Right A={self.right.last_a},B={self.right.last_b}")

    def _encoder_loop(self):
        """High-frequency encoder polling loop (2000Hz)"""
        poll_interval = 1.0 / self.ENCODER_POLL_RATE
        last_rpm_calc = time.time()

        while self.running:
            start = time.time()

            try:
                # Read all encoder pins
                left_a = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_A1)
                left_b = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_B1)
                right_a = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_A2)
                right_b = lgpio.gpio_read(self.gpio_handle, self.pins.ENCODER_B2)

                with self.lock:
                    # Decode quadrature for both motors
                    self._decode_quadrature(self.left, left_a, left_b)
                    self._decode_quadrature(self.right, right_a, right_b)

                # Calculate RPM periodically
                if start - last_rpm_calc >= self.RPM_CALC_INTERVAL:
                    self._calculate_rpm()
                    last_rpm_calc = start

            except Exception as e:
                logger.error(f"Encoder polling error: {e}")
                # Emergency stop on repeated errors
                if not hasattr(self, '_encoder_error_count'):
                    self._encoder_error_count = 0
                self._encoder_error_count += 1
                if self._encoder_error_count >= 10:
                    logger.critical("Too many encoder errors - emergency stop!")
                    self._emergency_stop()
                    self._encoder_error_count = 0

            # Maintain precise timing
            elapsed = time.time() - start
            sleep_time = poll_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _decode_quadrature(self, motor: MotorState, current_a: int, current_b: int):
        """
        Decode quadrature encoder signals for direction and count

        Standard quadrature decoding:
        - A rising edge + B=0 → forward
        - A rising edge + B=1 → backward
        - A falling edge + B=1 → forward
        - A falling edge + B=0 → backward
        """
        if current_a != motor.last_a or current_b != motor.last_b:
            # Detect state change and decode direction
            if motor.last_a == 0 and current_a == 1:
                # Rising edge on A
                motor.encoder_count += 1 if current_b == 0 else -1
            elif motor.last_a == 1 and current_a == 0:
                # Falling edge on A
                motor.encoder_count += 1 if current_b == 1 else -1

            motor.last_a = current_a
            motor.last_b = current_b

    def _calculate_rpm(self):
        """Calculate RPM from encoder counts using moving average"""
        now = time.time()

        with self.lock:
            for motor in [self.left, self.right]:
                dt = now - motor.last_rpm_time
                if dt > 0:
                    # Calculate instantaneous RPM
                    count_delta = motor.encoder_count - motor.last_encoder_count
                    revolutions = count_delta / self.ENCODER_PPR
                    instant_rpm = (revolutions / dt) * 60.0

                    # Add to moving average buffer
                    motor.rpm_history.append(instant_rpm)

                    # Calculate smoothed RPM
                    if len(motor.rpm_history) > 0:
                        motor.current_rpm = sum(motor.rpm_history) / len(motor.rpm_history)

                    # Update for next calculation
                    motor.last_encoder_count = motor.encoder_count
                    motor.last_rpm_time = now

    def _pid_loop(self):
        """PID control loop (50Hz)"""
        pid_interval = 1.0 / self.PID_UPDATE_RATE
        debug_counter = 0

        while self.running:
            start = time.time()

            try:
                current_time = time.time()

                # Watchdog safety check - no commands at all
                if current_time - self.last_command_time > self.watchdog_timeout:
                    if not self.emergency_stopped:
                        logger.warning("⏰ Watchdog timeout - no commands received, stopping motors")
                        self._emergency_stop()
                    time.sleep(pid_interval)
                    continue

                # Additional safety: If motors are running but last non-zero command was too long ago
                # This catches cases where the controller crashes while motors are moving
                if not self.motors_should_be_stopped:
                    if current_time - self.last_nonzero_command_time > 1.0:
                        # Haven't received a fresh movement command in 1 second while motors running
                        logger.warning("⏰ Stale movement command detected - forcing stop")
                        with self.lock:
                            self.left.target_rpm = 0
                            self.right.target_rpm = 0
                            self.motors_should_be_stopped = True

                with self.lock:
                    # Apply smooth target ramping
                    self._ramp_targets(pid_interval)

                    # Update PID for each motor
                    left_pwm = self._update_pid(self.left, self.left_gains, self.ramped_left_target)
                    right_pwm = self._update_pid(self.right, self.right_gains, self.ramped_right_target)

                # Apply PWM outputs to hardware
                self._apply_pwm(self.left, left_pwm)
                self._apply_pwm(self.right, right_pwm)

                # Debug logging every 1 second (50 cycles at 50Hz)
                debug_counter += 1
                if debug_counter >= 50:
                    logger.info(
                        f"🎯 PID: Target L={self.ramped_left_target:6.1f} R={self.ramped_right_target:6.1f} | "
                        f"Actual L={self.left.current_rpm:6.1f} R={self.right.current_rpm:6.1f} | "
                        f"PWM L={self.left.pwm_output:5.1f} R={self.right.pwm_output:5.1f} | "
                        f"Enc L={self.left.encoder_count} R={self.right.encoder_count}"
                    )
                    debug_counter = 0

            except Exception as e:
                logger.error(f"PID loop error: {e}")
                # Emergency stop on repeated errors
                if not hasattr(self, '_pid_error_count'):
                    self._pid_error_count = 0
                self._pid_error_count += 1
                if self._pid_error_count >= 5:
                    logger.critical("Too many PID errors - emergency stop!")
                    self._emergency_stop()
                    self._pid_error_count = 0

            # Maintain timing
            elapsed = time.time() - start
            sleep_time = pid_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _ramp_targets(self, dt: float):
        """Smooth target ramping to prevent jerky motion"""
        max_change = self.ramp_rate * dt

        # Ramp left target
        diff = self.left.target_rpm - self.ramped_left_target
        if abs(diff) <= max_change:
            self.ramped_left_target = self.left.target_rpm
        else:
            self.ramped_left_target += max_change if diff > 0 else -max_change

        # Ramp right target
        diff = self.right.target_rpm - self.ramped_right_target
        if abs(diff) <= max_change:
            self.ramped_right_target = self.right.target_rpm
        else:
            self.ramped_right_target += max_change if diff > 0 else -max_change

    def _update_pid(self, motor: MotorState, gains: PIDGains, target: float) -> float:
        """
        Update PID controller for one motor

        Returns: PWM value (0-PWM_MAX)
        """
        now = time.time()
        dt = now - motor.last_pid_time
        if dt <= 0:
            return motor.pwm_output
        motor.last_pid_time = now

        # Calculate error
        error = target - motor.current_rpm

        # Proportional term
        p_term = gains.kp * error

        # Integral term with anti-windup
        motor.integral += error * dt
        max_integral = self.PWM_MAX / (gains.ki + 0.001)
        motor.integral = max(-max_integral, min(max_integral, motor.integral))

        # Reset integral on zero crossing (prevent overshoot)
        if (motor.last_error > 0 and error < 0) or (motor.last_error < 0 and error > 0):
            motor.integral *= 0.5

        i_term = gains.ki * motor.integral

        # Derivative term
        d_term = gains.kd * (error - motor.last_error) / dt
        motor.last_error = error

        # Calculate raw PID output
        pid_output = p_term + i_term + d_term

        # Add feedforward for responsiveness
        # At MAX_RPM, need approximately PWM_MAX
        feedforward = (abs(target) / self.MAX_RPM) * self.PWM_MAX * 0.9

        # Total output
        output = pid_output + feedforward

        # Handle direction
        if target >= 0:
            motor.direction = 'forward'
            pwm = output
        else:
            motor.direction = 'backward'
            pwm = -output

        # Apply PWM limits
        if abs(target) < 5:  # Near zero = stop
            pwm = 0
        else:
            pwm = max(self.PWM_MIN, min(self.PWM_MAX, abs(pwm)))

        motor.pwm_output = pwm
        return pwm

    def _apply_pwm(self, motor: MotorState, pwm: float):
        """Apply PWM and direction to motor hardware"""
        # Safety check - don't try to apply PWM if GPIO not initialized
        if not self._gpio_initialized:
            return

        pwm_value = pwm / 100.0  # Convert percentage to 0.0-1.0

        if motor.name == 'left':
            if pwm == 0:
                self.left_in1.off()
                self.left_in2.off()
                self.left_ena.value = 0
                logger.debug(f"🔧 LEFT MOTOR: STOP (IN1=0, IN2=0, PWM=0)")
            elif motor.direction == 'forward':
                self.left_in1.off()   # IN1=0 for forward
                self.left_in2.on()    # IN2=1 for forward
                self.left_ena.value = pwm_value
            elif motor.direction == 'backward':
                self.left_in1.on()    # IN1=1 for backward
                self.left_in2.off()   # IN2=0 for backward
                self.left_ena.value = pwm_value

        elif motor.name == 'right':
            if pwm == 0:
                self.right_in3.off()
                self.right_in4.off()
                self.right_enb.value = 0
                logger.debug(f"🔧 RIGHT MOTOR: STOP (IN3=0, IN4=0, PWM=0)")
            elif motor.direction == 'forward':
                self.right_in3.on()   # IN3=1 for forward
                self.right_in4.off()  # IN4=0 for forward
                self.right_enb.value = pwm_value
            elif motor.direction == 'backward':
                self.right_in3.off()  # IN3=0 for backward
                self.right_in4.on()   # IN4=1 for backward
                self.right_enb.value = pwm_value

    def _emergency_stop(self):
        """Emergency stop all motors"""
        self.emergency_stopped = True

        # Clear targets
        self.left.target_rpm = 0
        self.right.target_rpm = 0
        self.ramped_left_target = 0
        self.ramped_right_target = 0

        # Clear PID state
        self.left.integral = 0
        self.right.integral = 0

        # Stop hardware
        self._apply_pwm(self.left, 0)
        self._apply_pwm(self.right, 0)

        logger.critical("🚨 EMERGENCY STOP - All motors halted")

    def emergency_stop(self):
        """Public emergency stop interface"""
        with self.lock:
            self._emergency_stop()

    def cleanup(self):
        """Cleanup GPIO and stop all operations"""
        logger.info("🧹 Cleaning up PID motor controller")

        # Stop control loops
        self.stop()

        # Emergency stop motors (only if GPIO is initialized)
        if self._gpio_initialized:
            self.emergency_stop()

        # Close GPIO devices
        self._cleanup_gpio()
        logger.info("✅ GPIO devices closed")


# Legacy compatibility interface
class MotorControllerPolling(ProperPIDMotorController):
    """
    Legacy compatibility wrapper for existing TreatBot code

    This maintains the same interface as the old motor controller
    while providing proper PID closed-loop control underneath.
    """

    def initialize(self) -> bool:
        """Initialize for Xbox controller compatibility"""
        return self.start()

    def set_motor_speed(self, motor: str, speed: int, direction: str) -> bool:
        """
        Legacy interface - converts PWM percentage to RPM targets

        This allows existing code to work while getting PID benefits
        """
        # Convert speed percentage to approximate RPM
        # 100% speed ≈ MAX_RPM (PWM_MAX provides the safety limit)
        target_rpm = (speed / 100.0) * self.MAX_RPM

        # Apply direction
        if direction == 'backward':
            target_rpm = -target_rpm
        elif direction == 'stop':
            target_rpm = 0

        # Set appropriate motor target
        if motor in ['A', 'left']:
            current_right = self.right.target_rpm if hasattr(self, 'right') else 0
            self.set_motor_rpm(target_rpm, current_right)
        elif motor in ['B', 'right']:
            current_left = self.left.target_rpm if hasattr(self, 'left') else 0
            self.set_motor_rpm(current_left, target_rpm)

        return True


if __name__ == "__main__":
    # Test the controller
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    print("🎯 Testing Proper PID Motor Controller")
    controller = ProperPIDMotorController()

    try:
        controller.start()

        # Test sequence
        print("Setting 30 RPM target for 5 seconds...")
        controller.set_motor_rpm(30, 30)
        time.sleep(5)

        print("Setting 0 RPM (stop)...")
        controller.set_motor_rpm(0, 0)
        time.sleep(2)

        print("Final status:")
        status = controller.get_status()
        for key, value in status.items():
            print(f"  {key}: {value}")

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        controller.cleanup()
        print("Test complete")