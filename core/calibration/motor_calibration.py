#!/usr/bin/env python3
"""
Motor Calibration System
Characterizes PWM-to-RPM curves for precise motor control
"""

import time
import threading
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class MotorCharacterization:
    """Single motor PWM-to-RPM characterization"""

    def __init__(self, motor_name: str):
        self.motor_name = motor_name
        self.pwm_points = []  # List of (pwm, rpm) tuples
        self.min_pwm = 30.0   # Minimum PWM that produces movement
        self.max_pwm = 80.0   # Maximum safe PWM
        self.rpm_per_pwm = 1.0  # Linear approximation slope
        self.calibrated = False

    def add_data_point(self, pwm: float, rpm: float):
        """Add a calibration data point"""
        self.pwm_points.append((pwm, rpm))
        self._update_linear_model()

    def _update_linear_model(self):
        """Update linear PWM-to-RPM model from data points"""
        if len(self.pwm_points) < 2:
            return

        # Simple linear regression for PWM-to-RPM relationship
        n = len(self.pwm_points)
        sum_pwm = sum(pwm for pwm, rpm in self.pwm_points)
        sum_rpm = sum(rpm for pwm, rpm in self.pwm_points)
        sum_pwm_rpm = sum(pwm * rpm for pwm, rpm in self.pwm_points)
        sum_pwm_sq = sum(pwm * pwm for pwm, rpm in self.pwm_points)

        if n * sum_pwm_sq - sum_pwm * sum_pwm != 0:
            self.rpm_per_pwm = (n * sum_pwm_rpm - sum_pwm * sum_rpm) / (n * sum_pwm_sq - sum_pwm * sum_pwm)

        self.calibrated = len(self.pwm_points) >= 3

    def pwm_for_target_rpm(self, target_rpm: float) -> float:
        """Calculate PWM needed for target RPM"""
        if not self.calibrated or self.rpm_per_pwm <= 0:
            # Fallback to simple proportional control
            return min(self.max_pwm, max(self.min_pwm, target_rpm * 0.5))

        # Use linear model
        required_pwm = target_rpm / self.rpm_per_pwm
        return min(self.max_pwm, max(self.min_pwm, required_pwm))

    def get_calibration_data(self) -> Dict:
        """Get calibration data for storage"""
        return {
            'motor_name': self.motor_name,
            'pwm_points': self.pwm_points,
            'min_pwm': self.min_pwm,
            'max_pwm': self.max_pwm,
            'rpm_per_pwm': self.rpm_per_pwm,
            'calibrated': self.calibrated
        }

    def load_calibration_data(self, data: Dict):
        """Load calibration data from storage"""
        self.pwm_points = data.get('pwm_points', [])
        self.min_pwm = data.get('min_pwm', 30.0)
        self.max_pwm = data.get('max_pwm', 80.0)
        self.rpm_per_pwm = data.get('rpm_per_pwm', 1.0)
        self.calibrated = data.get('calibrated', False)

class MotorCalibrationSystem:
    """Complete motor calibration system for differential drive"""

    def __init__(self, motor_controller, calibration_file: str = "motor_calibration.json"):
        self.motor_controller = motor_controller
        self.calibration_file = Path(calibration_file)

        # Motor characterizations
        self.left_motor = MotorCharacterization("left")
        self.right_motor = MotorCharacterization("right")

        # Calibration state
        self.is_calibrating = False
        self.calibration_thread = None
        self.calibration_lock = threading.Lock()

        # Load existing calibration if available
        self.load_calibration()

    def load_calibration(self) -> bool:
        """Load calibration data from file"""
        try:
            if self.calibration_file.exists():
                with open(self.calibration_file, 'r') as f:
                    data = json.load(f)

                if 'left' in data:
                    self.left_motor.load_calibration_data(data['left'])
                if 'right' in data:
                    self.right_motor.load_calibration_data(data['right'])

                logger.info(f"Motor calibration loaded from {self.calibration_file}")
                return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")

        return False

    def save_calibration(self) -> bool:
        """Save calibration data to file"""
        try:
            with self.calibration_lock:
                data = {
                    'left': self.left_motor.get_calibration_data(),
                    'right': self.right_motor.get_calibration_data(),
                    'timestamp': time.time(),
                    'version': '1.0'
                }

                with open(self.calibration_file, 'w') as f:
                    json.dump(data, f, indent=2)

                logger.info(f"Motor calibration saved to {self.calibration_file}")
                return True
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            return False

    def start_calibration(self, pwm_steps: List[float] = None) -> bool:
        """Start automatic motor calibration process"""
        if self.is_calibrating:
            logger.warning("Calibration already in progress")
            return False

        if pwm_steps is None:
            # Default calibration points
            pwm_steps = [35, 45, 55, 65, 75]

        self.is_calibrating = True
        self.calibration_thread = threading.Thread(
            target=self._calibration_routine,
            args=(pwm_steps,),
            daemon=False
        )
        self.calibration_thread.start()
        logger.info("Motor calibration started")
        return True

    def _calibration_routine(self, pwm_steps: List[float]):
        """Main calibration routine (runs in separate thread)"""
        try:
            logger.info("Starting motor calibration routine")

            # Clear existing calibration data
            self.left_motor.pwm_points.clear()
            self.right_motor.pwm_points.clear()

            # Calibrate each PWM step
            for pwm in pwm_steps:
                if not self.is_calibrating:  # Allow early termination
                    break

                logger.info(f"Calibrating at {pwm}% PWM")

                # Reset encoder counts for measurement
                self.motor_controller.reset_encoder_counts()

                # Run motors at this PWM for measurement period
                self.motor_controller.set_motor_speed('left', pwm, 'forward')
                self.motor_controller.set_motor_speed('right', pwm, 'forward')

                # Wait for motors to reach steady state
                time.sleep(2.0)

                # Measure RPM over stabilization period
                rpm_measurements = []
                measurement_duration = 3.0  # 3 seconds of measurement
                measurement_start = time.time()

                while time.time() - measurement_start < measurement_duration:
                    rpms = self.motor_controller.get_motor_rpm()
                    rpm_measurements.append((rpms['left'], rpms['right']))
                    time.sleep(0.1)  # 10Hz measurement rate

                # Stop motors
                self.motor_controller.emergency_stop()
                time.sleep(1.0)  # Rest between measurements

                # Calculate average RPM for this PWM setting
                if rpm_measurements:
                    avg_left_rpm = sum(left for left, right in rpm_measurements) / len(rpm_measurements)
                    avg_right_rpm = sum(right for left, right in rpm_measurements) / len(rpm_measurements)

                    # Store calibration points
                    self.left_motor.add_data_point(pwm, avg_left_rpm)
                    self.right_motor.add_data_point(pwm, avg_right_rpm)

                    logger.info(f"PWM {pwm}%: Left={avg_left_rpm:.1f} RPM, Right={avg_right_rpm:.1f} RPM")

            # Save calibration results
            self.save_calibration()

            logger.info("Motor calibration completed successfully")

        except Exception as e:
            logger.error(f"Calibration routine failed: {e}")
        finally:
            self.is_calibrating = False
            self.motor_controller.emergency_stop()

    def stop_calibration(self):
        """Stop ongoing calibration"""
        if self.is_calibrating:
            self.is_calibrating = False
            self.motor_controller.emergency_stop()
            if self.calibration_thread:
                self.calibration_thread.join(timeout=5.0)
            logger.info("Motor calibration stopped")

    def get_pwm_for_target_rpm(self, left_rpm: float, right_rpm: float) -> Tuple[float, float]:
        """Calculate PWM values needed for target RPM speeds"""
        left_pwm = self.left_motor.pwm_for_target_rpm(left_rpm)
        right_pwm = self.right_motor.pwm_for_target_rpm(right_rpm)
        return (left_pwm, right_pwm)

    def is_calibrated(self) -> bool:
        """Check if both motors are calibrated"""
        return self.left_motor.calibrated and self.right_motor.calibrated

    def get_calibration_status(self) -> Dict:
        """Get current calibration status"""
        return {
            'calibrated': self.is_calibrated(),
            'is_calibrating': self.is_calibrating,
            'left_motor': {
                'calibrated': self.left_motor.calibrated,
                'data_points': len(self.left_motor.pwm_points),
                'rpm_per_pwm': self.left_motor.rpm_per_pwm
            },
            'right_motor': {
                'calibrated': self.right_motor.calibrated,
                'data_points': len(self.right_motor.pwm_points),
                'rpm_per_pwm': self.right_motor.rpm_per_pwm
            },
            'calibration_file': str(self.calibration_file)
        }

    def quick_calibration_check(self) -> Dict:
        """Quick test to verify calibration accuracy"""
        if not self.is_calibrated():
            return {'status': 'not_calibrated'}

        try:
            # Test at 50% PWM
            test_pwm = 50.0

            self.motor_controller.reset_encoder_counts()
            self.motor_controller.set_motor_speed('left', test_pwm, 'forward')
            self.motor_controller.set_motor_speed('right', test_pwm, 'forward')

            time.sleep(2.0)  # Stabilize

            # Measure actual RPM
            rpms = self.motor_controller.get_motor_rpm()
            actual_left = rpms['left']
            actual_right = rpms['right']

            # Calculate predicted RPM from calibration
            predicted_left = test_pwm * self.left_motor.rpm_per_pwm
            predicted_right = test_pwm * self.right_motor.rpm_per_pwm

            self.motor_controller.emergency_stop()

            return {
                'status': 'tested',
                'test_pwm': test_pwm,
                'left': {
                    'actual_rpm': actual_left,
                    'predicted_rpm': predicted_left,
                    'error_percent': abs(actual_left - predicted_left) / max(predicted_left, 1) * 100
                },
                'right': {
                    'actual_rpm': actual_right,
                    'predicted_rpm': predicted_right,
                    'error_percent': abs(actual_right - predicted_right) / max(predicted_right, 1) * 100
                }
            }

        except Exception as e:
            logger.error(f"Calibration check failed: {e}")
            self.motor_controller.emergency_stop()
            return {'status': 'error', 'message': str(e)}