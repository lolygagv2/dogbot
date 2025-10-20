#!/usr/bin/env python3
"""
core/motor_controller.py - Tank motor control system
Refactored from proven test_motors500.py with all working optimizations
"""

import lgpio
import time
import threading
from enum import Enum

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.pins import TreatBotPins
from config.settings import SystemSettings

class MotorDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"

class MotorController:
    """Tank-style motor controller with proven 500Hz PWM and audio optimizations"""
    
    def __init__(self):
        self.pins = TreatBotPins()
        self.settings = SystemSettings()
        
        # Motor state tracking
        self.left_speed = 0
        self.right_speed = 0
        self.is_moving = False
        self._current_pwm_freq = self.settings.PWM_FREQUENCY
        
        # GPIO initialization
        self.gpio_chip = None
        self._initialize_motors()
    
    def _initialize_motors(self):
        """Initialize motor GPIO pins using proven lgpio method"""
        try:
            self.gpio_chip = lgpio.gpiochip_open(0)
            
            # Claim all motor control pins
            motor_pins = [
                self.pins.MOTOR_IN1, self.pins.MOTOR_IN2, 
                self.pins.MOTOR_IN3, self.pins.MOTOR_IN4,
                self.pins.MOTOR_ENA, self.pins.MOTOR_ENB
            ]
            
            for pin in motor_pins:
                lgpio.gpio_claim_output(self.gpio_chip, pin, lgpio.SET_PULL_NONE)
                lgpio.gpio_write(self.gpio_chip, pin, 0)  # Start LOW
            
            print(f"Motor controller initialized with {self._current_pwm_freq}Hz PWM")
            print(f"Motor A (Left):  IN1=GPIO{self.pins.MOTOR_IN1}, IN2=GPIO{self.pins.MOTOR_IN2}, ENA=GPIO{self.pins.MOTOR_ENA}")
            print(f"Motor B (Right): IN3=GPIO{self.pins.MOTOR_IN3}, IN4=GPIO{self.pins.MOTOR_IN4}, ENB=GPIO{self.pins.MOTOR_ENB}")
            
        except Exception as e:
            print(f"Motor controller initialization failed: {e}")
            self.gpio_chip = None
    
    def set_motor_speed(self, motor, speed, direction):
        """Control individual motor with corrected wiring from proven tests"""
        if not self.gpio_chip:
            print("Motor controller not initialized")
            return False
            
        speed = max(0, min(100, speed))
        
        try:
            if motor in ['A', 'left']:
                # Motor A (Left side) - REVERSED WIRING CORRECTION (from working tests)
                in1, in2, ena = self.pins.MOTOR_IN1, self.pins.MOTOR_IN2, self.pins.MOTOR_ENA
                motor_name = "A (Left)"
                # Apply proven wiring correction
                if direction == 'forward':
                    direction = 'backward'
                elif direction == 'backward':
                    direction = 'forward'
            elif motor in ['B', 'right']:
                # Motor B (Right side) - Normal wiring
                in1, in2, ena = self.pins.MOTOR_IN3, self.pins.MOTOR_IN4, self.pins.MOTOR_ENB
                motor_name = "B (Right)"
            else:
                print(f"Invalid motor: {motor}")
                return False
            
            # Execute motor control using proven method
            if direction == 'stop' or speed == 0:
                # Stop motor
                lgpio.gpio_write(self.gpio_chip, in1, 0)
                lgpio.gpio_write(self.gpio_chip, in2, 0)
                lgpio.tx_pwm(self.gpio_chip, ena, 0, 0)  # Stop PWM
                if motor in ['A', 'left']:
                    self.left_speed = 0
                else:
                    self.right_speed = 0
                    
            elif direction == 'forward':
                # Forward direction
                lgpio.gpio_write(self.gpio_chip, in1, 1)
                lgpio.gpio_write(self.gpio_chip, in2, 0)
                lgpio.tx_pwm(self.gpio_chip, ena, self._current_pwm_freq, speed)
                if motor in ['A', 'left']:
                    self.left_speed = speed
                else:
                    self.right_speed = speed
                    
            elif direction == 'backward':
                # Backward direction
                lgpio.gpio_write(self.gpio_chip, in1, 0)
                lgpio.gpio_write(self.gpio_chip, in2, 1)
                lgpio.tx_pwm(self.gpio_chip, ena, self._current_pwm_freq, speed)
                if motor in ['A', 'left']:
                    self.left_speed = -speed
                else:
                    self.right_speed = -speed
            
            # Debug output (can be removed in production)
            correction_note = "(corrected)" if motor in ['A', 'left'] else ""
            print(f"Motor {motor_name}: {direction} at {speed}% {correction_note}")
            return True
            
        except Exception as e:
            print(f"Motor control error: {e}")
            return False

    def tank_steering(self, direction: MotorDirection, speed=None, duration=None, audio_mode="normal"):
        """Execute tank-style movement with proven steering logic"""
        if speed is None:
            speed = self.settings.DEFAULT_MOTOR_SPEED
        
        print(f"Tank steering: {direction.value} at {speed}% (audio: {audio_mode})")
        
        # Handle PWM frequency based on audio requirements (from working tests)
        if audio_mode == "reduce_interference":
            self._current_pwm_freq = 500  # Proven frequency for reduced interference
        else:
            self._current_pwm_freq = self.settings.PWM_FREQUENCY  # Default optimized frequency
        
        # Execute steering using proven logic
        if direction == MotorDirection.FORWARD:
            self.set_motor_speed('A', speed, 'forward')
            self.set_motor_speed('B', speed, 'forward')
            
        elif direction == MotorDirection.BACKWARD:
            self.set_motor_speed('A', speed, 'backward')
            self.set_motor_speed('B', speed, 'backward')
            
        elif direction == MotorDirection.LEFT:
            # Turn left: left motor backward, right motor forward (proven method)
            self.set_motor_speed('A', speed, 'backward')
            self.set_motor_speed('B', speed, 'forward')
            
        elif direction == MotorDirection.RIGHT:
            # Turn right: left motor forward, right motor backward (proven method)
            self.set_motor_speed('A', speed, 'forward')
            self.set_motor_speed('B', speed, 'backward')
            
        elif direction == MotorDirection.STOP:
            self.set_motor_speed('A', 0, 'stop')
            self.set_motor_speed('B', 0, 'stop')
        
        # Update movement state
        self.is_moving = (direction != MotorDirection.STOP)
        
        # Auto-stop after duration (safety feature)
        if duration and direction != MotorDirection.STOP:
            def auto_stop():
                self.emergency_stop()
            threading.Timer(duration, auto_stop).start()
    
    def emergency_stop(self):
        """Emergency stop - immediately halt all motors"""
        print("EMERGENCY STOP - All motors halted")
        self.tank_steering(MotorDirection.STOP)
    
    def get_status(self):
        """Get current motor status"""
        return {
            'left_speed': self.left_speed,
            'right_speed': self.right_speed,
            'is_moving': self.is_moving,
            'pwm_frequency': self._current_pwm_freq,
            'initialized': self.gpio_chip is not None
        }
    
    def is_initialized(self):
        """Check if motor controller is properly initialized"""
        return self.gpio_chip is not None
    
    def cleanup(self):
        """Clean shutdown of motor system"""
        print("Cleaning up motor controller...")
        
        if self.gpio_chip:
            try:
                # Emergency stop
                self.emergency_stop()
                time.sleep(0.1)
                
                # Stop all PWM
                lgpio.tx_pwm(self.gpio_chip, self.pins.MOTOR_ENA, 0, 0)
                lgpio.tx_pwm(self.gpio_chip, self.pins.MOTOR_ENB, 0, 0)
                
                # Close GPIO chip
                lgpio.gpiochip_close(self.gpio_chip)
                self.gpio_chip = None
                print("Motor controller cleanup complete")
                
            except Exception as e:
                print(f"Motor cleanup error: {e}")

# Test function for individual module testing
def test_motors():
    """Simple test function for motor controller"""
    print("Testing Motor Controller...")
    
    motors = MotorController()
    if not motors.is_initialized():
        print("Motor initialization failed!")
        return
    
    try:
        # Test sequence
        print("Forward test...")
        motors.tank_steering(MotorDirection.FORWARD, 30, 2)
        time.sleep(3)
        
        print("Turn test...")
        motors.tank_steering(MotorDirection.LEFT, 30, 1)
        time.sleep(2)
        
        motors.tank_steering(MotorDirection.RIGHT, 30, 1)
        time.sleep(2)
        
        print("Stop test...")
        motors.emergency_stop()
        
        print("Motor test complete!")
        
    except KeyboardInterrupt:
        print("Test interrupted")
    finally:
        motors.cleanup()

if __name__ == "__main__":
    test_motors()
