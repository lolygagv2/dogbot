#!/usr/bin/env python3
"""
Servo Control Module for DogBot
Handles PCA9685 servo control with live adjustments
"""

import time
import logging
from typing import Optional, Dict, Tuple
from adafruit_servokit import ServoKit
from adafruit_pca9685 import PCA9685
import board
import busio
import numpy as np

logger = logging.getLogger(__name__)

class ServoController:
    """Control servos via PCA9685 PWM driver"""
    
    def __init__(self):
        """Initialize servo controller"""
        # PCA9685 channels - ONLY 3 SERVOS TOTAL
        self.CHANNEL_PAN = 0      # Pan servo (left/right)
        self.CHANNEL_TILT = 1     # Tilt servo (up/down)
        self.CHANNEL_CAROUSEL = 2 # Treat carousel rotation (continuous servo)
        
        # Servo configurations (adjustable)
        self.servo_config = {
            'pan': {
                'channel': self.CHANNEL_PAN,
                'min_angle': -90,
                'max_angle': 90,
                'center': 0,
                'min_pulse': 500,
                'max_pulse': 2500,
                'current_angle': 0,
                'speed': 50  # degrees per second
            },
            'tilt': {
                'channel': self.CHANNEL_TILT,
                'min_angle': -45,
                'max_angle': 45,
                'center': 0,
                'min_pulse': 500,
                'max_pulse': 2500,
                'current_angle': 0,
                'speed': 50
            },
            'carousel': {
                'channel': self.CHANNEL_CAROUSEL,
                'min_angle': 0,
                'max_angle': 360,
                'center': 0,
                'min_pulse': 500,
                'max_pulse': 2500,
                'current_angle': 0,
                'positions': 6,  # 6 treat compartments
                'speed': 30
            }
        }
        
        self.kit = None
        self.smooth_move_enabled = True
        self.calibration_mode = False
        
    def initialize(self):
        """Initialize PCA9685 and servos"""
        try:
            # Initialize I2C and PCA9685
            i2c = busio.I2C(board.SCL, board.SDA)
            
            # Initialize ServoKit with 16 channels
            self.kit = ServoKit(channels=16, i2c=i2c, address=0x40)
            
            # Configure each servo
            for servo_name, config in self.servo_config.items():
                channel = config['channel']
                self.kit.servo[channel].set_pulse_width_range(
                    config['min_pulse'],
                    config['max_pulse']
                )
                
            # Center all servos
            self.center_all()
            
            logger.info("Servo controller initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize servos: {e}")
            return False
            
    def set_servo_angle(self, servo_name: str, angle: float, smooth: bool = True):
        """
        Set servo to specific angle

        Args:
            servo_name: Name of servo ('pan', 'tilt', 'carousel')
            angle: Target angle in degrees
            smooth: Use smooth movement
        """
        if servo_name not in self.servo_config:
            logger.error(f"Unknown servo: {servo_name}")
            return

        # CRITICAL: Carousel is a continuous rotation servo - do NOT use set_servo_angle!
        if servo_name == 'carousel':
            logger.warning(f"Carousel is continuous rotation servo - use set_carousel_speed() instead!")
            return

        config = self.servo_config[servo_name]

        # Clamp angle to limits
        angle = np.clip(angle, config['min_angle'], config['max_angle'])

        try:
            if smooth and self.smooth_move_enabled:
                self._smooth_move(servo_name, angle)
            else:
                self.kit.servo[config['channel']].angle = self._map_angle(servo_name, angle)
                config['current_angle'] = angle

            logger.debug(f"Set {servo_name} to {angle}°")

        except Exception as e:
            logger.error(f"Failed to set {servo_name} angle: {e}")

    def set_carousel_speed(self, speed: float):
        """
        Control carousel continuous rotation servo

        Args:
            speed: -1.0 to +1.0 (negative = reverse, 0 = stop, positive = forward)
        """
        speed = np.clip(speed, -1.0, 1.0)
        try:
            self.kit.continuous_servo[self.CHANNEL_CAROUSEL].throttle = speed
            self._carousel_speed = speed  # Track current speed for safe stopping
            logger.debug(f"Set carousel speed to {speed}")
        except Exception as e:
            logger.error(f"Failed to set carousel speed: {e}")

    def stop_carousel(self, gradual: bool = True):
        """
        Stop carousel servo safely

        Args:
            gradual: If True, gradually reduce speed to prevent screeching
        """
        try:
            if gradual:
                # Gradually reduce speed to prevent mechanical stress
                current_speed = getattr(self, '_carousel_speed', 0.0)
                steps = 5
                for i in range(steps):
                    speed = current_speed * (1 - (i + 1) / steps)
                    self.kit.continuous_servo[self.CHANNEL_CAROUSEL].throttle = speed
                    time.sleep(0.1)  # 100ms delay between steps
            else:
                # Immediate stop (may cause screech)
                self.kit.continuous_servo[self.CHANNEL_CAROUSEL].throttle = 0.0

            self._carousel_speed = 0.0
            logger.debug("Carousel stopped safely")

        except Exception as e:
            logger.error(f"Failed to stop carousel: {e}")

    def emergency_stop_all(self):
        """Emergency stop all servos including carousel"""
        try:
            # Stop carousel immediately
            self.kit.continuous_servo[self.CHANNEL_CAROUSEL].throttle = 0.0
            # Disable all PWM outputs to prevent any movement
            for channel in range(4):  # 0-3 for our 4 servos
                self.kit._pca.channels[channel].duty_cycle = 0
            logger.warning("Emergency stop activated - all servos disabled")
        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")
            
    def _map_angle(self, servo_name: str, angle: float) -> float:
        """Map logical angle to servo angle (0-180)"""
        config = self.servo_config[servo_name]
        
        # Map from logical range to 0-180 servo range
        logical_range = config['max_angle'] - config['min_angle']
        servo_angle = ((angle - config['min_angle']) / logical_range) * 180
        
        return servo_angle
        
    def _smooth_move(self, servo_name: str, target_angle: float):
        """Smoothly move servo to target angle"""
        config = self.servo_config[servo_name]
        current = config['current_angle']
        
        if abs(target_angle - current) < 1:
            return
            
        # Calculate step size based on speed
        steps = int(abs(target_angle - current))
        step_delay = 1.0 / config['speed']  # Time per degree
        
        # Generate interpolated positions
        positions = np.linspace(current, target_angle, steps)
        
        for pos in positions:
            self.kit.servo[config['channel']].angle = self._map_angle(servo_name, pos)
            time.sleep(step_delay)
            
        config['current_angle'] = target_angle
        
    def set_pan_angle(self, angle: float, smooth: bool = True):
        """Set pan servo angle"""
        self.set_servo_angle('pan', angle, smooth)
        
    def set_tilt_angle(self, angle: float, smooth: bool = True):
        """Set tilt servo angle"""
        self.set_servo_angle('tilt', angle, smooth)
        
    def get_pan_angle(self) -> float:
        """Get current pan angle"""
        return self.servo_config['pan']['current_angle']
        
    def get_tilt_angle(self) -> float:
        """Get current tilt angle"""
        return self.servo_config['tilt']['current_angle']
        
    def center_all(self):
        """Center all servos to default positions (except continuous rotation servos)"""
        for servo_name, config in self.servo_config.items():
            # Skip carousel - it's a continuous rotation servo
            if servo_name == 'carousel':
                self.stop_carousel()  # Just stop it
                continue
            self.set_servo_angle(servo_name, config['center'], smooth=False)
            
    def rotate_carousel(self, steps: int = 1):
        """
        Rotate treat carousel by number of steps

        Args:
            steps: Number of compartments to rotate (1 step = 1 treat dispensed)
        """
        try:
            config = self.servo_config['carousel']

            # Calculate rotation time based on steps
            # Each step = 60° (360° / 6 positions)
            # CALIBRATED VALUES: pulse=1700μs, duration=80ms per treat
            # Based on: winch.duty_cycle = pulse_to_duty(1700)
            rotation_duration = 0.08 * steps  # 80ms per step (CALIBRATED)
            rotation_speed = 0.5  # Equivalent to 1700μs pulse (CALIBRATED)

            print(f"🔄 Rotating carousel {steps} step(s)...")

            # Start rotation
            self.set_carousel_speed(rotation_speed)
            time.sleep(rotation_duration)

            # Stop rotation gradually
            self.stop_carousel(gradual=True)

            # Update position tracking
            degrees_per_step = 360 / config['positions']
            config['current_angle'] = (config['current_angle'] + degrees_per_step * steps) % 360

            print(f"✅ Carousel rotated to position {config['current_angle']:.1f}°")
            return True

        except Exception as e:
            logger.error(f"Carousel rotation failed: {e}")
            return False
        
    def calibrate_servo(self, servo_name: str):
        """
        Interactive servo calibration
        
        Args:
            servo_name: Servo to calibrate
        """
        if servo_name not in self.servo_config:
            print(f"Unknown servo: {servo_name}")
            return
            
        config = self.servo_config[servo_name]
        print(f"\nCalibrating {servo_name} servo")
        print(f"Current settings:")
        print(f"  Min angle: {config['min_angle']}°")
        print(f"  Max angle: {config['max_angle']}°")
        print(f"  Min pulse: {config['min_pulse']}μs")
        print(f"  Max pulse: {config['max_pulse']}μs")
        
        self.calibration_mode = True
        
        while self.calibration_mode:
            print("\nCommands:")
            print("  a [angle] - Set angle (e.g., 'a 45')")
            print("  mp [value] - Set min pulse (e.g., 'mp 500')")
            print("  Mp [value] - Set max pulse (e.g., 'Mp 2500')")
            print("  ma [angle] - Set min angle")
            print("  Ma [angle] - Set max angle")
            print("  c - Center servo")
            print("  s - Save and exit")
            print("  q - Quit without saving")
            
            cmd = input("> ").strip().split()
            
            if not cmd:
                continue
                
            if cmd[0] == 'a' and len(cmd) == 2:
                try:
                    angle = float(cmd[1])
                    self.set_servo_angle(servo_name, angle, smooth=False)
                    print(f"Set to {angle}°")
                except ValueError:
                    print("Invalid angle")
                    
            elif cmd[0] == 'mp' and len(cmd) == 2:
                try:
                    config['min_pulse'] = int(cmd[1])
                    self._update_pulse_range(servo_name)
                    print(f"Min pulse set to {config['min_pulse']}μs")
                except ValueError:
                    print("Invalid pulse value")
                    
            elif cmd[0] == 'Mp' and len(cmd) == 2:
                try:
                    config['max_pulse'] = int(cmd[1])
                    self._update_pulse_range(servo_name)
                    print(f"Max pulse set to {config['max_pulse']}μs")
                except ValueError:
                    print("Invalid pulse value")
                    
            elif cmd[0] == 'ma' and len(cmd) == 2:
                try:
                    config['min_angle'] = float(cmd[1])
                    print(f"Min angle set to {config['min_angle']}°")
                except ValueError:
                    print("Invalid angle")
                    
            elif cmd[0] == 'Ma' and len(cmd) == 2:
                try:
                    config['max_angle'] = float(cmd[1])
                    print(f"Max angle set to {config['max_angle']}°")
                except ValueError:
                    print("Invalid angle")
                    
            elif cmd[0] == 'c':
                self.set_servo_angle(servo_name, config['center'], smooth=False)
                print(f"Centered at {config['center']}°")
                
            elif cmd[0] == 's':
                self._save_calibration(servo_name)
                self.calibration_mode = False
                print("Calibration saved")
                
            elif cmd[0] == 'q':
                self.calibration_mode = False
                print("Calibration cancelled")
                
    def _update_pulse_range(self, servo_name: str):
        """Update servo pulse range"""
        config = self.servo_config[servo_name]
        self.kit.servo[config['channel']].set_pulse_width_range(
            config['min_pulse'],
            config['max_pulse']
        )
        
    def _save_calibration(self, servo_name: str):
        """Save calibration to config file"""
        import yaml
        config_file = '/home/morgan/dogbot/config/servo_calibration.yaml'
        
        try:
            # Load existing config or create new
            try:
                with open(config_file, 'r') as f:
                    all_config = yaml.safe_load(f) or {}
            except FileNotFoundError:
                all_config = {}
                
            # Update with current servo config
            all_config[servo_name] = self.servo_config[servo_name]
            
            # Save to file
            with open(config_file, 'w') as f:
                yaml.dump(all_config, f, default_flow_style=False)
                
            logger.info(f"Saved calibration for {servo_name}")
            
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")
            
    def get_servo_info(self) -> Dict:
        """Get current servo states"""
        info = {}
        for name, config in self.servo_config.items():
            info[name] = {
                'angle': config['current_angle'],
                'min': config['min_angle'],
                'max': config['max_angle']
            }
        return info
        
    def cleanup(self):
        """Clean up servo resources"""
        try:
            # Return servos to safe positions
            self.center_all()
            
            # Deinitialize
            if self.kit:
                for i in range(16):
                    self.kit.servo[i].angle = None  # Release servo
                    
            logger.info("Servo controller cleaned up")
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Test function
def test_servos():
    """Test servo functionality"""
    controller = ServoController()
    
    if controller.initialize():
        print("Testing servos...")
        
        # Test pan
        print("Testing pan...")
        controller.set_pan_angle(-45)
        time.sleep(1)
        controller.set_pan_angle(45)
        time.sleep(1)
        controller.set_pan_angle(0)
        
        # Test tilt
        print("Testing tilt...")
        controller.set_tilt_angle(-30)
        time.sleep(1)
        controller.set_tilt_angle(30)
        time.sleep(1)
        controller.set_tilt_angle(0)
        
        # Test carousel
        print("Testing carousel...")
        for _ in range(6):
            controller.rotate_carousel(1)
            time.sleep(0.5)
            
        # Test treat dispensing
        print("Testing treat dispensing...")
        controller.rotate_carousel(1)  # Dispense one treat
        
        controller.cleanup()
        print("Test complete!")
    else:
        print("Failed to initialize")

if __name__ == "__main__":
    test_servos()