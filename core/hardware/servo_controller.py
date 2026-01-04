#!/usr/bin/env python3
"""
core/servo_controller.py - PCA9685 servo management
Refactored from proven pca9685_test.py with all working servo movements
"""

import time
import board
from adafruit_pca9685 import PCA9685
from core.hardware.i2c_bus import get_i2c_bus

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.pins import TreatBotPins
from config.settings import SystemSettings

class ServoController:
    """PCA9685-based servo control for camera positioning and treat carousel"""
    
    def __init__(self):
        self.pins = TreatBotPins()
        self.settings = SystemSettings()
        
        # Hardware components
        self.i2c = None
        self.pca = None
        self.pitch_channel = None
        self.pan_channel = None
        self.winch_channel = None
        
        # Servo positions tracking
        self.current_pitch = 90  # Degrees
        self.current_pan = 90    # Degrees
        
        self._initialize_pca9685()
    
    def _initialize_pca9685(self):
        """Initialize PCA9685 servo controller using proven configuration"""
        try:
            # Use shared I2C bus for compatibility with other I2C devices
            self.i2c = get_i2c_bus()
            
            # Initialize PCA9685 with reset to clear any bad state
            self.pca = PCA9685(self.i2c)
            self.pca.reset()
            time.sleep(0.1)
            # Ensure MODE2 has correct settings (OUTDRV=1, INVRT=0)
            self.pca.mode2 = 0x04
            self.pca.frequency = self.settings.SERVO_FREQUENCY  # 50Hz standard
            
            # Assign channels (from proven test script)
            self.pitch_channel = self.pca.channels[1]  # Camera pitch
            self.pan_channel = self.pca.channels[0]    # Camera pan  
            self.winch_channel = self.pca.channels[2]  # Continuous rotation winch
            
            print("PCA9685 servo controller initialized")
            print("Channel 0: Camera Pan")
            print("Channel 1: Camera Pitch")
            print("Channel 2: Winch/carousel")
            
            # Initialize to safe positions
            self.release_all_servos()
            return True
            
        except Exception as e:
            print(f"PCA9685 initialization failed: {e}")
            self.pca = None
            return False
    
    def _pulse_to_duty(self, pulse_us):
        """Convert pulse width in microseconds to duty cycle (proven formula)"""
        return int((pulse_us / 20000.0) * 0xFFFF)
    
    def _angle_to_pulse(self, angle):
        """Convert angle to pulse width - extended range for full servo movement"""
        # Map wider angle range to extended pulse range
        pulse_range = self.settings.SERVO_MAX_PULSE - self.settings.SERVO_MIN_PULSE
        # Allow -90 to 270 degree range (360 degree span)
        angle_normalized = max(-90, min(270, angle)) / 360.0  # Normalize to 0-1
        angle_normalized = (angle_normalized + 90.0/360.0)  # Shift to positive range
        return self.settings.SERVO_MIN_PULSE + (pulse_range * angle_normalized)
    
    def set_camera_pitch(self, angle, smooth=False):
        """Set camera pitch angle (0-180 degrees, 90 = level)"""
        if not self.pca:
            print("Servo controller not initialized")
            return False
        
        try:
            angle = max(-90, min(270, angle))  # Extended range
            
            if smooth and abs(angle - self.current_pitch) > 10:
                # Smooth movement for large angle changes
                return self._smooth_move_pitch(angle)
            else:
                # Direct movement
                pulse = self._angle_to_pulse(angle)
                self.pitch_channel.duty_cycle = self._pulse_to_duty(pulse)
                self.current_pitch = angle
                print(f"Camera pitch set to {angle} degrees")
                return True
                
        except Exception as e:
            print(f"Camera pitch error: {e}")
            return False
    
    def set_camera_pan(self, angle, smooth=False):
        """Set camera pan angle (0-180 degrees, 90 = center)"""
        if not self.pca:
            print("Servo controller not initialized")
            return False
            
        try:
            angle = max(-90, min(270, angle))  # Extended range
            
            if smooth and abs(angle - self.current_pan) > 10:
                # Smooth movement for large angle changes
                return self._smooth_move_pan(angle)
            else:
                # Direct movement
                pulse = self._angle_to_pulse(angle)
                self.pan_channel.duty_cycle = self._pulse_to_duty(pulse)
                self.current_pan = angle
                print(f"Camera pan set to {angle} degrees")
                return True
                
        except Exception as e:
            print(f"Camera pan error: {e}")
            return False
    
    def _smooth_move_pitch(self, target_angle):
        """Smooth pitch movement (from proven test script)"""
        if not self.pca:
            return False
            
        try:
            start_angle = self.current_pitch
            step_size = 2 if target_angle > start_angle else -2
            
            print(f"Smooth pitch: {start_angle} -> {target_angle} degrees")
            
            current = start_angle
            while abs(current - target_angle) > abs(step_size):
                current += step_size
                pulse = self._angle_to_pulse(current)
                self.pitch_channel.duty_cycle = self._pulse_to_duty(pulse)
                time.sleep(0.02)  # Proven smooth movement timing
            
            # Final position
            pulse = self._angle_to_pulse(target_angle)
            self.pitch_channel.duty_cycle = self._pulse_to_duty(pulse)
            self.current_pitch = target_angle
            
            print(f"Smooth pitch complete: {target_angle} degrees")
            return True
            
        except Exception as e:
            print(f"Smooth pitch error: {e}")
            return False
    
    def _smooth_move_pan(self, target_angle):
        """Smooth pan movement"""
        if not self.pca:
            return False
            
        try:
            start_angle = self.current_pan
            step_size = 2 if target_angle > start_angle else -2
            
            print(f"Smooth pan: {start_angle} -> {target_angle} degrees")
            
            current = start_angle
            while abs(current - target_angle) > abs(step_size):
                current += step_size
                pulse = self._angle_to_pulse(current)
                self.pan_channel.duty_cycle = self._pulse_to_duty(pulse)
                time.sleep(0.02)
            
            # Final position
            pulse = self._angle_to_pulse(target_angle)
            self.pan_channel.duty_cycle = self._pulse_to_duty(pulse)
            self.current_pan = target_angle
            
            print(f"Smooth pan complete: {target_angle} degrees")
            return True
            
        except Exception as e:
            print(f"Smooth pan error: {e}")
            return False
    
    def rotate_winch(self, direction='forward', duration=0.08):
        """Rotate continuous servo winch (from proven test script)"""
        if not self.pca:
            print("Servo controller not initialized")
            return False
            
        try:
            # Use proven pulse values from test script
            if direction == 'forward':
                pulse = 1700  # Forward rotation
            elif direction == 'backward':
                pulse = 1300  # Backward rotation
            elif direction == 'slow':
                pulse = 1590  # Slightly faster forward - controlled rotation
            else:
                print(f"Invalid winch direction: {direction}")
                return False

            # Start rotation
            self.winch_channel.duty_cycle = self._pulse_to_duty(pulse)
            print(f"Winch rotating {direction} for {duration}s")

            # Run for specified duration
            time.sleep(duration)

            # Stop rotation
            self.winch_channel.duty_cycle = 0
            print("Winch stopped")
            return True

        except Exception as e:
            print(f"Winch rotation error: {e}")
            return False
    
    def winch_burst_sequence(self, direction='forward', bursts=2, burst_duration=0.12, pause_duration=0.3):
        """Execute winch burst sequence (from proven test script)"""
        if not self.pca:
            return False
            
        try:
            print(f"Winch burst sequence: {bursts} bursts {direction}")
            
            for i in range(bursts):
                print(f"Burst {i+1}/{bursts}")
                self.rotate_winch(direction, burst_duration)
                if i < bursts - 1:  # Don't pause after last burst
                    time.sleep(pause_duration)
            
            print("Winch burst sequence complete")
            return True
            
        except Exception as e:
            print(f"Winch burst sequence error: {e}")
            return False
    
    def center_camera(self, smooth=True):
        """Center camera to neutral position"""
        print("Centering camera...")
        success_pitch = self.set_camera_pitch(55, smooth)
        success_pan = self.set_camera_pan(100, smooth)
        return success_pitch and success_pan
    
    def look_down(self, angle=70, smooth=True):
        """Point camera downward for ground-level monitoring"""
        print(f"Camera looking down {angle} degrees...")
        return self.set_camera_pitch(90 - angle, smooth)
    
    def look_up(self, angle=45, smooth=True):
        """Point camera upward"""
        print(f"Camera looking up {angle} degrees...")
        return self.set_camera_pitch(90 + angle, smooth)
    
    def scan_left_right(self, cycles=1, delay=1.0):
        """Scan camera left and right"""
        if not self.pca:
            return False
            
        try:
            print(f"Camera scanning {cycles} cycles...")
            
            for cycle in range(cycles):
                # Scan left
                self.set_camera_pan(200, smooth=True)  #160 ok
                time.sleep(delay)
                
                # Scan right
                self.set_camera_pan(10, smooth=True)    #30 ok
                time.sleep(delay)
            
            # Return to center
            self.set_camera_pan(90, smooth=True)
            print("Camera scan complete")
            return True
            
        except Exception as e:
            print(f"Camera scan error: {e}")
            return False
    
    def release_all_servos(self):
        """Release all servo channels (proven method)"""
        if not self.pca:
            return
            
        try:
            self.pitch_channel.duty_cycle = 0
            self.pan_channel.duty_cycle = 0
            self.winch_channel.duty_cycle = 0
            print("All servos released")
            
        except Exception as e:
            print(f"Servo release error: {e}")
    
    def get_status(self):
        """Get current servo positions and status"""
        return {
            'initialized': self.pca is not None,
            'current_pitch': self.current_pitch,
            'current_pan': self.current_pan,
            'servo_frequency': self.settings.SERVO_FREQUENCY
        }
    
    def is_initialized(self):
        """Check if servo controller is properly initialized"""
        return self.pca is not None
    
    def cleanup(self):
        """Clean shutdown of servo system"""
        print("Cleaning up servo controller...")
        
        # Release all servos
        self.release_all_servos()
        
        # Close I2C if we opened it
        if self.i2c:
            try:
                # Note: busio.I2C doesn't have explicit close method
                # The object will be garbage collected
                print("Servo controller cleanup complete")
            except Exception as e:
                print(f"Servo cleanup error: {e}")

# Test function for individual module testing
def test_servos():
    """Simple test function for servo controller"""
    print("Testing Servo Controller...")
    
    servos = ServoController()
    if not servos.is_initialized():
        print("Servo initialization failed!")
        return
    
    try:
        # Test camera positioning
        print("Testing camera pitch...")
        servos.set_camera_pitch(60)  # Look down
        time.sleep(1)
        
        servos.set_camera_pitch(120)  # Look up
        time.sleep(1)
        
        servos.center_camera()
        time.sleep(1)
        
        print("Testing camera pan...")
        servos.scan_left_right(cycles=1, delay=0.5)
        
        print("Testing winch...")
        servos.winch_burst_sequence('forward', bursts=2)
        time.sleep(1)
        
        print("Servo test complete!")
        
    except KeyboardInterrupt:
        print("Test interrupted")
    finally:
        servos.cleanup()

if __name__ == "__main__":
    test_servos()
