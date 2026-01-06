#!/usr/bin/env python3
"""
Motor, LED, and Camera Control Modules for DogBot
"""

import time
import logging
import numpy as np
from typing import Tuple, Optional, Dict
import RPi.GPIO as GPIO
import board
import neopixel
from picamera2 import Picamera2
from libcamera import controls
import threading

logger = logging.getLogger(__name__)

# ============== MOTOR CONTROL ==============

class MotorController:
    """Control Devastator tank chassis motors via L298N"""
    
    def __init__(self):
        """Initialize motor controller"""
        # L298N GPIO pins
        self.MOTOR_LEFT_EN = 17   # Enable left motor (PWM)
        self.MOTOR_LEFT_IN1 = 27  # Left motor direction 1
        self.MOTOR_LEFT_IN2 = 22  # Left motor direction 2
        
        self.MOTOR_RIGHT_EN = 18  # Enable right motor (PWM)
        self.MOTOR_RIGHT_IN1 = 23 # Right motor direction 1
        self.MOTOR_RIGHT_IN2 = 24 # Right motor direction 2
        
        self.pwm_left = None
        self.pwm_right = None
        self.current_speed = 0
        self.max_speed = 100
        
    def initialize(self):
        """Initialize motor GPIO pins"""
        try:
            GPIO.setmode(GPIO.BCM)
            
            # Setup motor pins
            GPIO.setup(self.MOTOR_LEFT_EN, GPIO.OUT)
            GPIO.setup(self.MOTOR_LEFT_IN1, GPIO.OUT)
            GPIO.setup(self.MOTOR_LEFT_IN2, GPIO.OUT)
            GPIO.setup(self.MOTOR_RIGHT_EN, GPIO.OUT)
            GPIO.setup(self.MOTOR_RIGHT_IN1, GPIO.OUT)
            GPIO.setup(self.MOTOR_RIGHT_IN2, GPIO.OUT)
            
            # Setup PWM for speed control (1000 Hz)
            self.pwm_left = GPIO.PWM(self.MOTOR_LEFT_EN, 1000)
            self.pwm_right = GPIO.PWM(self.MOTOR_RIGHT_EN, 1000)
            
            self.pwm_left.start(0)
            self.pwm_right.start(0)
            
            logger.info("Motor controller initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize motors: {e}")
            return False
            
    def set_motor_speed(self, left_speed: float, right_speed: float):
        """
        Set motor speeds
        
        Args:
            left_speed: Left motor speed (-100 to 100)
            right_speed: Right motor speed (-100 to 100)
        """
        # Clamp speeds
        left_speed = np.clip(left_speed, -100, 100)
        right_speed = np.clip(right_speed, -100, 100)
        
        # Set left motor
        if left_speed > 0:
            GPIO.output(self.MOTOR_LEFT_IN1, GPIO.HIGH)
            GPIO.output(self.MOTOR_LEFT_IN2, GPIO.LOW)
            self.pwm_left.ChangeDutyCycle(abs(left_speed))
        elif left_speed < 0:
            GPIO.output(self.MOTOR_LEFT_IN1, GPIO.LOW)
            GPIO.output(self.MOTOR_LEFT_IN2, GPIO.HIGH)
            self.pwm_left.ChangeDutyCycle(abs(left_speed))
        else:
            GPIO.output(self.MOTOR_LEFT_IN1, GPIO.LOW)
            GPIO.output(self.MOTOR_LEFT_IN2, GPIO.LOW)
            self.pwm_left.ChangeDutyCycle(0)
            
        # Set right motor
        if right_speed > 0:
            GPIO.output(self.MOTOR_RIGHT_IN1, GPIO.HIGH)
            GPIO.output(self.MOTOR_RIGHT_IN2, GPIO.LOW)
            self.pwm_right.ChangeDutyCycle(abs(right_speed))
        elif right_speed < 0:
            GPIO.output(self.MOTOR_RIGHT_IN1, GPIO.LOW)
            GPIO.output(self.MOTOR_RIGHT_IN2, GPIO.HIGH)
            self.pwm_right.ChangeDutyCycle(abs(right_speed))
        else:
            GPIO.output(self.MOTOR_RIGHT_IN1, GPIO.LOW)
            GPIO.output(self.MOTOR_RIGHT_IN2, GPIO.LOW)
            self.pwm_right.ChangeDutyCycle(0)
            
    def move_forward(self, duration: float = 1.0, speed: float = 50):
        """Move forward for duration"""
        self.set_motor_speed(speed, speed)
        time.sleep(duration)
        self.stop()
        
    def move_backward(self, duration: float = 1.0, speed: float = 50):
        """Move backward for duration"""
        self.set_motor_speed(-speed, -speed)
        time.sleep(duration)
        self.stop()
        
    def turn_left(self, angle: float = 90, speed: float = 50):
        """Turn left by angle degrees"""
        turn_time = angle / 90.0  # Approximate timing
        self.set_motor_speed(-speed, speed)
        time.sleep(turn_time)
        self.stop()
        
    def turn_right(self, angle: float = 90, speed: float = 50):
        """Turn right by angle degrees"""
        turn_time = angle / 90.0  # Approximate timing
        self.set_motor_speed(speed, -speed)
        time.sleep(turn_time)
        self.stop()
        
    def stop(self):
        """Stop all motors"""
        self.set_motor_speed(0, 0)
        
    def cleanup(self):
        """Clean up motor resources"""
        self.stop()
        if self.pwm_left:
            self.pwm_left.stop()
        if self.pwm_right:
            self.pwm_right.stop()
        GPIO.cleanup([
            self.MOTOR_LEFT_EN, self.MOTOR_LEFT_IN1, self.MOTOR_LEFT_IN2,
            self.MOTOR_RIGHT_EN, self.MOTOR_RIGHT_IN1, self.MOTOR_RIGHT_IN2
        ])
        logger.info("Motor controller cleaned up")

# ============== LED CONTROL ==============

class LEDController:
    """Control NeoPixel ring and dome LEDs"""
    
    def __init__(self):
        """Initialize LED controller"""
        self.NEOPIXEL_PIN = board.D10  # GPIO10 (Pin 19)
        self.NEOPIXEL_COUNT = 16       # 16 LED ring
        self.DOME_LED_PIN = 13         # GPIO 13 for blue dome
        
        self.pixels = None
        self.current_pattern = None
        self.pattern_thread = None
        self.running = False
        
        # Predefined colors
        self.colors = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'purple': (255, 0, 255),
            'cyan': (0, 255, 255),
            'white': (255, 255, 255),
            'orange': (255, 128, 0),
            'off': (0, 0, 0)
        }
        
    def initialize(self):
        """Initialize LED hardware"""
        try:
            # Initialize NeoPixels
            self.pixels = neopixel.NeoPixel(
                self.NEOPIXEL_PIN,
                self.NEOPIXEL_COUNT,
                brightness=0.5,
                auto_write=False
            )
            
            # Initialize dome LED
            GPIO.setup(self.DOME_LED_PIN, GPIO.OUT)
            GPIO.output(self.DOME_LED_PIN, GPIO.LOW)
            
            self.running = True
            logger.info("LED controller initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize LEDs: {e}")
            return False
            
    def set_pattern(self, pattern_name: str):
        """Set LED pattern"""
        # Stop current pattern
        if self.pattern_thread and self.pattern_thread.is_alive():
            self.running = False
            self.pattern_thread.join(timeout=1)
            
        self.running = True
        self.current_pattern = pattern_name
        
        # Start new pattern thread
        patterns = {
            'breathing': self._pattern_breathing,
            'pulse_green': self._pattern_pulse_green,
            'pulse_blue': self._pattern_pulse_blue,
            'rainbow': self._pattern_rainbow,
            'spin': self._pattern_spin,
            'celebration': self._pattern_celebration,
            'fade_in': self._pattern_fade_in,
            'fade_out': self._pattern_fade_out,
            'solid_white': self._pattern_solid_white,
            'searching': self._pattern_searching,
            'ready': self._pattern_ready,
            'off': self._pattern_off
        }
        
        if pattern_name in patterns:
            self.pattern_thread = threading.Thread(target=patterns[pattern_name])
            self.pattern_thread.start()
            
    def _pattern_breathing(self):
        """Breathing effect"""
        while self.running:
            # Fade in
            for i in range(0, 100, 2):
                brightness = i / 100.0
                color = tuple(int(c * brightness) for c in self.colors['white'])
                self.pixels.fill(color)
                self.pixels.show()
                time.sleep(0.02)
                
            # Fade out
            for i in range(100, 0, -2):
                brightness = i / 100.0
                color = tuple(int(c * brightness) for c in self.colors['white'])
                self.pixels.fill(color)
                self.pixels.show()
                time.sleep(0.02)
                
    def _pattern_pulse_green(self):
        """Pulse green for good behavior"""
        for _ in range(3):
            if not self.running:
                break
            self.pixels.fill(self.colors['green'])
            self.pixels.show()
            time.sleep(0.3)
            self.pixels.fill(self.colors['off'])
            self.pixels.show()
            time.sleep(0.3)
            
    def _pattern_rainbow(self):
        """Rainbow cycle effect"""
        while self.running:
            for j in range(255):
                if not self.running:
                    break
                for i in range(self.NEOPIXEL_COUNT):
                    pixel_index = (i * 256 // self.NEOPIXEL_COUNT) + j
                    self.pixels[i] = self._wheel(pixel_index & 255)
                self.pixels.show()
                time.sleep(0.02)
                
    def _pattern_spin(self):
        """Spinning effect"""
        while self.running:
            for i in range(self.NEOPIXEL_COUNT):
                if not self.running:
                    break
                self.pixels.fill(self.colors['off'])
                self.pixels[i] = self.colors['blue']
                if i > 0:
                    self.pixels[i-1] = tuple(c//2 for c in self.colors['blue'])
                self.pixels.show()
                time.sleep(0.05)
                
    def _pattern_celebration(self):
        """Celebration pattern for treats"""
        colors = ['red', 'green', 'blue', 'yellow', 'purple', 'cyan']
        for _ in range(10):
            if not self.running:
                break
            for color in colors:
                self.pixels.fill(self.colors[color])
                self.pixels.show()
                GPIO.output(self.DOME_LED_PIN, GPIO.HIGH)
                time.sleep(0.1)
                GPIO.output(self.DOME_LED_PIN, GPIO.LOW)
                time.sleep(0.1)
                
    def _pattern_searching(self):
        """Searching pattern"""
        while self.running:
            for i in range(self.NEOPIXEL_COUNT):
                if not self.running:
                    break
                self.pixels.fill(self.colors['off'])
                self.pixels[i] = self.colors['yellow']
                self.pixels[(i + self.NEOPIXEL_COUNT//2) % self.NEOPIXEL_COUNT] = self.colors['yellow']
                self.pixels.show()
                time.sleep(0.1)
                
    def _pattern_ready(self):
        """Ready state - solid green"""
        self.pixels.fill(self.colors['green'])
        self.pixels.show()
        GPIO.output(self.DOME_LED_PIN, GPIO.HIGH)
        
    def _pattern_solid_white(self):
        """Solid white"""
        self.pixels.fill(self.colors['white'])
        self.pixels.show()
        
    def _pattern_fade_in(self):
        """Fade in effect"""
        for i in range(0, 100, 5):
            brightness = i / 100.0
            color = tuple(int(c * brightness) for c in self.colors['blue'])
            self.pixels.fill(color)
            self.pixels.show()
            time.sleep(0.05)
            
    def _pattern_fade_out(self):
        """Fade out effect"""
        for i in range(100, 0, -5):
            brightness = i / 100.0
            color = tuple(int(c * brightness) for c in self.colors['blue'])
            self.pixels.fill(color)
            self.pixels.show()
            time.sleep(0.05)
        self.pixels.fill(self.colors['off'])
        self.pixels.show()
        
    def _pattern_off(self):
        """All LEDs off"""
        self.pixels.fill(self.colors['off'])
        self.pixels.show()
        GPIO.output(self.DOME_LED_PIN, GPIO.LOW)
        
    def _wheel(self, pos):
        """Generate rainbow colors"""
        if pos < 0 or pos > 255:
            return (0, 0, 0)
        if pos < 85:
            return (255 - pos * 3, pos * 3, 0)
        if pos < 170:
            pos -= 85
            return (0, 255 - pos * 3, pos * 3)
        pos -= 170
        return (pos * 3, 0, 255 - pos * 3)
        
    def set_dome_led(self, state: bool):
        """Control dome LED"""
        GPIO.output(self.DOME_LED_PIN, GPIO.HIGH if state else GPIO.LOW)
        
    def cleanup(self):
        """Clean up LED resources"""
        self.running = False
        if self.pattern_thread:
            self.pattern_thread.join(timeout=2)
        if self.pixels:
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
        GPIO.output(self.DOME_LED_PIN, GPIO.LOW)
        logger.info("LED controller cleaned up")

# ============== CAMERA CONTROL ==============

class IMX500Camera:
    """Control IMX500 AI camera with adjustable parameters"""
    
    def __init__(self, resolution: Tuple[int, int] = (1920, 1080), fps: int = 30):
        """Initialize camera"""
        self.resolution = resolution
        self.fps = fps
        self.camera = None
        self.config = None
        
        # Camera parameters (adjustable)
        self.parameters = {
            'brightness': 0.0,      # -1.0 to 1.0
            'contrast': 1.0,        # 0.0 to 2.0
            'saturation': 1.0,      # 0.0 to 2.0
            'sharpness': 1.0,       # 0.0 to 16.0
            'exposure_time': None,  # microseconds or None for auto
            'analogue_gain': 1.0,   # 1.0 to 16.0
            'awb_mode': 'auto',     # auto, tungsten, fluorescent, daylight, cloudy, custom
            'awb_gains': None,      # (red_gain, blue_gain) or None for auto
        }
        
    def initialize(self):
        """Initialize camera"""
        try:
            self.camera = Picamera2()
            
            # Create configuration
            self.config = self.camera.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={
                    "FrameRate": self.fps,
                }
            )
            
            self.camera.configure(self.config)
            
            logger.info(f"Camera initialized at {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")
            return False
            
    def start(self):
        """Start camera capture"""
        if self.camera:
            self.camera.start()
            self._apply_parameters()
            logger.info("Camera started")
            
    def stop(self):
        """Stop camera capture"""
        if self.camera:
            self.camera.stop()
            logger.info("Camera stopped")
            
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture single frame"""
        try:
            if self.camera:
                frame = self.camera.capture_array()
                return frame
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
        return None
        
    def set_parameter(self, param: str, value):
        """
        Set camera parameter
        
        Args:
            param: Parameter name
            value: Parameter value
        """
        if param not in self.parameters:
            logger.error(f"Unknown parameter: {param}")
            return
            
        self.parameters[param] = value
        
        try:
            if param == 'brightness':
                self.camera.set_controls({"Brightness": value})
            elif param == 'contrast':
                self.camera.set_controls({"Contrast": value})
            elif param == 'saturation':
                self.camera.set_controls({"Saturation": value})
            elif param == 'sharpness':
                self.camera.set_controls({"Sharpness": value})
            elif param == 'exposure_time':
                if value is None:
                    self.camera.set_controls({"AeEnable": True})
                else:
                    self.camera.set_controls({"AeEnable": False, "ExposureTime": value})
            elif param == 'analogue_gain':
                self.camera.set_controls({"AnalogueGain": value})
            elif param == 'awb_mode':
                awb_modes = {
                    'auto': controls.AwbModeEnum.Auto,
                    'tungsten': controls.AwbModeEnum.Tungsten,
                    'fluorescent': controls.AwbModeEnum.Fluorescent,
                    'daylight': controls.AwbModeEnum.Daylight,
                    'cloudy': controls.AwbModeEnum.Cloudy,
                    'custom': controls.AwbModeEnum.Custom
                }
                if value in awb_modes:
                    self.camera.set_controls({"AwbMode": awb_modes[value]})
            elif param == 'awb_gains' and value:
                self.camera.set_controls({
                    "AwbEnable": False,
                    "ColourGains": value
                })
                
            logger.info(f"Set {param} to {value}")
            
        except Exception as e:
            logger.error(f"Failed to set {param}: {e}")
            
    def _apply_parameters(self):
        """Apply all parameters to camera"""
        for param, value in self.parameters.items():
            if value is not None:
                self.set_parameter(param, value)
                
    def auto_adjust(self):
        """Auto-adjust camera parameters based on scene"""
        try:
            # Capture test frame
            frame = self.capture_frame()
            if frame is None:
                return
                
            # Calculate scene statistics
            brightness = np.mean(frame)
            contrast = np.std(frame)
            
            # Adjust parameters
            if brightness < 50:  # Too dark
                self.set_parameter('brightness', min(0.5, self.parameters['brightness'] + 0.2))
            elif brightness > 200:  # Too bright
                self.set_parameter('brightness', max(-0.5, self.parameters['brightness'] - 0.2))
                
            if contrast < 30:  # Low contrast
                self.set_parameter('contrast', min(2.0, self.parameters['contrast'] + 0.2))
                
            logger.info(f"Auto-adjusted camera: brightness={brightness:.1f}, contrast={contrast:.1f}")
            
        except Exception as e:
            logger.error(f"Auto-adjust failed: {e}")
            
    def get_parameters(self) -> Dict:
        """Get current camera parameters"""
        return self.parameters.copy()
        
    def save_snapshot(self, filename: str):
        """Save current frame to file"""
        try:
            frame = self.capture_frame()
            if frame is not None:
                import cv2
                cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                logger.info(f"Saved snapshot to {filename}")
                return True
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
        return False
        
    def cleanup(self):
        """Clean up camera resources"""
        self.stop()
        if self.camera:
            self.camera.close()
        logger.info("Camera cleaned up")

# Test functions
def test_motors():
    """Test motor control"""
    motor = MotorController()
    motor.initialize()
    
    print("Forward...")
    motor.move_forward(2)
    time.sleep(1)
    
    print("Backward...")
    motor.move_backward(2)
    time.sleep(1)
    
    print("Left turn...")
    motor.turn_left(90)
    time.sleep(1)
    
    print("Right turn...")
    motor.turn_right(90)
    
    motor.cleanup()
    
def test_leds():
    """Test LED patterns"""
    led = LEDController()
    led.initialize()
    
    patterns = ['breathing', 'rainbow', 'spin', 'celebration', 'searching']
    
    for pattern in patterns:
        print(f"Pattern: {pattern}")
        led.set_pattern(pattern)
        time.sleep(5)
        
    led.cleanup()
    
def test_camera():
    """Test camera adjustments"""
    cam = IMX500Camera()
    cam.initialize()
    cam.start()
    
    # Test different settings
    settings = [
        ('brightness', 0.5),
        ('contrast', 1.5),
        ('saturation', 1.2),
        ('sharpness', 2.0)
    ]
    
    for param, value in settings:
        print(f"Setting {param} to {value}")
        cam.set_parameter(param, value)
        time.sleep(2)
        cam.save_snapshot(f"/tmp/test_{param}.jpg")
        
    # Auto adjust
    print("Auto adjusting...")
    cam.auto_adjust()
    
    cam.cleanup()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "motor":
            test_motors()
        elif sys.argv[1] == "led":
            test_leds()
        elif sys.argv[1] == "camera":
            test_camera()
    else:
        print("Usage: python3 control_modules.py [motor|led|camera]")