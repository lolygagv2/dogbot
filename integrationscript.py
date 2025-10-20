#!/usr/bin/env python3
"""
Main DogBot AI System - Properly integrated for Pi 5
"""

import time
import threading
import numpy as np
from pathlib import Path

# Import your modules
from vision.imx500_camera import IMX500Camera
from ai.hailo_interface import HailoInterface, DogBehavior
from hardware.servo_control import ServoController
from hardware.audio_control import AudioController
from hardware.led_control import LEDController

class DogBotAI:
    """Main DogBot system with proper hardware integration"""
    
    def __init__(self):
        # Initialize all components
        self.camera = IMX500Camera()
        self.hailo = HailoInterface()
        self.servos = ServoController()
        self.audio = AudioController()
        self.leds = LEDController()
        
        # Behavior reward settings
        self.reward_behaviors = {
            DogBehavior.SITTING: {"duration": 2.0, "treats": 1},
            DogBehavior.LYING: {"duration": 3.0, "treats": 1},
            DogBehavior.SPINNING: {"duration": 1.0, "treats": 2},
            DogBehavior.STAY: {"duration": 5.0, "treats": 2},
            DogBehavior.HIGH_FIVE: {"duration": 0.5, "treats": 2}
        }
        
        self.last_reward_time = 0
        self.reward_cooldown = 10.0
        
    def start(self):
        """Start the complete system"""
        # Initialize hardware
        self.camera.initialize()
        self.servos.initialize()
        self.audio.initialize()
        self.leds.initialize()
        
        # Start camera
        self.camera.start()
        
        # Center camera
        self.servos.center_all()
        
        # Start processing
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        logger.info("DogBot AI fully operational on Pi 5")
    
    def _process_loop(self):
        """Main processing loop"""
        while True:
            try:
                # Get frame from IMX500
                frame, ai_metadata = self.camera.get_frame_with_ai()
                
                if frame is None:
                    continue
                
                # Use Hailo for advanced detection
                detection = self.hailo.detect_and_classify(frame)
                
                if detection:
                    # Track dog with servos
                    self._track_dog(detection.bbox)
                    
                    # Check for reward
                    if self._should_reward(detection):
                        self.dispense_treat(detection.behavior)
                    
                    # Update LEDs based on behavior
                    self._update_leds(detection.behavior)
                
            except Exception as e:
                logger.error(f"Process loop error: {e}")
                
            time.sleep(0.03)  # ~30 FPS
    
    def _should_reward(self, detection):
        """Check if behavior warrants reward"""
        if detection.behavior not in self.reward_behaviors:
            return False
        
        required_duration = self.reward_behaviors[detection.behavior]["duration"]
        
        # Check duration and cooldown
        if detection.duration >= required_duration:
            if time.time() - self.last_reward_time > self.reward_cooldown:
                return True
        
        return False
    
    def dispense_treat(self, behavior):
        """Dispense treat with audio/visual feedback"""
        self.audio.play_sound("good_dog")
        self.leds.set_pattern("celebration")
        
        # Rotate carousel
        treat_count = self.reward_behaviors[behavior]["treats"]
        for _ in range(treat_count):
            self.servos.rotate_carousel(1)
            time.sleep(0.5)
        
        self.last_reward_time = time.time()