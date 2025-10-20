#!/usr/bin/env python3
"""
DogBot Autonomous Mission System
Orchestrates AI detection, behavior analysis, and treat dispensing
"""

import threading
import time
import json
import cv2
from datetime import datetime
from ai.dog_detection import DogBehaviorDetector
from hardware.servo_control import ServoController
from hardware.audio_control import AudioController
from hardware.led_control import LEDController
import logging

class MissionController:
    def __init__(self):
        # Initialize subsystems
        self.detector = DogBehaviorDetector()
        self.servo_controller = ServoController()
        self.audio_controller = AudioController()
        self.led_controller = LEDController()
        
        # Mission state
        self.running = False
        self.camera = None
        self.mission_log = []
        
        # Performance metrics
        self.treats_dispensed = 0
        self.behaviors_detected = 0
        self.session_start = time.time()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/home/pi/dogbot/logs/mission.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('MissionController')
    
    def initialize_camera(self):
        """Initialize camera with optimal settings"""
        self.camera = cv2.VideoCapture(0)
        
        # Optimize camera settings for AI
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
        
        # Test camera
        ret, frame = self.camera.read()
        if not ret:
            self.logger.error("❌ Camera initialization failed")
            return False
        
        self.logger.info("✅ Camera initialized")
        return True
    
    def dispense_treat(self, behavior_info):
        """Dispense treat with full feedback system"""
        try:
            self.logger.info(f"🍪 Dispensing treat for: {behavior_info}")
            
            # LED feedback - excitement pattern
            self.led_controller.set_pattern('excitement')
            
            # Audio feedback - positive sound
            self.audio_controller.play_sound('good_dog')
            
            # Rotate carousel to dispense treat
            self.servo_controller.rotate_carousel(45)  # 45 degree rotation
            time.sleep(0.5)
            
            # Activate treat pusher/dropper mechanism
            self.servo_controller.dispense_treat()
            
            # Update metrics
            self.treats_dispensed += 1
            
            # Log the event
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': 'treat_dispensed',
                'behavior': behavior_info,
                'treat_count': self.treats_dispensed
            }
            self.mission_log.append(event)
            
            # LED feedback - success pattern
            time.sleep(1)
            self.led_controller.set_pattern('success')
            
            self.logger.info(f"✅ Treat dispensed successfully (Total: {self.treats_dispensed})")
            
        except Exception as e:
            self.logger.error(f"❌ Treat dispensing failed: {e}")
            self.led_controller.set_pattern('error')
    
    def process_frame(self, frame):
        """Process single frame through AI pipeline with action decisions"""
        try:
            # Run AI detection
            detections, behavior_analysis = self.detector.detect_frame(frame)
            
            if detections:
                self.behaviors_detected += len(detections)
                
                # Check each behavior for reward criteria
                for behavior_name, analysis in behavior_analysis.items():
                    if analysis['should_reward']:
                        self.dispense_treat({
                            'behavior': behavior_name,
                            'confidence': analysis['confidence'],
                            'stability': analysis['stability'],
                            'duration': analysis['duration']
                        })
                        
                        # Update last treat time to prevent spam
                        self.detector.last_treat_time = time.time()
                        break  # Only one treat per frame
            
            return detections, behavior_analysis
            
        except Exception as e:
            self.logger.error(f"❌ Frame processing error: {e}")
            return [], {}
    
    def autonomous_mission_loop(self):
        """Main autonomous operation loop"""
        self.logger.info("🤖 Starting autonomous mission loop")
        frame_count = 0
        
        while self.running:
            try:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    self.logger.warning("⚠️ Failed to capture frame")
                    continue
                
                frame_count += 1
                
                # Process every frame for real-time response
                detections, analysis = self.process_frame(frame)
                
                # Log significant events
                if detections:
                    self.logger.info(f"🔍 Frame {frame_count}: {len(detections)} detections")
                
                # Add small delay to prevent CPU overload
                time.sleep(0.033)  # ~30 FPS
                
            except KeyboardInterrupt:
                self.logger.info("🛑 Mission interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"❌ Mission loop error: {e}")
                time.sleep(1)  # Prevent rapid error spam
        
        self.logger.info("🏁 Autonomous mission completed")
    
    def start_mission(self):
        """Start autonomous mission"""
        if self.running:
            self.logger.warning("Mission already running")
            return False
        
        # Initialize systems
        if not self.initialize_camera():
            return False
        
        # Startup sequence
        self.led_controller.set_pattern('startup')
        self.audio_controller.play_sound('startup')
        
        # Start mission
        self.running = True
        self.session_start = time.time()
        
        # Run mission in separate thread
        self.mission_thread = threading.Thread(target=self.autonomous_mission_loop)
        self.mission_thread.start()
        
        self.logger.info("🚀 Autonomous mission started")
        return True
    
    def stop_mission(self):
        """Stop autonomous mission"""
        self.running = False
        
        if hasattr(self, 'mission_thread'):
            self.mission_thread.join(timeout=5)
        
        if self.camera:
            self.camera.release()
        
        # Shutdown sequence
        self.led_controller.set_pattern('shutdown')
        self.audio_controller.play_sound('shutdown')
        
        # Save mission log
        self.save_mission_log()
        
        self.logger.info("🛑 Mission stopped")
    
    def save_mission_log(self):
        """Save mission log to file"""
        try:
            log_filename = f"/home/pi/dogbot/logs/mission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            session_summary = {
                'session_duration': time.time() - self.session_start,
                'treats_dispensed': self.treats_dispensed,
                'behaviors_detected': self.behaviors_detected,
                'events': self.mission_log
            }
            
            with open(log_filename, 'w') as f:
                json.dump(session_summary, f, indent=2)
            
            self.logger.info(f"📝 Mission log saved: {log_filename}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save mission log: {e}")
    
    def get_mission_status(self):
        """Get current mission status"""
        return {
            'running': self.running,
            'treats_dispensed': self.treats_dispensed,
            'behaviors_detected': self.behaviors_detected,
            'session_duration': time.time() - self.session_start if self.running else 0
        }

# CLI interface for testing
if __name__ == "__main__":
    import sys
    
    mission = MissionController()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'start':
            mission.start_mission()
            try:
                while mission.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                mission.stop_mission()
        
        elif command == 'test':
            print("🧪 Running test mode...")
            if mission.initialize_camera():
                # Run for 30 seconds
                mission.start_mission()
                time.sleep(30)
                mission.stop_mission()
    else:
        print("Usage: python mission_system.py [start|test]")