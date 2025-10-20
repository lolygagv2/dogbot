#!/usr/bin/env python3
"""
main.py - TreatSensei Core Application
Integrates all proven subsystems into unified robot control
"""

import time
import signal
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Clear GPIO 16 before starting
os.system("sudo gpioset gpiochip0 16=0 2>/dev/null")
time.sleep(0.1)
# Import all core modules
from core.hardware.motor_controller import MotorController, MotorDirection
from core.hardware.audio_controller import AudioController
from core.hardware.led_controller import LEDController, LEDMode
from core.hardware.servo_controller import ServoController
from core.ai_controller import AIController

class TreatSenseiCore:
    """Main application class integrating all proven subsystems"""
    
    def __init__(self):
        print("Initializing TreatSensei Core Systems...")
        print("=" * 50)
        
        # Initialize subsystems in order
        self.motors = None
        self.audio = None
        self.leds = None
        self.servos = None
        self.ai = None
        
        # System state
        self.running = True
        self.initialization_successful = False
        
        # Initialize all subsystems
        self._initialize_subsystems()
        
        # Set up signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _initialize_subsystems(self):
        """Initialize all robot subsystems"""
        try:
            # Initialize motor controller
            print("Initializing motor controller...")
            self.motors = MotorController()
            if not self.motors.is_initialized():
                print("WARNING: Motor controller initialization incomplete")
            else:
                print("Motor controller ready")
            
            # Initialize audio controller
            print("\nInitializing audio controller...")
            self.audio = AudioController()
            if not self.audio.is_initialized():
                print("WARNING: Audio controller initialization incomplete")
            else:
                print("Audio controller ready")
            
            # Initialize LED controller
            print("\nInitializing LED controller...")
            self.leds = LEDController()
            if not self.leds.is_initialized():
                print("WARNING: LED controller initialization incomplete")
            else:
                print("LED controller ready")
            
            # Initialize servo controller
            print("\nInitializing servo controller...")
            self.servos = ServoController()
            if not self.servos.is_initialized():
                print("WARNING: Servo controller initialization incomplete")
            else:
                print("Servo controller ready")

            # Initialize AI controller
            print("\nInitializing AI controller...")
            self.ai = AIController()
            if not self.ai.initialize():
                print("WARNING: AI controller initialization incomplete")
            else:
                print("AI controller ready")

            # Check overall initialization
            basic_systems_ok = (self.motors.is_initialized() and
                               self.audio.is_initialized() and
                               self.leds.is_initialized() and
                               self.servos.is_initialized())

            ai_ok = self.ai.is_initialized() if self.ai else False

            if basic_systems_ok:
                
                self.initialization_successful = True
                print("\n" + "=" * 50)
                print("TreatSensei initialization COMPLETE!")
                print("All basic subsystems operational")
                if ai_ok:
                    print("🤖 AI system ready - dog detection enabled!")
                else:
                    print("⚠️  AI system not available - basic robot functions only")
                
                # Set initial system state
                self._set_initial_state()
                
            else:
                print("\n" + "=" * 50)
                print("TreatSensei initialization PARTIAL")
                print("Some subsystems may not be fully operational")
                self._set_initial_state()  # Still try to set safe state
                
        except Exception as e:
            print(f"\nCRITICAL: Initialization failed: {e}")
            self.initialization_successful = False
    
    def _set_initial_state(self):
        """Set robot to safe initial state"""
        try:
            # Stop all motors
            if self.motors and self.motors.is_initialized():
                self.motors.emergency_stop()
            
            # Set LEDs to idle mode
            if self.leds and self.leds.is_initialized():
                self.leds.set_mode(LEDMode.IDLE)
            
            # Switch audio to DFPlayer
            if self.audio and self.audio.is_initialized():
                self.audio.switch_to_dfplayer()
                # Play startup sound
                self.audio.play_sound("door_scan")
            
            # Center camera
            if self.servos and self.servos.is_initialized():
                self.servos.center_camera()
            
            print("Robot set to safe initial state")
            
        except Exception as e:
            print(f"Initial state setup error: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.running = False
        self.shutdown()
        sys.exit(0)
    
    def emergency_stop(self):
        """Emergency stop all robot systems"""
        print("EMERGENCY STOP ACTIVATED")
        
        # Stop motors immediately
        if self.motors:
            self.motors.emergency_stop()
        
        # Set error LED mode
        if self.leds:
            self.leds.set_mode(LEDMode.ERROR)
        
        # Release servos
        if self.servos:
            self.servos.release_all_servos()
        
        print("Emergency stop complete")
    
    def get_system_status(self):
        """Get comprehensive system status"""
        status = {
            'initialization_successful': self.initialization_successful,
            'running': self.running,
            'subsystems': {}
        }
        
        if self.motors:
            status['subsystems']['motors'] = self.motors.get_status()
        
        if self.audio:
            status['subsystems']['audio'] = self.audio.get_status()
        
        if self.leds:
            status['subsystems']['leds'] = self.leds.get_status()
        
        if self.servos:
            status['subsystems']['servos'] = self.servos.get_status()

        if self.ai:
            status['subsystems']['ai'] = self.ai.get_status()

        return status
    
    def run_basic_test_sequence(self):
        """Run basic test of all subsystems"""
        if not self.initialization_successful:
            print("Cannot run test - initialization incomplete")
            return False
        
        print("\nRunning basic system test...")
        
        try:
            # LED test
            print("Testing LEDs...")
            self.leds.set_mode(LEDMode.SEARCHING)
            time.sleep(2)
            
            # Audio test
            print("Testing audio...")
            self.audio.play_sound("good_dog")
            time.sleep(2)
            
            # Servo test
            print("Testing servos...")
            self.servos.look_down(30)
            time.sleep(1)
            self.servos.center_camera()
            time.sleep(1)
            
            # Motor test (brief)
            print("Testing motors...")
            self.motors.tank_steering(MotorDirection.FORWARD, 30, 1)
            time.sleep(2)
            
            # Return to idle state
            self.leds.set_mode(LEDMode.IDLE)
            print("Basic test sequence complete!")
            return True
            
        except Exception as e:
            print(f"Test sequence error: {e}")
            self.emergency_stop()
            return False
    
    def interactive_mode(self):
        """Run interactive control mode"""
        print("\nTreatSensei Interactive Mode")
        print("Commands:")
        print("  forward/backward/left/right [speed] - Motor control")
        print("  stop                               - Stop motors")
        print("  audio [command]                    - Audio control")
        print("  led [mode]                         - LED control")
        print("  servo [action]                     - Servo control")
        print("  ai test                            - Test AI detection")
        print("  ai status                          - AI system status")
        print("  status                             - System status")
        print("  test                               - Run test sequence")
        print("  emergency                          - Emergency stop")
        print("  quit                               - Exit")
        
        while self.running:
            try:
                cmd = input("\nTreatSensei> ").strip().lower().split()
                if not cmd:
                    continue
                
                if cmd[0] == 'quit':
                    break
                elif cmd[0] == 'stop':
                    if self.motors:
                        self.motors.emergency_stop()
                elif cmd[0] in ['forward', 'backward', 'left', 'right']:
                    if self.motors:
                        speed = int(cmd[1]) if len(cmd) > 1 else 50
                        direction = MotorDirection(cmd[0])
                        self.motors.tank_steering(direction, speed)
                elif cmd[0] == 'audio':
                    if len(cmd) > 1 and self.audio:
                        if cmd[1] == 'good':
                            self.audio.play_sound("good_dog")
                        elif cmd[1] == 'no': 
                            self.audio.play_sound("no")
                        elif cmd[1] == 'scooby':  
                            self.audio.play_sound("scooby_snacks")
                        elif cmd[1] == 'door_scan':
                            self.audio.play_sound("door_scan")    
                        elif cmd[1] == 'hi_scan': 
                            self.audio.play_sound("hi_scan")
                        elif cmd[1] == 'busy_scan':
                            self.audio.play_sound("busy_scan")
                        elif cmd[1] == 'progress_scan': #YES seems to be working YES
                            self.audio.play_sound("progress_scan")
                        elif cmd[1] == 'robo_scan': #YES
                            self.audio.play_sound("robo_scan")
                        elif cmd[1] == 'concerto':
                            self.audio.play_sound("mozart_piano")
                        elif cmd[1] == 'flute':
                            self.audio.play_sound("mozart_concerto")
                        elif cmd[1] == 'milkshake':  #YES
                            self.audio.play_sound("milkshake")
                        elif cmd[1] == 'startup':
                            self.audio.play_sound("hi_scan")
                        elif cmd[1] == 'pi':
                            self.audio.switch_to_pi_audio()
                        elif cmd[1] == 'df':
                            self.audio.switch_to_dfplayer()
                elif cmd[0] == 'led':
                    if len(cmd) > 1 and self.leds:
                        try:
                            mode = LEDMode(cmd[1])
                            self.leds.set_mode(mode)
                        except ValueError:
                            print(f"Invalid LED mode: {cmd[1]}")
                elif cmd[0] == 'servo':
                    if len(cmd) > 1 and self.servos:
                        if cmd[1] == 'center':
                            self.servos.center_camera()
                        elif cmd[1] == 'down':
                            self.servos.look_down()
                        elif cmd[1] == 'up':
                            self.servos.look_up()
                        elif cmd[1] == 'scan':
                            self.servos.scan_left_right()
                        elif cmd[1] == 'winch':
                            self.servos.rotate_winch()
                elif cmd[0] == 'ai':
                    if len(cmd) > 1 and self.ai:
                        if cmd[1] == 'test':
                            print("🤖 Testing AI detection...")
                            # Create test frame
                            import numpy as np
                            test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
                            detections = self.ai.detect_objects(test_frame)
                            print(f"AI test complete. Detections: {len(detections)}")
                        elif cmd[1] == 'status':
                            if self.ai:
                                ai_status = self.ai.get_status()
                                print("AI Status:")
                                for key, value in ai_status.items():
                                    print(f"  {key}: {value}")
                            else:
                                print("AI system not initialized")
                    else:
                        print("AI system not available")
                elif cmd[0] == 'status':
                    status = self.get_system_status()
                    print("System Status:")
                    for subsystem, info in status['subsystems'].items():
                        print(f"  {subsystem}: {info}")
                elif cmd[0] == 'test':
                    self.run_basic_test_sequence()
                elif cmd[0] == 'emergency':
                    self.emergency_stop()
                else:
                    print("Unknown command")
                    
            except KeyboardInterrupt:
                print("\nShutdown requested...")
                break
            except Exception as e:
                print(f"Command error: {e}")
    
    def shutdown(self):
        """Clean shutdown of all systems"""
        print("\nShutting down TreatSensei...")
        
        # Stop all motion
        self.emergency_stop()
        
        # Cleanup all subsystems
        if self.motors:
            self.motors.cleanup()
        
        if self.audio:
            self.audio.cleanup()
        
        if self.leds:
            self.leds.cleanup()
        
        if self.servos:
            self.servos.cleanup()

        if self.ai:
            self.ai.cleanup()

        print("TreatSensei shutdown complete")

def main():
    """Main application entry point"""
    print("TreatSensei AI Robot - Core System")
    print("Based on proven hardware integration")
    print("")
    
    try:
        # Initialize robot
        robot = TreatSenseiCore()
        
        if robot.initialization_successful:
            # Run basic test
            robot.run_basic_test_sequence()
            
            # Enter interactive mode
            robot.interactive_mode()
        else:
            print("Initialization failed - exiting")
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        print(f"System error: {e}")
    finally:
        if 'robot' in locals():
            robot.shutdown()

if __name__ == "__main__":
    main()