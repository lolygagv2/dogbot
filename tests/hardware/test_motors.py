#!/usr/bin/env python3
"""
Devastator Tank Motor Control Test - Clean Version
Uses lgpio for motors, gpioset for MAX4544 (the method that works)

GPIO Pin Assignments (VERIFIED):
- GPIO17 (Pin 11): IN1 (Motor A Direction 1)
- GPIO18 (Pin 12): IN2 (Motor A Direction 2) 
- GPIO27 (Pin 13): IN3 (Motor B Direction 1)
- GPIO22 (Pin 15): IN4 (Motor B Direction 2)
- GPIO13 (Pin 33): ENA (Motor A Enable/Speed PWM)
- GPIO19 (Pin 35): ENB (Motor B Enable/Speed PWM)
- GPIO16 (Pin 36): MAX4544 audio switches (gpioset - CONFIRMED WORKING)
"""

import lgpio
import time
import threading
import subprocess
from enum import Enum

class MotorDirection(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"
    LEFT = "left"
    RIGHT = "right"
    STOP = "stop"

class DevastatorMotorController:
    def __init__(self, motor_in1=17, motor_in2=18, motor_in3=27, motor_in4=22, 
                 motor_ena=13, motor_enb=19, audio_relay_pin=16):
        """
        Initialize with lgpio for motors and gpioset for MAX4544 (confirmed working)
        """
        
        # Store pin assignments
        self.motor_in1 = motor_in1    # GPIO17 - Motor A Direction 1
        self.motor_in2 = motor_in2    # GPIO18 - Motor A Direction 2
        self.motor_in3 = motor_in3    # GPIO27 - Motor B Direction 1
        self.motor_in4 = motor_in4    # GPIO22 - Motor B Direction 2
        self.motor_ena = motor_ena    # GPIO13 - Motor A Enable/Speed
        self.motor_enb = motor_enb    # GPIO19 - Motor B Enable/Speed
        self.audio_relay_pin = audio_relay_pin  # GPIO16 - MAX4544 switches
        
        # Initialize lgpio for motors only
        self.gpio_chip = None
        try:
            self.gpio_chip = lgpio.gpiochip_open(0)
            
            # Claim motor control pins with lgpio
            motor_pins = [self.motor_in1, self.motor_in2, self.motor_in3, 
                         self.motor_in4, self.motor_ena, self.motor_enb]
            
            for pin in motor_pins:
                lgpio.gpio_claim_output(self.gpio_chip, pin, lgpio.SET_PULL_NONE)
                lgpio.gpio_write(self.gpio_chip, pin, 0)  # Start LOW
            
            print("✅ Motor controller initialized with lgpio")
            print(f"🔧 Motor A (Left):  IN1=GPIO{self.motor_in1}, IN2=GPIO{self.motor_in2}, ENA=GPIO{self.motor_ena}")
            print(f"🔧 Motor B (Right): IN3=GPIO{self.motor_in3}, IN4=GPIO{self.motor_in4}, ENB=GPIO{self.motor_enb}")
            
        except Exception as e:
            print(f"❌ Motor controller initialization failed: {e}")
        
        # Initialize MAX4544 using gpioset (the method that works)
        self.gpio_method = None
        try:
            # Test if gpioset works for GPIO16
            result = subprocess.run(['gpioset', 'gpiochip0', '16=0'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                self.gpio_method = 'gpioset'
                print(f"✅ MAX4544 audio switches ready on GPIO{self.audio_relay_pin} (using gpioset)")
            else:
                print(f"⚠️  gpioset failed: {result.stderr}")
                self.gpio_method = None
        except Exception as e:
            print(f"⚠️  MAX4544 initialization failed: {e}")
            self.gpio_method = None
            
        # Motor state
        self.left_speed = 0
        self.right_speed = 0
        self.is_moving = False
        self._current_pwm_freq = 1000  # Default PWM frequency

    def set_motor_speed(self, motor, speed, direction):
        """Set individual motor speed and direction with corrected wiring"""
        if not self.gpio_chip:
            return
            
        speed = max(0, min(100, speed))
        
        try:
            if motor == 'A' or motor == 'left':
                # Motor A (Left side) - REVERSED WIRING CORRECTION
                in1, in2, ena = self.motor_in1, self.motor_in2, self.motor_ena
                motor_name = "A (Left)"
                # Swap forward/backward for Motor A due to reversed wiring
                if direction == 'forward':
                    direction = 'backward'
                elif direction == 'backward':
                    direction = 'forward'
            elif motor == 'B' or motor == 'right':
                # Motor B (Right side) - Normal wiring
                in1, in2, ena = self.motor_in3, self.motor_in4, self.motor_enb
                motor_name = "B (Right)"
            else:
                print(f"❌ Invalid motor: {motor}")
                return
            
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
                pwm_freq = getattr(self, '_current_pwm_freq', 1000)
                lgpio.tx_pwm(self.gpio_chip, ena, pwm_freq, speed)
                if motor in ['A', 'left']:
                    self.left_speed = speed
                else:
                    self.right_speed = speed
                    
            elif direction == 'backward':
                # Backward direction
                lgpio.gpio_write(self.gpio_chip, in1, 0)
                lgpio.gpio_write(self.gpio_chip, in2, 1)
                pwm_freq = getattr(self, '_current_pwm_freq', 1000)
                lgpio.tx_pwm(self.gpio_chip, ena, pwm_freq, speed)
                if motor in ['A', 'left']:
                    self.left_speed = -speed
                else:
                    self.right_speed = -speed
                    
            print(f"🔧 Motor {motor_name}: {direction} at {speed}% {'(corrected)' if motor in ['A', 'left'] else ''}")
            
        except Exception as e:
            print(f"❌ Motor control error: {e}")

    def tank_steering(self, direction: MotorDirection, speed=50, duration=None, audio_mode="normal"):
        """Tank-style steering with flexible audio handling"""
        print(f"🚗 Tank steering: {direction.value} at {speed}% (audio: {audio_mode})")
        
        # Choose PWM frequency based on audio requirements
        if audio_mode == "reduce_interference":
            pwm_frequency = 500   # Lower frequency, less interference
            motor_speed = speed   # No compensation needed
            print(f"   Using {pwm_frequency}Hz PWM for reduced audio interference")
        else:
            pwm_frequency = 1000  # Proven frequency
            motor_speed = speed
        
        # Apply frequency setting
        self._current_pwm_freq = pwm_frequency
        
        # Handle audio switching for mute mode only
        if audio_mode == "mute" and direction != MotorDirection.STOP:
            self.mute_audio(True)
        
        if direction == MotorDirection.FORWARD:
            self.set_motor_speed('A', motor_speed, 'forward')
            self.set_motor_speed('B', motor_speed, 'forward')
            
        elif direction == MotorDirection.BACKWARD:
            self.set_motor_speed('A', motor_speed, 'backward')
            self.set_motor_speed('B', motor_speed, 'backward')
            
        elif direction == MotorDirection.LEFT:
            # Turn left: left motor backward, right motor forward
            self.set_motor_speed('A', motor_speed, 'backward')
            self.set_motor_speed('B', motor_speed, 'forward')
            
        elif direction == MotorDirection.RIGHT:
            # Turn right: left motor forward, right motor backward
            self.set_motor_speed('A', motor_speed, 'forward')
            self.set_motor_speed('B', motor_speed, 'backward')
            
        elif direction == MotorDirection.STOP:
            self.set_motor_speed('A', 0, 'stop')
            self.set_motor_speed('B', 0, 'stop')
            # Unmute audio when stopped
            if audio_mode == "mute":
                self.mute_audio(False)
        
        self.is_moving = (direction != MotorDirection.STOP)
        
        # Auto-stop after duration
        if duration and direction != MotorDirection.STOP:
            def stop_and_unmute():
                self.stop_all_motors()
                if audio_mode == "mute":
                    self.mute_audio(False)
            threading.Timer(duration, stop_and_unmute).start()

    def stop_all_motors(self):
        """Emergency stop - immediately stop both motors"""
        print("🛑 STOP - All motors halted")
        self.tank_steering(MotorDirection.STOP)

    def mute_audio(self, mute=True):
        """
        Control MAX4544 using gpioset (the method that works)
        """
        if not self.gpio_method:
            print("⚠️  MAX4544 audio switches not available")
            return
            
        try:
            # Use gpioset for GPIO control
            switch_state = 1 if mute else 0
            result = subprocess.run(['gpioset', 'gpiochip0', f'{self.audio_relay_pin}={switch_state}'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                if switch_state:
                    print(f"🔇 Audio: Switched to Pi USB Audio (GPIO{self.audio_relay_pin}=HIGH)")
                else:
                    print(f"🔇 Audio: Switched to DFPlayer (GPIO{self.audio_relay_pin}=LOW)")
            else:
                print(f"❌ GPIO control failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ MAX4544 control error: {e}")

    def check_audio_relay_status(self):
        """Check current MAX4544 switch position using gpioget"""
        if not self.gpio_method:
            print("⚠️  MAX4544 audio switches not available")
            return
            
        try:
            result = subprocess.run(['gpioget', 'gpiochip0', str(self.audio_relay_pin)], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                current_state = int(result.stdout.strip())
                print(f"📊 MAX4544 Audio Switch Status:")
                print(f"   GPIO{self.audio_relay_pin} state: {'HIGH' if current_state else 'LOW'}")
                
                if current_state:
                    print(f"   Audio path: Pi USB Audio → MAX4544 → Amp → Speakers")
                    print(f"   DFPlayer: Switched out")
                else:
                    print(f"   Audio path: DFPlayer → MAX4544 → Amp → Speakers") 
                    print(f"   Pi USB Audio: Switched out")
                    
                return current_state
            else:
                print(f"❌ Cannot read GPIO status: {result.stderr}")
                return None
        except Exception as e:
            print(f"❌ Cannot read MAX4544 status: {e}")
            return None

    def test_working_method(self):
        """Test MAX4544 using gpioset (the method that works)"""
        print("🔧 Testing MAX4544 with gpioset commands...")
        print("   🎵 Make sure DFPlayer is playing music...")
        input("   Press Enter when ready...")
        
        try:
            print("   Setting GPIO16 LOW (DFPlayer path)...")
            subprocess.run(['gpioset', 'gpiochip0', '16=0'], check=True)
            time.sleep(1)
            
            print("   Setting GPIO16 HIGH (Pi audio path)...")
            subprocess.run(['gpioset', 'gpiochip0', '16=1'], check=True)
            time.sleep(1)
            
            audio_changed = input("   Did audio switch off/change? (y/n): ").lower()
            
            print("   Setting GPIO16 LOW (back to DFPlayer)...")
            subprocess.run(['gpioset', 'gpiochip0', '16=0'], check=True)
            
            if audio_changed == 'y':
                print("   ✅ MAX4544 confirmed working with gpioset!")
                self.gpio_method = 'gpioset'
                return True
            else:
                print("   ❌ No audio change detected")
                return False
                
        except Exception as e:
            print(f"❌ gpioset test failed: {e}")
            return False

    def cleanup(self):
        """Clean shutdown"""
        try:
            print("🧹 Cleaning up motor controller...")
            self.stop_all_motors()
            
            if self.gpio_chip is not None:
                lgpio.gpiochip_close(self.gpio_chip)
                print("GPIO cleanup complete")
        except Exception as e:
            print(f"Cleanup error: {e}")

def interactive_test():
    """Interactive motor testing"""
    print("🚗 Devastator Tank Motor Test")
    print("========================================")
    print("⚠️  SAFETY WARNING:")
    print("   - Ensure robot is elevated (wheels off ground) OR")
    print("   - Ensure clear space around robot")
    print("   - Be ready to press Ctrl+C for emergency stop")
    print("\nCommands:")
    print("  forward [speed]       - Move forward (1000Hz PWM)")
    print("  backward [speed]      - Move backward") 
    print("  left [speed]          - Turn left")
    print("  right [speed]         - Turn right")
    print("  stop                  - Stop all motors")
    print("  forward_audio [speed] - Move forward optimized for audio (500Hz PWM)")
    print("  backward_audio [speed]- Move backward optimized for audio")
    print("  left_audio [speed]    - Turn left optimized for audio")
    print("  right_audio [speed]   - Turn right optimized for audio")
    print("  forward_mute [speed]  - Move forward with audio muted")
    print("  backward_mute [speed] - Move backward with audio muted")
    print("  left_mute [speed]     - Turn left with audio muted")
    print("  right_mute [speed]    - Turn right with audio muted")
    print("  test_working_method   - Test MAX4544 using gpioset")
    print("  mute_on               - Switch to Pi USB Audio")
    print("  mute_off              - Switch to DFPlayer")
    print("  relay_status          - Check MAX4544 status")
    print("  relay_toggle          - Toggle relay to test both positions")
    print("  status                - Show motor status")
    print("  quit                  - Exit")
    
    motors = DevastatorMotorController()
    
    try:
        while True:
            cmd = input("\nMotor Command: ").strip().lower().split()
            if not cmd:
                continue
                
            if cmd[0] == 'quit':
                break
            elif cmd[0] == 'stop':
                motors.stop_all_motors()
            elif cmd[0] in ['forward', 'backward', 'left', 'right']:
                speed = int(cmd[1]) if len(cmd) > 1 else 50
                direction = MotorDirection(cmd[0])
                motors.tank_steering(direction, speed, audio_mode="normal")
            elif cmd[0] in ['forward_audio', 'backward_audio', 'left_audio', 'right_audio']:
                speed = int(cmd[1]) if len(cmd) > 1 else 50
                direction_name = cmd[0].replace('_audio', '')
                direction = MotorDirection(direction_name)
                motors.tank_steering(direction, speed, audio_mode="reduce_interference")
            elif cmd[0] in ['forward_mute', 'backward_mute', 'left_mute', 'right_mute']:
                speed = int(cmd[1]) if len(cmd) > 1 else 50
                direction_name = cmd[0].replace('_mute', '')
                direction = MotorDirection(direction_name)
                motors.tank_steering(direction, speed, audio_mode="mute")
            elif cmd[0] == 'mute_on':
                motors.mute_audio(True)
            elif cmd[0] == 'mute_off':
                motors.mute_audio(False)
            elif cmd[0] == 'relay_status':
                motors.check_audio_relay_status()
            elif cmd[0] == 'test_working_method':
                if motors.test_working_method():
                    print("🎉 MAX4544 confirmed working!")
                else:
                    print("❌ MAX4544 still not working - check hardware")
            elif cmd[0] == 'relay_toggle':
                print("🔄 Testing relay positions...")
                motors.check_audio_relay_status()
                input("   Press Enter to toggle relay...")
                motors.mute_audio(True)
                time.sleep(1)
                input("   Press Enter to toggle back...")
                motors.mute_audio(False)
                motors.check_audio_relay_status()
            elif cmd[0] == 'status':
                print(f"🔧 Left Motor Speed: {motors.left_speed}%")
                print(f"🔧 Right Motor Speed: {motors.right_speed}%")
                print(f"🚗 Moving: {motors.is_moving}")
                print(f"🎵 PWM Frequency: {motors._current_pwm_freq}Hz")
            else:
                print("❌ Unknown command")
                
    except KeyboardInterrupt:
        print("\n⏹️ Emergency stop!")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        motors.cleanup()

if __name__ == "__main__":
    interactive_test()