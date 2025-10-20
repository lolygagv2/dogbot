#!/usr/bin/env python3
"""
core/audio_controller.py - DFPlayer Pro and audio relay management
Refactored from proven DFPlayer commands and MAX4544 switching
"""

import serial
import subprocess
import time

# Import configuration
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.pins import TreatBotPins
from config.settings import SystemSettings, AudioFiles

class AudioController:
    """DFPlayer Pro control and MAX4544 audio path switching"""
    
    def __init__(self):
        self.pins = TreatBotPins()
        self.settings = SystemSettings()
        
        # Connection states
        self.serial_connection = None
        self.relay_available = False
        
        # Initialize subsystems
        self._initialize_dfplayer()
        self._initialize_relay()
    
    def _initialize_dfplayer(self):
        """Initialize DFPlayer Pro using proven serial configuration"""
        try:
            # Use proven baud rate and settings from working script
            self.serial_connection = serial.Serial(
                '/dev/ttyAMA0', 
                self.settings.DFPLAYER_BAUD_RATE, 
                timeout=self.settings.DFPLAYER_TIMEOUT
            )
            time.sleep(2)  # Critical delay for DFPlayer initialization
            
            # Apply proven configuration sequence
            self._send_command('AT+VOL=22')      # Set volume
            self._send_command('AT+LED=OFF')     # Turn off LED
            self._send_command('AT+PLAYMODE=3')  # Play one song and pause
            self._send_command('AT+PROMPT=OFF')  # Turn off prompt tone
            self._send_command('AT+AMP=ON')      # Turn on amplifier
            
            print("DFPlayer Pro initialized successfully")
            return True
            
        except Exception as e:
            print(f"DFPlayer initialization failed: {e}")
            self.serial_connection = None
            return False
    
    def _initialize_relay(self):
        """Initialize MAX4544 audio relay using proven gpioset method"""
        try:
            # Test the proven gpioset method that works
            result = subprocess.run(
                ['gpioset', 'gpiochip0', f'{self.pins.RELAY_AUDIO_SWITCH}=0'], 
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                self.relay_available = True
                print(f"MAX4544 audio relay initialized on GPIO{self.pins.RELAY_AUDIO_SWITCH} (gpioset method)")
                return True
            else:
                print(f"Audio relay initialization failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"Audio relay initialization failed: {e}")
            return False
    
    def _send_command(self, command):
        """Send AT command to DFPlayer with proven timing"""
        if not self.serial_connection:
            print("DFPlayer not connected")
            return False
            
        try:
            # Use proven command format
            full_command = f"{command}\r\n"
            self.serial_connection.write(full_command.encode())
            time.sleep(self.settings.AUDIO_COMMAND_DELAY)  # Proven delay
            
            # Read response (optional)
            if self.serial_connection.in_waiting > 0:
                response = self.serial_connection.read(100)
                print(f"DFPlayer response: {response}")
            
            return True
            
        except Exception as e:
            print(f"DFPlayer command error '{command}': {e}")
            return False
    
    def play_sound(self, sound_name):
            """Play sound by name, using settings.py mappings"""
            # Check if it's an attribute of AudioFiles
            if hasattr(AudioFiles, sound_name.upper()):
                filepath = getattr(AudioFiles, sound_name.upper())
                return self.play_file_by_path(filepath)
            
            # Direct path fallback
            if sound_name.startswith('/'):
                return self.play_file_by_path(sound_name)
                
            print(f"Unknown sound: {sound_name}")
            return False
  
    def play_file_by_path(self, filepath):
        """Play specific audio file by path (proven method)"""
        command = f'AT+PLAYFILE={filepath}'
        success = self._send_command(command)
        if success:
            print(f"Playing audio file: {filepath}")
        return success
    
    def play_file_by_number(self, number):
        """Play file by number (proven method)"""
        command = f'AT+PLAYNUM={number}'
        success = self._send_command(command)
        if success:
            print(f"Playing audio file number: {number}")
        return success
    
    def set_volume(self, volume):
        """Set DFPlayer volume (0-30, proven range)"""
        volume = max(0, min(30, volume))
        command = f'AT+VOL={volume}'
        success = self._send_command(command)
        if success:
            print(f"Volume set to: {volume}")
        return success
    
    def play_pause_toggle(self):
        """Toggle play/pause (proven command)"""
        return self._send_command('AT+PLAY=PP')
    
    def play_next(self):
        """Play next track"""
        return self._send_command('AT+PLAY=NEXT')
    
    def play_previous(self):
        """Play previous track"""
        return self._send_command('AT+PLAY=LAST')
    
    def query_current_file(self):
        """Query currently playing file"""
        return self._send_command('AT+QUERY=5')
    
    def query_play_time(self):
        """Query current play time"""
        return self._send_command('AT+QUERY=3')
    
    def switch_to_pi_audio(self):
        """Switch MAX4544 to Raspberry Pi USB audio path"""
        if not self.relay_available:
            print("Audio relay not available")
            return False
            
        try:
            # Use proven gpioset HIGH command
            result = subprocess.run(
                ['gpioset', 'gpiochip0', f'{self.pins.RELAY_AUDIO_SWITCH}=1'], 
                capture_output=True, text=True, check=True
            )
            print("Audio switched to Pi USB Audio")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Audio switch to Pi failed: {e}")
            return False
        except Exception as e:
            print(f"Audio switch error: {e}")
            return False
    
    def switch_to_dfplayer(self):
        """Switch MAX4544 to DFPlayer audio path (default)"""
        if not self.relay_available:
            print("Audio relay not available")
            return False
            
        try:
            # Use proven gpioset LOW command  
            result = subprocess.run(
                ['gpioset', 'gpiochip0', f'{self.pins.RELAY_AUDIO_SWITCH}=0'], 
                capture_output=True, text=True, check=True
            )
            print("Audio switched to DFPlayer")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Audio switch to DFPlayer failed: {e}")
            return False
        except Exception as e:
            print(f"Audio switch error: {e}")
            return False
    
    def get_relay_status(self):
        """Check current MAX4544 relay position using proven gpioget"""
        if not self.relay_available:
            return None
            
        try:
            result = subprocess.run(
                ['gpioget', 'gpiochip0', str(self.pins.RELAY_AUDIO_SWITCH)], 
                capture_output=True, text=True, check=True
            )
            
            current_state = int(result.stdout.strip())
            status = {
                'gpio_state': current_state,
                'audio_path': 'Pi USB Audio' if current_state else 'DFPlayer',
                'relay_working': True
            }
            
            print(f"Audio relay status: GPIO{self.pins.RELAY_AUDIO_SWITCH}={'HIGH' if current_state else 'LOW'} ({status['audio_path']})")
            return status
            
        except Exception as e:
            print(f"Cannot read relay status: {e}")
            return {'relay_working': False, 'error': str(e)}
    
    def test_relay_switching(self):
        """Test MAX4544 switching functionality"""
        if not self.relay_available:
            print("Relay not available for testing")
            return False
            
        print("Testing MAX4544 audio relay switching...")
        print("Make sure audio is playing, then observe switching...")
        
        try:
            # Test sequence
            print("1. Setting to DFPlayer...")
            self.switch_to_dfplayer()
            time.sleep(1)
            
            print("2. Setting to Pi USB Audio...")
            self.switch_to_pi_audio()
            time.sleep(1)
            
            print("3. Back to DFPlayer...")
            self.switch_to_dfplayer()
            
            print("Relay test complete!")
            return True
            
        except Exception as e:
            print(f"Relay test failed: {e}")
            return False
    
    
        """Play treat dispensed confirmation"""
        return self.play_file_by_path(AudioFiles.TREAT_DISPENSED)
    
    def get_status(self):
        """Get comprehensive audio system status"""
        return {
            'dfplayer_connected': self.serial_connection is not None,
            'relay_available': self.relay_available,
            'relay_status': self.get_relay_status() if self.relay_available else None
        }
    
    def is_initialized(self):
        """Check if audio system is properly initialized"""
        return self.serial_connection is not None and self.relay_available
    
    def cleanup(self):
        """Clean shutdown of audio system"""
        print("Cleaning up audio controller...")
        
        # Stop any playing audio
        if self.serial_connection:
            try:
                self._send_command('AT+PLAY=PP')  # Pause
                self.serial_connection.close()
                self.serial_connection = None
                print("DFPlayer connection closed")
            except Exception as e:
                print(f"DFPlayer cleanup error: {e}")
        
        # Set relay to default position (DFPlayer)
        if self.relay_available:
            try:
                self.switch_to_dfplayer()
                print("Audio relay set to default position")
            except Exception as e:
                print(f"Relay cleanup error: {e}")

# Test function for individual module testing
def test_audio():
    """Simple test function for audio controller"""
    print("Testing Audio Controller...")
    
    audio = AudioController()
    if not audio.is_initialized():
        print("Audio initialization incomplete!")
        return
    
    try:
        # Test DFPlayer
        print("Testing DFPlayer...")
        audio.set_volume(15)
        audio.play_file_by_number(1)
        time.sleep(2)
        
        # Test relay switching
        print("Testing relay switching...")
        audio.test_relay_switching()
        
        # Test status
        status = audio.get_status()
        print(f"Audio system status: {status}")
        
        print("Audio test complete!")
        
    except KeyboardInterrupt:
        print("Test interrupted")
    finally:
        audio.cleanup()

if __name__ == "__main__":
    test_audio()
