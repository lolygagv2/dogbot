 Test DFPlayer Pro via UART
import serial
import time

class DFPlayerController:
    def __init__(self):
        # Your documented pins
        self.serial = serial.Serial('/dev/serial0', 
                                   baudrate=115200,
                                   timeout=1)
        self.relay_pin = 37  # Audio source switcher
        GPIO.setup(self.relay_pin, GPIO.OUT)
    
    def play_sound(self, file_num):
        """Play MP3 from SD card"""
        cmd = f"AT+PLAYNUM={file_num}\r\n"
        self.serial.write(cmd.encode())
        
    def set_volume(self, vol):
        """Volume 0-30"""
        cmd = f"AT+VOL={vol}\r\n"
        self.serial.write(cmd.encode())
    
    def switch_to_dfplayer(self):
        """Use relay to switch speaker to DFPlayer"""
        GPIO.output(self.relay_pin, GPIO.LOW)
    
    def switch_to_respeaker(self):
        """Switch speaker to ReSpeaker for voice"""
        GPIO.output(self.relay_pin, GPIO.HIGH)

# Test sequence
player = DFPlayerController()
player.set_volume(20)
player.switch_to_dfplayer()
player.play_sound(1)  # Play first MP3
