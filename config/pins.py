#!/usr/bin/env python3
"""
config/pins.py - Single source of truth for ALL pin assignments
Based on verified working hardware configuration
"""

class TreatBotPins:
    """Pin assignments for TreatSensei robot - NEVER MODIFY WITHOUT TESTING"""
    
    # Audio Control - MAX4544 analog switches
    RELAY_AUDIO_SWITCH = 16  # GPIO16 (Pin 36) - Confirmed working with gpioset
    
    # Motor Control - L298N Motor Driver
    MOTOR_IN1 = 17  # GPIO17 (Pin 11) - Motor A Direction 1
    MOTOR_IN2 = 18  # GPIO18 (Pin 12) - Motor A Direction 2
    MOTOR_IN3 = 27  # GPIO27 (Pin 13) - Motor B Direction 1  
    MOTOR_IN4 = 22  # GPIO22 (Pin 15) - Motor B Direction 2
    MOTOR_ENA = 13  # GPIO13 (Pin 33) - Motor A Enable/Speed PWM
    MOTOR_ENB = 19  # GPIO19 (Pin 35) - Motor B Enable/Speed PWM
    
    # Serial/UART - DFPlayer Pro
    DFPLAYER_TX = 14  # GPIO14 (Pin 8) -> DFPlayer RX (Orange wire)
    DFPLAYER_RX = 15  # GPIO15 (Pin 10) -> DFPlayer TX (Yellow wire)
    
    # LED Control
    NEOPIXEL = 12    # GPIO12 (Pin 32) - NeoPixel ring data
    BLUE_LED = 25    # GPIO25 (Pin 22) - Blue LED strip
    
    # I2C Bus - PCA9685 Servo Controller + Camera
    I2C_SDA = 2      # GPIO2 (Pin 3) - I2C Data
    I2C_SCL = 3      # GPIO3 (Pin 5) - I2C Clock
    
    # Available pins for future expansion
    FREE_PINS = [5, 6, 20, 21, 26]  # GPIO numbers available for sensors, etc.
    
    @classmethod
    def validate_pins(cls):
        """Validate no pin conflicts exist"""
        used_pins = [
            cls.RELAY_AUDIO_SWITCH, cls.MOTOR_IN1, cls.MOTOR_IN2, 
            cls.MOTOR_IN3, cls.MOTOR_IN4, cls.MOTOR_ENA, cls.MOTOR_ENB,
            cls.DFPLAYER_TX, cls.DFPLAYER_RX, cls.NEOPIXEL, cls.BLUE_LED,
            cls.I2C_SDA, cls.I2C_SCL
        ]
        
        if len(used_pins) != len(set(used_pins)):
            raise ValueError("Pin conflict detected!")
        
        print(f"Pin validation passed - {len(used_pins)} pins assigned")
        return True

# Validate on import
TreatBotPins.validate_pins()
