#!/usr/bin/env python3
"""
config/pins.py - Single source of truth for ALL pin assignments
Based on verified working hardware configuration
"""

class TreatBotPins:
    """Pin assignments for TreatSensei robot - NEVER MODIFY WITHOUT TESTING"""
    
    # Motor Control - L298N Motor Driver
    # CORRECTED MAPPING: Motor A = LEFT motor, Motor B = RIGHT motor
    MOTOR_IN1 = 17  # GPIO17 (Pin 11) - Motor A Direction 1 (LEFT MOTOR)
    MOTOR_IN2 = 18  # GPIO18 (Pin 12) - Motor A Direction 2 (LEFT MOTOR)
    MOTOR_IN3 = 27  # GPIO27 (Pin 13) - Motor B Direction 1 (RIGHT MOTOR)
    MOTOR_IN4 = 22  # GPIO22 (Pin 15) - Motor B Direction 2 (RIGHT MOTOR)
    MOTOR_ENA = 13  # GPIO13 (Pin 33) - Motor A Enable/Speed PWM (LEFT MOTOR)
    MOTOR_ENB = 19  # GPIO19 (Pin 35) - Motor B Enable/Speed PWM (RIGHT MOTOR)

    # Encoder Pins - DFRobot Motors with Built-in Encoders
    # CORRECTED MAPPING: A1/B1 = LEFT motor, A2/B2 = RIGHT motor
    ENCODER_A1 = 4   # GPIO4 (Pin 7) - Motor A (LEFT) Encoder A (GREEN wire)
    ENCODER_B1 = 23  # GPIO23 (Pin 16) - Motor A (LEFT) Encoder B (YELLOW wire)
    ENCODER_A2 = 5   # GPIO5 (Pin 29) - Motor B (RIGHT) Encoder A (GREEN wire)
    ENCODER_B2 = 6   # GPIO6 (Pin 31) - Motor B (RIGHT) Encoder B (YELLOW wire)

    # LED Control
    NEOPIXEL = 12    # GPIO12 (Pin 32) - NeoPixel ring data
    BLUE_LED = 25    # GPIO25 (Pin 22) - Blue LED strip
    
    # I2C Bus - PCA9685 Servo Controller + Camera
    I2C_SDA = 2      # GPIO2 (Pin 3) - I2C Data
    I2C_SCL = 3      # GPIO3 (Pin 5) - I2C Clock
    
    # Available pins for future expansion
    FREE_PINS = [14, 15, 16, 20, 21, 26]  # GPIO numbers available for sensors, etc.
    # Note: GPIO5, GPIO6 now used for Motor 2 encoders

    @classmethod
    def validate_pins(cls):
        """Validate no pin conflicts exist"""
        used_pins = [
            cls.MOTOR_IN1, cls.MOTOR_IN2, cls.MOTOR_IN3, cls.MOTOR_IN4,
            cls.MOTOR_ENA, cls.MOTOR_ENB,
            cls.ENCODER_A1, cls.ENCODER_B1, cls.ENCODER_A2, cls.ENCODER_B2,
            cls.NEOPIXEL, cls.BLUE_LED, cls.I2C_SDA, cls.I2C_SCL
        ]
        
        if len(used_pins) != len(set(used_pins)):
            raise ValueError("Pin conflict detected!")
        
        print(f"Pin validation passed - {len(used_pins)} pins assigned")
        return True

# Validate on import
TreatBotPins.validate_pins()
