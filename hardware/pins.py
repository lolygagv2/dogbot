# treatbot_pins.py - ALWAYS USE THIS
class TreatBotPins:
    """Single source of truth for ALL pin assignments"""
    
    # Audio Control
    RELAY_AUDIO_SWITCH = 16  # GPIO16 (Pin 36) - DPDT Relay for audio switching
    
    # Motor Control  
    MOTOR_IN1 = 17  # GPIO17 (Pin 11)
    MOTOR_IN2 = 18  # GPIO18 (Pin 12)
    MOTOR_IN3 = 27  # GPIO27 (Pin 13)
    MOTOR_IN4 = 22  # GPIO22 (Pin 15)
    MOTOR_ENA = 13  # GPIO13 (Pin 33)
    MOTOR_ENB = 19  # GPIO19 (Pin 35)
    
    # Serial/UART
    DFPLAYER_TX = 14  # GPIO14 (Pin 8) -> DFPlayer RX
    DFPLAYER_RX = 15  # GPIO15 (Pin 10) -> DFPlayer TX
    
    # LEDs
    NEOPIXEL = 12  # GPIO12 (Pin 32)
    BLUE_LED = 25  # GPIO25 (Pin 22) maybe?
