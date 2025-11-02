#!/usr/bin/env python3
"""
config/settings.py - System configuration and constants
"""

class SystemSettings:
    """System-wide configuration settings"""
    
    # Motor Configuration
    PWM_FREQUENCY = 500  # Optimized to reduce audio whine
    DEFAULT_MOTOR_SPEED = 50  # Default speed percentage
    MOTOR_TIMEOUT = 10.0  # Max time for motor operations (safety)
    
    # Audio Configuration  
    DFPLAYER_BAUD_RATE = 115200
    DFPLAYER_TIMEOUT = 1.0
    DEFAULT_VOLUME = 23  # DFPlayer volume (0-30)
    AUDIO_COMMAND_DELAY = 0.1  # Delay between AT commands
    
    # LED Configuration
    NEOPIXEL_COUNT = 75
    NEOPIXEL_BRIGHTNESS = 0.3  # 0.1 to 1.0
    LED_ANIMATION_SPEED = 0.05  # Default animation delay
    
    # Servo Configuration
    SERVO_FREQUENCY = 50  # Standard servo PWM frequency
    SERVO_MIN_PULSE = 500   # Extended range for wider movement
    SERVO_MAX_PULSE = 2500  # Extended range for wider movement
    SERVO_CENTER_PULSE = 1500  # Microseconds
    
    # Camera Configuration (future)
    CAMERA_FPS = 30
    CAMERA_RESOLUTION = (640, 480)
    
    # AI Configuration (future)
    AI_CONFIDENCE_THRESHOLD = 0.7
    POSE_DETECTION_FPS = 10
    BARK_DETECTION_SENSITIVITY = 0.8
    
    # System Safety
    EMERGENCY_STOP_TIMEOUT = 0.5  # Max time to execute emergency stop
    SYSTEM_HEALTH_CHECK_INTERVAL = 5.0  # Seconds
    
    # File Paths
    AUDIO_FILES_PATH = "/media/audio/"
    LOG_FILE_PATH = "/var/log/treatsensei.log"
    CONFIG_FILE_PATH = "/etc/treatsensei/config.json"

class AudioFiles:
    """DFPlayer audio file mappings"""
    
   # ===== TALKS FOLDER =====
    # System/intro sounds
    SCOOBY_INTRO =  "/talks/0001.mp3"  # Scooby Snacks INTRO
  
     # Dog names/calls
    ELSA =          "/talks/0003.mp3"  # Elsa
    BEZIK =         "/talks/0004.mp3"  # Bezik
    BEZIK_COME =    "/talks/0005.mp3"  # Bezik Come
    ELSA_COME =     "/talks/0006.mp3"  # Elsa COME
    DOGS_COME =     "/talks/0007.mp3"  # Bezik Elsa Come (both dogs)

    # Positive reinforcement
    GOOD_DOG =      "/talks/0008.mp3"  # GOOD
    KAHNSHIK =      "/talks/0009.mp3"  # KAHNSHIK (good/praise?)

    # Commands
    LIE_DOWN =      "/talks/0010.mp3"  # Lie Down
    QUIET =         "/talks/0011.mp3"  # Quiet
    NO =            "/talks/0012.mp3"  # No
    TREAT =         "/talks/0013.mp3"  # Treat
    KOKOMA =        "/talks/0014.mp3"  # Kokoma potatoe
    SIT =           "/talks/0015.mp3"  # Sit
    SPIN =          "/talks/0016.mp3"  # Spin
    STAY =          "/talks/0017.mp3"  # Stay

    # ===== 02 AUDIO FOLDER =====
    # Background music
    MOZART_CONCERTO =   "/02/0019.mp3"  # Mozart: Concerto for flute, h
    MOZART_PIANO =      "/02/0018.mp3"  # Mozart: Piano concerto No. 26
    MILKSHAKE =         "/02/0020.mp3"  # Milkshake
    YUMMY =             "/02/0021.mp3"  # Justin Bieber - Yummy (Lyric Video)
    HUNGRY =            "/02/0022.mp3"  # Duran Duran - Hungry like the Wolf [Audio]
    DNCE =              "/02/0023.mp3"  # DNCE - Cake By The Ocean (Lyrics)
    DOGS_OUT =          "/02/0024.mp3"  # Baha Men - Who Let the Dogs Out Lyrics
    SCOOBY_SNACKS =     "/02/0030.mp3"  # Scooby Snacks

    # System FX
    PROGRESS_SCAN =     "/02/0025.mp3"  # Progress minutes of scanning
    ROBO_SCAN =         "/02/0026.mp3"  # RobotScanning
    DOOR_SCAN =         "/02/0027.mp3"  # Short Door Scan beep
    HI_SCAN =           "/02/0028.mp3"  # A load high pitch scan
    BUSY_SCAN =         "/02/0029.mp3"  # A busy noisey scan - like 30 seconds or so


    @classmethod
    def get_file_number(cls, filename):
        """Extract file number from filename for PLAYNUM command"""
        if "0001" in filename:
            return 1
        elif "0002" in filename:
            return 2
        # Add more mappings as needed
        return 1  # Default

class Colors:
    """RGB color definitions for LEDs"""
    
    # Basic colors
    OFF = (0, 0, 0)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    
    # Status colors
    IDLE = (30, 30, 30)        # Dim white
    SEARCHING = (0, 255, 255)  # Cyan
    DOG_DETECTED = (0, 255, 0) # Green
    TREAT_LAUNCH = (255, 255, 255)  # Bright white
    ERROR = (255, 0, 0)        # Red
    CHARGING = (255, 165, 0)   # Orange
    
    # Special colors
    PURPLE = (128, 0, 128)
    YELLOW = (255, 255, 0)
    PINK = (255, 192, 203)
    WARM_WHITE = (255, 180, 120)
