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
    
    # Audio Configuration (USB Audio via pygame)
    DEFAULT_VOLUME = 70  # USB audio volume (0-100)
    AUDIO_SAMPLE_RATE = 22050  # Sample rate for pygame mixer
    
    # LED Configuration (Adafruit 332 LED/m strip, 0.5m = 165 LEDs)
    NEOPIXEL_COUNT = 165
    NEOPIXEL_BRIGHTNESS = 0.25  # Reduced for power safety with 165 LEDs
    NEOPIXEL_PIN = 10  # GPIO 10 (SPI MOSI) - required for Pi 5
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
    """USB Audio file mappings for pygame-based playback"""

   # ===== TALKS FOLDER (Voice Commands & Dog Names) =====
    # System/intro sounds
    SCOOBY_INTRO =  "/home/morgan/dogbot/VOICEMP3/talks/scooby_intro.mp3"  # Scooby Snacks INTRO

    # Dog names/calls
    ELSA =          "/home/morgan/dogbot/VOICEMP3/talks/elsa.mp3"  # Elsa
    BEZIK =         "/home/morgan/dogbot/VOICEMP3/talks/bezik.mp3"  # Bezik
    BEZIK_COME =    "/home/morgan/dogbot/VOICEMP3/talks/bezik_come.mp3"  # Bezik Come
    ELSA_COME =     "/home/morgan/dogbot/VOICEMP3/talks/elsa_come.mp3"  # Elsa COME
    DOGS_COME =     "/home/morgan/dogbot/VOICEMP3/talks/dogs_come.mp3"  # Bezik Elsa Come (both dogs)

    # Positive reinforcement
    GOOD_DOG =      "/home/morgan/dogbot/VOICEMP3/talks/good_dog.mp3"  # GOOD
    KAHNSHIK =      "/home/morgan/dogbot/VOICEMP3/talks/kahnshik.mp3"  # KAHNSHIK (good/praise?)

    # Commands
    LIE_DOWN =      "/home/morgan/dogbot/VOICEMP3/talks/lie_down.mp3"  # Lie Down
    QUIET =         "/home/morgan/dogbot/VOICEMP3/talks/quiet.mp3"  # Quiet
    NO =            "/home/morgan/dogbot/VOICEMP3/talks/no.mp3"  # No
    TREAT =         "/home/morgan/dogbot/VOICEMP3/talks/treat.mp3"  # Treat
    KOKOMA =        "/home/morgan/dogbot/VOICEMP3/talks/kokoma.mp3"  # Kokoma potato
    SIT =           "/home/morgan/dogbot/VOICEMP3/talks/sit.mp3"  # Sit
    SPIN =          "/home/morgan/dogbot/VOICEMP3/talks/spin.mp3"  # Spin
    STAY =          "/home/morgan/dogbot/VOICEMP3/talks/stay.mp3"  # Stay

    # ===== SONGS FOLDER (Background Music & Entertainment) =====
    # Background music
    MOZART_CONCERTO =   "/home/morgan/dogbot/VOICEMP3/songs/mozart_concerto.mp3"  # Mozart: Concerto for flute, h
    MOZART_PIANO =      "/home/morgan/dogbot/VOICEMP3/songs/mozart_piano.mp3"  # Mozart: Piano concerto No. 26
    MILKSHAKE =         "/home/morgan/dogbot/VOICEMP3/songs/milkshake.mp3"  # Milkshake
    YUMMY =             "/home/morgan/dogbot/VOICEMP3/songs/yummy.mp3"  # Justin Bieber - Yummy (Lyric Video)
    HUNGRY =            "/home/morgan/dogbot/VOICEMP3/songs/hungry_like_wolf.mp3"  # Duran Duran - Hungry like the Wolf [Audio]
    DNCE =              "/home/morgan/dogbot/VOICEMP3/songs/cake_by_ocean.mp3"  # DNCE - Cake By The Ocean (Lyrics)
    DOGS_OUT =          "/home/morgan/dogbot/VOICEMP3/songs/who_let_dogs_out.mp3"  # Baha Men - Who Let the Dogs Out Lyrics
    SCOOBY_SNACKS =     "/home/morgan/dogbot/VOICEMP3/songs/scooby_snacks.mp3"  # Scooby Snacks

    # System FX
    PROGRESS_SCAN =     "/home/morgan/dogbot/VOICEMP3/songs/progress_scan.mp3"  # Progress minutes of scanning
    ROBO_SCAN =         "/home/morgan/dogbot/VOICEMP3/songs/robot_scan.mp3"  # RobotScanning
    DOOR_SCAN =         "/home/morgan/dogbot/VOICEMP3/songs/door_scan.mp3"  # Short Door Scan beep
    HI_SCAN =           "/home/morgan/dogbot/VOICEMP3/songs/hi_scan.mp3"  # A load high pitch scan
    BUSY_SCAN =         "/home/morgan/dogbot/VOICEMP3/songs/busy_scan.mp3"  # A busy noisey scan - like 30 seconds or so


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
    WARNING = (255, 200, 0)    # Amber/Yellow for warnings
