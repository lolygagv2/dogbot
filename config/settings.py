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
    # Dog names/calls (dog-specific, stay in root talks folder)
    ELSA =          "/home/morgan/dogbot/VOICEMP3/talks/elsa.mp3"  # Elsa
    BEZIK =         "/home/morgan/dogbot/VOICEMP3/talks/bezik.mp3"  # Bezik
    BEZIK_COME =    "/home/morgan/dogbot/VOICEMP3/talks/bezik_come.mp3"  # Bezik Come
    ELSA_COME =     "/home/morgan/dogbot/VOICEMP3/talks/elsa_come.mp3"  # Elsa COME
    DOGS_COME =     "/home/morgan/dogbot/VOICEMP3/talks/dogs_come.mp3"  # Bezik Elsa Come (both dogs)
    DOG_0 =         "/home/morgan/dogbot/VOICEMP3/talks/dog_0.mp3"  # Unknown dog fallback

    # Positive reinforcement (default folder)
    GOOD_DOG =      "/home/morgan/dogbot/VOICEMP3/talks/default/good.mp3"  # GOOD
    KAHNSHIK =      "/home/morgan/dogbot/VOICEMP3/talks/default/kahnshik.mp3"  # KAHNSHIK (Korean praise)

    # Commands (default folder)
    COME =          "/home/morgan/dogbot/VOICEMP3/talks/default/come.mp3"  # Come
    DOWN =          "/home/morgan/dogbot/VOICEMP3/talks/default/down.mp3"  # Down/Lie Down
    QUIET =         "/home/morgan/dogbot/VOICEMP3/talks/default/quiet.mp3"  # Quiet
    NO =            "/home/morgan/dogbot/VOICEMP3/talks/default/no.mp3"  # No
    TREAT =         "/home/morgan/dogbot/VOICEMP3/talks/default/treat.mp3"  # Treat
    SIT =           "/home/morgan/dogbot/VOICEMP3/talks/default/sit.mp3"  # Sit
    SPIN =          "/home/morgan/dogbot/VOICEMP3/talks/default/spin.mp3"  # Spin
    STAY =          "/home/morgan/dogbot/VOICEMP3/talks/default/stay.mp3"  # Stay
    SPEAK =         "/home/morgan/dogbot/VOICEMP3/talks/default/speak.mp3"  # Speak
    CROSSES =       "/home/morgan/dogbot/VOICEMP3/talks/default/crosses.mp3"  # Cross paws

    # ===== SONGS FOLDER (Background Music & Entertainment) =====
    # Background music (default folder)
    MOZART_CONCERTO =   "/home/morgan/dogbot/VOICEMP3/songs/default/mozart_concerto.mp3"
    MOZART_PIANO =      "/home/morgan/dogbot/VOICEMP3/songs/default/mozart_piano.mp3"
    MILKSHAKE =         "/home/morgan/dogbot/VOICEMP3/songs/default/milkshake.mp3"
    YUMMY =             "/home/morgan/dogbot/VOICEMP3/songs/default/yummy.mp3"
    HUNGRY =            "/home/morgan/dogbot/VOICEMP3/songs/default/hungry_like_wolf.mp3"
    DNCE =              "/home/morgan/dogbot/VOICEMP3/songs/default/cake_by_ocean.mp3"
    DOGS_OUT =          "/home/morgan/dogbot/VOICEMP3/songs/default/who_let_dogs_out.mp3"
    SCOOBY_SNACKS =     "/home/morgan/dogbot/VOICEMP3/songs/default/scooby_snacks.mp3"
    WIMZ_THEME =        "/home/morgan/dogbot/VOICEMP3/songs/default/Wimz_theme.mp3"
    TOKYO =             "/home/morgan/dogbot/VOICEMP3/songs/default/3lau_tokyo.mp3"
    OCEAN_EYES =        "/home/morgan/dogbot/VOICEMP3/songs/default/eilish_ocean_eyes.mp3"
    EDM_REMIXES =       "/home/morgan/dogbot/VOICEMP3/songs/default/edm_remixes.mp3"


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
