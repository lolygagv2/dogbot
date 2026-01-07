"""
Robot Configuration Loader
Loads robot-specific settings from YAML profiles.
Automatically detects which robot it's running on.
"""

import os
import yaml
import socket
from pathlib import Path


class RobotConfig:
    """
    Singleton configuration loader for robot-specific settings.

    Usage:
        from config.config_loader import RobotConfig
        config = RobotConfig()

        # Access values
        duration = config.dispenser.dispense_duration
        pulse = config.servo.slow_pulse
        max_speed = config.controller.max_speed
        deadzone = config.controller.xbox_deadzone
        kp = config.controller.pid_kp
    """

    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """Load configuration from YAML file based on robot profile."""
        profile_name = self._detect_profile()
        config_path = self._get_config_path(profile_name)

        if not config_path.exists():
            raise FileNotFoundError(
                f"Robot profile not found: {config_path}\n"
                f"Set ROBOT_PROFILE environment variable or create the profile."
            )

        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)

        print(f"[Config] Loaded profile: {self._config.get('robot_id', 'unknown')}")
        print(f"[Config] Description: {self._config.get('description', 'No description')}")

    def _detect_profile(self) -> str:
        """
        Detect which robot profile to use.
        Priority:
        1. ROBOT_PROFILE environment variable
        2. Hostname mapping
        3. /etc/robot_id file
        4. Default to 'treatbot'
        """
        # Check environment variable first
        env_profile = os.environ.get('ROBOT_PROFILE')
        if env_profile:
            print(f"[Config] Using profile from ROBOT_PROFILE env: {env_profile}")
            return env_profile

        # Check hostname
        hostname = socket.gethostname().lower()
        hostname_map = {
            'treatbot': 'treatbot',
            'treatbot1': 'treatbot',
            'treatbot2': 'treatbot2',
            'wimz-alpha': 'treatbot',
            'wimz-beta': 'treatbot2',
        }
        if hostname in hostname_map:
            print(f"[Config] Using profile from hostname '{hostname}': {hostname_map[hostname]}")
            return hostname_map[hostname]

        # Check /etc/robot_id file
        robot_id_file = Path('/etc/robot_id')
        if robot_id_file.exists():
            profile = robot_id_file.read_text().strip()
            print(f"[Config] Using profile from /etc/robot_id: {profile}")
            return profile

        # Default
        print("[Config] No profile specified, defaulting to 'treatbot'")
        return 'treatbot'

    def _get_config_path(self, profile_name: str) -> Path:
        """Get the full path to the config file."""
        possible_paths = [
            Path(__file__).parent / 'robot_profiles' / f'{profile_name}.yaml',
            Path('config/robot_profiles') / f'{profile_name}.yaml',
            Path('/home/morgan/dogbot/config/robot_profiles') / f'{profile_name}.yaml',
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return possible_paths[0]

    # --- Camera Settings ---
    @property
    def camera(self):
        return CameraConfig(self._config.get('camera', {}))

    # --- Dispenser Settings ---
    @property
    def dispenser(self):
        return DispenserConfig(self._config.get('dispenser', {}))

    # --- Servo Settings ---
    @property
    def servo(self):
        return ServoConfig(self._config.get('servo', {}))

    # --- Controller Settings ---
    @property
    def controller(self):
        return ControllerConfig(self._config.get('controller', {}))

    # --- Raw Access ---
    @property
    def robot_id(self) -> str:
        return self._config.get('robot_id', 'unknown')

    @property
    def raw(self) -> dict:
        return self._config


class CameraConfig:
    """Camera-related configuration."""

    def __init__(self, config: dict):
        self._config = config

    @property
    def rotation(self) -> int:
        """Camera rotation in degrees (0, 90, 180, 270). Default 90 for backward compat."""
        return self._config.get('rotation', 90)


class DispenserConfig:
    """Dispenser-related configuration."""

    def __init__(self, config: dict):
        self._config = config

    @property
    def dispense_duration(self) -> float:
        return self._config.get('dispense_duration', 0.12)


class ServoConfig:
    """Servo-related configuration."""

    def __init__(self, config: dict):
        self._config = config

    @property
    def slow_pulse(self) -> int:
        return self._config.get('slow_pulse', 1590)


class ControllerConfig:
    """Controller/motor-related configuration."""

    def __init__(self, config: dict):
        self._config = config

    @property
    def max_speed(self) -> int:
        return self._config.get('max_speed', 72)

    @property
    def turn_speed_factor(self) -> float:
        return self._config.get('turn_speed_factor', 0.8)

    @property
    def max_rpm(self) -> int:
        return self._config.get('max_rpm', 110)

    @property
    def use_pid_control(self) -> bool:
        return self._config.get('use_pid_control', True)

    @property
    def min_pwm_threshold(self) -> int:
        return self._config.get('min_pwm_threshold', 0)

    @property
    def left_motor_multiplier(self) -> float:
        calibration = self._config.get('motor_calibration', {})
        return calibration.get('left_multiplier', 1.0)

    @property
    def right_motor_multiplier(self) -> float:
        calibration = self._config.get('motor_calibration', {})
        return calibration.get('right_multiplier', 1.0)

    # Xbox controller settings
    @property
    def xbox_deadzone(self) -> float:
        xbox = self._config.get('xbox', {})
        return xbox.get('joystick_deadzone', 0.1)

    @property
    def xbox_scale(self) -> float:
        xbox = self._config.get('xbox', {})
        return xbox.get('joystick_scale', 1.0)

    # PID settings
    @property
    def pid_kp(self) -> float:
        pid = self._config.get('pid', {})
        return pid.get('kp', 1.0)

    @property
    def pid_ki(self) -> float:
        pid = self._config.get('pid', {})
        return pid.get('ki', 0.0)

    @property
    def pid_kd(self) -> float:
        pid = self._config.get('pid', {})
        return pid.get('kd', 0.0)


def get_config() -> RobotConfig:
    """Get the robot configuration singleton."""
    return RobotConfig()
