#!/usr/bin/env python3
"""
Unified Mission API for TreatBot Training
All training scripts use the same API
"""

import time
import json
import yaml
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
import logging

# Import hardware interfaces
try:
    from hardware.treat_dispenser import TreatDispenser
    TREAT_DISPENSER_AVAILABLE = True
except ImportError:
    TREAT_DISPENSER_AVAILABLE = False

try:
    from hardware.audio_controller import AudioController
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

try:
    from hardware.led_controller import LEDController
    LED_AVAILABLE = True
except ImportError:
    LED_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class MissionEvent:
    """Event data structure"""
    event_type: str
    timestamp: float
    data: Dict[str, Any]
    success: bool = True

@dataclass
class PoseCondition:
    """Pose detection condition"""
    pose: str
    duration: float = 0.0
    confidence_threshold: float = 0.5

@dataclass
class RewardAction:
    """Reward action specification"""
    treat: bool = False
    audio: Optional[str] = None
    lights: Optional[str] = None
    delay_s: float = 0.0

class MissionController:
    """
    Unified API for all training missions
    """

    def __init__(self, mission_name: str, config: Dict = None):
        self.mission_name = mission_name
        self.config = config or self.load_config(mission_name)
        self.logger = self._setup_logger()

        # Initialize hardware interfaces
        self.hardware = self._init_hardware()

        # Mission state
        self.mission_id = None
        self.start_time = None
        self.events = []
        self.is_active = False

        # Database for logging
        self.db_path = Path("data/missions.db")
        self._init_database()

        # Callbacks
        self.pose_callbacks = {}
        self.event_callbacks = {}

    def _setup_logger(self):
        """Setup mission-specific logger"""
        mission_logger = logging.getLogger(f"mission.{self.mission_name}")
        if not mission_logger.handlers:
            handler = logging.FileHandler(f"logs/mission_{self.mission_name}.log")
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            mission_logger.addHandler(handler)
            mission_logger.setLevel(logging.INFO)
        return mission_logger

    def _init_hardware(self):
        """Initialize hardware interfaces"""
        hardware = {}

        if TREAT_DISPENSER_AVAILABLE:
            try:
                hardware['treats'] = TreatDispenser()
                self.logger.info("Treat dispenser initialized")
            except Exception as e:
                self.logger.warning(f"Treat dispenser failed: {e}")

        if AUDIO_AVAILABLE:
            try:
                hardware['audio'] = AudioController()
                self.logger.info("Audio controller initialized")
            except Exception as e:
                self.logger.warning(f"Audio controller failed: {e}")

        if LED_AVAILABLE:
            try:
                hardware['leds'] = LEDController()
                self.logger.info("LED controller initialized")
            except Exception as e:
                self.logger.warning(f"LED controller failed: {e}")

        return hardware

    def _init_database(self):
        """Initialize SQLite database for mission logging"""
        self.db_path.parent.mkdir(exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS missions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_name TEXT NOT NULL,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    success BOOLEAN,
                    total_events INTEGER,
                    config TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id INTEGER,
                    event_type TEXT,
                    timestamp TIMESTAMP,
                    data TEXT,
                    success BOOLEAN,
                    FOREIGN KEY (mission_id) REFERENCES missions (id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS detections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id INTEGER,
                    timestamp TIMESTAMP,
                    pose TEXT,
                    confidence FLOAT,
                    duration FLOAT,
                    bbox TEXT,
                    FOREIGN KEY (mission_id) REFERENCES missions (id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS rewards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id INTEGER,
                    timestamp TIMESTAMP,
                    treat_dispensed BOOLEAN,
                    audio_played TEXT,
                    lights_activated TEXT,
                    FOREIGN KEY (mission_id) REFERENCES missions (id)
                )
            """)

    def load_config(self, mission_name: str) -> Dict:
        """Load mission configuration from YAML file"""
        config_path = Path(f"missions/configs/{mission_name}.yaml")

        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Return default config
            return {
                'timeout_minutes': 10,
                'max_rewards_per_session': 5,
                'cooldown_between_rewards': 15,
                'success_criteria': {
                    'min_successful_detections': 3
                }
            }

    def start(self) -> str:
        """
        Initialize mission and log start time

        Returns:
            Mission ID
        """
        self.start_time = time.time()
        self.is_active = True

        # Create mission record in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO missions (mission_name, start_time, config)
                VALUES (?, ?, ?)
            """, (
                self.mission_name,
                datetime.fromtimestamp(self.start_time),
                json.dumps(self.config)
            ))
            self.mission_id = cursor.lastrowid

        self.logger.info(f"Mission '{self.mission_name}' started (ID: {self.mission_id})")

        # Log start event
        self.log_event("mission_start", {
            "mission_name": self.mission_name,
            "config": self.config
        })

        return str(self.mission_id)

    def wait_for_condition(self, pose: str, duration: float = 0, timeout: float = 60) -> bool:
        """
        Wait until pose detected for X seconds

        Args:
            pose: Target pose to detect
            duration: Minimum duration to maintain pose
            timeout: Maximum time to wait

        Returns:
            Success status
        """
        self.logger.info(f"Waiting for pose '{pose}' for {duration}s (timeout: {timeout}s)")

        start_wait = time.time()
        pose_start = None

        while time.time() - start_wait < timeout:
            if not self.is_active:
                return False

            # This would be connected to your AI detection system
            # For now, we'll simulate with a callback system
            current_pose = self._get_current_pose()

            if current_pose == pose:
                if pose_start is None:
                    pose_start = time.time()
                    self.log_event("pose_detected", {
                        "pose": pose,
                        "timestamp": pose_start
                    })
                elif time.time() - pose_start >= duration:
                    self.logger.info(f"Pose condition met: {pose} for {duration}s")
                    self.log_event("pose_condition_met", {
                        "pose": pose,
                        "duration": time.time() - pose_start
                    })
                    return True
            else:
                if pose_start is not None:
                    self.log_event("pose_lost", {
                        "pose": pose,
                        "held_duration": time.time() - pose_start
                    })
                pose_start = None

            time.sleep(0.1)  # Check every 100ms

        self.logger.warning(f"Timeout waiting for pose '{pose}'")
        self.log_event("pose_timeout", {
            "pose": pose,
            "waited_duration": timeout
        })
        return False

    def _get_current_pose(self) -> Optional[str]:
        """
        Get current pose from AI system
        This is a placeholder - connect to your AI controller
        """
        # TODO: Connect to AI3StageControllerFixed
        # For now, return None or implement callback system
        if hasattr(self, '_current_pose'):
            return self._current_pose
        return None

    def set_current_pose(self, pose: str, confidence: float):
        """
        Set current pose (called by AI system)

        Args:
            pose: Detected pose
            confidence: Detection confidence
        """
        self._current_pose = pose
        self._current_confidence = confidence

        # Log detection
        if self.mission_id:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO detections (mission_id, timestamp, pose, confidence)
                    VALUES (?, ?, ?, ?)
                """, (
                    self.mission_id,
                    datetime.now(),
                    pose,
                    confidence
                ))

    def reward(self, treat: bool = True, audio: str = None, lights: str = None) -> bool:
        """
        Trigger reward actions

        Args:
            treat: Dispense treat
            audio: Audio file to play
            lights: Light pattern to activate

        Returns:
            Success status
        """
        success = True
        actions_taken = []

        # Dispense treat
        if treat and 'treats' in self.hardware:
            try:
                self.hardware['treats'].dispense()
                actions_taken.append("treat_dispensed")
                self.logger.info("Treat dispensed")
            except Exception as e:
                self.logger.error(f"Treat dispense failed: {e}")
                success = False

        # Play audio
        if audio and 'audio' in self.hardware:
            try:
                self.hardware['audio'].play(audio)
                actions_taken.append(f"audio_{audio}")
                self.logger.info(f"Audio played: {audio}")
            except Exception as e:
                self.logger.error(f"Audio failed: {e}")
                success = False

        # Activate lights
        if lights and 'leds' in self.hardware:
            try:
                self.hardware['leds'].pattern(lights)
                actions_taken.append(f"lights_{lights}")
                self.logger.info(f"Lights activated: {lights}")
            except Exception as e:
                self.logger.error(f"Lights failed: {e}")
                success = False

        # Log reward
        self.log_event("reward_given", {
            "treat": treat,
            "audio": audio,
            "lights": lights,
            "actions_taken": actions_taken,
            "success": success
        })

        # Database log
        if self.mission_id:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO rewards (mission_id, timestamp, treat_dispensed, audio_played, lights_activated)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    self.mission_id,
                    datetime.now(),
                    treat,
                    audio,
                    lights
                ))

        return success

    def log_event(self, event_type: str, data: Dict = None) -> None:
        """
        Log to database and memory

        Args:
            event_type: Type of event
            data: Event data
        """
        event = MissionEvent(
            event_type=event_type,
            timestamp=time.time(),
            data=data or {}
        )

        self.events.append(event)
        self.logger.info(f"Event: {event_type} - {data}")

        # Database log
        if self.mission_id:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO events (mission_id, event_type, timestamp, data, success)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    self.mission_id,
                    event_type,
                    datetime.fromtimestamp(event.timestamp),
                    json.dumps(event.data),
                    event.success
                ))

    def end(self, success: bool = True) -> Dict:
        """
        Cleanup and log completion

        Args:
            success: Mission success status

        Returns:
            Mission summary
        """
        end_time = time.time()
        duration = end_time - self.start_time if self.start_time else 0

        self.is_active = False

        # Update mission record
        if self.mission_id:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE missions
                    SET end_time = ?, success = ?, total_events = ?
                    WHERE id = ?
                """, (
                    datetime.fromtimestamp(end_time),
                    success,
                    len(self.events),
                    self.mission_id
                ))

        # Create summary
        summary = {
            "mission_id": self.mission_id,
            "mission_name": self.mission_name,
            "duration_seconds": duration,
            "total_events": len(self.events),
            "success": success,
            "events": [
                {
                    "type": e.event_type,
                    "timestamp": e.timestamp,
                    "data": e.data
                } for e in self.events
            ]
        }

        self.logger.info(f"Mission '{self.mission_name}' ended - Success: {success}, Duration: {duration:.1f}s")
        self.log_event("mission_end", {
            "success": success,
            "duration": duration,
            "total_events": len(self.events)
        })

        return summary

    def get_status(self) -> Dict:
        """Get current mission status"""
        return {
            "mission_id": self.mission_id,
            "mission_name": self.mission_name,
            "is_active": self.is_active,
            "start_time": self.start_time,
            "duration": time.time() - self.start_time if self.start_time else 0,
            "total_events": len(self.events),
            "hardware_status": {
                name: "available" for name in self.hardware.keys()
            }
        }

# Convenience functions for common mission patterns
def simple_sit_mission(duration: float = 3.0) -> MissionController:
    """Create a simple sit training mission"""
    config = {
        "pose_target": "sit",
        "duration_required": duration,
        "timeout_minutes": 5,
        "reward": {
            "treat": True,
            "audio": "good_dog.mp3",
            "lights": "celebration"
        }
    }
    return MissionController("sit_training", config)

def quiet_training_mission(silence_duration: float = 10.0) -> MissionController:
    """Create a quiet training mission"""
    config = {
        "pose_target": "sit",
        "silence_duration": silence_duration,
        "bark_threshold_db": 60,
        "timeout_minutes": 10,
        "reward": {
            "treat": True,
            "audio": "good_quiet.mp3",
            "lights": "blue_pulse"
        }
    }
    return MissionController("quiet_training", config)