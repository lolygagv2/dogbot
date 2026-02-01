#!/usr/bin/env python3
"""
Mission engine for managing training missions and sequences
Loads mission definitions from JSON and coordinates with reward logic
"""

import json
import threading
import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import os
from core.bus import get_bus, publish_system_event, AudioEvent
from core.state import get_state, SystemMode
from core.store import get_store
from core.behavior_interpreter import get_behavior_interpreter
from orchestrators.sequence_engine import get_sequence_engine
from orchestrators.reward_logic import get_reward_logic
from services.media.usb_audio import get_usb_audio_service
from services.reward.dispenser import get_dispenser_service
from services.media.led import get_led_service
from services.cloud.relay_client import get_relay_client


class MissionState(Enum):
    """Mission execution states - mirrors coaching engine for consistency"""
    IDLE = "idle"
    STARTING = "starting"
    # Coach-style states for trick execution
    WAITING_FOR_DOG = "waiting_for_dog"
    ATTENTION_CHECK = "attention_check"
    GREETING = "greeting"
    COMMAND = "command"
    WATCHING = "watching"
    SUCCESS = "success"
    FAILURE = "failure"
    # Retry states
    RETRY_GREETING = "retry_greeting"
    RETRY_COMMAND = "retry_command"
    RETRY_WATCHING = "retry_watching"
    FINAL_FAILURE = "final_failure"
    # Mission-level states
    STAGE_COOLDOWN = "stage_cooldown"
    EXECUTING_REWARD = "executing_reward"
    COOLDOWN = "cooldown"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


@dataclass
class MissionStage:
    """Individual stage within a mission"""
    name: str
    timeout: float = 60.0  # seconds
    success_event: str = ""
    min_duration: float = 0.0  # minimum time in this stage
    max_attempts: int = 1
    sequence: Optional[str] = None  # sequence to execute
    cooldown: float = 0.0  # cooldown after stage
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Mission:
    """Mission definition"""
    name: str
    description: str = ""
    enabled: bool = True
    schedule: str = "manual"  # daily, weekly, manual
    max_rewards: int = 5
    duration_minutes: int = 30
    stages: List[MissionStage] = field(default_factory=list)
    completion: Dict[str, str] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)


# Maps mission JSON success_event pose names to trick_rules.yaml trick names
# "Down" -> trick "down" -> behavior "lie" (interpreter handles internally)
POSE_TO_TRICK = {
    "Sit": "sit",
    "Down": "down",
    "Spin": "spin",
    "Stand": "stand",
}

# BUILD 36: Mission name aliases - maps app/display names to actual mission filenames
# Resolves Issue 1 from Build 35 - app sends "stay_training" but robot has "sit_training"
MISSION_ALIASES = {
    # App name variations -> actual mission name
    "stay_training": "sit_training",
    "Stay Training": "sit_training",
    "Basic Sit": "sit_training",
    "basic_sit": "sit_training",
    "Sit Training": "sit_training",
    # Down training
    "lie_training": "down_sustained",
    "Lie Training": "down_sustained",
    "Down Training": "down_sustained",
    "down_training": "down_sustained",
    # Bark prevention
    "quiet_training": "bark_prevention",
    "Quiet Training": "bark_prevention",
    "Stop Barking": "stop_barking",
    # Come training
    "come_training": "come_and_sit",
    "Come Training": "come_and_sit",
}


@dataclass
class MissionSession:
    """Active mission session - includes coach-style tracking"""
    mission_id: int
    mission: Mission
    dog_id: Optional[str] = None
    dog_name: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    current_stage: int = 0
    stage_start_time: float = field(default_factory=time.time)
    state: MissionState = MissionState.IDLE
    rewards_given: int = 0
    attempts: int = 0  # Attempts for current stage (max 2)
    events_log: List[Dict[str, Any]] = field(default_factory=list)
    last_event_time: float = field(default_factory=time.time)
    pose_tracking_reset: bool = False  # Whether interpreter.reset_tracking() called for current pose stage
    # Coach-style tracking
    trick_requested: Optional[str] = None  # Current trick being requested
    command_time: Optional[float] = None  # When command was given
    behavior_detected: Optional[str] = None  # What behavior was actually detected
    attention_start: Optional[float] = None  # When attention check started


# Mission engine singleton
_mission_engine = None
_engine_lock = threading.Lock()


class MissionEngine:
    """
    Mission state machine and coordination engine

    NOW uses coach-style flow for trick stages:
    1. Wait for dog (with presence check)
    2. Attention check (2-3s)
    3. Greeting (play dog name)
    4. Command (play trick audio)
    5. Watch for behavior (10s default)
    6. Success/Failure handling
    7. Retry logic (2 attempts)

    Manages:
    - Loading mission definitions from JSON
    - Tracking mission progress through stages
    - Coordinating with reward logic and sequence engine
    - Enforcing daily limits and cooldowns
    - Mission state persistence
    """

    # Coach-style timing constants
    # BUILD 36: Reduced from 3.0s/66% to 1.5s/50% to speed up dog detection
    # Issue 5f/5g from Build 35 - detection taking too long
    DETECTION_TIME_SEC = 1.5  # Dog must be visible this long (was 3.0)
    PRESENCE_RATIO_MIN = 0.50  # Dog must be in 50% of frames (was 0.66)
    ATTENTION_DURATION_SEC = 2.0  # Attention check duration
    WATCH_DURATION_SEC = 10.0  # Default watch window
    STALE_TIMEOUT_SEC = 6.0  # Dog considered gone after this (must be > detection_event_interval of 5s)

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.store = get_store()
        self.sequence_engine = get_sequence_engine()
        self.reward_logic = get_reward_logic()
        self.interpreter = get_behavior_interpreter()
        self.audio = get_usb_audio_service()
        self.dispenser = get_dispenser_service()
        self.led = get_led_service()

        self.logger = logging.getLogger(__name__)

        # Mission management
        self.missions: Dict[str, Mission] = {}
        self.active_session: Optional[MissionSession] = None
        self.mission_thread: Optional[threading.Thread] = None
        self.running = False
        self._lock = threading.Lock()

        # Coach-style dog tracking (same as coaching_engine.py)
        self.dogs_in_view: Dict[str, Dict[str, Any]] = {}

        # Dog name mapping
        self.dog_names = {
            'aruco_315': 'Elsa',
            'aruco_832': 'Bezik',
            315: 'Elsa',
            832: 'Bezik'
        }

        # Bark tracking for speak missions
        self.bark_count = 0
        self.bark_timestamps: List[float] = []
        self.listening_for_barks = False
        self._listening_started_at = 0.0

        # Load missions
        self._load_missions()

        # Subscribe to events
        self._setup_event_handlers()

        self.logger.info("Mission engine initialized")

    def _load_missions(self):
        """Load mission definitions from JSON files"""
        missions_dir = Path("/home/morgan/dogbot/missions")

        if not missions_dir.exists():
            self.logger.warning(f"Missions directory not found: {missions_dir}")
            return

        for mission_file in missions_dir.glob("*.json"):
            try:
                with open(mission_file, 'r') as f:
                    mission_data = json.load(f)

                mission = self._parse_mission(mission_data)
                self.missions[mission.name] = mission
                self.logger.info(f"Loaded mission: {mission.name}")

            except Exception as e:
                self.logger.error(f"Failed to load mission {mission_file}: {e}")

    def _parse_mission(self, data: Dict[str, Any]) -> Mission:
        """Parse mission from JSON data"""
        stages = []
        for stage_data in data.get("stages", []):
            stage = MissionStage(
                name=stage_data["name"],
                timeout=stage_data.get("timeout", 60.0),
                success_event=stage_data.get("success_event", ""),
                min_duration=stage_data.get("min_duration", 0.0),
                max_attempts=stage_data.get("max_attempts", 1),
                sequence=stage_data.get("sequence"),
                cooldown=stage_data.get("cooldown", 0.0),
                conditions=stage_data.get("conditions", {})
            )
            stages.append(stage)

        return Mission(
            name=data["name"],
            description=data.get("description", ""),
            enabled=data.get("enabled", True),
            schedule=data.get("schedule", "manual"),
            max_rewards=data.get("max_rewards", 5),
            duration_minutes=data.get("duration_minutes", 30),
            stages=stages,
            completion=data.get("completion", {}),
            config=data.get("config", {})
        )

    def _setup_event_handlers(self):
        """Subscribe to relevant events"""
        self.bus.subscribe("vision", self._on_vision_event)
        self.bus.subscribe("audio", self._on_audio_event)
        self.bus.subscribe("reward", self._on_reward_event)
        self.bus.subscribe("system", self._on_system_event)

    def start_mission(self, mission_name: str, dog_id: str = None) -> bool:
        """
        Start a training mission

        Args:
            mission_name: Name of mission to start (supports aliases)
            dog_id: Optional specific dog ID

        Returns:
            True if mission started successfully
        """
        self.logger.info(f"[MISSION] start_mission called: name={mission_name}, dog_id={dog_id}")
        self.logger.info(f"[MISSION] Available missions: {list(self.missions.keys())}")

        # BUILD 36: Check mission aliases if direct name not found
        actual_mission_name = mission_name
        if mission_name not in self.missions:
            # Try alias lookup
            if mission_name in MISSION_ALIASES:
                actual_mission_name = MISSION_ALIASES[mission_name]
                self.logger.info(f"[MISSION] Using alias: {mission_name} -> {actual_mission_name}")
            else:
                self.logger.error(f"[MISSION] Mission not found: {mission_name}")
                self.logger.error(f"[MISSION] Hint: Available missions are: {list(self.missions.keys())}")
                return False

        if actual_mission_name not in self.missions:
            self.logger.error(f"[MISSION] Aliased mission not found: {actual_mission_name}")
            return False

        # Use the resolved mission name
        mission_name = actual_mission_name

        mission = self.missions[mission_name]

        if not mission.enabled:
            self.logger.error(f"Mission disabled: {mission_name}")
            return False

        with self._lock:
            if self.active_session:
                self.logger.error("Mission already active")
                return False

            # Check daily limits
            if self._check_daily_limits(dog_id):
                self.logger.warning("Daily reward limit reached")
                return False

            # Create database mission record
            mission_id = self.store.start_mission(
                name=mission_name,
                target_rewards=mission.max_rewards,
                config=mission.config
            )

            # Create session
            self.active_session = MissionSession(
                mission_id=mission_id,
                mission=mission,
                dog_id=dog_id
            )

            # Set mode to MISSION and lock it to prevent interruption
            self.state.set_mode(SystemMode.MISSION, reason=f'Mission: {mission_name}')
            self.state.lock_mode(f'Mission active: {mission_name}')

            self.logger.info(f"[MISSION] Mode set to MISSION, locked={self.state.is_mode_locked()}")

            # Start mission thread
            self.running = True
            self.mission_thread = threading.Thread(target=self._mission_loop, daemon=True)
            self.mission_thread.start()

            # Verify detector is running
            try:
                from services.perception.detector import get_detector_service
                detector = get_detector_service()
                self.logger.info(f"[MISSION] Detector status: initialized={detector.ai_initialized}, running={detector.running}")
            except Exception as e:
                self.logger.warning(f"[MISSION] Could not check detector status: {e}")

            self.logger.info(f"[MISSION] Started mission: {mission_name} (ID: {mission_id}), thread started")
            publish_system_event("mission.started", {
                "mission_name": mission_name,
                "mission_id": mission_id,
                "dog_id": dog_id
            })

            return True

    def stop_mission(self, reason: str = "user_requested") -> bool:
        """Stop active mission"""
        with self._lock:
            if not self.active_session:
                return False

            session = self.active_session

            # Update mission state
            session.state = MissionState.STOPPED
            self.running = False

            # End database record
            self.store.end_mission(
                session.mission_id,
                status="stopped",
                results={
                    "reason": reason,
                    "rewards_given": session.rewards_given,
                    "duration": time.time() - session.start_time,
                    "stages_completed": session.current_stage
                }
            )

            # Unlock mode FIRST, then set to idle
            self.state.unlock_mode()
            self.state.set_mode(SystemMode.IDLE, reason=f'Mission stopped: {reason}')

            self.logger.info(f"Stopped mission: {session.mission.name} ({reason}), mode unlocked")
            publish_system_event("mission.stopped", {
                "mission_name": session.mission.name,
                "mission_id": session.mission_id,
                "reason": reason,
                "rewards_given": session.rewards_given
            })

            self.active_session = None
            return True

    def pause_mission(self) -> bool:
        """Pause active mission"""
        with self._lock:
            if not self.active_session:
                return False

            self.active_session.state = MissionState.PAUSED
            self.logger.info("Mission paused")
            publish_system_event("mission.paused", {
                "mission_id": self.active_session.mission_id
            })
            return True

    def resume_mission(self) -> bool:
        """Resume paused mission"""
        with self._lock:
            if not self.active_session or self.active_session.state != MissionState.PAUSED:
                return False

            self.active_session.state = MissionState.WAITING_FOR_DOG
            self.active_session.stage_start_time = time.time()
            self.logger.info("Mission resumed")
            publish_system_event("mission.resumed", {
                "mission_id": self.active_session.mission_id
            })
            return True

    def get_mission_status(self) -> Dict[str, Any]:
        """Get current mission status"""
        with self._lock:
            if not self.active_session:
                return {"active": False}

            session = self.active_session
            current_stage = None

            if session.current_stage < len(session.mission.stages):
                stage = session.mission.stages[session.current_stage]
                current_stage = {
                    "name": stage.name,
                    "timeout": stage.timeout,
                    "elapsed": time.time() - session.stage_start_time,
                    "success_event": stage.success_event
                }

            return {
                "active": True,
                "mission_id": session.mission_id,
                "mission_name": session.mission.name,
                "dog_id": session.dog_id,
                "state": session.state.value,
                "current_stage": session.current_stage,
                "total_stages": len(session.mission.stages),
                "stage_info": current_stage,
                "rewards_given": session.rewards_given,
                "max_rewards": session.mission.max_rewards,
                "duration": time.time() - session.start_time,
                "max_duration": session.mission.duration_minutes * 60
            }

    def _mission_loop(self):
        """
        Main mission execution loop - NOW uses coach-style flow!

        For each pose/trick stage:
        1. WAITING_FOR_DOG - wait for dog presence (3s, 66% visibility)
        2. ATTENTION_CHECK - verify dog is attentive
        3. GREETING - play dog's name audio
        4. COMMAND - play trick command audio
        5. WATCHING - poll for behavior (10s default)
        6. SUCCESS/FAILURE - handle outcome
        7. Retry on first failure (2 attempts per stage)
        """
        session = self.active_session

        try:
            session.state = MissionState.STARTING
            session.stage_start_time = time.time()

            self.logger.info(f"🎯 Mission loop started: {session.mission.name}")

            while self.running and session.state not in [
                MissionState.COMPLETED, MissionState.FAILED, MissionState.STOPPED
            ]:
                # Check for paused state
                if session.state == MissionState.PAUSED:
                    time.sleep(0.1)
                    continue

                # Check mission timeout
                if time.time() - session.start_time > session.mission.duration_minutes * 60:
                    self._complete_mission("timeout")
                    break

                # Check max rewards
                if session.rewards_given >= session.mission.max_rewards:
                    self._complete_mission("max_rewards_reached")
                    break

                # Clean stale dogs (like coaching engine)
                self._cleanup_stale_dogs()

                # BUILD 35: Removed frames_total increment from main loop
                # frames_total now increments in event handler alongside frames_seen
                # This ensures presence ratio works correctly with detection event timing

                # Process current state (coach-style state machine)
                self._process_state()

                time.sleep(0.1)  # 10Hz loop

        except Exception as e:
            self.logger.error(f"Mission loop error: {e}", exc_info=True)
            session.state = MissionState.FAILED

        finally:
            self.running = False
            self.listening_for_barks = False

    def _cleanup_stale_dogs(self):
        """Remove dogs not seen recently (same as coaching engine)"""
        cutoff = time.time() - self.STALE_TIMEOUT_SEC
        stale = [
            dog_id for dog_id, info in self.dogs_in_view.items()
            if info['last_seen'] < cutoff
        ]
        for dog_id in stale:
            del self.dogs_in_view[dog_id]

    def _get_dog_name(self, dog_id: str) -> str:
        """Get friendly name for dog (same as coaching engine)"""
        if dog_id in self.dogs_in_view:
            tracked_name = self.dogs_in_view[dog_id].get('name')
            if tracked_name:
                return tracked_name

        if dog_id in self.dog_names:
            return self.dog_names[dog_id]

        if dog_id.startswith('aruco_'):
            try:
                aruco_id = int(dog_id.split('_')[1])
                if aruco_id in self.dog_names:
                    return self.dog_names[aruco_id]
            except (ValueError, IndexError):
                pass

        return dog_id

    def _play_audio(self, filename: str, wait: bool = True, timeout: float = 5.0):
        """Play audio file (same as coaching engine)"""
        if not self.audio:
            return

        try:
            base_path = '/home/morgan/dogbot/VOICEMP3/talks'
            full_path = os.path.join(base_path, filename)

            if not os.path.exists(full_path):
                # Try default directory
                full_path = os.path.join(base_path, 'default', filename)

            if os.path.exists(full_path):
                self.audio.play_file(full_path)
                if wait:
                    self.audio.wait_for_completion(timeout=timeout)
            else:
                self.logger.warning(f"Audio file not found: {filename}")
        except Exception as e:
            self.logger.error(f"Audio playback error: {e}")

    def _process_state(self):
        """Process current FSM state (coach-style state machine)"""
        session = self.active_session
        if not session:
            return

        # Check if we've completed all stages
        if session.current_stage >= len(session.mission.stages):
            self._complete_mission("all_stages_completed")
            return

        stage = session.mission.stages[session.current_stage]

        # Route to state handler
        if session.state == MissionState.STARTING:
            self._state_starting()
        elif session.state == MissionState.WAITING_FOR_DOG:
            self._state_waiting_for_dog()
        elif session.state == MissionState.ATTENTION_CHECK:
            self._state_attention_check()
        elif session.state == MissionState.GREETING:
            self._state_greeting()
        elif session.state == MissionState.COMMAND:
            self._state_command()
        elif session.state == MissionState.WATCHING:
            self._state_watching()
        elif session.state == MissionState.SUCCESS:
            self._state_success()
        elif session.state == MissionState.FAILURE:
            self._state_failure()
        elif session.state == MissionState.RETRY_GREETING:
            self._state_retry_greeting()
        elif session.state == MissionState.RETRY_COMMAND:
            self._state_retry_command()
        elif session.state == MissionState.RETRY_WATCHING:
            self._state_retry_watching()
        elif session.state == MissionState.FINAL_FAILURE:
            self._state_final_failure()
        elif session.state == MissionState.STAGE_COOLDOWN:
            self._state_stage_cooldown()

    def _state_starting(self):
        """Initialize first stage"""
        session = self.active_session
        stage = session.mission.stages[session.current_stage]

        self.logger.info(f"📋 Starting stage {session.current_stage + 1}: {stage.name}")
        session.state = MissionState.WAITING_FOR_DOG
        session.stage_start_time = time.time()
        session.attempts = 0

        # Send status to app
        self._send_mission_status("waiting_for_dog", stage.name)

    def _state_waiting_for_dog(self):
        """Wait for dog to be visible and meet presence requirements (matches coaching_engine)"""
        session = self.active_session
        stage = session.mission.stages[session.current_stage]
        now = time.time()

        # Check stage timeout
        if now - session.stage_start_time > stage.timeout:
            self._handle_stage_timeout(stage)
            return

        # Find eligible dog using SAME logic as coaching_engine:
        # Requirements:
        # 1. Dog has been tracked for DETECTION_TIME_SEC (3s)
        # 2. Dog present in >= PRESENCE_RATIO_MIN (66%) of frames since first seen
        eligible_dogs = []
        for dog_id, info in self.dogs_in_view.items():
            time_elapsed = now - info['first_seen']
            presence_ratio = info['frames_seen'] / max(info['frames_total'], 1)

            # Match coaching_engine presence check
            if time_elapsed >= self.DETECTION_TIME_SEC and presence_ratio >= self.PRESENCE_RATIO_MIN:
                has_aruco_name = info.get('name') is not None
                eligible_dogs.append((dog_id, info, has_aruco_name, time_elapsed))

        if not eligible_dogs:
            # Log waiting status periodically
            if int(now * 2) % 10 == 0:  # Every 5 seconds
                dogs_status = [(d, f"{now - i['first_seen']:.1f}s") for d, i in self.dogs_in_view.items()]
                self.logger.debug(f"[MISSION] Waiting for dog... tracked: {dogs_status}")
            return

        # Prefer ArUco-identified dogs
        eligible_dogs.sort(key=lambda x: (not x[2], -x[3]))
        dog_id, info, has_aruco_name, time_elapsed = eligible_dogs[0]
        dog_name = self._get_dog_name(dog_id)

        self.logger.info(f"🐕 Dog ready for mission: {dog_name} "
                        f"(visible {time_elapsed:.1f}s, presence={presence_ratio:.0%}, aruco={'yes' if has_aruco_name else 'no'})")

        # Update session
        session.dog_id = dog_id
        session.dog_name = dog_name
        session.trick_requested = self._get_trick_name(stage)
        session.attention_start = now
        session.state = MissionState.ATTENTION_CHECK

    def _state_attention_check(self):
        """Verify dog is still attentive (brief check)"""
        session = self.active_session
        dog_id = session.dog_id

        # Check dog is still visible
        if dog_id not in self.dogs_in_view:
            self.logger.info("Dog left during attention check")
            session.state = MissionState.WAITING_FOR_DOG
            return

        # Attention confirmed - move to greeting
        session.state = MissionState.GREETING

    def _state_greeting(self):
        """Greet the dog by name"""
        session = self.active_session
        dog_name = session.dog_name or 'dog'
        trick = session.trick_requested or 'trick'

        self.logger.info(f"👋 Greeting {dog_name} for {trick}")

        # LED attention pattern
        if self.led:
            try:
                self.led.set_pattern('attention', duration=2.0)
            except Exception:
                pass

        # Play dog's name
        dog_id = session.dog_id
        name_played = False

        if dog_id and self.audio:
            result = self.audio.play_command(dog_name.lower(), dog_id=dog_id)
            if result.get('success'):
                name_played = True
                self.audio.wait_for_completion(timeout=5.0)

        if not name_played:
            name_audio = f'{dog_name.lower()}.mp3'
            base_path = '/home/morgan/dogbot/VOICEMP3/talks'
            if not os.path.exists(os.path.join(base_path, name_audio)):
                name_audio = 'dogs_come.mp3'
            self._play_audio(name_audio, wait=True, timeout=5.0)

        time.sleep(0.5)
        session.state = MissionState.COMMAND

    def _state_command(self):
        """Give the trick command"""
        session = self.active_session
        trick = session.trick_requested

        if not trick:
            # Non-pose stage - skip to watching
            session.command_time = time.time()
            session.state = MissionState.WATCHING
            return

        self.logger.info(f"🎤 Commanding: {trick}")

        # Play trick command audio
        trick_rules = self.interpreter.get_trick_rules(trick)
        audio_file = trick_rules.get('audio_command', f'{trick}.mp3') if trick_rules else f'{trick}.mp3'
        self._play_audio(audio_file, wait=True, timeout=5.0)

        # Reset behavior tracking AFTER audio
        self.interpreter.reset_tracking()

        # Start bark listening for speak trick
        if trick == 'speak':
            time.sleep(0.3)
            self.bark_count = 0
            self.bark_timestamps = []
            self._listening_started_at = time.time()
            self.listening_for_barks = True
            self.logger.info("🎤 Listening for barks (speak mission)...")

        session.command_time = time.time()
        session.pose_tracking_reset = True
        session.state = MissionState.WATCHING

        self._send_mission_status("watching", trick)

    def _state_watching(self):
        """Watch for behavior response (same as coaching engine)"""
        session = self.active_session
        stage = session.mission.stages[session.current_stage]
        trick = session.trick_requested
        dog_name = session.dog_name or 'unknown'

        watch_elapsed = time.time() - session.command_time
        trick_rules = self.interpreter.get_trick_rules(trick) if trick else {}
        detection_window = trick_rules.get('detection_window_sec', self.WATCH_DURATION_SEC) if trick_rules else self.WATCH_DURATION_SEC

        # Use stage timeout if longer than trick default
        if stage.timeout > detection_window:
            detection_window = min(stage.timeout, 30.0)  # Cap at 30s

        # Special handling for speak trick
        if trick == 'speak':
            speak_rules = trick_rules or {}
            min_barks = speak_rules.get('min_barks', 1)
            max_barks = speak_rules.get('max_barks', 2)
            speak_timeout = speak_rules.get('detection_window_sec', 5.0)

            if self.bark_count > max_barks:
                self.listening_for_barks = False
                session.state = MissionState.FAILURE
                self.logger.info(f"Speak failed - too many barks ({self.bark_count})")
                return

            if self.bark_count >= min_barks:
                self.listening_for_barks = False
                session.behavior_detected = 'bark'
                session.state = MissionState.SUCCESS
                self.logger.info(f"✅ Success! Dog spoke with {self.bark_count} bark(s)")
                return

            if watch_elapsed >= speak_timeout:
                self.listening_for_barks = False
                session.state = MissionState.FAILURE
                self.logger.info(f"Speak timeout - only {self.bark_count} bark(s)")
                return
            return

        # Standard pose-based tricks
        if trick:
            result = self.interpreter.check_trick(trick, dog_id=dog_name)

            # Check for sustained hold requirement
            hold_required = stage.min_duration if stage.min_duration > 0 else 0

            if hold_required > 0:
                if result.behavior_detected and result.hold_duration >= hold_required:
                    session.behavior_detected = result.behavior_detected
                    session.state = MissionState.SUCCESS
                    self.logger.info(f"✅ Success! {dog_name} held {trick} for {result.hold_duration:.1f}s")
                    return
            else:
                if result.completed:
                    session.behavior_detected = result.behavior_detected
                    session.state = MissionState.SUCCESS
                    self.logger.info(f"✅ Success! {dog_name} performed {trick}")
                    return

            # Log progress
            if result.behavior_detected:
                self.logger.debug(f"Watching: {result.reason}")

        # Check timeout
        if watch_elapsed >= detection_window:
            session.state = MissionState.FAILURE
            self.logger.info(f"⏱️ Watch timeout for {trick}")

    def _state_success(self):
        """Handle successful behavior detection"""
        session = self.active_session
        stage = session.mission.stages[session.current_stage]
        dog_name = session.dog_name or 'unknown'
        trick = session.trick_requested

        self.logger.info(f"🎉 Mission stage success: {trick} by {dog_name}")

        # Play success audio
        self._play_audio('good_dog.mp3', wait=True, timeout=3.0)

        # LED celebration
        if self.led:
            try:
                self.led.set_pattern('celebrate', duration=2.0)
            except Exception:
                pass

        # Dispense treat
        if self.dispenser:
            try:
                self.dispenser.dispense_treat()
                session.rewards_given += 1
                self.logger.info(f"🍖 Treat dispensed ({session.rewards_given}/{session.mission.max_rewards})")
            except Exception as e:
                self.logger.error(f"Dispenser error: {e}")

        # Log to store
        self.store.log_reward(
            dog_id=session.dog_id,
            behavior=session.behavior_detected or trick,
            confidence=0.8,
            success=True,
            treats_dispensed=1,
            mission_name=session.mission.name
        )

        # Send success to app
        self._send_mission_status("success", trick)

        # Advance to next stage
        self._advance_stage()

    def _state_failure(self):
        """Handle first failure - retry once"""
        session = self.active_session
        session.attempts += 1

        if session.attempts >= 2:
            # Already retried - final failure
            session.state = MissionState.FINAL_FAILURE
            return

        self.logger.info(f"😔 First attempt failed, retrying...")
        self._play_audio('good_dog.mp3', wait=True, timeout=2.0)  # "Good try!"
        time.sleep(1.0)
        session.state = MissionState.RETRY_GREETING

    def _state_retry_greeting(self):
        """Retry - greet again"""
        session = self.active_session
        dog_name = session.dog_name or 'dog'

        # Check dog still visible
        if session.dog_id not in self.dogs_in_view:
            self.logger.info("Dog left during retry")
            session.state = MissionState.WAITING_FOR_DOG
            return

        self.logger.info(f"🔄 Retry greeting {dog_name}")
        name_audio = f'{dog_name.lower()}.mp3'
        self._play_audio(name_audio, wait=True, timeout=3.0)
        time.sleep(0.3)
        session.state = MissionState.RETRY_COMMAND

    def _state_retry_command(self):
        """Retry - give command again"""
        session = self.active_session
        trick = session.trick_requested

        self.logger.info(f"🔄 Retry command: {trick}")

        trick_rules = self.interpreter.get_trick_rules(trick) if trick else {}
        audio_file = trick_rules.get('audio_command', f'{trick}.mp3') if trick_rules else f'{trick}.mp3'
        self._play_audio(audio_file, wait=True, timeout=5.0)

        self.interpreter.reset_tracking()

        if trick == 'speak':
            time.sleep(0.3)
            self.bark_count = 0
            self.bark_timestamps = []
            self._listening_started_at = time.time()
            self.listening_for_barks = True

        session.command_time = time.time()
        session.state = MissionState.RETRY_WATCHING

    def _state_retry_watching(self):
        """Retry watching - same as regular watching"""
        self._state_watching()
        # After watching completes, it sets SUCCESS or FAILURE
        # If FAILURE again, we'll hit FINAL_FAILURE

    def _state_final_failure(self):
        """Handle final failure after retry"""
        session = self.active_session
        stage = session.mission.stages[session.current_stage]
        trick = session.trick_requested

        self.logger.info(f"❌ Mission stage failed after retry: {trick}")

        # Play consolation
        self._play_audio('good_dog.mp3', wait=True, timeout=2.0)

        # Send failure to app
        self._send_mission_status("failed", trick)

        # Decide: advance to next stage or fail mission
        # For now, advance to give other stages a chance
        self._advance_stage()

    def _state_stage_cooldown(self):
        """Brief cooldown between stages"""
        session = self.active_session
        stage = session.mission.stages[session.current_stage]

        if stage.cooldown > 0:
            time.sleep(min(stage.cooldown, 5.0))

        self._advance_stage()

    def _send_mission_status(self, status: str, trick: str = None):
        """Send mission status to app via relay"""
        try:
            relay = get_relay_client()
            if relay and relay.connected:
                session = self.active_session
                relay.send_event("mission_progress", {
                    "mission_name": session.mission.name if session and session.mission else None,
                    "status": status,
                    "trick": trick,
                    "stage": session.current_stage + 1 if session else 0,
                    "total_stages": len(session.mission.stages) if session else 0,
                    "dog_name": session.dog_name if session else None,
                    "rewards": session.rewards_given if session else 0,
                })
        except Exception:
            pass

    def _is_pose_stage(self, stage: MissionStage) -> bool:
        """Check if a stage expects a pose/trick detection"""
        return stage.success_event.startswith("VisionEvent.Pose.")

    def _get_trick_name(self, stage: MissionStage) -> Optional[str]:
        """Extract trick name from stage success_event (e.g. 'VisionEvent.Pose.Sit' -> 'sit')"""
        if not self._is_pose_stage(stage):
            return None
        # Extract pose name: "VisionEvent.Pose.Sit" -> "Sit"
        parts = stage.success_event.split(".")
        if len(parts) >= 3:
            pose_name = parts[2]  # "Sit", "Down", "Spin", "Stand"
            return POSE_TO_TRICK.get(pose_name)
        return None

    def _process_stage(self):
        """Process current mission stage"""
        session = self.active_session

        if session.current_stage >= len(session.mission.stages):
            self._complete_mission("all_stages_completed")
            return

        stage = session.mission.stages[session.current_stage]
        stage_elapsed = time.time() - session.stage_start_time

        # Check stage timeout
        if stage_elapsed > stage.timeout:
            self._handle_stage_timeout(stage)
            return

        # Update state based on current stage
        if session.state == MissionState.STARTING:
            session.state = MissionState.WAITING_FOR_DOG
            self.logger.info(f"Starting stage: {stage.name}")

        # For pose stages in WAITING_FOR_BEHAVIOR, poll the interpreter
        if session.state == MissionState.WAITING_FOR_BEHAVIOR and self._is_pose_stage(stage):
            self._poll_pose_stage(stage)

        # Other stage types (bark, quiet, dog detection) handled by event handlers

    def _poll_pose_stage(self, stage: MissionStage):
        """
        Poll BehaviorInterpreter for pose/trick stages.

        Replicates coaching engine pattern:
        - Reset tracking once on stage entry
        - Poll check_trick() each loop iteration (10Hz from mission_loop sleep)
        - For sustained hold missions, check hold_duration >= stage.min_duration
        - Send progress events to app via relay
        """
        session = self.active_session
        trick_name = self._get_trick_name(stage)

        if not trick_name:
            self.logger.warning(f"Could not map pose stage to trick: {stage.success_event}")
            return

        # Reset tracking once on stage entry
        if not session.pose_tracking_reset:
            self.interpreter.reset_tracking()
            session.pose_tracking_reset = True
            self.logger.info(f"Mission pose stage: watching for '{trick_name}' (reset tracking)")

            # Send initial watching event to app
            hold_required = stage.min_duration if stage.min_duration > 0 else 0
            try:
                relay = get_relay_client()
                if relay and relay.connected:
                    relay.send_event("mission_progress", {
                        "mission_name": session.mission.name if session.mission else None,
                        "status": "watching",
                        "trick": trick_name,
                        "stage": session.current_stage + 1,
                        "total_stages": len(session.mission.stages),
                        "dog_name": session.dog_name,
                        "rewards": session.rewards_given,
                        "progress": 0,
                        "target_sec": hold_required,
                    })
            except Exception:
                pass

        # Poll the interpreter
        result = self.interpreter.check_trick(trick_name)

        # Determine hold requirement: stage.min_duration overrides trick_rules default
        # For sustained missions (e.g. down_sustained.json min_duration=30s),
        # the interpreter reports completed=True after trick_rules hold (e.g. 1.5s),
        # but we need to wait for the full stage.min_duration
        hold_required = stage.min_duration if stage.min_duration > 0 else 0
        trick_rules = self.interpreter.get_trick_rules(trick_name)
        trick_hold = trick_rules.get('hold_duration_sec', 1.0) if trick_rules else 1.0

        # Send progress updates (throttled to ~1Hz via stage_start_time check)
        if result.behavior_detected and result.hold_duration > 0:
            stage_elapsed = time.time() - session.stage_start_time
            # Send progress every ~1 second
            if int(stage_elapsed) != int(stage_elapsed - 0.1):
                try:
                    relay = get_relay_client()
                    if relay and relay.connected:
                        relay.send_event("mission_progress", {
                            "mission_name": session.mission.name if session.mission else None,
                            "status": "watching",
                            "trick": trick_name,
                            "stage": session.current_stage + 1,
                            "total_stages": len(session.mission.stages),
                            "dog_name": session.dog_name,
                            "rewards": session.rewards_given,
                            "progress": round(result.hold_duration, 1),
                            "target_sec": hold_required if hold_required > 0 else trick_hold,
                        })
                except Exception:
                    pass

        # Check completion
        if hold_required > 0:
            # Sustained hold: interpreter must show the right behavior AND hold_duration >= min_duration
            if result.behavior_detected and result.hold_duration >= hold_required:
                self._handle_pose_success(stage, trick_name, result)
                return
        else:
            # Standard hold: rely on interpreter's completed flag (uses trick_rules hold)
            if result.completed:
                self._handle_pose_success(stage, trick_name, result)
                return

        # Log progress for debugging
        if result.behavior_detected:
            self.logger.debug(f"Mission pose watch: {result.reason}")

    def _handle_pose_success(self, stage: MissionStage, trick_name: str, result):
        """Handle successful pose detection in a mission stage"""
        session = self.active_session

        self.logger.info(
            f"Mission pose success: {trick_name} "
            f"(detected: {result.behavior_detected}, "
            f"held: {result.hold_duration:.1f}s, conf: {result.confidence:.2f})"
        )

        # Trigger reward logic
        reward_given = self.reward_logic.evaluate_reward(
            behavior=result.behavior_detected,
            confidence=result.confidence,
            dog_id=session.dog_id
        )

        if reward_given:
            session.rewards_given += 1
            self.store.log_reward(
                dog_id=session.dog_id,
                behavior=result.behavior_detected,
                confidence=result.confidence,
                success=True,
                treats_dispensed=1,
                mission_name=session.mission.name
            )

        # Log event
        session.events_log.append({
            "type": "pose_success",
            "trick": trick_name,
            "behavior": result.behavior_detected,
            "hold_duration": result.hold_duration,
            "confidence": result.confidence,
            "reward_given": reward_given,
            "time": time.time()
        })

        # Send success event to app
        try:
            relay = get_relay_client()
            if relay and relay.connected:
                relay.send_event("mission_progress", {
                    "mission_name": session.mission.name if session.mission else None,
                    "status": "success",
                    "trick": trick_name,
                    "stage": session.current_stage + 1,
                    "total_stages": len(session.mission.stages),
                    "dog_name": session.dog_name,
                    "rewards": session.rewards_given,
                    "progress": round(result.hold_duration, 1),
                    "target_sec": result.hold_duration,
                })
        except Exception:
            pass

        # Advance to next stage
        self._advance_stage()

    def _advance_stage(self):
        """Advance to next mission stage"""
        session = self.active_session

        if session.current_stage < len(session.mission.stages):
            stage = session.mission.stages[session.current_stage]

            # Execute stage sequence if defined
            if stage.sequence:
                self.sequence_engine.execute_sequence(stage.sequence)

            # Handle cooldown
            if stage.cooldown > 0:
                session.state = MissionState.COOLDOWN
                time.sleep(stage.cooldown)

        # Move to next stage
        session.current_stage += 1
        session.stage_start_time = time.time()
        session.attempts = 0
        session.pose_tracking_reset = False  # Fresh reset for next pose stage

        if session.current_stage >= len(session.mission.stages):
            self._complete_mission("all_stages_completed")
        else:
            session.state = MissionState.WAITING_FOR_DOG
            next_stage = session.mission.stages[session.current_stage]
            self.logger.info(f"Advanced to stage: {next_stage.name}")

    def _handle_stage_timeout(self, stage: MissionStage):
        """Handle stage timeout"""
        session = self.active_session

        # Optional stages skip to next on timeout instead of failing
        if stage.conditions.get("optional", False):
            self.logger.info(f"Optional stage '{stage.name}' timed out - skipping")
            # Send skip event to app for this stage
            trick_name = self._get_trick_name(stage)
            if trick_name:
                try:
                    relay = get_relay_client()
                    if relay and relay.connected:
                        relay.send_event("mission_progress", {
                            "mission_name": session.mission.name if session.mission else None,
                            "status": "skipped",
                            "trick": trick_name,
                            "stage": session.current_stage + 1,
                            "total_stages": len(session.mission.stages),
                            "dog_name": session.dog_name,
                            "rewards": session.rewards_given,
                        })
                except Exception:
                    pass
            self._advance_stage()
            return

        session.attempts += 1

        # Send failure event to app
        trick_name = self._get_trick_name(stage)
        if trick_name:
            try:
                relay = get_relay_client()
                if relay and relay.connected:
                    relay.send_event("mission_progress", {
                        "mission_name": session.mission.name if session.mission else None,
                        "status": "failed",
                        "trick": trick_name,
                        "stage": session.current_stage + 1,
                        "total_stages": len(session.mission.stages),
                        "dog_name": session.dog_name,
                        "rewards": session.rewards_given,
                    })
            except Exception:
                pass

        if session.attempts >= stage.max_attempts:
            self._complete_mission("stage_timeout")
        else:
            # Retry stage
            session.stage_start_time = time.time()
            session.state = MissionState.WAITING_FOR_DOG
            session.pose_tracking_reset = False  # Fresh reset on retry
            self.logger.warning(f"Stage timeout, retrying: {stage.name}")

    def _complete_mission(self, reason: str):
        """Complete mission with given reason"""
        session = self.active_session

        success = reason in ["all_stages_completed", "max_rewards_reached"]
        session.state = MissionState.COMPLETED if success else MissionState.FAILED

        # Update database
        self.store.end_mission(
            session.mission_id,
            status="completed" if success else "failed",
            results={
                "reason": reason,
                "rewards_given": session.rewards_given,
                "duration": time.time() - session.start_time,
                "stages_completed": session.current_stage
            }
        )

        # Unlock mode and set to idle
        self.state.unlock_mode()
        self.state.set_mode(SystemMode.IDLE, reason=f'Mission {reason}')

        self.logger.info(f"Mission completed: {session.mission.name} ({reason}), mode unlocked")
        publish_system_event("mission.completed", {
            "mission_name": session.mission.name,
            "mission_id": session.mission_id,
            "success": success,
            "reason": reason,
            "rewards_given": session.rewards_given
        })

        # Send mission_complete event to app via relay
        try:
            relay = get_relay_client()
            if relay and relay.connected:
                relay.send_event("mission_complete", {
                    "mission_name": session.mission.name if session.mission else None,
                    "success": success,
                    "tricks_completed": session.current_stage,
                    "treats_given": session.rewards_given,
                    "reason": reason,
                })
        except Exception:
            pass

        # Play mission complete audio on success
        if success:
            try:
                audio = get_usb_audio_service()
                if audio and audio.is_initialized:
                    audio.play_file("/home/morgan/dogbot/VOICEMP3/wimz/Wimz_missioncomplete.mp3")
                    self.logger.info("Played mission complete audio")
            except Exception as e:
                self.logger.debug(f"Could not play mission complete audio: {e}")

    def _check_daily_limits(self, dog_id: str = None) -> bool:
        """Check if daily reward limits are reached"""
        # Get today's rewards from store
        rewards_today = len(self.store.get_reward_history(dog_id, days=1))

        # Get daily limit from active mission config, or use system default
        daily_limit = 30  # System default
        if self.active_session and self.active_session.mission:
            daily_limit = self.active_session.mission.config.get('daily_limit', 30)

        return rewards_today >= daily_limit

    # Event handlers
    def _on_vision_event(self, event):
        """Handle vision events (dog detection, pose, etc.)"""
        if not self.active_session:
            return

        event_data = event.data
        subtype = event.subtype

        if subtype == "dog_detected":
            self._handle_dog_detected(event_data)
        elif subtype == "dog_lost":
            self._handle_dog_lost(event_data)
        # Pose stages are now polled via _poll_pose_stage(), not event-driven

    def _on_reward_event(self, event):
        """Handle reward events"""
        if not self.active_session:
            return

        if event.subtype == "Completed":
            self._handle_reward_completed(event.data)

    def _on_audio_event(self, event):
        """Handle audio events (bark detection, quiet periods)"""
        if not self.active_session:
            return

        event_data = event.data
        subtype = event.subtype

        if subtype == "bark_detected":
            # Coach-style bark tracking for speak trick
            if self.listening_for_barks:
                session = self.active_session
                if session and session.trick_requested == 'speak':
                    # Reject stale bark events from before listening started
                    if hasattr(event, 'timestamp') and event.timestamp < self._listening_started_at:
                        self.logger.debug(f"Ignoring stale bark event")
                        return

                    self.bark_count += 1
                    self.bark_timestamps.append(time.time())
                    self.logger.info(f"🐕 Bark detected during speak mission! Count: {self.bark_count}")

            # Also handle for bark-based stages
            self._handle_bark_for_mission(event_data)
        elif subtype == "quiet_period":
            self._handle_quiet_period(event_data)

    def _on_system_event(self, event):
        """Handle system events"""
        if event.subtype == "EmergencyStop":
            self._handle_emergency_stop(event.data)

    def _handle_dog_detected(self, event_data: Dict[str, Any]):
        """Handle dog detection event - track dogs like coaching engine"""
        dog_id = event_data.get('dog_id')
        dog_name = event_data.get('dog_name')

        if not dog_id:
            return

        now = time.time()

        if dog_id not in self.dogs_in_view:
            # New dog - initialize tracking
            self.dogs_in_view[dog_id] = {
                'first_seen': now,
                'last_seen': now,
                'frames_seen': 1,
                'frames_total': 1,
                'name': dog_name if dog_name not in ['unknown', None] else None
            }
            display_name = dog_name if dog_name and dog_name not in ['unknown', None] else dog_id
            self.logger.info(f"🐕 Dog entered view for mission: {display_name}")
        else:
            # Existing dog - update tracking
            entry = self.dogs_in_view[dog_id]
            entry['last_seen'] = now
            entry['frames_seen'] += 1
            entry['frames_total'] += 1  # BUILD 35: Increment both counters together

            # Update name if ArUco identified
            if dog_name and dog_name not in ['unknown', None] and entry['name'] is None:
                entry['name'] = dog_name
                self.logger.info(f"🏷️ Dog {dog_id} identified as {dog_name}")

        # Check if dog detection is the success event (non-pose stages)
        session = self.active_session
        if session and session.state == MissionState.WATCHING:
            if session.current_stage < len(session.mission.stages):
                stage = session.mission.stages[session.current_stage]
                if stage.success_event == "VisionEvent.DogDetected":
                    self._advance_stage()
                    self.logger.info(f"Dog detected (success), advanced to stage {session.current_stage + 1}")

    def _handle_dog_lost(self, event_data: Dict[str, Any]):
        """Handle dog lost event"""
        session = self.active_session

        if session.state == MissionState.WAITING_FOR_BEHAVIOR:
            session.state = MissionState.WAITING_FOR_DOG
            self.logger.info("Dog lost, waiting for detection")

    def _handle_reward_completed(self, event_data: Dict[str, Any]):
        """Handle reward completion"""
        session = self.active_session

        if session.state == MissionState.EXECUTING_REWARD:
            self._advance_stage()

    def _handle_emergency_stop(self, event_data: Dict[str, Any]):
        """Handle emergency stop"""
        if self.active_session:
            self.stop_mission("emergency_stop")

    def _handle_bark_for_mission(self, event_data: Dict[str, Any]):
        """
        Handle bark detection for mission stage advancement

        Supports success_event formats:
        - "AudioEvent.BarkDetected" - any bark
        - "AudioEvent.BarkDetected.alert" - specific emotion
        - "AudioEvent.BarkDetected.scared" - specific emotion
        """
        session = self.active_session

        if session.state != MissionState.WAITING_FOR_BEHAVIOR:
            return

        if session.current_stage >= len(session.mission.stages):
            return

        stage = session.mission.stages[session.current_stage]
        success_event = stage.success_event

        # Check if this stage expects a bark event
        if not success_event.startswith("AudioEvent.Bark"):
            return

        emotion = event_data.get("emotion", "")
        confidence = event_data.get("confidence", 0.0)
        dog_id = event_data.get("dog_id")
        dog_name = event_data.get("dog_name")

        # Check emotion filter (e.g., "AudioEvent.BarkDetected.alert")
        parts = success_event.split(".")
        if len(parts) >= 3:
            required_emotion = parts[2].lower()
            if emotion.lower() != required_emotion:
                self.logger.debug(f"Bark emotion {emotion} doesn't match required {required_emotion}")
                return

        # Check confidence threshold from conditions
        conditions = stage.conditions
        min_confidence = conditions.get("min_confidence", 0.5)
        if confidence < min_confidence:
            self.logger.debug(f"Bark confidence {confidence:.2f} below threshold {min_confidence}")
            return

        # Bark matches stage criteria - advance!
        self.logger.info(f"Bark matched stage {stage.name}: {emotion} (conf: {confidence:.2f})")

        # Update session dog if detected
        if dog_id:
            session.dog_id = dog_id

        # Log event
        session.events_log.append({
            "type": "bark_detected",
            "emotion": emotion,
            "confidence": confidence,
            "dog_id": dog_id,
            "dog_name": dog_name,
            "time": time.time()
        })

        # Execute sequence if defined
        if stage.sequence:
            self.sequence_engine.execute_sequence(stage.sequence, {
                "dog_id": dog_id,
                "dog_name": dog_name,
                "emotion": emotion
            })

        # Advance to next stage
        self._advance_stage()

    def _handle_quiet_period(self, event_data: Dict[str, Any]):
        """Handle quiet period detection for missions"""
        session = self.active_session

        if session.state != MissionState.WAITING_FOR_BEHAVIOR:
            return

        if session.current_stage >= len(session.mission.stages):
            return

        stage = session.mission.stages[session.current_stage]

        # Check if this stage expects quiet period
        if stage.success_event != "AudioEvent.QuietPeriod":
            return

        quiet_duration = event_data.get("duration", 0.0)
        required_duration = stage.conditions.get("quiet_duration", 5.0)

        if quiet_duration >= required_duration:
            self.logger.info(f"Quiet period matched: {quiet_duration:.1f}s >= {required_duration}s")
            self._advance_stage()

    def get_available_missions(self) -> List[Dict[str, Any]]:
        """Get list of available missions"""
        missions = []
        for name, mission in self.missions.items():
            missions.append({
                "name": name,
                "description": mission.description,
                "enabled": mission.enabled,
                "max_rewards": mission.max_rewards,
                "duration_minutes": mission.duration_minutes,
                "stages": len(mission.stages)
            })
        return missions


def get_mission_engine() -> MissionEngine:
    """Get singleton mission engine"""
    global _mission_engine

    if _mission_engine is None:
        with _engine_lock:
            if _mission_engine is None:
                _mission_engine = MissionEngine()

    return _mission_engine


# Example usage
if __name__ == "__main__":
    engine = get_mission_engine()
    print("Available missions:")
    for mission in engine.get_available_missions():
        print(f"- {mission['name']}: {mission['description']}")