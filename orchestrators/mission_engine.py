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

from core.bus import get_bus, publish_system_event
from core.state import get_state, SystemMode
from core.store import get_store
from orchestrators.sequence_engine import get_sequence_engine
from orchestrators.reward_logic import get_reward_logic


class MissionState(Enum):
    """Mission execution states"""
    IDLE = "idle"
    STARTING = "starting"
    WAITING_FOR_DOG = "waiting_for_dog"
    WAITING_FOR_BEHAVIOR = "waiting_for_behavior"
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


@dataclass
class MissionSession:
    """Active mission session"""
    mission_id: int
    mission: Mission
    dog_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    current_stage: int = 0
    stage_start_time: float = field(default_factory=time.time)
    state: MissionState = MissionState.IDLE
    rewards_given: int = 0
    attempts: int = 0
    events_log: List[Dict[str, Any]] = field(default_factory=list)
    last_event_time: float = field(default_factory=time.time)


# Mission engine singleton
_mission_engine = None
_engine_lock = threading.Lock()


class MissionEngine:
    """
    Mission state machine and coordination engine

    Manages:
    - Loading mission definitions from JSON
    - Tracking mission progress through stages
    - Coordinating with reward logic and sequence engine
    - Enforcing daily limits and cooldowns
    - Mission state persistence
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.store = get_store()
        self.sequence_engine = get_sequence_engine()
        self.reward_logic = get_reward_logic()

        self.logger = logging.getLogger(__name__)

        # Mission management
        self.missions: Dict[str, Mission] = {}
        self.active_session: Optional[MissionSession] = None
        self.mission_thread: Optional[threading.Thread] = None
        self.running = False
        self._lock = threading.Lock()

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
        self.bus.subscribe("reward", self._on_reward_event)
        self.bus.subscribe("system", self._on_system_event)

    def start_mission(self, mission_name: str, dog_id: str = None) -> bool:
        """
        Start a training mission

        Args:
            mission_name: Name of mission to start
            dog_id: Optional specific dog ID

        Returns:
            True if mission started successfully
        """
        if mission_name not in self.missions:
            self.logger.error(f"Mission not found: {mission_name}")
            return False

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

            # Start mission thread
            self.running = True
            self.mission_thread = threading.Thread(target=self._mission_loop, daemon=True)
            self.mission_thread.start()

            self.logger.info(f"Started mission: {mission_name} (ID: {mission_id})")
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

            self.logger.info(f"Stopped mission: {session.mission.name} ({reason})")
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
        """Main mission execution loop"""
        session = self.active_session

        try:
            session.state = MissionState.STARTING
            session.stage_start_time = time.time()

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

                # Process current stage
                self._process_stage()

                time.sleep(0.1)  # Prevent tight loop

        except Exception as e:
            self.logger.error(f"Mission loop error: {e}")
            session.state = MissionState.FAILED

        finally:
            self.running = False

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

        # Stage-specific logic handled by event handlers
        # Events will trigger state transitions

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

        if session.current_stage >= len(session.mission.stages):
            self._complete_mission("all_stages_completed")
        else:
            session.state = MissionState.WAITING_FOR_DOG
            next_stage = session.mission.stages[session.current_stage]
            self.logger.info(f"Advanced to stage: {next_stage.name}")

    def _handle_stage_timeout(self, stage: MissionStage):
        """Handle stage timeout"""
        session = self.active_session
        session.attempts += 1

        if session.attempts >= stage.max_attempts:
            self._complete_mission("stage_timeout")
        else:
            # Retry stage
            session.stage_start_time = time.time()
            session.state = MissionState.WAITING_FOR_DOG
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

        self.logger.info(f"Mission completed: {session.mission.name} ({reason})")
        publish_system_event("mission.completed", {
            "mission_name": session.mission.name,
            "mission_id": session.mission_id,
            "success": success,
            "reason": reason,
            "rewards_given": session.rewards_given
        })

    def _check_daily_limits(self, dog_id: str = None) -> bool:
        """Check if daily reward limits are reached"""
        # Get today's rewards from store
        rewards_today = len(self.store.get_reward_history(dog_id, days=1))
        daily_limit = 10  # configurable limit

        return rewards_today >= daily_limit

    # Event handlers
    def _on_vision_event(self, event):
        """Handle vision events (dog detection, pose, etc.)"""
        if not self.active_session:
            return

        event_data = event.data
        subtype = event.subtype

        if subtype == "DogDetected":
            self._handle_dog_detected(event_data)
        elif subtype == "DogLost":
            self._handle_dog_lost(event_data)
        elif subtype == "Pose":
            self._handle_pose_detected(event_data)

    def _on_reward_event(self, event):
        """Handle reward events"""
        if not self.active_session:
            return

        if event.subtype == "Completed":
            self._handle_reward_completed(event.data)

    def _on_system_event(self, event):
        """Handle system events"""
        if event.subtype == "EmergencyStop":
            self._handle_emergency_stop(event.data)

    def _handle_dog_detected(self, event_data: Dict[str, Any]):
        """Handle dog detection event"""
        session = self.active_session

        if session.state == MissionState.WAITING_FOR_DOG:
            session.state = MissionState.WAITING_FOR_BEHAVIOR
            session.dog_id = event_data.get("dog_id")
            # Advance to next stage when dog is detected
            if session.current_stage < len(session.mission.stages) - 1:
                session.current_stage += 1
                session.stage_start_time = time.time()
            self.logger.info(f"Dog detected, advanced to stage {session.current_stage + 1}, waiting for behavior")

    def _handle_dog_lost(self, event_data: Dict[str, Any]):
        """Handle dog lost event"""
        session = self.active_session

        if session.state == MissionState.WAITING_FOR_BEHAVIOR:
            session.state = MissionState.WAITING_FOR_DOG
            self.logger.info("Dog lost, waiting for detection")

    def _handle_pose_detected(self, event_data: Dict[str, Any]):
        """Handle pose detection event"""
        session = self.active_session

        if session.state != MissionState.WAITING_FOR_BEHAVIOR:
            return

        if session.current_stage >= len(session.mission.stages):
            return

        stage = session.mission.stages[session.current_stage]
        pose = event_data.get("pose", "")
        confidence = event_data.get("confidence", 0.0)

        # Check if this pose matches stage success criteria
        if stage.success_event == f"VisionEvent.Pose.{pose.capitalize()}":
            # Check minimum duration
            if stage.min_duration > 0:
                # Would need pose duration tracking
                duration = event_data.get("duration", 0.0)
                if duration < stage.min_duration:
                    return

            # Trigger reward logic
            reward_given = self.reward_logic.evaluate_reward(
                behavior=pose,
                confidence=confidence,
                dog_id=session.dog_id
            )

            if reward_given:
                session.rewards_given += 1
                session.state = MissionState.EXECUTING_REWARD

                # Log reward
                self.store.log_reward(
                    dog_id=session.dog_id,
                    behavior=pose,
                    confidence=confidence,
                    treat_dispensed=True,
                    audio_played="good_dog.mp3",
                    lights_activated="celebration"
                )

                self.logger.info(f"Reward given for {pose}")

    def _handle_reward_completed(self, event_data: Dict[str, Any]):
        """Handle reward completion"""
        session = self.active_session

        if session.state == MissionState.EXECUTING_REWARD:
            self._advance_stage()

    def _handle_emergency_stop(self, event_data: Dict[str, Any]):
        """Handle emergency stop"""
        if self.active_session:
            self.stop_mission("emergency_stop")

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