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

from core.bus import get_bus, publish_system_event, AudioEvent
from core.state import get_state, SystemMode
from core.store import get_store
from core.behavior_interpreter import get_behavior_interpreter
from orchestrators.sequence_engine import get_sequence_engine
from orchestrators.reward_logic import get_reward_logic
from services.media.usb_audio import get_usb_audio_service
from services.cloud.relay_client import get_relay_client


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


# Maps mission JSON success_event pose names to trick_rules.yaml trick names
# "Down" -> trick "down" -> behavior "lie" (interpreter handles internally)
POSE_TO_TRICK = {
    "Sit": "sit",
    "Down": "down",
    "Spin": "spin",
    "Stand": "stand",
}


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
    pose_tracking_reset: bool = False  # Whether interpreter.reset_tracking() called for current pose stage


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
        self.interpreter = get_behavior_interpreter()

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
        self.bus.subscribe("audio", self._on_audio_event)
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

            # Set mode to MISSION and lock it to prevent interruption
            self.state.set_mode(SystemMode.MISSION, reason=f'Mission: {mission_name}')
            self.state.lock_mode(f'Mission active: {mission_name}')

            # Start mission thread
            self.running = True
            self.mission_thread = threading.Thread(target=self._mission_loop, daemon=True)
            self.mission_thread.start()

            self.logger.info(f"Started mission: {mission_name} (ID: {mission_id}), mode locked to MISSION")
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
                        "stage": "watching",
                        "trick": trick_name,
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
                            "stage": "watching",
                            "trick": trick_name,
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
                    "stage": "success",
                    "trick": trick_name,
                    "hold_time": round(result.hold_duration, 1),
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
            # Send failure event to app for this stage
            trick_name = self._get_trick_name(stage)
            if trick_name:
                try:
                    relay = get_relay_client()
                    if relay and relay.connected:
                        relay.send_event("mission_progress", {
                            "stage": "failure",
                            "trick": trick_name,
                            "reason": "timeout_optional_skipped",
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
                        "stage": "failure",
                        "trick": trick_name,
                        "reason": "timeout",
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
            self._handle_bark_for_mission(event_data)
        elif subtype == "quiet_period":
            self._handle_quiet_period(event_data)

    def _on_system_event(self, event):
        """Handle system events"""
        if event.subtype == "EmergencyStop":
            self._handle_emergency_stop(event.data)

    def _handle_dog_detected(self, event_data: Dict[str, Any]):
        """Handle dog detection event"""
        session = self.active_session

        if session.state == MissionState.WAITING_FOR_DOG:
            session.dog_id = event_data.get("dog_id")

            # Check if dog detection is the success event for current stage
            if session.current_stage < len(session.mission.stages):
                stage = session.mission.stages[session.current_stage]
                if stage.success_event == "VisionEvent.DogDetected":
                    # Dog detection IS the success event - advance stage
                    self._advance_stage()
                    self.logger.info(f"Dog detected (success), advanced to stage {session.current_stage + 1}")
                else:
                    # Dog detection is not the goal - just switch to behavior waiting
                    session.state = MissionState.WAITING_FOR_BEHAVIOR
                    self.logger.info(f"Dog detected, waiting for behavior on stage {session.current_stage + 1}")

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