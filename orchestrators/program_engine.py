#!/usr/bin/env python3
"""
Program Engine - Manages training programs (collections of sequential missions)

A program is a curated collection of missions that run sequentially,
designed for comprehensive training sessions.
"""

import json
import threading
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.bus import get_bus
from core.store import get_store
from services.media.usb_audio import get_usb_audio_service


class ProgramState(Enum):
    """Program execution states"""
    IDLE = "idle"
    RUNNING = "running"
    RESTING = "resting"  # Between missions
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class Program:
    """Training program definition"""
    name: str
    display_name: str
    description: str
    missions: List[str]
    created_by: str = "preset"  # "preset" or "user"
    repeat: bool = False
    daily_treat_limit: int = 20
    rest_between_missions_sec: int = 60


@dataclass
class ProgramSession:
    """Active program session"""
    program: Program
    start_time: float = field(default_factory=time.time)
    current_mission_index: int = 0
    state: ProgramState = ProgramState.IDLE
    treats_dispensed: int = 0
    missions_completed: List[str] = field(default_factory=list)
    missions_failed: List[str] = field(default_factory=list)
    rest_until: float = 0.0


# Singleton
_program_engine = None
_engine_lock = threading.Lock()


class ProgramEngine:
    """
    Orchestrates sequential execution of training programs.

    Programs are collections of missions that run one after another,
    with configurable rest periods between them.
    """

    def __init__(self):
        self.bus = get_bus()
        self.store = get_store()
        self.logger = logging.getLogger(__name__)

        # Program management
        self.programs: Dict[str, Program] = {}
        self.active_session: Optional[ProgramSession] = None
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._running = False

        # Mission engine reference (set later to avoid circular import)
        self._mission_engine = None

        # Load programs
        self._load_programs()

        # Subscribe to mission events
        self.bus.subscribe("mission_completed", self._on_mission_completed)
        self.bus.subscribe("mission_failed", self._on_mission_failed)
        self.bus.subscribe("mission_stopped", self._on_mission_stopped)
        self.bus.subscribe("reward_given", self._on_reward_given)

        self.logger.info(f"Program engine initialized with {len(self.programs)} programs")

    def _get_mission_engine(self):
        """Lazy load mission engine to avoid circular import"""
        if self._mission_engine is None:
            from orchestrators.mission_engine import get_mission_engine
            self._mission_engine = get_mission_engine()
        return self._mission_engine

    def _load_programs(self):
        """Load program definitions from JSON files"""
        programs_dir = Path("/home/morgan/dogbot/programs")

        if not programs_dir.exists():
            self.logger.warning(f"Programs directory not found: {programs_dir}")
            programs_dir.mkdir(parents=True, exist_ok=True)
            return

        for program_file in programs_dir.glob("*.json"):
            try:
                with open(program_file, 'r') as f:
                    data = json.load(f)

                program = Program(
                    name=data["name"],
                    display_name=data.get("display_name", data["name"]),
                    description=data.get("description", ""),
                    missions=data.get("missions", []),
                    created_by=data.get("created_by", "preset"),
                    repeat=data.get("repeat", False),
                    daily_treat_limit=data.get("daily_treat_limit", 20),
                    rest_between_missions_sec=data.get("rest_between_missions_sec", 60)
                )
                self.programs[program.name] = program
                self.logger.info(f"Loaded program: {program.name} ({len(program.missions)} missions)")

            except Exception as e:
                self.logger.error(f"Failed to load program {program_file}: {e}")

    def reload_programs(self):
        """Reload all programs from disk"""
        with self._lock:
            self.programs.clear()
            self._load_programs()

    def get_available_programs(self) -> List[Dict[str, Any]]:
        """Get list of all available programs for API/phone app"""
        programs = []
        mission_engine = self._get_mission_engine()
        available_missions = list(mission_engine.missions.keys())

        for program in self.programs.values():
            # Check which missions are valid
            valid_missions = [m for m in program.missions if m in available_missions]
            invalid_missions = [m for m in program.missions if m not in available_missions]

            programs.append({
                "name": program.name,
                "display_name": program.display_name,
                "description": program.description,
                "missions": program.missions,
                "mission_count": len(program.missions),
                "valid_missions": len(valid_missions),
                "invalid_missions": invalid_missions,
                "created_by": program.created_by,
                "daily_treat_limit": program.daily_treat_limit,
                "rest_between_missions_sec": program.rest_between_missions_sec,
                "is_valid": len(invalid_missions) == 0
            })

        return programs

    def get_program(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a specific program"""
        if name not in self.programs:
            return None

        program = self.programs[name]
        mission_engine = self._get_mission_engine()

        # Get details for each mission
        mission_details = []
        for mission_name in program.missions:
            if mission_name in mission_engine.missions:
                m = mission_engine.missions[mission_name]
                mission_details.append({
                    "name": m.name,
                    "description": m.description,
                    "max_rewards": m.max_rewards,
                    "duration_minutes": m.duration_minutes
                })
            else:
                mission_details.append({
                    "name": mission_name,
                    "description": "Mission not found",
                    "max_rewards": 0,
                    "duration_minutes": 0,
                    "error": "invalid"
                })

        return {
            "name": program.name,
            "display_name": program.display_name,
            "description": program.description,
            "created_by": program.created_by,
            "daily_treat_limit": program.daily_treat_limit,
            "rest_between_missions_sec": program.rest_between_missions_sec,
            "missions": mission_details
        }

    def create_program(self, name: str, display_name: str, description: str,
                       missions: List[str], daily_treat_limit: int = 20,
                       rest_between_missions_sec: int = 60) -> Dict[str, Any]:
        """Create a new custom program"""
        # Validate name
        safe_name = name.lower().replace(" ", "_").replace("-", "_")
        if safe_name in self.programs:
            return {"success": False, "error": f"Program '{safe_name}' already exists"}

        # Validate missions exist
        mission_engine = self._get_mission_engine()
        invalid = [m for m in missions if m not in mission_engine.missions]
        if invalid:
            return {"success": False, "error": f"Invalid missions: {invalid}"}

        if len(missions) < 1:
            return {"success": False, "error": "Program must have at least 1 mission"}

        # Create program
        program = Program(
            name=safe_name,
            display_name=display_name,
            description=description,
            missions=missions,
            created_by="user",
            daily_treat_limit=daily_treat_limit,
            rest_between_missions_sec=rest_between_missions_sec
        )

        # Save to file
        program_path = Path(f"/home/morgan/dogbot/programs/{safe_name}.json")
        try:
            with open(program_path, 'w') as f:
                json.dump({
                    "name": program.name,
                    "display_name": program.display_name,
                    "description": program.description,
                    "missions": program.missions,
                    "created_by": program.created_by,
                    "repeat": program.repeat,
                    "daily_treat_limit": program.daily_treat_limit,
                    "rest_between_missions_sec": program.rest_between_missions_sec
                }, f, indent=2)

            self.programs[safe_name] = program
            self.logger.info(f"Created custom program: {safe_name}")

            return {"success": True, "name": safe_name, "program": self.get_program(safe_name)}

        except Exception as e:
            self.logger.error(f"Failed to save program: {e}")
            return {"success": False, "error": str(e)}

    def delete_program(self, name: str) -> Dict[str, Any]:
        """Delete a custom program (presets cannot be deleted)"""
        if name not in self.programs:
            return {"success": False, "error": f"Program '{name}' not found"}

        program = self.programs[name]
        if program.created_by == "preset":
            return {"success": False, "error": "Cannot delete preset programs"}

        # Stop if running
        if self.active_session and self.active_session.program.name == name:
            self.stop_program()

        # Delete file
        program_path = Path(f"/home/morgan/dogbot/programs/{name}.json")
        try:
            if program_path.exists():
                program_path.unlink()
            del self.programs[name]
            self.logger.info(f"Deleted program: {name}")
            return {"success": True}
        except Exception as e:
            self.logger.error(f"Failed to delete program: {e}")
            return {"success": False, "error": str(e)}

    def start_program(self, name: str, dog_id: Optional[str] = None) -> bool:
        """Start a training program"""
        with self._lock:
            if self.active_session and self.active_session.state == ProgramState.RUNNING:
                self.logger.warning("Program already running")
                return False

            if name not in self.programs:
                self.logger.error(f"Program not found: {name}")
                return False

            program = self.programs[name]

            # Validate all missions exist
            mission_engine = self._get_mission_engine()
            for mission_name in program.missions:
                if mission_name not in mission_engine.missions:
                    self.logger.error(f"Mission not found: {mission_name}")
                    return False

            # Create session
            self.active_session = ProgramSession(
                program=program,
                start_time=time.time(),
                state=ProgramState.RUNNING
            )

            self.logger.info(f"Starting program: {program.display_name} ({len(program.missions)} missions)")

            # Play announcement
            try:
                audio = get_usb_audio_service()
                audio.play_file("/wimz/Wimz_missioncomplete.mp3")  # Reuse for program start
            except:
                pass

            # Start first mission
            self._start_next_mission(dog_id)

            # Start monitor thread
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()

            return True

    def _start_next_mission(self, dog_id: Optional[str] = None):
        """Start the next mission in sequence"""
        if not self.active_session:
            return

        session = self.active_session
        if session.current_mission_index >= len(session.program.missions):
            # All missions complete
            self._complete_program()
            return

        mission_name = session.program.missions[session.current_mission_index]
        mission_engine = self._get_mission_engine()

        self.logger.info(f"Starting mission {session.current_mission_index + 1}/{len(session.program.missions)}: {mission_name}")

        success = mission_engine.start_mission(mission_name, dog_id=dog_id)
        if not success:
            self.logger.error(f"Failed to start mission: {mission_name}")
            session.missions_failed.append(mission_name)
            session.current_mission_index += 1
            # Try next mission
            self._start_next_mission(dog_id)

    def _on_mission_completed(self, data: Dict[str, Any]):
        """Handle mission completion"""
        with self._lock:
            if not self.active_session:
                return

            session = self.active_session
            mission_name = data.get("mission_name", "")

            # Check if this mission is part of our program
            if session.current_mission_index < len(session.program.missions):
                expected = session.program.missions[session.current_mission_index]
                if mission_name == expected:
                    session.missions_completed.append(mission_name)
                    session.current_mission_index += 1

                    self.logger.info(f"Mission completed: {mission_name} ({session.current_mission_index}/{len(session.program.missions)})")

                    # Check if program complete
                    if session.current_mission_index >= len(session.program.missions):
                        self._complete_program()
                    else:
                        # Rest before next mission
                        session.state = ProgramState.RESTING
                        session.rest_until = time.time() + session.program.rest_between_missions_sec
                        self.logger.info(f"Resting for {session.program.rest_between_missions_sec}s before next mission")

    def _on_mission_failed(self, data: Dict[str, Any]):
        """Handle mission failure"""
        with self._lock:
            if not self.active_session:
                return

            session = self.active_session
            mission_name = data.get("mission_name", "")

            if session.current_mission_index < len(session.program.missions):
                expected = session.program.missions[session.current_mission_index]
                if mission_name == expected:
                    session.missions_failed.append(mission_name)
                    session.current_mission_index += 1

                    self.logger.warning(f"Mission failed: {mission_name}")

                    # Continue to next mission
                    if session.current_mission_index >= len(session.program.missions):
                        self._complete_program()
                    else:
                        session.state = ProgramState.RESTING
                        session.rest_until = time.time() + session.program.rest_between_missions_sec

    def _on_mission_stopped(self, data: Dict[str, Any]):
        """Handle mission being stopped externally"""
        # Treat as failure for program purposes
        self._on_mission_failed(data)

    def _on_reward_given(self, data: Dict[str, Any]):
        """Track treats dispensed across program"""
        with self._lock:
            if not self.active_session:
                return

            session = self.active_session
            treats = data.get("treats_dispensed", 1)
            session.treats_dispensed += treats

            # Check daily limit
            if session.treats_dispensed >= session.program.daily_treat_limit:
                self.logger.info(f"Daily treat limit reached ({session.treats_dispensed}/{session.program.daily_treat_limit})")
                self._complete_program()

    def _monitor_loop(self):
        """Background thread to handle rest periods"""
        while self._running:
            time.sleep(1)

            with self._lock:
                if not self.active_session:
                    continue

                session = self.active_session

                # Handle rest period completion
                if session.state == ProgramState.RESTING:
                    if time.time() >= session.rest_until:
                        session.state = ProgramState.RUNNING
                        self._start_next_mission()

    def _complete_program(self):
        """Mark program as complete"""
        if not self.active_session:
            return

        session = self.active_session
        session.state = ProgramState.COMPLETED
        self._running = False

        self.logger.info(f"Program completed: {session.program.display_name}")
        self.logger.info(f"  Missions completed: {len(session.missions_completed)}/{len(session.program.missions)}")
        self.logger.info(f"  Treats dispensed: {session.treats_dispensed}")

        # Play completion audio
        try:
            audio = get_usb_audio_service()
            audio.play_file("/wimz/Wimz_missioncomplete.mp3")
        except:
            pass

        # Publish event
        self.bus.publish("program_completed", {
            "program_name": session.program.name,
            "missions_completed": session.missions_completed,
            "missions_failed": session.missions_failed,
            "treats_dispensed": session.treats_dispensed,
            "duration_sec": time.time() - session.start_time
        })

    def stop_program(self) -> bool:
        """Stop the current program"""
        with self._lock:
            if not self.active_session:
                return False

            session = self.active_session
            session.state = ProgramState.STOPPED
            self._running = False

            # Stop current mission
            mission_engine = self._get_mission_engine()
            mission_engine.stop_mission("program_stopped")

            self.logger.info(f"Program stopped: {session.program.display_name}")
            return True

    def pause_program(self) -> bool:
        """Pause the current program"""
        with self._lock:
            if not self.active_session:
                return False

            session = self.active_session
            if session.state not in [ProgramState.RUNNING, ProgramState.RESTING]:
                return False

            session.state = ProgramState.PAUSED

            # Pause current mission
            mission_engine = self._get_mission_engine()
            mission_engine.pause_mission()

            self.logger.info(f"Program paused: {session.program.display_name}")
            return True

    def resume_program(self) -> bool:
        """Resume a paused program"""
        with self._lock:
            if not self.active_session:
                return False

            session = self.active_session
            if session.state != ProgramState.PAUSED:
                return False

            session.state = ProgramState.RUNNING

            # Resume current mission
            mission_engine = self._get_mission_engine()
            mission_engine.resume_mission()

            self.logger.info(f"Program resumed: {session.program.display_name}")
            return True

    def get_status(self) -> Dict[str, Any]:
        """Get current program status for API/phone app"""
        if not self.active_session:
            return {
                "active": False,
                "state": "idle"
            }

        session = self.active_session
        program = session.program

        # Get current mission info
        current_mission = None
        if session.current_mission_index < len(program.missions):
            current_mission = program.missions[session.current_mission_index]

        # Calculate progress
        total = len(program.missions)
        completed = len(session.missions_completed)
        progress_pct = int((completed / total) * 100) if total > 0 else 0

        return {
            "active": True,
            "state": session.state.value,
            "program": {
                "name": program.name,
                "display_name": program.display_name,
                "description": program.description
            },
            "progress": {
                "current_mission_index": session.current_mission_index,
                "total_missions": total,
                "missions_completed": completed,
                "missions_failed": len(session.missions_failed),
                "progress_percent": progress_pct
            },
            "current_mission": current_mission,
            "treats_dispensed": session.treats_dispensed,
            "daily_treat_limit": program.daily_treat_limit,
            "elapsed_sec": int(time.time() - session.start_time),
            "missions_completed_list": session.missions_completed,
            "missions_failed_list": session.missions_failed
        }


def get_program_engine() -> ProgramEngine:
    """Get or create the program engine singleton"""
    global _program_engine
    with _engine_lock:
        if _program_engine is None:
            _program_engine = ProgramEngine()
        return _program_engine
