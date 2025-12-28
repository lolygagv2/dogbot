#!/usr/bin/env python3
"""
Silent Guardian Mode - WIM-Z Primary Product Mode
Bark-focused passive monitoring with escalating intervention and treat rewards

State Machine:
LISTENING -> INTERVENTION_START -> WAITING_FOR_QUIET -> REWARD_SEQUENCE -> COOLDOWN
                                          |
                                  ESCALATION_RESPONSE (if bark continues)
"""

import os
import sys
import time
import threading
import logging
import yaml
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports
from core.bus import get_bus, AudioEvent, publish_system_event
from core.state import get_state, SystemMode
from core.store import get_store
from core.bark_frequency_tracker import get_bark_frequency_tracker

# Services
from services.media.usb_audio import get_usb_audio_service, set_agc
from services.reward.dispenser import get_dispenser_service
from services.media.led import get_led_service

logger = logging.getLogger(__name__)

# Dog visibility timeout - how recently must we have seen a dog
DOG_VISIBILITY_TIMEOUT = 10.0  # 10 seconds - dog seen within last 10 seconds


class SGState(Enum):
    """Silent Guardian internal state machine"""
    LISTENING = "listening"
    INTERVENTION_START = "intervention_start"
    WAITING_FOR_QUIET = "waiting_for_quiet"
    CALLING_DOG = "calling_dog"          # New: Calling dog to come
    WAITING_FOR_DOG = "waiting_for_dog"  # New: Waiting for dog visibility
    REWARD_SEQUENCE = "reward_sequence"
    ESCALATION_RESPONSE = "escalation_response"
    COOLDOWN = "cooldown"


@dataclass
class InterventionState:
    """State for current intervention"""
    started_at: float = 0.0
    dog_id: Optional[str] = None
    dog_name: Optional[str] = None
    escalation_level: int = 1
    quiet_start_time: Optional[float] = None
    quiet_periods_achieved: int = 0
    barks_since_intervention: int = 0
    # Escalating quiet command tracking
    quiet_commands_issued: int = 1  # Initial command counts as 1
    last_quiet_command_time: float = 0.0
    gave_up: bool = False
    # Dog calling/visibility tracking
    calling_started_at: float = 0.0
    come_commands_issued: int = 0
    last_come_command_time: float = 0.0
    dog_visible: bool = False
    dog_last_seen_time: float = 0.0
    dog_behavior_detected: Optional[str] = None
    dog_sitting_confirmed: bool = False


class SilentGuardianMode:
    """
    Silent Guardian - Primary WIM-Z product mode

    Passive bark monitoring that:
    1. Listens for barks continuously
    2. When bark threshold exceeded, triggers intervention
    3. Uses escalating responses based on frequency
    4. Rewards quiet behavior with treats
    5. Tracks per-dog statistics
    """

    def __init__(self, config_path: str = None):
        """Initialize Silent Guardian mode"""
        self.bus = get_bus()
        self.state = get_state()
        self.store = get_store()
        self.bark_tracker = get_bark_frequency_tracker()

        # Services
        self.audio = get_usb_audio_service()
        self.dispenser = get_dispenser_service()
        self.led = get_led_service()

        # Load configuration
        if config_path is None:
            config_path = '/home/morgan/dogbot/configs/rules/silent_guardian_rules.yaml'
        self.config = self._load_config(config_path)

        # Mode state
        self.running = False
        self.mode_thread = None
        self._state_lock = threading.Lock()

        # Internal FSM state
        self.fsm_state = SGState.LISTENING
        self.intervention = InterventionState()

        # Session tracking
        self.session_id = None
        self.session_start_time = None
        self.treats_dispensed = 0
        self.interventions_triggered = 0

        # Bark tracking
        self.last_bark_time = 0.0
        self.last_intervention_time = 0.0

        # Escalation tracking (hourly window)
        self.escalation_events: List[float] = []  # timestamps of interventions

        # Music state
        self.calming_music_playing = False

        # Session duration limit (8 hours in seconds)
        self.max_session_duration = 8 * 60 * 60  # 8 hours
        self.last_treat_time = 0.0  # Track last treat for min interval

        # Dog visibility tracking (from vision events)
        self.last_dog_seen_time = 0.0
        self.last_dog_id = None
        self.last_dog_behavior = None
        self.last_dog_confidence = 0.0

        # Non-blocking cooldown timer (to avoid blocking sleeps)
        self._cooldown_start = None
        self._cooldown_duration = 0.0

        logger.info("Silent Guardian mode initialized")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Loaded Silent Guardian config from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration if YAML fails to load"""
        return {
            'bark_detection': {
                'threshold': 5,  # Barks in window before intervention (2.5x more patient)
                'loudness_threshold_db': -15,  # Only count barks louder than this (dB)
                'window_seconds': 60
            },
            'session_limits': {
                'max_treats': 11,
                'min_time_between_treats': 30
            },
            'escalation': {
                'window_minutes': 60,
                'max_level': 3
            },
            'intervention_sequences': {
                'level_1': {
                    'quiet_required_seconds': 10,
                    'reward_on_success': True
                },
                'level_2': {
                    'quiet_required_seconds': 30,
                    'reward_on_success': True
                },
                'level_3': {
                    'quiet_periods_required': 2,
                    'quiet_period_duration': 20,
                    'play_calming_music': True,
                    'reward_on_success': True
                }
            },
            'dog_identification': {
                'elsa': {'aruco_id': 315, 'audio_name': 'elsa.mp3'},
                'bezik': {'aruco_id': 832, 'audio_name': 'bezik.mp3'}
            }
        }

    def start(self) -> bool:
        """Start Silent Guardian mode"""
        if self.running:
            logger.warning("Silent Guardian already running")
            return True

        try:
            # Disable AGC for bark detection (raw energy levels needed)
            set_agc(False)
            logger.info("AGC disabled for Silent Guardian mode (bark detection)")

            # Start a new session in database
            self.session_id = self.store.start_silent_guardian_session()
            self.session_start_time = time.time()
            self.treats_dispensed = 0
            self.interventions_triggered = 0

            # Subscribe to bark events (use string 'audio', not AudioEvent class)
            self.bus.subscribe('audio', self._on_audio_event)

            # Subscribe to vision events for dog visibility tracking
            self.bus.subscribe('vision', self._on_vision_event)

            # Set system mode
            self.state.set_mode(SystemMode.SILENT_GUARDIAN, "Silent Guardian started")

            # Start mode thread
            self.running = True
            self.mode_thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="SilentGuardian"
            )
            self.mode_thread.start()

            # Publish start event
            publish_system_event('silent_guardian_started', {
                'session_id': self.session_id,
                'max_treats': self.config.get('session_limits', {}).get('max_treats', 11)
            }, 'silent_guardian')

            logger.info(f"Silent Guardian started (session: {self.session_id})")
            return True

        except Exception as e:
            logger.error(f"Failed to start Silent Guardian: {e}")
            return False

    def stop(self):
        """Stop Silent Guardian mode"""
        if not self.running:
            return

        logger.info("Stopping Silent Guardian...")
        self.running = False

        # End session in database
        if self.session_id:
            self.store.end_silent_guardian_session(
                self.session_id,
                self.interventions_triggered,
                self.treats_dispensed
            )

        # Wait for thread
        if self.mode_thread and self.mode_thread.is_alive():
            self.mode_thread.join(timeout=2.0)

        # Re-enable AGC when leaving Silent Guardian mode
        set_agc(True)
        logger.info("AGC re-enabled (leaving Silent Guardian mode)")

        # Publish stop event
        publish_system_event('silent_guardian_stopped', {
            'session_id': self.session_id,
            'interventions': self.interventions_triggered,
            'treats': self.treats_dispensed
        }, 'silent_guardian')

        logger.info("Silent Guardian stopped")

    def _on_audio_event(self, event):
        """Handle audio events from bark detector"""
        if event.subtype != 'bark_detected':
            return

        # Only process if we're in LISTENING, WAITING_FOR_QUIET, or WAITING_FOR_DOG state
        if self.fsm_state not in [SGState.LISTENING, SGState.WAITING_FOR_QUIET, SGState.WAITING_FOR_DOG]:
            return

        # Extract bark info
        dog_id = event.data.get('dog_id')
        dog_name = event.data.get('dog_name')
        confidence = event.data.get('confidence', 0.0)
        loudness_db = event.data.get('loudness_db', -30)  # dB level

        # Check loudness threshold - ignore quiet barks
        bark_config = self.config.get('bark_detection', {})
        loudness_threshold = bark_config.get('loudness_threshold_db', -15)
        if loudness_db < loudness_threshold:
            logger.debug(f"Ignoring quiet bark: {loudness_db:.1f}dB < {loudness_threshold}dB")
            return

        logger.info(f"Bark detected: dog={dog_name or dog_id} (confidence: {confidence:.2f}, loudness: {loudness_db:.1f}dB)")

        self.last_bark_time = time.time()

        # Stop calming music if playing - bark interrupts it
        if self.calming_music_playing:
            self._stop_calming_music()
            logger.info("Calming music stopped due to bark")

        # If in WAITING_FOR_QUIET or WAITING_FOR_DOG, reset quiet timer and increment bark counter
        if self.fsm_state in [SGState.WAITING_FOR_QUIET, SGState.WAITING_FOR_DOG]:
            self.intervention.quiet_start_time = None
            self.intervention.barks_since_intervention += 1
            logger.info(f"Bark during intervention - quiet timer reset (barks: {self.intervention.barks_since_intervention})")
            return

        # In LISTENING state - check if threshold exceeded
        bark_config = self.config.get('bark_detection', {})
        threshold = bark_config.get('threshold', 2)

        # Use bark frequency tracker
        result = self.bark_tracker.check_threshold(dog_id or 'unknown', threshold)

        if result:
            # Threshold exceeded - start intervention
            self._start_intervention(dog_id, dog_name)

    def _on_vision_event(self, event):
        """Handle vision events for dog visibility tracking"""
        if event.subtype == 'dog_detected':
            dog_id = event.data.get('dog_id')
            dog_name = event.data.get('dog_name')
            confidence = event.data.get('confidence', 0.0)

            # Update visibility tracking
            self.last_dog_seen_time = time.time()
            self.last_dog_id = dog_id
            self.last_dog_confidence = confidence

            logger.debug(f"Dog visible: {dog_name or dog_id} (conf: {confidence:.2f})")

            # Update intervention state if active
            if self.intervention:
                self.intervention.dog_visible = True
                self.intervention.dog_last_seen_time = time.time()

                # Update dog identity from ArUco if we have a valid name
                # This ensures we know WHO came when called
                if dog_name and dog_name != 'unknown':
                    self.intervention.dog_name = dog_name
                    self.intervention.dog_id = dog_id

        elif event.subtype == 'behavior_detected':
            behavior = event.data.get('behavior')
            confidence = event.data.get('confidence', 0.0)
            dog_name = event.data.get('dog_name')
            dog_id = event.data.get('dog_id')

            if behavior:
                self.last_dog_behavior = behavior
                logger.debug(f"Dog behavior: {behavior} (conf: {confidence:.2f})")

                # Update intervention state if active
                if self.intervention:
                    self.intervention.dog_behavior_detected = behavior

                    # Update dog identity from ArUco-identified behavior
                    # This captures WHO did the behavior, not just who barked
                    if dog_name and dog_name != 'unknown':
                        self.intervention.dog_name = dog_name
                        self.intervention.dog_id = dog_id
                        logger.debug(f"Updated intervention dog: {dog_name}")

                    if behavior.lower() == 'sit' and confidence > 0.6:
                        self.intervention.dog_sitting_confirmed = True
                        logger.info(f"Dog sitting confirmed: {dog_name or 'unknown'} (conf: {confidence:.2f})")

    def _is_dog_visible(self) -> bool:
        """Check if a dog has been seen recently"""
        if self.last_dog_seen_time == 0:
            return False
        elapsed = time.time() - self.last_dog_seen_time
        return elapsed < DOG_VISIBILITY_TIMEOUT

    def _start_intervention(self, dog_id: Optional[str], dog_name: Optional[str]):
        """Start an intervention sequence"""
        with self._state_lock:
            if self.fsm_state != SGState.LISTENING:
                return

            # Calculate escalation level
            escalation_level = self._get_escalation_level()

            # Record intervention start
            self.fsm_state = SGState.INTERVENTION_START
            self.intervention = InterventionState(
                started_at=time.time(),
                dog_id=dog_id,
                dog_name=dog_name,
                escalation_level=escalation_level
            )

            self.interventions_triggered += 1
            self.last_intervention_time = time.time()
            self.escalation_events.append(time.time())

            # Log to database (intervention start - outcome not yet known)
            self.store.log_sg_intervention(
                session_id=self.session_id,
                dog_id=dog_id,
                dog_name=dog_name,
                escalation_level=escalation_level
            )

            logger.info(f"Starting intervention (level {escalation_level}) for {dog_name or dog_id or 'unknown dog'}")

    def _get_escalation_level(self) -> int:
        """Calculate current escalation level based on recent interventions"""
        escalation_config = self.config.get('escalation', {})
        window_minutes = escalation_config.get('window_minutes', 60)
        max_level = escalation_config.get('max_level', 3)

        # Clean old events outside window
        cutoff = time.time() - (window_minutes * 60)
        self.escalation_events = [t for t in self.escalation_events if t > cutoff]

        # Count events in window
        event_count = len(self.escalation_events)

        # Level 1: 0-2 events, Level 2: 3 events, Level 3: 4+ events
        if event_count <= 2:
            return 1
        elif event_count == 3:
            return 2
        else:
            return min(3, max_level)

    def _run_loop(self):
        """Main state machine loop"""
        logger.info("Silent Guardian loop started")

        while self.running:
            try:
                # Check system mode hasn't changed
                if self.state.get_mode() != SystemMode.SILENT_GUARDIAN:
                    logger.info("Mode changed externally, stopping Silent Guardian")
                    break

                # Check 8-hour session limit
                if self._check_session_expired():
                    self._reset_session()

                # Run state machine
                self._process_state()

                # Brief sleep
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Silent Guardian loop error: {e}")
                time.sleep(1.0)

        logger.info("Silent Guardian loop ended")

    def _check_session_expired(self) -> bool:
        """Check if 8-hour session has expired"""
        if self.session_start_time is None:
            return False

        elapsed = time.time() - self.session_start_time
        return elapsed >= self.max_session_duration

    def _reset_session(self):
        """Reset session after 8 hours"""
        logger.info("8-hour session expired - resetting")

        # End old session in database
        if self.session_id:
            self.store.end_silent_guardian_session(
                self.session_id,
                self.interventions_triggered,
                self.treats_dispensed
            )

        # Start new session
        self.session_id = self.store.start_silent_guardian_session()
        self.session_start_time = time.time()
        self.treats_dispensed = 0
        self.interventions_triggered = 0
        self.escalation_events = []  # Reset escalation tracking too

        # Publish event
        publish_system_event('silent_guardian_session_reset', {
            'new_session_id': self.session_id,
            'max_treats': self.config.get('session_limits', {}).get('max_treats', 11)
        }, 'silent_guardian')

        logger.info(f"New session started: {self.session_id}")

    def _process_state(self):
        """Process current FSM state"""
        if self.fsm_state == SGState.LISTENING:
            # Passive - waiting for bark events
            pass

        elif self.fsm_state == SGState.INTERVENTION_START:
            # Execute intervention commands
            self._execute_intervention()
            self.fsm_state = SGState.WAITING_FOR_QUIET
            self.intervention.quiet_start_time = time.time()

        elif self.fsm_state == SGState.WAITING_FOR_QUIET:
            # Check for timeout (90 seconds of persistent barking)
            if self._check_intervention_timeout():
                return  # Moved to cooldown

            # Check if we need to issue another quiet command
            self._check_escalating_quiet()

            # Check if quiet achieved
            if self._check_quiet_achieved():
                # Stop calming music before calling sequence
                if self.calming_music_playing:
                    self._stop_calming_music()
                # Start calling dog sequence
                self.fsm_state = SGState.CALLING_DOG
                self.intervention.calling_started_at = time.time()
                self.intervention.come_commands_issued = 0
                logger.info("Quiet achieved - starting dog calling sequence")

        elif self.fsm_state == SGState.CALLING_DOG:
            # Execute the calling sequence with proper timing
            self._process_calling_sequence()

        elif self.fsm_state == SGState.WAITING_FOR_DOG:
            # Wait for dog to be visible before reward
            self._process_waiting_for_dog()

        elif self.fsm_state == SGState.REWARD_SEQUENCE:
            # Execute reward sequence (only if dog visible)
            self._execute_reward_sequence()
            self.fsm_state = SGState.COOLDOWN

        elif self.fsm_state == SGState.COOLDOWN:
            # NON-BLOCKING cooldown using timestamp check
            # (Blocking sleep was causing safety heartbeat failures)

            # Initialize cooldown timer if not set
            if not hasattr(self, '_cooldown_start') or self._cooldown_start is None:
                self._cooldown_start = time.time()
                if self.intervention and self.intervention.gave_up:
                    self._cooldown_duration = 120.0  # 2 minutes for gave_up
                    logger.info("2-minute shutdown cooldown started (dog ignored commands)")
                else:
                    self._cooldown_duration = 2.0  # Brief cooldown otherwise
                    logger.info("Brief cooldown started")
                return  # Wait for next iteration

            # Check if cooldown elapsed
            elapsed = time.time() - self._cooldown_start
            if elapsed < self._cooldown_duration:
                return  # Still in cooldown, wait for next iteration

            # Cooldown complete - reset and return to listening
            self._cooldown_start = None
            self._cooldown_duration = 0
            self.intervention = None  # Reset intervention state
            self.fsm_state = SGState.LISTENING
            logger.info("Returning to LISTENING state")

    def _process_calling_sequence(self):
        """
        Execute the dog calling sequence with proper timing:
        1. DOG COME (dogs_come.mp3)
        2. Wait 20 seconds
        3. COME AGAIN (dogs_come.mp3)
        4. Wait 15 seconds
        5. TREAT (treat.mp3) - "Do you want a treat?"
        Then transition to WAITING_FOR_DOG
        """
        if self.intervention is None:
            self.fsm_state = SGState.LISTENING
            return

        now = time.time()
        elapsed = now - self.intervention.calling_started_at
        commands_issued = self.intervention.come_commands_issued

        # If dog is already visible and sitting, skip ahead
        if self._is_dog_visible() and self.intervention.dog_sitting_confirmed:
            logger.info("Dog already visible and sitting - proceeding to reward")
            self.fsm_state = SGState.REWARD_SEQUENCE
            return

        # Timing: 0s: COME, 20s: COME AGAIN, 35s: TREAT, then wait for dog
        if commands_issued == 0:
            # First COME command
            logger.info("Calling dog: COME (1st)")
            self._play_audio('dogs_come.mp3')
            self.intervention.come_commands_issued = 1
            self.intervention.last_come_command_time = now

        elif commands_issued == 1 and elapsed >= 20.0:
            # Second COME command after 20 seconds
            logger.info("Calling dog: COME (2nd)")
            self._play_audio('dogs_come.mp3')
            self.intervention.come_commands_issued = 2
            self.intervention.last_come_command_time = now

        elif commands_issued == 2 and elapsed >= 35.0:
            # TREAT prompt after 35 seconds (20 + 15)
            logger.info("Calling dog: TREAT?")
            self._play_audio('treat.mp3')
            self.intervention.come_commands_issued = 3
            self.intervention.last_come_command_time = now

        elif commands_issued >= 3 and elapsed >= 38.0:
            # After TREAT prompt, wait for dog to be visible
            self.fsm_state = SGState.WAITING_FOR_DOG
            logger.info("Calling sequence complete - waiting for dog visibility")

    def _process_waiting_for_dog(self):
        """
        Wait for dog to be visible and SITTING before dispensing treat.

        Reward conditions (must meet ONE):
        1. Dog is visible AND sitting (behavior == 'sit')
        2. Dog is visible AND has been quiet for 20+ seconds

        Does NOT reward just for being visible!
        """
        if self.intervention is None:
            self.fsm_state = SGState.LISTENING
            return

        now = time.time()

        # Check timeout - give up after 3 minutes of waiting
        elapsed = now - self.intervention.calling_started_at
        if elapsed > 180.0:  # 3 minutes total
            logger.warning("Timeout waiting for dog to sit - giving up")
            self.intervention.gave_up = True
            self.fsm_state = SGState.COOLDOWN
            return

        # Check if dog is visible
        if not self._is_dog_visible():
            # Dog not visible - reset quiet timer and keep waiting
            self.intervention.quiet_start_time = None
            if int(elapsed) % 10 == 0:
                logger.info(f"Waiting for dog to be visible... ({elapsed:.0f}s)")
            return

        # Dog is visible! Check if sitting
        if self.intervention.dog_sitting_confirmed:
            logger.info("Dog confirmed SITTING - proceeding to reward!")
            self.fsm_state = SGState.REWARD_SEQUENCE
            return

        # Dog visible but not sitting - issue SIT command once
        if not hasattr(self.intervention, 'sit_command_issued') or not self.intervention.sit_command_issued:
            logger.info("Dog visible - issuing SIT command")
            self._play_audio('sit.mp3')
            self.intervention.sit_command_issued = True
            self.intervention.quiet_start_time = now  # Start quiet timer
            return

        # Check if dog has been quiet for 20 seconds while visible
        if self.intervention.quiet_start_time:
            quiet_duration = now - self.intervention.quiet_start_time

            # Log progress every 5 seconds
            if int(quiet_duration) % 5 == 0 and int(quiet_duration) > 0:
                logger.info(f"Dog visible, waiting for sit or quiet... ({quiet_duration:.0f}s/20s)")

            if quiet_duration >= 20.0:
                logger.info(f"Dog quiet for {quiet_duration:.1f}s while visible - proceeding to reward")
                self.fsm_state = SGState.REWARD_SEQUENCE
                return
        else:
            # Start quiet timer
            self.intervention.quiet_start_time = now

    def _execute_intervention(self):
        """Execute intervention based on escalation level"""
        level = self.intervention.escalation_level
        dog_name = self.intervention.dog_name

        sequences = self.config.get('intervention_sequences', {})
        level_config = sequences.get(f'level_{level}', {})

        logger.info(f"Executing level {level} intervention")

        # Track first quiet command time for escalation
        self.intervention.last_quiet_command_time = time.time()

        # LED indication
        if self.led:
            self.led.set_pattern('attention', duration=3.0)

        if level == 1:
            # Level 1: "[Dog name], quiet" or just "quiet"
            if dog_name:
                self._play_audio(f'{dog_name.lower()}.mp3')
                time.sleep(0.5)
            self._play_audio('quiet.mp3')

        elif level == 2:
            # Level 2: quiet x2, no, come, quiet
            self._play_audio('quiet.mp3')
            time.sleep(0.5)
            self._play_audio('quiet.mp3')
            time.sleep(0.5)
            self._play_audio('no.mp3')
            time.sleep(1.0)
            self._play_audio('dogs_come.mp3')
            time.sleep(2.0)
            self._play_audio('quiet.mp3')

        elif level >= 3:
            # Level 3: quiet + calming music (loops until quiet achieved or bark)
            self._play_audio('quiet.mp3')
            time.sleep(1.0)

            # Start calming music - will loop until bark or quiet achieved
            if level_config.get('play_calming_music', True):
                self._start_calming_music()

    def _check_quiet_achieved(self) -> bool:
        """Check if quiet requirement has been met"""
        if self.intervention.quiet_start_time is None:
            return False

        level = self.intervention.escalation_level
        sequences = self.config.get('intervention_sequences', {})
        level_config = sequences.get(f'level_{level}', {})

        now = time.time()
        quiet_duration = now - self.intervention.quiet_start_time

        if level < 3:
            # Levels 1-2: Single quiet period required
            required_seconds = level_config.get('quiet_required_seconds', 10)

            if quiet_duration >= required_seconds:
                logger.info(f"Quiet achieved! {quiet_duration:.1f}s >= {required_seconds}s")
                return True
        else:
            # Level 3: Multiple quiet periods required
            period_duration = level_config.get('quiet_period_duration', 20)
            periods_required = level_config.get('quiet_periods_required', 2)

            if quiet_duration >= period_duration:
                self.intervention.quiet_periods_achieved += 1
                self.intervention.quiet_start_time = now  # Reset for next period

                logger.info(f"Quiet period {self.intervention.quiet_periods_achieved}/{periods_required} achieved")

                if self.intervention.quiet_periods_achieved >= periods_required:
                    return True

        return False

    def _check_intervention_timeout(self) -> bool:
        """Check if intervention has timed out (90 seconds of persistent barking)"""
        if self.intervention is None:
            return False

        elapsed = time.time() - self.intervention.started_at
        max_intervention_time = 90.0  # 90 seconds max

        if elapsed >= max_intervention_time:
            logger.warning(f"Intervention timeout after {elapsed:.1f}s - dog ignored {self.intervention.quiet_commands_issued} quiet commands")
            self.intervention.gave_up = True

            # Log the failed intervention
            self.store.log_sg_intervention(
                session_id=self.session_id,
                dog_id=self.intervention.dog_id,
                dog_name=self.intervention.dog_name,
                escalation_level=self.intervention.escalation_level,
                quiet_achieved=False,
                treat_given=False
            )

            # Enter extended cooldown (2 minutes)
            self.fsm_state = SGState.COOLDOWN
            logger.info("Entering 2-minute shutdown cooldown")
            return True

        return False

    def _check_escalating_quiet(self):
        """Issue escalating quiet commands when dog keeps barking"""
        if self.intervention is None:
            return

        # Only escalate if dog has barked since last quiet command
        if self.intervention.barks_since_intervention == 0:
            return

        # Max 10 quiet commands total
        if self.intervention.quiet_commands_issued >= 10:
            return

        now = time.time()
        time_since_last_command = now - self.intervention.last_quiet_command_time

        # Escalating frequency: intervals get shorter as we issue more commands
        # Command 1: initial (at start)
        # Command 2: after 15s, Command 3: +12s, Command 4: +10s, Command 5: +8s
        # Commands 6-10: every 6 seconds
        intervals = [15, 12, 10, 8, 6, 6, 6, 6, 6]  # 9 additional intervals
        cmd_index = self.intervention.quiet_commands_issued - 1  # 0-indexed
        if cmd_index < len(intervals):
            required_interval = intervals[cmd_index]
        else:
            required_interval = 6

        if time_since_last_command >= required_interval:
            # Issue another quiet command
            self.intervention.quiet_commands_issued += 1
            self.intervention.last_quiet_command_time = now
            self.intervention.barks_since_intervention = 0  # Reset bark counter

            logger.info(f"Escalating: quiet command #{self.intervention.quiet_commands_issued}/10")

            # LED flash for attention
            if self.led:
                self.led.set_pattern('attention', duration=1.0)

            # Play quiet with increasing firmness
            if self.intervention.quiet_commands_issued >= 7:
                # Very firm: quiet + no
                self._play_audio('quiet.mp3')
                time.sleep(0.3)
                self._play_audio('no.mp3')
            elif self.intervention.quiet_commands_issued >= 4:
                # Firm: double quiet
                self._play_audio('quiet.mp3')
                time.sleep(0.3)
                self._play_audio('quiet.mp3')
            else:
                # Normal quiet
                self._play_audio('quiet.mp3')

            # Reset quiet timer since we just gave a command
            self.intervention.quiet_start_time = time.time()

    def _execute_reward_sequence(self):
        """
        Execute reward sequence for successful quiet + dog visible.
        By this point, the dog calling sequence (COME → COME → TREAT) is done
        and the dog should be visible. Now we just:
        1. Final QUIET reminder
        2. Dispense treat (only if dog confirmed visible)
        3. GOOD DOG
        """
        level = self.intervention.escalation_level
        sequences = self.config.get('intervention_sequences', {})
        level_config = sequences.get(f'level_{level}', {})

        if not level_config.get('reward_on_success', True):
            logger.info("Reward disabled for this level")
            return

        # Check session treat limit
        max_treats = self.config.get('session_limits', {}).get('max_treats', 11)
        if self.treats_dispensed >= max_treats:
            logger.info(f"Session treat limit reached ({self.treats_dispensed}/{max_treats})")
            self._play_audio('good_dog.mp3')
            return

        # CRITICAL: Verify dog is visible before dispensing
        if not self._is_dog_visible():
            logger.warning("Dog not visible - skipping treat dispense")
            self._play_audio('quiet.mp3')
            return

        logger.info("Executing reward sequence (dog visible)")

        # LED celebration
        if self.led:
            self.led.celebration_sequence(3.0)

        # Final QUIET reminder
        self._play_audio('quiet.mp3')
        time.sleep(1.5)

        # Dispense treat
        self._dispense_treat()

        # Play "GOOD DOG" after treat
        time.sleep(0.5)
        self._play_audio('good_dog.mp3')

        # Log success
        self.store.log_sg_intervention(
            session_id=self.session_id,
            dog_id=self.intervention.dog_id,
            dog_name=self.intervention.dog_name,
            escalation_level=level,
            quiet_achieved=True,
            treat_given=True
        )

    def _play_audio(self, filename: str, is_music: bool = False):
        """Play audio file"""
        try:
            audio_config = self.config.get('audio_paths', {})

            if is_music:
                base = audio_config.get('songs', '/home/morgan/dogbot/VOICEMP3/songs')
            else:
                base = audio_config.get('talks', '/home/morgan/dogbot/VOICEMP3/talks')

            # Handle if filename already has path
            if filename.startswith('/') or filename.startswith('songs/'):
                full_path = os.path.join('/home/morgan/dogbot/VOICEMP3', filename)
            else:
                full_path = os.path.join(base, filename)

            if os.path.exists(full_path):
                if self.audio:
                    self.audio.play_file(full_path)
                    logger.debug(f"Playing: {full_path}")
            else:
                logger.warning(f"Audio file not found: {full_path}")

        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    def _start_calming_music(self):
        """Start playing calming music (loops until stopped)"""
        try:
            # Use the dog desensitization music
            music_file = '/home/morgan/dogbot/VOICEMP3/songs/DOG DESENSITISATION MUSIC! to help train your dog, improve behaviour. Sound effects included.mp3'

            if os.path.exists(music_file):
                if self.audio:
                    # Play with looping enabled
                    self.audio.play_file(music_file, loop=True)
                    self.calming_music_playing = True
                    logger.info("Started calming music (looping)")
            else:
                logger.warning(f"Calming music file not found: {music_file}")

        except Exception as e:
            logger.error(f"Calming music start error: {e}")

    def _stop_calming_music(self):
        """Stop calming music playback"""
        try:
            if self.audio and self.calming_music_playing:
                self.audio.stop()
                self.calming_music_playing = False
                logger.info("Calming music stopped")
        except Exception as e:
            logger.error(f"Calming music stop error: {e}")

    def _dispense_treat(self):
        """Dispense a treat"""
        try:
            if self.dispenser:
                self.dispenser.dispense_treat(
                    dog_id=self.intervention.dog_id if self.intervention else None,
                    reason='silent_guardian_reward'
                )
                self.treats_dispensed += 1
                logger.info(f"Treat dispensed ({self.treats_dispensed} this session)")

                # Log treat to store
                self.store.log_reward(
                    dog_id=self.intervention.dog_id or 'unknown',
                    behavior='quiet',
                    confidence=1.0,
                    success=True,
                    treats_dispensed=1,
                    mission_name='silent_guardian'
                )

        except Exception as e:
            logger.error(f"Treat dispense error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current mode status"""
        return {
            'running': self.running,
            'fsm_state': self.fsm_state.value if self.fsm_state else None,
            'session_id': self.session_id,
            'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
            'interventions_triggered': self.interventions_triggered,
            'treats_dispensed': self.treats_dispensed,
            'treats_remaining': max(0, self.config.get('session_limits', {}).get('max_treats', 11) - self.treats_dispensed),
            'current_escalation_level': self._get_escalation_level(),
            'last_bark_time': self.last_bark_time,
            'last_intervention_time': self.last_intervention_time
        }

    def cleanup(self):
        """Clean up resources"""
        self.stop()


# Singleton instance
_silent_guardian_instance = None
_silent_guardian_lock = threading.Lock()


def get_silent_guardian_mode() -> SilentGuardianMode:
    """Get or create Silent Guardian mode instance (singleton)"""
    global _silent_guardian_instance
    if _silent_guardian_instance is None:
        with _silent_guardian_lock:
            if _silent_guardian_instance is None:
                _silent_guardian_instance = SilentGuardianMode()
    return _silent_guardian_instance


def main():
    """Test Silent Guardian mode"""
    import signal

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    mode = SilentGuardianMode()

    def signal_handler(sig, frame):
        print("\nShutting down...")
        mode.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("\n=== SILENT GUARDIAN MODE TEST ===")
    print("Press Ctrl+C to exit\n")

    mode.start()

    # Keep main thread alive
    while mode.running:
        status = mode.get_status()
        print(f"\rState: {status['fsm_state']} | Interventions: {status['interventions_triggered']} | Treats: {status['treats_dispensed']}", end='')
        time.sleep(1)


if __name__ == "__main__":
    main()
