#!/usr/bin/env python3
"""
Silent Guardian Mode - WIM-Z Primary Product Mode
Bark intervention with 3-level escalation system + anti-treat-farming

Escalation Flow:
- Level 1 (1st-2nd intervention in hour): "[dog name], quiet" → quiet period → reward
- Level 2 (3rd intervention in hour): quiet, quiet, no, come, quiet → quiet period → reward
- Level 3 (4th+ intervention in hour): quiet + CALMING MUSIC → 2x quiet periods → reward

Anti-Farming Features:
- After a treat: 10 min eligibility cooldown (verbal praise only, no treats)
- Progressive quiet: Each intervention requires longer quiet (20s → 30s → 45s → 60s)
- Min 2 min between treats as backup

Resets:
- If dog barks during intervention → restart that level's sequence
- After 90s with no success → give up, enter 2-min cooldown
- After 60 minutes of quiet → reset escalation level and progressive requirements
- Mode switch → full reset (new session)
"""

import os
import sys
import time
import threading
import logging
import yaml
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports
from core.bus import get_bus, publish_system_event
from core.state import get_state, SystemMode
from core.store import get_store
from core.bark_frequency_tracker import get_bark_frequency_tracker

# Services
from services.media.usb_audio import get_usb_audio_service, set_agc
from services.reward.dispenser import get_dispenser_service
from services.media.led import get_led_service

logger = logging.getLogger(__name__)


class SGState(Enum):
    """Silent Guardian states"""
    LISTENING = "listening"           # Waiting for bark threshold
    INTERVENTION = "intervention"     # Running intervention sequence
    COOLDOWN = "cooldown"            # Brief pause after intervention
    GAVE_UP = "gave_up"              # Intervention timed out


class SilentGuardianMode:
    """
    Silent Guardian - Bark intervention with 3-level escalation

    Level 1: Gentle reminder (dog name + quiet)
    Level 2: Firm request (multiple commands)
    Level 3: Calming music mode
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

        # FSM state
        self.fsm_state = SGState.LISTENING

        # Intervention tracking
        self.intervention_start_time = 0.0
        self.intervention_step = 0
        self.last_step_time = 0.0
        self.bark_during_intervention = False
        self.intervention_dog_id = None
        self.intervention_dog_name = None
        self.quiet_start_time = 0.0  # When dog started being quiet

        # Escalation tracking
        self.escalation_events: List[float] = []  # Timestamps of interventions
        self.current_escalation_level = 1
        self.quiet_periods_achieved = 0  # For Level 3 (need 2 periods)
        self.calming_music_playing = False
        self.last_intervention_time = 0.0  # For escalation reset

        # Session tracking
        self.session_id = None
        self.session_start_time = None
        self.treats_dispensed = 0
        self.interventions_triggered = 0

        # Bark tracking
        self.last_bark_time = 0.0

        # Session duration limit (8 hours)
        self.max_session_duration = 8 * 60 * 60

        # Cooldown tracking
        self._cooldown_start = None
        self._cooldown_duration = 2.0  # 2 seconds between interventions
        self._gave_up_cooldown = 120.0  # 2 minutes after giving up

        # Timeouts
        self.intervention_timeout = 90.0  # Max 90 seconds per intervention

        # Anti-treat-farming: eligibility cooldown after treats
        self.last_treat_time = 0.0  # When last treat was dispensed
        self.treat_eligibility_cooldown = 600.0  # 10 minutes before treats available again
        self.consecutive_interventions = 0  # For progressive quiet requirements
        self.treat_eligible = True  # Whether treats can be given

        logger.info("Silent Guardian mode initialized (3-level escalation + anti-farming)")

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
        """Default configuration"""
        return {
            'bark_detection': {
                'threshold': 2,
                'window_seconds': 60,
                'loudness_threshold_db': -25,
                'confidence_minimum': 0.35
            },
            'escalation': {
                'window_minutes': 60,
                'reset_after_quiet_minutes': 60,
                'max_level': 3
            },
            'intervention_sequences': {
                'level_1': {'quiet_required_seconds': 20},
                'level_2': {'quiet_required_seconds': 30},
                'level_3': {'quiet_period_duration': 20, 'quiet_periods_required': 2}
            },
            'session_limits': {
                'max_treats': 11,
                'min_time_between_treats': 120
            },
            'audio_paths': {
                'talks': '/home/morgan/dogbot/VOICEMP3/talks',
                'songs': '/home/morgan/dogbot/VOICEMP3/songs',
                'calming_music': 'songs/DOG DESENSITISATION MUSIC! to help train your dog, improve behaviour. Sound effects included.mp3'
            }
        }

    def _get_escalation_level(self) -> int:
        """
        Calculate current escalation level based on recent interventions

        Returns:
            1 = 0-2 interventions in last hour
            2 = 3 interventions in last hour
            3 = 4+ interventions in last hour
        """
        now = time.time()
        escalation_config = self.config.get('escalation', {})
        window_minutes = escalation_config.get('window_minutes', 60)
        max_level = escalation_config.get('max_level', 3)

        # Clean old events outside window
        window_seconds = window_minutes * 60
        self.escalation_events = [t for t in self.escalation_events if now - t < window_seconds]

        # Calculate level based on count
        count = len(self.escalation_events)
        if count <= 2:
            level = 1
        elif count == 3:
            level = 2
        else:
            level = 3

        return min(level, max_level)

    def _check_escalation_reset(self):
        """Reset escalation level after extended quiet period"""
        if self.last_intervention_time == 0.0:
            return

        now = time.time()
        escalation_config = self.config.get('escalation', {})
        reset_minutes = escalation_config.get('reset_after_quiet_minutes', 60)

        elapsed_minutes = (now - self.last_intervention_time) / 60

        if elapsed_minutes >= reset_minutes:
            if self.escalation_events:
                logger.info(f"Escalation reset after {elapsed_minutes:.1f} minutes of quiet")
                self.escalation_events = []
            # Also reset consecutive interventions when escalation resets
            if self.consecutive_interventions > 0:
                self.consecutive_interventions = 0
                logger.info("Progressive quiet requirements reset")

    def _check_treat_eligibility(self):
        """
        Check and update treat eligibility based on time since last treat.
        After a treat, dogs must be quiet for 10 minutes before treats are available again.
        Still corrects barking, but only gives verbal praise during cooldown.
        """
        if self.last_treat_time == 0.0:
            self.treat_eligible = True
            return

        now = time.time()
        elapsed = now - self.last_treat_time

        if elapsed >= self.treat_eligibility_cooldown:
            if not self.treat_eligible:
                logger.info(f"Treat eligibility restored after {elapsed/60:.1f} minutes")
                self.treat_eligible = True
                # Reset consecutive interventions when eligibility restores
                self.consecutive_interventions = 0
        else:
            self.treat_eligible = False

    def _get_progressive_quiet_duration(self, base_duration: float) -> float:
        """
        Get the required quiet duration based on consecutive interventions.
        Each successful intervention increases the required quiet time.

        Progressive scale:
        - 1st intervention: base (e.g., 20s)
        - 2nd intervention: base * 1.5 (e.g., 30s)
        - 3rd intervention: base * 2.25 (e.g., 45s)
        - 4th+ intervention: base * 3 (e.g., 60s) - capped

        Returns:
            Required quiet duration in seconds
        """
        multipliers = [1.0, 1.5, 2.25, 3.0]  # Progressive multipliers
        idx = min(self.consecutive_interventions, len(multipliers) - 1)
        duration = base_duration * multipliers[idx]

        if self.consecutive_interventions > 0:
            logger.debug(f"Progressive quiet: {duration:.0f}s (intervention #{self.consecutive_interventions + 1})")

        return duration

    def start(self) -> bool:
        """Start Silent Guardian mode"""
        if self.running:
            logger.warning("Silent Guardian already running")
            return True

        try:
            # Disable AGC for bark detection
            set_agc(False)
            logger.info("AGC disabled for bark detection")

            # Start database session
            self.session_id = self.store.start_silent_guardian_session()
            self.session_start_time = time.time()
            self.treats_dispensed = 0
            self.interventions_triggered = 0
            self.escalation_events = []

            # Reset anti-farming state
            self.last_treat_time = 0.0
            self.consecutive_interventions = 0
            self.treat_eligible = True

            # Subscribe to bark events
            self.bus.subscribe('audio', self._on_audio_event)

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

        # Stop calming music if playing
        self._stop_calming_music()

        # End database session
        if self.session_id:
            self.store.end_silent_guardian_session(
                self.session_id,
                self.interventions_triggered,
                self.treats_dispensed
            )

        # Wait for thread
        if self.mode_thread and self.mode_thread.is_alive():
            self.mode_thread.join(timeout=2.0)

        # Re-enable AGC
        set_agc(True)
        logger.info("AGC re-enabled")

        # Publish stop event
        publish_system_event('silent_guardian_stopped', {
            'session_id': self.session_id,
            'interventions': self.interventions_triggered,
            'treats': self.treats_dispensed
        }, 'silent_guardian')

        logger.info("Silent Guardian stopped")

    def _on_audio_event(self, event):
        """Handle bark events"""
        if event.subtype != 'bark_detected':
            return

        # Extract bark info
        dog_id = event.data.get('dog_id')
        dog_name = event.data.get('dog_name')
        confidence = event.data.get('confidence', 0.0)
        loudness_db = event.data.get('loudness_db', -30)

        # Check thresholds
        bark_config = self.config.get('bark_detection', {})
        loudness_threshold = bark_config.get('loudness_threshold_db', -25)
        confidence_minimum = bark_config.get('confidence_minimum', 0.35)

        if loudness_db < loudness_threshold:
            logger.debug(f"Ignoring quiet bark: {loudness_db:.1f}dB < {loudness_threshold}dB")
            return

        if confidence < confidence_minimum:
            logger.debug(f"Ignoring low-confidence: {confidence:.2f} < {confidence_minimum}")
            return

        logger.info(f"Bark detected: {dog_name or dog_id or 'unknown'} (conf: {confidence:.2f}, loud: {loudness_db:.1f}dB)")
        self.last_bark_time = time.time()

        # Handle based on current state
        if self.fsm_state == SGState.LISTENING:
            # Check if threshold exceeded
            threshold = bark_config.get('threshold', 2)
            result = self.bark_tracker.check_threshold(dog_id or 'unknown', threshold)

            if result:
                # Start intervention
                self._start_intervention(dog_id, dog_name)

        elif self.fsm_state == SGState.INTERVENTION:
            # Bark during intervention - reset quiet timer and sequence
            self.bark_during_intervention = True
            self.quiet_start_time = 0.0
            self.quiet_periods_achieved = 0
            logger.info("Bark during intervention - restarting sequence")

    def _start_intervention(self, dog_id: Optional[str], dog_name: Optional[str]):
        """Start the intervention sequence"""
        with self._state_lock:
            if self.fsm_state != SGState.LISTENING:
                return

            # Record escalation event
            now = time.time()
            self.escalation_events.append(now)
            self.last_intervention_time = now

            # Calculate escalation level
            self.current_escalation_level = self._get_escalation_level()

            self.fsm_state = SGState.INTERVENTION
            self.intervention_start_time = now
            self.intervention_step = 0
            self.last_step_time = 0.0
            self.bark_during_intervention = False
            self.intervention_dog_id = dog_id
            self.intervention_dog_name = dog_name
            self.quiet_start_time = 0.0
            self.quiet_periods_achieved = 0

            self.interventions_triggered += 1

            # Log to database with actual escalation level
            self.store.log_sg_intervention(
                session_id=self.session_id,
                dog_id=dog_id,
                dog_name=dog_name,
                escalation_level=self.current_escalation_level
            )

            logger.info(f"Starting Level {self.current_escalation_level} intervention for {dog_name or dog_id or 'unknown'}")

    def _run_loop(self):
        """Main loop"""
        logger.info("Silent Guardian loop started")

        while self.running:
            try:
                # Check mode hasn't changed
                if self.state.get_mode() != SystemMode.SILENT_GUARDIAN:
                    logger.info("Mode changed, stopping")
                    break

                # Check 8-hour session limit
                if self._check_session_expired():
                    self._reset_session()

                # Check escalation reset (60 min quiet)
                self._check_escalation_reset()

                # Check treat eligibility (10 min cooldown after treats)
                self._check_treat_eligibility()

                # Process state machine
                self._process_state()

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(1.0)

        logger.info("Silent Guardian loop ended")

    def _check_session_expired(self) -> bool:
        """Check if 8-hour session expired"""
        if self.session_start_time is None:
            return False
        elapsed = time.time() - self.session_start_time
        return elapsed >= self.max_session_duration

    def _reset_session(self):
        """Reset session after 8 hours"""
        logger.info("8-hour session expired - resetting")

        if self.session_id:
            self.store.end_silent_guardian_session(
                self.session_id,
                self.interventions_triggered,
                self.treats_dispensed
            )

        self.session_id = self.store.start_silent_guardian_session()
        self.session_start_time = time.time()
        self.treats_dispensed = 0
        self.interventions_triggered = 0
        self.escalation_events = []

        logger.info(f"New session: {self.session_id}")

    def _process_state(self):
        """Process current state"""
        if self.fsm_state == SGState.LISTENING:
            # Passive - waiting for bark events
            pass

        elif self.fsm_state == SGState.INTERVENTION:
            self._process_intervention()

        elif self.fsm_state == SGState.COOLDOWN:
            self._process_cooldown()

        elif self.fsm_state == SGState.GAVE_UP:
            self._process_gave_up()

    def _process_intervention(self):
        """Process the intervention based on escalation level"""
        now = time.time()

        # Check intervention timeout (90 seconds)
        if now - self.intervention_start_time >= self.intervention_timeout:
            logger.warning(f"Intervention timed out after {self.intervention_timeout}s - giving up")
            self._stop_calming_music()
            self.fsm_state = SGState.GAVE_UP
            self._cooldown_start = now

            # Log failed intervention
            self.store.log_sg_intervention(
                session_id=self.session_id,
                dog_id=self.intervention_dog_id,
                dog_name=self.intervention_dog_name,
                escalation_level=self.current_escalation_level,
                quiet_achieved=False,
                treat_given=False
            )
            return

        # Check for bark during intervention - restart sequence
        if self.bark_during_intervention:
            logger.info(f"Restarting Level {self.current_escalation_level} sequence due to bark")
            self.intervention_step = 0
            self.last_step_time = 0.0
            self.bark_during_intervention = False
            self.quiet_start_time = 0.0
            self.quiet_periods_achieved = 0
            # Fall through to execute step 0

        # Route to level-specific handler
        if self.current_escalation_level == 1:
            self._process_level_1(now)
        elif self.current_escalation_level == 2:
            self._process_level_2(now)
        else:
            self._process_level_3(now)

    def _process_level_1(self, now: float):
        """
        Level 1: Gentle reminder
        Step 0: Play dog name (if known) + quiet
        Step 1: Wait for quiet period (progressive) → reward
        """
        level_config = self.config.get('intervention_sequences', {}).get('level_1', {})
        base_quiet = level_config.get('quiet_required_seconds', 20)
        quiet_required = self._get_progressive_quiet_duration(base_quiet)

        if self.intervention_step == 0:
            if self.last_step_time == 0.0:
                logger.info("Level 1 - Step 0: Playing name + quiet")
                if self.led:
                    self.led.set_pattern('attention', duration=2.0)

                # Play dog name if known
                if self.intervention_dog_name:
                    name_file = f"{self.intervention_dog_name.lower()}.mp3"
                    self._play_audio(name_file)
                    time.sleep(0.5)

                self._play_audio('quiet.mp3')
                self.last_step_time = now
                self.quiet_start_time = now
                self.intervention_step = 1

        elif self.intervention_step == 1:
            # Check if quiet period achieved
            if self.quiet_start_time > 0:
                quiet_duration = now - self.quiet_start_time
                if quiet_duration >= quiet_required:
                    logger.info(f"Level 1 - SUCCESS: {quiet_duration:.1f}s quiet achieved")
                    self._give_reward()
                    self.fsm_state = SGState.COOLDOWN
                    self._cooldown_start = now

    def _process_level_2(self, now: float):
        """
        Level 2: Firm request
        Step 0: quiet
        Step 1: quiet (after 0.5s)
        Step 2: no (after 0.5s)
        Step 3: come (after 1s)
        Step 4: quiet (after 2s)
        Step 5: Wait for quiet period (progressive) → reward
        """
        level_config = self.config.get('intervention_sequences', {}).get('level_2', {})
        base_quiet = level_config.get('quiet_required_seconds', 30)
        quiet_required = self._get_progressive_quiet_duration(base_quiet)

        if self.intervention_step == 0:
            if self.last_step_time == 0.0:
                logger.info("Level 2 - Step 0: quiet")
                if self.led:
                    self.led.set_pattern('attention', duration=2.0)
                self._play_audio('quiet.mp3')
                self.last_step_time = now
                self.intervention_step = 1

        elif self.intervention_step == 1:
            if now - self.last_step_time >= 0.5:
                logger.info("Level 2 - Step 1: quiet")
                self._play_audio('quiet.mp3')
                self.last_step_time = now
                self.intervention_step = 2

        elif self.intervention_step == 2:
            if now - self.last_step_time >= 0.5:
                logger.info("Level 2 - Step 2: no")
                self._play_audio('no.mp3')
                self.last_step_time = now
                self.intervention_step = 3

        elif self.intervention_step == 3:
            if now - self.last_step_time >= 1.0:
                logger.info("Level 2 - Step 3: come")
                self._play_audio('dogs_come.mp3')
                self.last_step_time = now
                self.intervention_step = 4

        elif self.intervention_step == 4:
            if now - self.last_step_time >= 2.0:
                logger.info("Level 2 - Step 4: quiet")
                self._play_audio('quiet.mp3')
                self.last_step_time = now
                self.quiet_start_time = now
                self.intervention_step = 5

        elif self.intervention_step == 5:
            # Check if quiet period achieved
            if self.quiet_start_time > 0:
                quiet_duration = now - self.quiet_start_time
                if quiet_duration >= quiet_required:
                    logger.info(f"Level 2 - SUCCESS: {quiet_duration:.1f}s quiet achieved")
                    self._give_reward()
                    self.fsm_state = SGState.COOLDOWN
                    self._cooldown_start = now

    def _process_level_3(self, now: float):
        """
        Level 3: Calming music mode
        Step 0: quiet + start calming music
        Step 1: Wait for 2x quiet periods (progressive duration) → reward
        """
        level_config = self.config.get('intervention_sequences', {}).get('level_3', {})
        base_period = level_config.get('quiet_period_duration', 20)
        quiet_period_duration = self._get_progressive_quiet_duration(base_period)
        quiet_periods_required = level_config.get('quiet_periods_required', 2)

        if self.intervention_step == 0:
            if self.last_step_time == 0.0:
                logger.info("Level 3 - Step 0: quiet + starting calming music")
                if self.led:
                    self.led.set_pattern('calm', duration=60.0)
                self._play_audio('quiet.mp3')
                time.sleep(1.0)
                self._start_calming_music()
                self.last_step_time = now
                self.quiet_start_time = now
                self.quiet_periods_achieved = 0
                self.intervention_step = 1

        elif self.intervention_step == 1:
            # Check if quiet period achieved
            if self.quiet_start_time > 0:
                quiet_duration = now - self.quiet_start_time
                if quiet_duration >= quiet_period_duration:
                    self.quiet_periods_achieved += 1
                    logger.info(f"Level 3 - Quiet period {self.quiet_periods_achieved}/{quiet_periods_required} achieved")

                    if self.quiet_periods_achieved >= quiet_periods_required:
                        logger.info("Level 3 - SUCCESS: All quiet periods achieved")
                        self._stop_calming_music()
                        self._give_reward()
                        self.fsm_state = SGState.COOLDOWN
                        self._cooldown_start = now
                    else:
                        # Reset for next quiet period
                        self.quiet_start_time = now

    def _start_calming_music(self):
        """Start playing calming music"""
        if self.calming_music_playing:
            return

        try:
            audio_config = self.config.get('audio_paths', {})
            music_file = audio_config.get('calming_music',
                'songs/DOG DESENSITISATION MUSIC! to help train your dog, improve behaviour. Sound effects included.mp3')

            base = audio_config.get('base', '/home/morgan/dogbot/VOICEMP3')
            full_path = os.path.join(base, music_file)

            if os.path.exists(full_path):
                if self.audio:
                    self.audio.play_file(full_path, loop=True)
                    self.calming_music_playing = True
                    logger.info(f"Calming music started: {music_file}")
            else:
                logger.warning(f"Calming music file not found: {full_path}")

        except Exception as e:
            logger.error(f"Failed to start calming music: {e}")

    def _stop_calming_music(self):
        """Stop calming music"""
        if not self.calming_music_playing:
            return

        try:
            if self.audio:
                self.audio.stop()
            self.calming_music_playing = False
            logger.info("Calming music stopped")
        except Exception as e:
            logger.error(f"Failed to stop calming music: {e}")

    def _give_reward(self):
        """Give reward for successful quiet"""
        # Stop calming music if playing
        self._stop_calming_music()

        # Track consecutive interventions for progressive quiet
        self.consecutive_interventions += 1

        # Check treat limit
        max_treats = self.config.get('session_limits', {}).get('max_treats', 11)
        if self.treats_dispensed >= max_treats:
            logger.info(f"Treat limit reached ({self.treats_dispensed}/{max_treats})")
            self._play_audio('good_dog.mp3')
            return

        # Check treat eligibility (anti-farming)
        # Also enforce min_time_between_treats as backup
        min_time = self.config.get('session_limits', {}).get('min_time_between_treats', 120)
        time_since_treat = time.time() - self.last_treat_time if self.last_treat_time > 0 else float('inf')

        if not self.treat_eligible or time_since_treat < min_time:
            # Verbal praise only - no treat during eligibility cooldown
            cooldown_remaining = self.treat_eligibility_cooldown - time_since_treat
            logger.info(f"Verbal praise only - treat cooldown ({cooldown_remaining/60:.1f} min remaining)")
            if self.led:
                self.led.set_pattern('success', duration=2.0)
            self._play_audio('good_dog.mp3')

            # Log intervention without treat
            self.store.log_sg_intervention(
                session_id=self.session_id,
                dog_id=self.intervention_dog_id,
                dog_name=self.intervention_dog_name,
                escalation_level=self.current_escalation_level,
                quiet_achieved=True,
                treat_given=False,  # No treat during cooldown
                music_played=(self.current_escalation_level == 3)
            )
            return

        # Full reward: LED celebration + audio + treat
        if self.led:
            self.led.celebration_sequence(3.0)

        # Play good.mp3
        logger.info("Playing good.mp3")
        self._play_audio('good.mp3')
        time.sleep(1.0)

        # Dispense treat
        if self.dispenser:
            self.dispenser.dispense_treat(
                dog_id=self.intervention_dog_id,
                reason='silent_guardian_reward'
            )
            self.treats_dispensed += 1
            self.last_treat_time = time.time()  # Start eligibility cooldown
            self.treat_eligible = False  # Immediately mark ineligible
            logger.info(f"Treat dispensed ({self.treats_dispensed} this session) - 10 min cooldown started")

            # Log to database
            self.store.log_reward(
                dog_id=self.intervention_dog_id or 'unknown',
                behavior='quiet',
                confidence=1.0,
                success=True,
                treats_dispensed=1,
                mission_name='silent_guardian'
            )

        # Log successful intervention with actual escalation level
        self.store.log_sg_intervention(
            session_id=self.session_id,
            dog_id=self.intervention_dog_id,
            dog_name=self.intervention_dog_name,
            escalation_level=self.current_escalation_level,
            quiet_achieved=True,
            treat_given=True,
            music_played=(self.current_escalation_level == 3)
        )

    def _process_cooldown(self):
        """Process cooldown between interventions"""
        if self._cooldown_start is None:
            self._cooldown_start = time.time()
            if self.led:
                self.led.set_pattern('off')
            return

        elapsed = time.time() - self._cooldown_start
        if elapsed >= self._cooldown_duration:
            # Cooldown complete
            self._cooldown_start = None
            self.fsm_state = SGState.LISTENING
            logger.info("Returning to LISTENING state")

    def _process_gave_up(self):
        """Process gave-up cooldown (longer than normal)"""
        if self._cooldown_start is None:
            self._cooldown_start = time.time()
            if self.led:
                self.led.set_pattern('error', duration=5.0)
            logger.info(f"Entered gave-up cooldown ({self._gave_up_cooldown}s)")
            return

        elapsed = time.time() - self._cooldown_start
        if elapsed >= self._gave_up_cooldown:
            # Gave-up cooldown complete
            self._cooldown_start = None
            self.fsm_state = SGState.LISTENING
            logger.info("Gave-up cooldown complete, returning to LISTENING state")

    def _play_audio(self, filename: str):
        """Play audio file from talks directory"""
        try:
            audio_config = self.config.get('audio_paths', {})
            base = audio_config.get('talks', '/home/morgan/dogbot/VOICEMP3/talks')

            if filename.startswith('/'):
                full_path = filename
            else:
                full_path = os.path.join(base, filename)

            if os.path.exists(full_path):
                if self.audio:
                    self.audio.play_file(full_path)
                    logger.debug(f"Playing: {full_path}")
            else:
                logger.warning(f"Audio file not found: {full_path}")

        except Exception as e:
            logger.error(f"Audio error: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        # Calculate time until treat eligibility
        if self.last_treat_time > 0:
            time_since_treat = time.time() - self.last_treat_time
            treat_cooldown_remaining = max(0, self.treat_eligibility_cooldown - time_since_treat)
        else:
            treat_cooldown_remaining = 0

        return {
            'running': self.running,
            'fsm_state': self.fsm_state.value if self.fsm_state else None,
            'session_id': self.session_id,
            'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
            'interventions_triggered': self.interventions_triggered,
            'treats_dispensed': self.treats_dispensed,
            'treats_remaining': max(0, self.config.get('session_limits', {}).get('max_treats', 11) - self.treats_dispensed),
            'current_escalation_level': self.current_escalation_level,
            'escalation_events_count': len(self.escalation_events),
            'quiet_periods_achieved': self.quiet_periods_achieved,
            'calming_music_playing': self.calming_music_playing,
            'intervention_step': self.intervention_step if self.fsm_state == SGState.INTERVENTION else None,
            'last_bark_time': self.last_bark_time,
            # Anti-farming tracking
            'treat_eligible': self.treat_eligible,
            'treat_cooldown_remaining': treat_cooldown_remaining,
            'consecutive_interventions': self.consecutive_interventions
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

    print("\n=== SILENT GUARDIAN MODE TEST (3-Level Escalation) ===")
    print("Level 1: [name] + quiet → 20s quiet → reward")
    print("Level 2: quiet, quiet, no, come, quiet → 30s quiet → reward")
    print("Level 3: quiet + CALMING MUSIC → 2x 20s quiet → reward")
    print("Press Ctrl+C to exit\n")

    mode.start()

    while mode.running:
        status = mode.get_status()
        level = status.get('current_escalation_level', 1)
        step = status.get('intervention_step', '-')
        music = "MUSIC" if status.get('calming_music_playing') else ""
        print(f"\rState: {status['fsm_state']} | Level: {level} | Step: {step} | Interventions: {status['interventions_triggered']} | Treats: {status['treats_dispensed']} {music}", end='')
        time.sleep(1)


if __name__ == "__main__":
    main()
