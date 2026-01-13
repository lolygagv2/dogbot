#!/usr/bin/env python3
"""
Silent Guardian Mode - WIM-Z Primary Product Mode
Simple bark intervention with fixed-timing reward sequence

Flow:
1. Listen for 2 barks in 60 seconds
2. After 2nd bark: "QUIET"
3. Wait 5s: "QUIET" again
4. Wait 5s: "treat.mp3"
5. Wait 5s: "QUIET" again
6. Wait 5s: if no barking → "good.mp3" + DISPENSE TREAT

If dog barks during sequence → restart from step 2
"""

import os
import sys
import time
import threading
import logging
import yaml
from enum import Enum
from typing import Dict, Any, Optional
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
    """Silent Guardian states - simplified"""
    LISTENING = "listening"           # Waiting for bark threshold
    INTERVENTION = "intervention"     # Running fixed-timing sequence
    COOLDOWN = "cooldown"            # Brief pause after intervention


class SilentGuardianMode:
    """
    Silent Guardian - Simple bark intervention mode

    Fixed sequence on bark detection:
    QUIET → 5s → QUIET → 5s → treat.mp3 → 5s → QUIET → 5s → good.mp3 + treat
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
        self.intervention_step = 0  # 0-4 for the 5 steps
        self.last_step_time = 0.0
        self.bark_during_intervention = False
        self.intervention_dog_id = None
        self.intervention_dog_name = None

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

        logger.info("Silent Guardian mode initialized (simple flow)")

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
                'threshold': 2,  # 2 barks triggers intervention
                'window_seconds': 60,
                'loudness_threshold_db': -15,
                'confidence_minimum': 0.80
            },
            'session_limits': {
                'max_treats': 11,
                'min_time_between_treats': 30
            },
            'audio_paths': {
                'talks': '/home/morgan/dogbot/VOICEMP3/talks',
                'songs': '/home/morgan/dogbot/VOICEMP3/songs'
            }
        }

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
        loudness_threshold = bark_config.get('loudness_threshold_db', -15)
        confidence_minimum = bark_config.get('confidence_minimum', 0.80)

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
            # Bark during intervention - will restart sequence
            self.bark_during_intervention = True
            logger.info("Bark during intervention - will restart sequence")

    def _start_intervention(self, dog_id: Optional[str], dog_name: Optional[str]):
        """Start the intervention sequence"""
        with self._state_lock:
            if self.fsm_state != SGState.LISTENING:
                return

            self.fsm_state = SGState.INTERVENTION
            self.intervention_start_time = time.time()
            self.intervention_step = 0
            self.last_step_time = 0.0
            self.bark_during_intervention = False
            self.intervention_dog_id = dog_id
            self.intervention_dog_name = dog_name

            self.interventions_triggered += 1

            # Log to database
            self.store.log_sg_intervention(
                session_id=self.session_id,
                dog_id=dog_id,
                dog_name=dog_name,
                escalation_level=1
            )

            logger.info(f"Starting intervention for {dog_name or dog_id or 'unknown'}")

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

    def _process_intervention(self):
        """
        Process the intervention sequence:
        Step 0: "QUIET" (immediate)
        Step 1: Wait 5s, "QUIET"
        Step 2: Wait 5s, "treat.mp3"
        Step 3: Wait 5s, "QUIET"
        Step 4: Wait 5s, check for success → "good.mp3" + treat OR restart
        """
        now = time.time()

        # Check for bark during intervention - restart sequence
        if self.bark_during_intervention:
            logger.info("Restarting intervention sequence due to bark")
            self.intervention_step = 0
            self.last_step_time = 0.0
            self.bark_during_intervention = False
            # Fall through to execute step 0

        # Step timing
        step_interval = 5.0  # 5 seconds between steps

        # Execute steps
        if self.intervention_step == 0:
            # Step 0: First QUIET (immediate)
            if self.last_step_time == 0.0:
                logger.info("Intervention Step 0: QUIET")
                if self.led:
                    self.led.set_pattern('attention', duration=2.0)
                self._play_audio('quiet.mp3')
                self.last_step_time = now
                self.intervention_step = 1

        elif self.intervention_step == 1:
            # Step 1: Second QUIET after 5s
            if now - self.last_step_time >= step_interval:
                logger.info("Intervention Step 1: QUIET")
                self._play_audio('quiet.mp3')
                self.last_step_time = now
                self.intervention_step = 2

        elif self.intervention_step == 2:
            # Step 2: "treat.mp3" after 5s
            if now - self.last_step_time >= step_interval:
                logger.info("Intervention Step 2: treat.mp3")
                self._play_audio('treat.mp3')
                self.last_step_time = now
                self.intervention_step = 3

        elif self.intervention_step == 3:
            # Step 3: Third QUIET after 5s
            if now - self.last_step_time >= step_interval:
                logger.info("Intervention Step 3: QUIET")
                self._play_audio('quiet.mp3')
                self.last_step_time = now
                self.intervention_step = 4

        elif self.intervention_step == 4:
            # Step 4: Final check after 5s
            if now - self.last_step_time >= step_interval:
                # Success! No bark during entire sequence
                logger.info("Intervention Step 4: SUCCESS - giving reward")
                self._give_reward()
                self.fsm_state = SGState.COOLDOWN
                self._cooldown_start = now

    def _give_reward(self):
        """Give reward for successful quiet"""
        # Check treat limit
        max_treats = self.config.get('session_limits', {}).get('max_treats', 11)
        if self.treats_dispensed >= max_treats:
            logger.info(f"Treat limit reached ({self.treats_dispensed}/{max_treats})")
            self._play_audio('good_dog.mp3')
            return

        # LED celebration
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
            logger.info(f"Treat dispensed ({self.treats_dispensed} this session)")

            # Log to database
            self.store.log_reward(
                dog_id=self.intervention_dog_id or 'unknown',
                behavior='quiet',
                confidence=1.0,
                success=True,
                treats_dispensed=1,
                mission_name='silent_guardian'
            )

        # Log successful intervention
        self.store.log_sg_intervention(
            session_id=self.session_id,
            dog_id=self.intervention_dog_id,
            dog_name=self.intervention_dog_name,
            escalation_level=1,
            quiet_achieved=True,
            treat_given=True
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
        return {
            'running': self.running,
            'fsm_state': self.fsm_state.value if self.fsm_state else None,
            'session_id': self.session_id,
            'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
            'interventions_triggered': self.interventions_triggered,
            'treats_dispensed': self.treats_dispensed,
            'treats_remaining': max(0, self.config.get('session_limits', {}).get('max_treats', 11) - self.treats_dispensed),
            'intervention_step': self.intervention_step if self.fsm_state == SGState.INTERVENTION else None,
            'last_bark_time': self.last_bark_time
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
    print("Simple flow: QUIET → 5s → QUIET → 5s → treat → 5s → QUIET → 5s → good + TREAT")
    print("Press Ctrl+C to exit\n")

    mode.start()

    while mode.running:
        status = mode.get_status()
        step = status.get('intervention_step', '-')
        print(f"\rState: {status['fsm_state']} | Step: {step} | Interventions: {status['interventions_triggered']} | Treats: {status['treats_dispensed']}", end='')
        time.sleep(1)


if __name__ == "__main__":
    main()
