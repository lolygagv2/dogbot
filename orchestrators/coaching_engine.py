#!/usr/bin/env python3
"""
Coaching Engine for WIM-Z
Opportunistic trick training when dog approaches camera

State Machine:
WAITING_FOR_DOG -> ATTENTION_CHECK (2-3s) -> GREETING -> COMMAND -> WATCHING (10s) -> RESULT -> COOLDOWN

Features:
- ArUco-based dog identification
- Trick rotation (avoids repeating last 3 tricks per dog)
- Per-dog cooldowns
- Success/failure tracking
"""

import os
import sys
import time
import random
import threading
import logging
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports
from core.bus import get_bus, VisionEvent, publish_system_event
from core.state import get_state, SystemMode
from core.store import get_store
from core.behavior_interpreter import get_behavior_interpreter

# Services
from services.media.usb_audio import get_usb_audio_service, set_agc
from services.reward.dispenser import get_dispenser_service
from services.media.led import get_led_service

logger = logging.getLogger(__name__)


class CoachState(Enum):
    """Coaching session state machine"""
    WAITING_FOR_DOG = "waiting_for_dog"
    ATTENTION_CHECK = "attention_check"
    GREETING = "greeting"
    COMMAND = "command"
    WATCHING = "watching"
    SUCCESS = "success"
    FAILURE = "failure"
    COOLDOWN = "cooldown"


@dataclass
class DogSession:
    """Tracking for individual dog coaching session"""
    dog_id: str
    dog_name: Optional[str] = None
    trick_requested: Optional[str] = None
    attention_start: Optional[float] = None
    command_time: Optional[float] = None
    behavior_detected: Optional[str] = None
    success: bool = False


@dataclass
class DogHistory:
    """Per-dog coaching history"""
    last_tricks: List[str] = field(default_factory=list)  # Last 3 tricks
    last_session_time: float = 0.0
    total_sessions: int = 0
    successful_sessions: int = 0


class CoachingEngine:
    """
    Coaching Engine for opportunistic trick training

    Behavior:
    1. Wait for dog to enter camera view (via ArUco)
    2. Check for attention (still + facing camera for 2-3s)
    3. Greet dog by name
    4. Request ONE trick from rotation
    5. Watch for 10 seconds for behavior
    6. Success: celebrate + treat
    7. Failure: "Good try!" (no treat)
    8. 5-minute cooldown per dog
    """

    # Default tricks for rotation (loaded from config at runtime)
    # These are fallbacks if config not loaded
    DEFAULT_TRICKS = ['sit', 'down', 'crosses', 'spin', 'speak']

    # Speak trick settings (loaded from config)
    SPEAK_TIMEOUT = 5.0      # Seconds to wait for barks
    SPEAK_MIN_BARKS = 1      # Minimum barks required
    SPEAK_MAX_BARKS = 2      # Maximum barks allowed (more = fail)

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize coaching engine"""
        self.bus = get_bus()
        self.state = get_state()
        self.store = get_store()

        # Behavior interpreter (Layer 1) - handles all detection logic
        self.interpreter = get_behavior_interpreter()

        # Services
        self.audio = get_usb_audio_service()
        self.dispenser = get_dispenser_service()
        self.led = get_led_service()

        # Load tricks from interpreter config (Layer 2)
        self.TRICKS = self.interpreter.get_all_tricks()
        if not self.TRICKS:
            self.TRICKS = self.DEFAULT_TRICKS
        # Filter to only coachable tricks (exclude 'stand' for now)
        self.TRICKS = [t for t in self.TRICKS if t not in ['stand']]
        logger.info(f"Available tricks for coaching: {self.TRICKS}")

        # Configuration - load from interpreter's trick_rules.yaml
        self.config = config or {}
        trick_config = self.interpreter.trick_rules
        coaching_config = getattr(self.interpreter, 'coaching_config', {})

        # Time dog must be visible before starting session (no stillness required)
        self.attention_duration = self.config.get('attention_duration',
            coaching_config.get('attention_duration_sec', 3.5))
        self.watch_duration = self.config.get('watch_duration', 10.0)  # Per-trick override below
        self.cooldown_duration = self.config.get('cooldown_duration',
            coaching_config.get('cooldown_duration_sec', 300.0))

        # Dog identification mapping
        self.dog_names = {
            'aruco_315': 'Elsa',
            'aruco_832': 'Bezik',
            315: 'Elsa',
            832: 'Bezik'
        }

        # Engine state
        self.running = False
        self.engine_thread = None
        self._lock = threading.Lock()

        # FSM state
        self.fsm_state = CoachState.WAITING_FOR_DOG
        self.current_session: Optional[DogSession] = None

        # Dog tracking
        self.dog_history: Dict[str, DogHistory] = {}
        self.dogs_in_view: Dict[str, float] = {}  # dog_id -> first_seen_time
        self.dog_names_in_view: Dict[str, str] = {}  # dog_id -> dog_name (from ArUco)

        # Session statistics
        self.sessions_today = 0
        self.successes_today = 0

        # Testing/debug: force specific trick (None = random)
        self._forced_trick: Optional[str] = None

        # Bark tracking for 'speak' trick
        self.bark_count = 0
        self.bark_timestamps: List[float] = []
        self.listening_for_barks = False

        logger.info("Coaching Engine initialized")

    def start(self) -> bool:
        """Start coaching engine"""
        if self.running:
            logger.warning("Coaching engine already running")
            return True

        try:
            # Disable AGC for bark detection (raw energy levels needed)
            set_agc(False)
            logger.info("AGC disabled for coaching mode (bark detection)")

            # Reset FSM state on start (fixes mode re-entry)
            self.fsm_state = CoachState.WAITING_FOR_DOG
            self.current_session = None
            self.dogs_in_view.clear()
            self.dog_names_in_view.clear()
            self.bark_count = 0
            self.bark_timestamps.clear()
            self.listening_for_barks = False
            logger.info("Coaching engine state reset to WAITING_FOR_DOG")

            # Subscribe to vision events (use string, not class)
            self.bus.subscribe('vision', self._on_vision_event)
            # Subscribe to audio events for bark detection (speak trick)
            self.bus.subscribe('audio', self._on_audio_event)

            # Start engine thread
            self.running = True
            self.engine_thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="CoachingEngine"
            )
            self.engine_thread.start()

            # Set system mode
            self.state.set_mode(SystemMode.COACH, "Coaching engine started")

            publish_system_event('coaching_started', {
                'tricks_available': self.TRICKS
            }, 'coaching_engine')

            logger.info("Coaching engine started")
            return True

        except Exception as e:
            logger.error(f"Failed to start coaching engine: {e}")
            return False

    def stop(self):
        """Stop coaching engine"""
        if not self.running:
            return

        logger.info("Stopping coaching engine...")
        self.running = False

        if self.engine_thread and self.engine_thread.is_alive():
            self.engine_thread.join(timeout=2.0)

        # Re-enable AGC when leaving coaching mode
        set_agc(True)
        logger.info("AGC re-enabled (leaving coaching mode)")

        publish_system_event('coaching_stopped', {
            'sessions_today': self.sessions_today,
            'successes_today': self.successes_today
        }, 'coaching_engine')

        logger.info("Coaching engine stopped")

    def _on_vision_event(self, event):
        """Handle vision events for dog detection"""
        if event.subtype == 'dog_detected':
            dog_id = event.data.get('dog_id')
            dog_name = event.data.get('dog_name')  # ArUco-identified name
            if not dog_id:
                return

            # Track dog visibility
            is_new = dog_id not in self.dogs_in_view
            self.dogs_in_view[dog_id] = time.time()

            # Store ArUco-identified name if available
            if dog_name and dog_name not in ['unknown', None]:
                self.dog_names_in_view[dog_id] = dog_name

            if is_new:
                display_name = self.dog_names_in_view.get(dog_id, dog_id)
                logger.info(f"🐕 Dog entered view for coaching: {display_name}")

        elif event.subtype == 'behavior_detected':
            # Track detected behaviors for coaching response
            if self.fsm_state == CoachState.WATCHING and self.current_session:
                behavior = event.data.get('behavior')
                dog_name = event.data.get('dog_name')

                # Match by dog_name if available, otherwise accept any behavior
                if behavior:
                    self.current_session.behavior_detected = behavior
                    logger.info(f"Behavior detected during coaching: {behavior}")

    def _on_audio_event(self, event):
        """Handle audio events for bark detection (speak trick)"""
        if event.subtype == 'bark_detected':
            # Only count barks if we're actively listening (during 'speak' trick)
            if self.listening_for_barks and self.current_session:
                if self.current_session.trick_requested == 'speak':
                    # Require minimum confidence to count as real bark
                    # Filters out random sounds that might trigger the energy-based detector
                    confidence = event.data.get('confidence', 0.0) if event.data else 0.0
                    min_speak_confidence = 0.50  # Require 50% confidence for speak trick

                    if confidence < min_speak_confidence:
                        logger.debug(f"🔇 Bark ignored for speak (conf={confidence:.2f} < {min_speak_confidence})")
                        return

                    self.bark_count += 1
                    self.bark_timestamps.append(time.time())
                    logger.info(f"🐕 Bark detected during speak trick! Count: {self.bark_count} (conf={confidence:.2f})")

                    # Check for too many barks (immediate fail)
                    if self.bark_count > self.SPEAK_MAX_BARKS:
                        logger.info(f"Too many barks ({self.bark_count}) - speak trick failed")
                        self.current_session.behavior_detected = 'too_many_barks'

    def _get_dog_name(self, dog_id: str) -> str:
        """Get friendly name for dog"""
        # First check ArUco-identified names from vision events
        if dog_id in self.dog_names_in_view:
            return self.dog_names_in_view[dog_id]

        # Then check static mapping
        if dog_id in self.dog_names:
            return self.dog_names[dog_id]

        # Try extracting ArUco ID
        if dog_id.startswith('aruco_'):
            aruco_id = int(dog_id.split('_')[1])
            if aruco_id in self.dog_names:
                return self.dog_names[aruco_id]

        return dog_id

    def _get_or_create_history(self, dog_id: str) -> DogHistory:
        """Get or create dog history"""
        if dog_id not in self.dog_history:
            self.dog_history[dog_id] = DogHistory()
        return self.dog_history[dog_id]

    def _can_coach_dog(self, dog_id: str) -> bool:
        """Check if dog is eligible for coaching (not in cooldown)"""
        history = self._get_or_create_history(dog_id)

        if history.last_session_time == 0:
            return True

        elapsed = time.time() - history.last_session_time
        return elapsed >= self.cooldown_duration

    def _select_trick(self, dog_id: str) -> str:
        """Select a trick avoiding recent ones for this dog"""
        # Check for forced trick (testing mode)
        if self._forced_trick:
            logger.info(f"Using forced trick: {self._forced_trick}")
            return self._forced_trick

        history = self._get_or_create_history(dog_id)

        # Get available tricks (not in last 3)
        available = [t for t in self.TRICKS if t not in history.last_tricks]

        if not available:
            # All tricks used recently, pick any
            available = self.TRICKS.copy()

        return random.choice(available)

    def _run_loop(self):
        """Main engine loop"""
        logger.info("Coaching engine loop started")

        while self.running:
            try:
                # Check system mode
                if self.state.get_mode() != SystemMode.COACH:
                    logger.info("Mode changed, stopping coaching engine")
                    break

                # Clean stale dog visibility
                self._cleanup_stale_dogs()

                # Run state machine
                self._process_state()

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Coaching engine error: {e}")
                time.sleep(1.0)

        logger.info("Coaching engine loop ended")

    def _cleanup_stale_dogs(self):
        """Remove dogs not seen recently"""
        cutoff = time.time() - 2.0  # 2 second timeout

        stale = [
            dog_id for dog_id, seen_time in self.dogs_in_view.items()
            if seen_time < cutoff
        ]

        for dog_id in stale:
            del self.dogs_in_view[dog_id]
            if dog_id in self.dog_names_in_view:
                del self.dog_names_in_view[dog_id]

    def _process_state(self):
        """Process current FSM state"""
        if self.fsm_state == CoachState.WAITING_FOR_DOG:
            self._state_waiting_for_dog()

        elif self.fsm_state == CoachState.ATTENTION_CHECK:
            self._state_attention_check()

        elif self.fsm_state == CoachState.GREETING:
            self._state_greeting()

        elif self.fsm_state == CoachState.COMMAND:
            self._state_command()

        elif self.fsm_state == CoachState.WATCHING:
            self._state_watching()

        elif self.fsm_state == CoachState.SUCCESS:
            self._state_success()

        elif self.fsm_state == CoachState.FAILURE:
            self._state_failure()

        elif self.fsm_state == CoachState.COOLDOWN:
            self._state_cooldown()

    def _state_waiting_for_dog(self):
        """Wait for a dog to be visible and eligible"""
        # Find an eligible dog in view
        for dog_id, first_seen_time in self.dogs_in_view.items():
            if self._can_coach_dog(dog_id):
                # Found eligible dog - check how long it's been visible
                time_in_view = time.time() - first_seen_time
                logger.info(f"🎯 Dog {dog_id} in view for {time_in_view:.1f}s (need {self.attention_duration}s)")

                if time_in_view >= self.attention_duration:
                    # Dog has been visible long enough - start session
                    self._start_session(dog_id)
                    return

    def _start_session(self, dog_id: str):
        """Start a coaching session with a dog"""
        dog_name = self._get_dog_name(dog_id)

        # Don't start session if dog isn't properly identified via ArUco
        # Wait for ArUco identification (up to 10 seconds grace period)
        if dog_id.startswith('dog_') and dog_id not in self.dog_names_in_view:
            # Dog not yet identified - keep waiting
            logger.debug(f"Dog {dog_id} not yet identified via ArUco, waiting...")
            return

        trick = self._select_trick(dog_id)

        self.current_session = DogSession(
            dog_id=dog_id,
            dog_name=dog_name,
            trick_requested=trick
        )

        self.fsm_state = CoachState.GREETING
        logger.info(f"Starting coaching session: {dog_name} - trick: {trick}")

    def _state_attention_check(self):
        """Verify dog is still attentive"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        dog_id = self.current_session.dog_id

        # Check dog is still visible
        if dog_id not in self.dogs_in_view:
            logger.info("Dog left during attention check")
            self.fsm_state = CoachState.WAITING_FOR_DOG
            self.current_session = None
            return

        # Attention confirmed
        self.fsm_state = CoachState.GREETING

    def _state_greeting(self):
        """Greet the dog by name"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        dog_name = self.current_session.dog_name

        # LED attention pattern
        if self.led:
            self.led.set_pattern('attention', duration=2.0)

        # Say dog's name
        self._play_audio(f'{dog_name.lower()}.mp3')
        time.sleep(1.5)

        self.fsm_state = CoachState.COMMAND

    def _state_command(self):
        """Give the trick command"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        trick = self.current_session.trick_requested

        # Say the trick command using audio file from config (Layer 2)
        trick_rules = self.interpreter.get_trick_rules(trick)
        audio_file = trick_rules.get('audio_command', f'{trick}.mp3')
        self._play_audio(audio_file)
        time.sleep(1.0)

        # Start listening for barks if speak trick
        if trick == 'speak':
            self.bark_count = 0
            self.bark_timestamps = []
            self.listening_for_barks = True
            logger.info("Listening for barks (speak trick)...")

        self.current_session.command_time = time.time()
        self.fsm_state = CoachState.WATCHING
        logger.info(f"Command given: {trick}")

    def _state_watching(self):
        """Watch for behavior response using BehaviorInterpreter (Layer 1)"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        watch_elapsed = time.time() - self.current_session.command_time
        expected_trick = self.current_session.trick_requested
        dog_name = self.current_session.dog_name or 'unknown'

        # Get trick-specific timeout from config (Layer 2)
        trick_rules = self.interpreter.get_trick_rules(expected_trick)
        detection_window = trick_rules.get('detection_window_sec', self.watch_duration)

        # Special handling for speak trick (bark-based)
        if expected_trick == 'speak':
            speak_rules = trick_rules
            min_barks = speak_rules.get('min_barks', self.SPEAK_MIN_BARKS)
            max_barks = speak_rules.get('max_barks', self.SPEAK_MAX_BARKS)
            speak_timeout = speak_rules.get('detection_window_sec', self.SPEAK_TIMEOUT)

            # Check for too many barks (fail immediately)
            if self.bark_count > max_barks:
                self.listening_for_barks = False
                self.fsm_state = CoachState.FAILURE
                logger.info(f"Speak failed - too many barks ({self.bark_count})")
                return

            # Check if we have enough barks (success!)
            if self.bark_count >= min_barks:
                self.listening_for_barks = False
                self.current_session.success = True
                self.current_session.behavior_detected = 'bark'
                self.fsm_state = CoachState.SUCCESS
                logger.info(f"Success! Dog spoke with {self.bark_count} bark(s)")
                return

            # Check timeout (no barks or not enough)
            if watch_elapsed >= speak_timeout:
                self.listening_for_barks = False
                self.fsm_state = CoachState.FAILURE
                logger.info(f"Speak timeout - only {self.bark_count} bark(s), need {min_barks}")
                return

            return  # Keep watching for barks

        # Standard pose-based tricks - use BehaviorInterpreter (Layer 1)
        result = self.interpreter.check_trick(expected_trick, dog_id=dog_name)

        if result.completed:
            self.current_session.success = True
            self.current_session.behavior_detected = result.behavior_detected
            self.fsm_state = CoachState.SUCCESS
            logger.info(f"Success! {dog_name} performed {expected_trick} "
                       f"(detected: {result.behavior_detected}, "
                       f"held: {result.hold_duration:.1f}s, conf: {result.confidence:.2f})")
            return

        # Log progress for debugging
        if result.behavior_detected:
            logger.debug(f"Watching {expected_trick}: {result.reason}")

        # Check timeout
        if watch_elapsed >= detection_window:
            self.fsm_state = CoachState.FAILURE
            if result.behavior_detected:
                logger.info(f"Timeout - {result.reason}")
            else:
                logger.info(f"Timeout - no behavior detected for {expected_trick}")

    def _state_success(self):
        """Handle successful trick"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        # LED celebration
        if self.led:
            self.led.celebration_sequence(3.0)

        # Play celebration audio
        self._play_audio('good_dog.mp3')
        time.sleep(1.5)

        # Dispense treat
        self._dispense_treat()

        # Log success
        self._log_session(success=True)

        self.fsm_state = CoachState.COOLDOWN

    def _state_failure(self):
        """Handle failed trick attempt"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        # Say "no" on failure - no treat
        self._play_audio('no.mp3')
        time.sleep(1.0)
        logger.info(f"Session failed - played 'no', no treat")

        # Log failure
        self._log_session(success=False)

        self.fsm_state = CoachState.COOLDOWN

    def _state_cooldown(self):
        """Brief pause before returning to waiting"""
        if self.current_session:
            # Update dog history
            dog_id = self.current_session.dog_id
            history = self._get_or_create_history(dog_id)
            history.last_session_time = time.time()

            # Track trick rotation
            trick = self.current_session.trick_requested
            if trick:
                history.last_tricks.append(trick)
                # Keep only last 3
                if len(history.last_tricks) > 3:
                    history.last_tricks = history.last_tricks[-3:]

            self.current_session = None

        time.sleep(2.0)
        self.fsm_state = CoachState.WAITING_FOR_DOG
        logger.info("Returning to WAITING_FOR_DOG state")

    def _play_audio(self, filename: str):
        """Play audio file from talks directory"""
        try:
            base = '/home/morgan/dogbot/VOICEMP3/talks'
            full_path = os.path.join(base, filename)

            if os.path.exists(full_path):
                if self.audio:
                    self.audio.play_file(full_path)
            else:
                logger.warning(f"Audio file not found: {full_path}")

        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    def _dispense_treat(self):
        """Dispense a treat"""
        try:
            if self.dispenser:
                dog_id = self.current_session.dog_id if self.current_session else None
                self.dispenser.dispense_treat(
                    dog_id=dog_id,
                    reason='coaching_reward'
                )
                logger.info("Treat dispensed for successful trick")

        except Exception as e:
            logger.error(f"Treat dispense error: {e}")

    def _log_session(self, success: bool):
        """Log coaching session to database"""
        if not self.current_session:
            return

        self.sessions_today += 1
        if success:
            self.successes_today += 1

        # Update dog history
        dog_id = self.current_session.dog_id
        history = self._get_or_create_history(dog_id)
        history.total_sessions += 1
        if success:
            history.successful_sessions += 1

        # Log to database
        try:
            response_time = None
            if self.current_session.command_time:
                response_time = time.time() - self.current_session.command_time

            self.store.log_coaching_session(
                dog_id=dog_id,
                dog_name=self.current_session.dog_name,
                trick_requested=self.current_session.trick_requested,
                trick_completed=success,
                response_time=response_time
            )
        except Exception as e:
            logger.error(f"Failed to log coaching session: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get coaching engine status"""
        current = None
        if self.current_session:
            current = {
                'dog_id': self.current_session.dog_id,
                'dog_name': self.current_session.dog_name,
                'trick': self.current_session.trick_requested
            }

        return {
            'running': self.running,
            'fsm_state': self.fsm_state.value,
            'current_session': current,
            'dogs_in_view': list(self.dogs_in_view.keys()),
            'sessions_today': self.sessions_today,
            'successes_today': self.successes_today,
            'success_rate': self.successes_today / self.sessions_today if self.sessions_today > 0 else 0,
            'dog_cooldowns': {
                dog_id: max(0, self.cooldown_duration - (time.time() - h.last_session_time))
                for dog_id, h in self.dog_history.items()
                if h.last_session_time > 0
            }
        }

    def reset_cooldowns(self, dog_id: str = None) -> Dict[str, Any]:
        """Reset cooldowns for testing - either specific dog or all dogs"""
        if dog_id:
            if dog_id in self.dog_history:
                self.dog_history[dog_id].last_session_time = 0.0
                logger.info(f"Cooldown reset for {dog_id}")
                return {'reset': [dog_id]}
            else:
                return {'reset': [], 'error': f'Dog {dog_id} not found'}
        else:
            for history in self.dog_history.values():
                history.last_session_time = 0.0
            reset_dogs = list(self.dog_history.keys())
            logger.info(f"Cooldowns reset for all dogs: {reset_dogs}")
            return {'reset': reset_dogs}

    def set_forced_trick(self, trick: str = None) -> Dict[str, Any]:
        """Force a specific trick for testing (None to clear)"""
        if trick and trick not in self.TRICKS:
            return {'error': f'Invalid trick: {trick}', 'valid_tricks': self.TRICKS}

        self._forced_trick = trick
        if trick:
            logger.info(f"Forced trick set: {trick}")
            return {'forced_trick': trick, 'message': f"Next session will use '{trick}'"}
        else:
            logger.info("Forced trick cleared")
            return {'forced_trick': None, 'message': "Random trick selection restored"}

    def cleanup(self):
        """Clean up resources"""
        self.stop()


# Singleton instance
_coaching_instance = None
_coaching_lock = threading.Lock()


def get_coaching_engine() -> CoachingEngine:
    """Get or create coaching engine instance (singleton)"""
    global _coaching_instance
    if _coaching_instance is None:
        with _coaching_lock:
            if _coaching_instance is None:
                _coaching_instance = CoachingEngine()
    return _coaching_instance


def main():
    """Test coaching engine"""
    import signal

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    engine = CoachingEngine()

    def signal_handler(sig, frame):
        print("\nShutting down...")
        engine.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("\n=== COACHING ENGINE TEST ===")
    print("Press Ctrl+C to exit\n")

    engine.start()

    while engine.running:
        status = engine.get_status()
        print(f"\rState: {status['fsm_state']} | Sessions: {status['sessions_today']} | Successes: {status['successes_today']}", end='')
        time.sleep(1)


if __name__ == "__main__":
    main()
