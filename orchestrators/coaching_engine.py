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
from services.media.video_recorder import get_video_recorder
from services.cloud.relay_client import get_relay_client

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
    # Retry states - give dog second chance on first failure
    RETRY_GREETING = "retry_greeting"
    RETRY_COMMAND = "retry_command"
    RETRY_WATCHING = "retry_watching"
    FINAL_FAILURE = "final_failure"
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
    attempt: int = 1  # Track attempt number (1 or 2)


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
    # NOTE: 'crosses' removed - unreliable detection
    DEFAULT_TRICKS = ['sit', 'down', 'spin', 'speak']

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
        self.video_recorder = get_video_recorder()
        self._session_video_path: Optional[str] = None

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
        # Note: Per-dog cooldown removed - only global session cooldown (3 min) applies now

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
        # dog_id -> {first_seen, last_seen, frames_seen, frames_total, name}
        # Tracks presence ratio for session eligibility (3s + 66% in-frame)
        self.dogs_in_view: Dict[str, dict] = {}

        # Detection timing config
        # BUILD 38: Adjusted to 2.0s/55% per user feedback (1.5s/50% felt too fast)
        self.detection_time_sec = 2.0      # Time dog must be visible
        self.presence_ratio_min = 0.55     # Min percentage in-frame
        self.stale_timeout_sec = 5.0       # Remove dog after this long unseen

        # Session statistics
        self.sessions_today = 0
        self.successes_today = 0

        # Testing/debug: force specific trick (None = random)
        self._forced_trick: Optional[str] = None

        # Bark tracking for 'speak' trick
        self.bark_count = 0
        self.bark_timestamps: List[float] = []
        self.listening_for_barks = False
        self._listening_started_at: float = 0.0  # When bark listening started (for stale event filtering)

        # Global session cooldown - only one automatic session per 3 minutes
        # Guide button can reset this to allow manual triggering
        self._last_session_end: float = 0.0
        self.global_session_cooldown_sec = 180.0  # 3 minutes between automatic sessions

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

        # Stop video recording if active (engine stopped mid-session)
        if self._session_video_path:
            try:
                self.video_recorder.stop_recording()
                logger.info(f"Coaching video saved (engine stopped): {self._session_video_path}")
            except Exception as e:
                logger.warning(f"Error stopping video on engine stop: {e}")
            self._session_video_path = None

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
                logger.info(f"🐕 Dog entered view for coaching: {display_name}")
            else:
                # Existing dog - update tracking
                entry = self.dogs_in_view[dog_id]
                entry['last_seen'] = now
                entry['frames_seen'] += 1
                entry['frames_total'] += 1  # BUILD 35: Increment both counters together

                # Update name if ArUco identified (can happen anytime)
                if dog_name and dog_name not in ['unknown', None] and entry['name'] is None:
                    entry['name'] = dog_name
                    logger.info(f"🏷️ Dog {dog_id} identified as {dog_name}")

                    # Announce name if session is active for this dog (late ArUco identification)
                    if (self.current_session and
                        self.current_session.dog_id == dog_id and
                        self.fsm_state in [CoachState.WATCHING, CoachState.RETRY_WATCHING]):
                        self._play_audio(f'{dog_name}.mp3')
                        self.current_session.dog_name = dog_name

        elif event.subtype == 'behavior_detected':
            # Track detected behaviors for coaching response
            watching_states = [CoachState.WATCHING, CoachState.RETRY_WATCHING]
            if self.fsm_state in watching_states and self.current_session:
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
                    # CRITICAL: Reject stale bark events from before listening started
                    # This prevents race conditions where threaded callbacks from old events
                    # (e.g., audio artifacts during command playback) execute after we start listening
                    if event.timestamp < self._listening_started_at:
                        logger.debug(f"Ignoring stale bark event from {event.timestamp:.3f} "
                                    f"(listening started {self._listening_started_at:.3f})")
                        return

                    # Bark gate already validated this is a real bark (energy-based detection)
                    # The confidence here is emotion classification, not bark detection
                    # So we count any bark the gate detected
                    self.bark_count += 1
                    self.bark_timestamps.append(time.time())
                    logger.info(f"🐕 Bark detected during speak trick! Count: {self.bark_count}")

                    # Check for too many barks (immediate fail)
                    if self.bark_count > self.SPEAK_MAX_BARKS:
                        logger.info(f"Too many barks ({self.bark_count}) - speak trick failed")
                        self.current_session.behavior_detected = 'too_many_barks'

    def _get_dog_name(self, dog_id: str) -> str:
        """Get friendly name for dog"""
        # First check ArUco-identified name from dogs_in_view tracking
        if dog_id in self.dogs_in_view:
            tracked_name = self.dogs_in_view[dog_id].get('name')
            if tracked_name:
                return tracked_name

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
        """Check if dog is eligible for coaching.
        Per-dog cooldown removed - only global session cooldown (3 min) applies now."""
        return True

    def _select_trick(self, dog_id: str) -> str:
        """Select next trick in sequential rotation"""
        # Check for forced trick (testing mode)
        if self._forced_trick:
            logger.info(f"Using forced trick: {self._forced_trick}")
            return self._forced_trick

        history = self._get_or_create_history(dog_id)

        # Sequential rotation: get last trick and pick the next one
        if history.last_tricks:
            last_trick = history.last_tricks[-1]
            try:
                last_idx = self.TRICKS.index(last_trick)
                next_idx = (last_idx + 1) % len(self.TRICKS)
            except ValueError:
                # Last trick not in current list, start from beginning
                next_idx = 0
        else:
            # No history, start from first trick
            next_idx = 0

        trick = self.TRICKS[next_idx]
        logger.info(f"Sequential rotation: {trick} (index {next_idx}/{len(self.TRICKS)-1})")
        return trick

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

                # BUILD 35: Removed frames_total increment from main loop
                # frames_total now increments in event handler alongside frames_seen
                # This ensures presence ratio works correctly with detection event timing

                # Run state machine
                self._process_state()

                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Coaching engine error: {e}")
                time.sleep(1.0)

        logger.info("Coaching engine loop ended")

    def _cleanup_stale_dogs(self):
        """Remove dogs not seen recently"""
        cutoff = time.time() - self.stale_timeout_sec

        stale = [
            dog_id for dog_id, info in self.dogs_in_view.items()
            if info['last_seen'] < cutoff
        ]

        for dog_id in stale:
            del self.dogs_in_view[dog_id]

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

        # Retry states - give dog second chance
        elif self.fsm_state == CoachState.RETRY_GREETING:
            self._state_retry_greeting()

        elif self.fsm_state == CoachState.RETRY_COMMAND:
            self._state_retry_command()

        elif self.fsm_state == CoachState.RETRY_WATCHING:
            self._state_retry_watching()

        elif self.fsm_state == CoachState.FINAL_FAILURE:
            self._state_final_failure()

        elif self.fsm_state == CoachState.COOLDOWN:
            self._state_cooldown()

    def _state_waiting_for_dog(self):
        """Wait for a dog to be visible and eligible"""
        now = time.time()

        # Check global cooldown - only one session per 3 minutes unless reset by Guide button
        time_since_last = now - self._last_session_end
        if self._last_session_end > 0 and time_since_last < self.global_session_cooldown_sec:
            # Still in cooldown - don't start new sessions automatically
            return

        # Collect all eligible dogs that meet presence requirements
        eligible_dogs = []
        for dog_id, info in self.dogs_in_view.items():
            if self._can_coach_dog(dog_id):
                time_elapsed = now - info['first_seen']
                presence_ratio = info['frames_seen'] / max(info['frames_total'], 1)

                # 3 seconds elapsed AND minimum presence ratio
                if time_elapsed >= self.detection_time_sec and presence_ratio >= self.presence_ratio_min:
                    has_aruco_name = info.get('name') is not None
                    eligible_dogs.append((dog_id, info, has_aruco_name, time_elapsed, presence_ratio))

        if not eligible_dogs:
            return

        # Sort to prefer ArUco-identified dogs (those with name field set)
        # ArUco dogs (has_aruco_name=True) sort before generic dogs (has_aruco_name=False)
        eligible_dogs.sort(key=lambda x: (not x[2], -x[3]))  # ArUco first, then longest visible

        # Select the best candidate
        dog_id, info, has_aruco_name, time_elapsed, presence_ratio = eligible_dogs[0]
        display_name = info.get('name') or dog_id
        logger.info(f"🎯 Selected {display_name} for coaching "
                   f"({time_elapsed:.1f}s, {presence_ratio*100:.0f}% presence, "
                   f"ArUco: {has_aruco_name})")

        self._start_session(dog_id)

    def _start_session(self, dog_id: str):
        """Start a coaching session with a dog"""
        dog_name = self._get_dog_name(dog_id)

        # Note: Previously required ArUco identification for dog_0/dog_1 etc.
        # Removed - coach any detected dog, ArUco just gives us the name

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

        dog_name = self.current_session.dog_name or 'dog'
        trick = self.current_session.trick_requested or "trick"

        # BUILD 40: Send coach_progress event for greeting stage
        try:
            relay = get_relay_client()
            if relay and relay.connected:
                relay.send_event('coach_progress', {
                    'stage': 'greeting',
                    'dog_name': dog_name,
                    'trick': trick
                })
        except Exception:
            pass

        # Start video recording for this coaching session
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = dog_name.lower().replace(' ', '_')
            filename = f"coach_{safe_name}_{trick}_{timestamp}.mp4"
            self._session_video_path = self.video_recorder.start_recording(filename)
            logger.info(f"Started coaching video: {self._session_video_path}")
        except Exception as e:
            logger.warning(f"Could not start video recording: {e}")
            self._session_video_path = None

        # LED attention pattern
        if self.led:
            self.led.set_pattern('attention', duration=2.0)

        # Say dog's name - try custom voice first, then default files
        dog_id = self.current_session.dog_id if self.current_session else None
        name_played = False

        # Try custom voice for dog name via play_command
        if dog_id and self.audio:
            result = self.audio.play_command(dog_name.lower(), dog_id=dog_id)
            if result.get('success'):
                name_played = True
                if result.get('voice_source') == 'custom':
                    logger.info(f"Using custom voice for '{dog_name}'")
                self.audio.wait_for_completion(timeout=5.0)

        if not name_played:
            # Fallback: try default name audio file
            name_audio = f'{dog_name.lower()}.mp3'
            base_path = '/home/morgan/dogbot/VOICEMP3/talks'
            if not os.path.exists(os.path.join(base_path, name_audio)):
                name_audio = 'dogs_come.mp3'
                logger.info(f"No audio for '{dog_name}', using generic greeting")
            self._play_audio(name_audio, wait=True, timeout=5.0)

        time.sleep(0.5)  # Brief pause between name and command

        self.fsm_state = CoachState.COMMAND

    def _state_command(self):
        """Give the trick command"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        trick = self.current_session.trick_requested

        # BUILD 40: Send coach_progress event for command stage
        try:
            relay = get_relay_client()
            if relay and relay.connected:
                relay.send_event('coach_progress', {
                    'stage': 'command',
                    'trick': trick,
                    'dog_name': self.current_session.dog_name
                })
        except Exception:
            pass

        # Say the trick command using audio file from config (Layer 2)
        trick_rules = self.interpreter.get_trick_rules(trick)
        audio_file = trick_rules.get('audio_command', f'{trick}.mp3')
        self._play_audio(audio_file, wait=True, timeout=5.0)

        # CRITICAL: Reset behavior tracking AFTER audio finishes
        # If we reset before audio, the dog's pose gets tracked during the 3s audio playback
        # and by the time we enter WATCHING, hold time has already accumulated
        # BUILD 38 DEBUG: Log before/after reset
        pre_status = self.interpreter.get_status()
        logger.info(f"📍 PRE-RESET state: behavior={pre_status['current_behavior']}, hold={pre_status['hold_duration']:.1f}s")
        self.interpreter.reset_tracking()
        logger.info(f"📍 POST-RESET: now watching for {trick}")

        # Start listening for barks if speak trick
        if trick == 'speak':
            # Wait briefly to let any audio playback residue die down
            # Otherwise the speaker audio gets picked up by bark detector
            time.sleep(0.3)
            self.bark_count = 0
            self.bark_timestamps = []
            self._listening_started_at = time.time()  # Record BEFORE enabling (for stale event filtering)
            self.listening_for_barks = True
            logger.info(f"Listening for barks (speak trick) from {self._listening_started_at:.3f}...")

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

        # BUILD 40: Send periodic coach_progress during watching (~every 500ms)
        # Throttle by checking if half-second boundary crossed
        if int(watch_elapsed * 2) != int((watch_elapsed - 0.1) * 2):
            try:
                relay = get_relay_client()
                if relay and relay.connected:
                    relay.send_event('coach_progress', {
                        'stage': 'watching',
                        'trick': expected_trick,
                        'dog_name': dog_name,
                        'confidence': result.confidence if result else 0.0,
                        'hold_duration': result.hold_duration if result else 0.0,
                        'elapsed': round(watch_elapsed, 1)
                    })
            except Exception:
                pass

        if result.completed:
            self.current_session.success = True
            self.current_session.behavior_detected = result.behavior_detected
            self.fsm_state = CoachState.SUCCESS
            watch_elapsed = time.time() - self.current_session.command_time
            # BUILD 38 DEBUG: Detailed success logging
            logger.info(f"🎉 SUCCESS! {dog_name} → {expected_trick} "
                       f"(behavior={result.behavior_detected}, "
                       f"held={result.hold_duration:.1f}s, conf={result.confidence:.2f}, "
                       f"watch_time={watch_elapsed:.1f}s)")
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

        trick = self.current_session.trick_requested
        dog_name = self.current_session.dog_name
        behavior = self.current_session.behavior_detected

        # BUILD 40: Send coach_reward event for success
        try:
            relay = get_relay_client()
            if relay and relay.connected:
                relay.send_event('coach_reward', {
                    'behavior': trick,
                    'dog_name': dog_name,
                    'success': True
                })
        except Exception:
            pass

        # LED celebration
        if self.led:
            self.led.celebration_sequence(3.0)

        # Play celebration audio
        self._play_audio('good.mp3')
        time.sleep(1.5)

        # Dispense treat
        self._dispense_treat()

        # Log success
        self._log_session(success=True)

        self.fsm_state = CoachState.COOLDOWN

    def _state_failure(self):
        """Handle failed trick attempt - give second chance on first failure"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        if self.current_session.attempt == 1:
            # First failure - give dog another chance
            dog_name = self.current_session.dog_name or 'dog'
            logger.info(f"First attempt failed - giving {dog_name} another chance")
            self.current_session.attempt = 2
            self.current_session.behavior_detected = None
            self.fsm_state = CoachState.RETRY_GREETING
        else:
            # Second failure - final failure, no treat
            self.fsm_state = CoachState.FINAL_FAILURE

    def _state_retry_greeting(self):
        """Re-greet dog for second attempt"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        dog_name = self.current_session.dog_name or 'dog'
        logger.info(f"Retry greeting: {dog_name}")

        # LED attention pattern
        if self.led:
            self.led.set_pattern('attention', duration=2.0)

        # Say dog's name again - try custom voice first, then default files
        dog_id = self.current_session.dog_id if self.current_session else None
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

        time.sleep(0.5)  # Brief pause between name and command

        self.fsm_state = CoachState.RETRY_COMMAND

    def _state_retry_command(self):
        """Give trick command again for second attempt"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        trick = self.current_session.trick_requested
        logger.info(f"Retry command: {trick}")

        # Say the trick command again - wait for completion
        trick_rules = self.interpreter.get_trick_rules(trick)
        audio_file = trick_rules.get('audio_command', f'{trick}.mp3')
        self._play_audio(audio_file, wait=True, timeout=5.0)

        # Reset behavior tracking AFTER audio finishes
        self.interpreter.reset_tracking()

        # Reset bark tracking for speak trick retry
        if trick == 'speak':
            # Wait briefly to let any audio playback residue die down
            time.sleep(0.3)
            self.bark_count = 0
            self.bark_timestamps = []
            self._listening_started_at = time.time()  # Record BEFORE enabling (for stale event filtering)
            self.listening_for_barks = True
            logger.info(f"Listening for barks (speak trick retry) from {self._listening_started_at:.3f}...")

        self.current_session.command_time = time.time()
        self.fsm_state = CoachState.RETRY_WATCHING

    def _state_retry_watching(self):
        """Watch for behavior on second attempt (same logic as WATCHING)"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        watch_elapsed = time.time() - self.current_session.command_time
        expected_trick = self.current_session.trick_requested
        dog_name = self.current_session.dog_name or 'unknown'

        # Get trick-specific timeout from config
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
                self.fsm_state = CoachState.FINAL_FAILURE
                logger.info(f"Retry speak failed - too many barks ({self.bark_count})")
                return

            # Check if we have enough barks (success!)
            if self.bark_count >= min_barks:
                self.listening_for_barks = False
                self.current_session.success = True
                self.current_session.behavior_detected = 'bark'
                self.fsm_state = CoachState.SUCCESS
                logger.info(f"Success on retry! Dog spoke with {self.bark_count} bark(s)")
                return

            # Check timeout
            if watch_elapsed >= speak_timeout:
                self.listening_for_barks = False
                self.fsm_state = CoachState.FINAL_FAILURE
                logger.info(f"Retry speak timeout - only {self.bark_count} bark(s)")
                return

            return  # Keep watching for barks

        # Standard pose-based tricks - use BehaviorInterpreter
        result = self.interpreter.check_trick(expected_trick, dog_id=dog_name)

        if result.completed:
            self.current_session.success = True
            self.current_session.behavior_detected = result.behavior_detected
            self.fsm_state = CoachState.SUCCESS
            logger.info(f"Success on retry! {dog_name} performed {expected_trick}")
            return

        # Check timeout - go to final failure
        if watch_elapsed >= detection_window:
            self.fsm_state = CoachState.FINAL_FAILURE
            logger.info(f"Retry timeout - {expected_trick} not performed")

    def _state_final_failure(self):
        """Handle final failure after retry - no treat"""
        if not self.current_session:
            self.fsm_state = CoachState.WAITING_FOR_DOG
            return

        # Play "no" on final failure
        self._play_audio('no.mp3')
        time.sleep(1.5)

        dog_name = self.current_session.dog_name or 'dog'
        logger.info(f"Final failure - no treat for {dog_name}")

        # Log failure
        self._log_session(success=False)

        self.fsm_state = CoachState.COOLDOWN

    def _state_cooldown(self):
        """Brief pause before returning to waiting"""
        # Stop video recording if active
        if self._session_video_path:
            try:
                self.video_recorder.stop_recording()
                logger.info(f"Coaching video saved: {self._session_video_path}")
            except Exception as e:
                logger.warning(f"Error stopping video: {e}")
            self._session_video_path = None

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

        # CRITICAL: Clear dogs_in_view to prevent immediate re-detection
        # Without this, the same dog (possibly with different ID) would
        # immediately start a new session since it already has 3+ seconds of presence
        self.dogs_in_view.clear()
        logger.info("Cleared dogs_in_view - dogs must be re-detected")

        # Set post-session grace period timestamp
        self._last_session_end = time.time()

        time.sleep(2.0)
        self.fsm_state = CoachState.WAITING_FOR_DOG
        logger.info("Returning to WAITING_FOR_DOG state")

    def _play_audio(self, filename: str, wait: bool = False, timeout: float = 5.0):
        """Play audio file, using custom voice when dog_id is available

        Args:
            filename: Audio filename (e.g., 'bezik.mp3') or command name
            wait: If True, block until audio finishes (up to timeout)
            timeout: Max seconds to wait for audio completion (default 5s)
        """
        try:
            if not self.audio:
                return

            # Try custom voice via play_command when session has dog_id
            dog_id = self.current_session.dog_id if self.current_session else None
            if dog_id:
                # Strip .mp3 extension to get command name for play_command()
                command = filename.replace('.mp3', '').replace('.wav', '')
                result = self.audio.play_command(command, dog_id=dog_id)
                if result.get('success'):
                    if wait:
                        self.audio.wait_for_completion(timeout=timeout)
                    return

            # Fallback to direct file path
            base = '/home/morgan/dogbot/VOICEMP3/talks'
            full_path = os.path.join(base, filename)

            if os.path.exists(full_path):
                self.audio.play_file(full_path)
                if wait:
                    self.audio.wait_for_completion(timeout=timeout)
            else:
                logger.warning(f"Audio file not found: {full_path}")

        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    def _dispense_treat(self):
        """Dispense a treat"""
        try:
            if self.dispenser:
                dog_id = self.current_session.dog_id if self.current_session else None
                trick = self.current_session.trick_requested if self.current_session else 'unknown'
                behavior = self.current_session.behavior_detected if self.current_session else 'unknown'
                # BUILD 38 DEBUG: Log exactly what triggered the treat
                logger.info(f"🍖 DISPENSING TREAT: dog={dog_id}, trick={trick}, behavior={behavior}")
                self.dispenser.dispense_treat(
                    dog_id=dog_id,
                    reason='coaching_reward'
                )
                logger.info("✅ Treat dispensed successfully")

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

        # Calculate remaining global cooldown
        global_cooldown_remaining = 0
        if self._last_session_end > 0:
            elapsed = time.time() - self._last_session_end
            global_cooldown_remaining = max(0, self.global_session_cooldown_sec - elapsed)

        return {
            'running': self.running,
            'fsm_state': self.fsm_state.value,
            'current_session': current,
            'dogs_in_view': list(self.dogs_in_view.keys()),
            'sessions_today': self.sessions_today,
            'successes_today': self.successes_today,
            'success_rate': self.successes_today / self.sessions_today if self.sessions_today > 0 else 0,
            'global_cooldown_remaining': global_cooldown_remaining
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

    def reset_session_cooldown(self) -> Dict[str, Any]:
        """FULL RESET - Cancel current session and start fresh.
        Called by Guide button to manually trigger new coaching sessions.

        This is a HARD RESET that:
        - Cancels any in-progress session
        - Clears FSM state back to WAITING_FOR_DOG
        - Resets all tracking (barks, behaviors)
        - Clears cooldowns
        - Fast-tracks visible dogs for immediate eligibility
        """
        # Stop any video recording in progress
        if self._session_video_path:
            try:
                self.video_recorder.stop_recording()
                logger.info(f"Video stopped due to reset: {self._session_video_path}")
            except Exception as e:
                logger.warning(f"Error stopping video on reset: {e}")
            self._session_video_path = None

        # Cancel current session completely
        self.current_session = None

        # Reset FSM to waiting state
        self.fsm_state = CoachState.WAITING_FOR_DOG

        # Reset bark tracking
        self.bark_count = 0
        self.bark_timestamps = []
        self.listening_for_barks = False

        # Reset behavior interpreter tracking
        if self.interpreter:
            self.interpreter.reset_tracking()

        # Clear cooldown
        self._last_session_end = 0.0

        # Fast-track any visible dogs to be immediately eligible
        now = time.time()
        for dog_id, info in self.dogs_in_view.items():
            # Set first_seen to 4 seconds ago so detection_time_sec (3s) is already met
            info['first_seen'] = now - 4.0
            # Update last_seen to NOW so dog isn't immediately cleaned up as stale
            info['last_seen'] = now
            # RESET presence counters to guarantee 100% ratio
            info['frames_seen'] = 10
            info['frames_total'] = 10

        dogs_count = len(self.dogs_in_view)
        logger.info(f"🔄 FULL RESET - Session cancelled, {dogs_count} dogs ready for new session")
        return {'reset': True, 'dogs_ready': dogs_count, 'message': 'Full reset - ready for new session'}

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
