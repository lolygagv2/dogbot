#!/usr/bin/env python3
"""
Silent Guardian Mode - WIM-Z Primary Product Mode
Bark intervention with 3-level escalation system + anti-treat-farming

Escalation Flow:
- Level 1 (1st-2nd intervention in hour): "[dog name], quiet" → quiet period → reward
- Level 2 (3rd intervention in hour): quiet, quiet, no, come, quiet → quiet period → reward
- Level 3 (4th intervention in hour): quiet + PHYSICAL MOVEMENT SEQUENCE → quiet period → reward
- Level 4 (5th+ intervention in hour): quiet + CALMING MUSIC → 2x quiet periods → reward

Anti-Farming Features:
- After a treat: 10 min eligibility cooldown (verbal praise only, no treats)
- Progressive quiet: Each intervention requires longer quiet (20s → 30s → 45s → 60s)
- Min 2 min between treats as backup

Resets:
- If dog barks during intervention → restart that level's sequence
- After 90s with no success → give up, enter 2-min cooldown
- After 15 min continuous quiet → reset escalation to 0 (LED pulse + app event)
- Mode switch → full reset (new session)
"""

import os
import sys
import time
import random
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
from core.data import get_wimz_store
from core.bark_frequency_tracker import get_bark_frequency_tracker

# Services
from services.media.usb_audio import get_usb_audio_service, set_agc
from services.reward.dispenser import get_dispenser_service
from services.media.led import get_led_service
from services.perception.bark_detector import get_bark_detector_service

logger = logging.getLogger(__name__)


class SGState(Enum):
    """Silent Guardian states"""
    LISTENING = "listening"           # Waiting for bark threshold
    INTERVENTION = "intervention"     # Running intervention sequence
    COOLDOWN = "cooldown"            # Brief pause after intervention
    GAVE_UP = "gave_up"              # Intervention timed out
    PANIC = "panic"                  # Distress episode - interventions suppressed


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
        self.calming_music_index = 0  # Current index in calming music playlist
        self.last_intervention_time = 0.0  # For escalation reset

        # Session tracking
        self.session_id = None
        self.session_start_time = None
        self.treats_dispensed = 0
        self.interventions_triggered = 0
        # Summary counters persisted at session end. These were never tracked,
        # so end_silent_guardian_session() always wrote 0 for them (compounded by
        # the positional-arg mismatch at the call sites — see stop()).
        self.total_barks = 0          # barks passing the loudness threshold
        self.successful_quiets = 0    # quiet periods achieved (treat OR praise)
        self.max_escalation_level = 0 # highest escalation level reached

        # Bark tracking
        self.last_bark_time = 0.0

        # Session duration limit (8 hours)
        self.max_session_duration = 8 * 60 * 60

        # Cooldown tracking
        self._cooldown_start = None
        self._cooldown_duration = 2.0  # 2 seconds between interventions
        self._gave_up_cooldown = 45.0  # 45 seconds after giving up

        # Timeouts
        self.intervention_timeout = 90.0  # Max 90 seconds per intervention

        # Anti-treat-farming: eligibility cooldown after treats
        self.last_treat_time = 0.0  # When last treat was dispensed
        # 10 min default before treats available again; per-unit override via profile session_limits
        self.treat_eligibility_cooldown = float(
            self.config.get('session_limits', {}).get('treat_eligibility_cooldown', 600.0)
        )
        self.consecutive_interventions = 0  # For progressive quiet requirements
        self.treat_eligible = True  # Whether treats can be given

        # Quiet reset tracking (continuous quiet → reset escalation)
        escalation_config = self.config.get('escalation', {})
        self.quiet_reset_minutes = escalation_config.get('reset_after_quiet_minutes', 15)
        self.quiet_since = 0.0  # Timestamp of when continuous quiet started (0 = not quiet)
        self.total_escalation_resets = 0  # Resets this session

        # Bark-type session log: every bark passing the loudness threshold,
        # with classifier label + confidence + reporting group. ALL summary
        # aggregation reads from this (session-scoped — never the cumulative
        # counters, which have a known persistence bug).
        self.session_bark_log: List[Dict[str, Any]] = []
        # Loud non-bark sounds (notbark veto with high loudness): (ts, db).
        # Context for panic alerts ("after hearing loud noises").
        self.loud_noises: List[tuple] = []

        # Level-4 once-per-session summary guard (also persisted in DB row)
        self.summary_sent = False

        # Panic detection state
        self.panic_active = False
        self.panic_start_time = 0.0
        self.panic_trigger = None            # 'burst' | 'sustained_rate' | 'futility'
        self.panic_loud_noise_prior = False
        self.panic_episodes_this_session = 0
        self.panic_songs_played = 0
        self.post_panic_cooldown_until = 0.0
        self.sustained_rate_since = 0.0      # weighted BPM above threshold since
        self.futility_count = 0              # consecutive futile intervention cycles
        self.futility_pending_since = 0.0    # intervention-end ts awaiting resume check

        # Calming music chaining (Level 4 + panic): last track for no-repeat
        self.calming_last_track = None
        self.calming_track_started_at = 0.0

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
                'threshold': 3,
                'window_seconds': 60,
                'loudness_threshold_db': -20,
                'confidence_minimum': 0.45
            },
            'escalation': {
                'window_minutes': 60,
                'reset_after_quiet_minutes': 15,
                'max_level': 4
            },
            'intervention_sequences': {
                'level_1': {'quiet_required_seconds': 20},
                'level_2': {'quiet_required_seconds': 30},
                'level_3': {
                    'movement_cycles': 3,
                    'move_duration_seconds': 0.4,
                    'move_speed_pct': 50,
                    'cycle_pause_seconds': 5.0,
                    'quiet_required_seconds': 30,
                },
                'level_4': {'quiet_period_duration': 20, 'quiet_periods_required': 2}
            },
            'session_limits': {
                'max_treats': 11,
                'min_time_between_treats': 120
            },
            'audio_paths': {
                'talks': '/home/morgan/dogbot/VOICEMP3/talks',
                'songs': '/home/morgan/dogbot/VOICEMP3/songs',
                'calming_music_playlist': [
                    'songs/default/dog_music_01.mp3',
                    'songs/default/dog_music_02.mp3',
                    'songs/default/dog_music_03.mp3',
                    'songs/default/dog_music_04.mp3',
                ]
            }
        }

    def _get_escalation_level(self) -> int:
        """
        Calculate current escalation level based on recent interventions

        Returns:
            1 = 0-2 interventions in last hour
            2 = 3 interventions in last hour
            3 = 4 interventions in last hour (physical movement)
            4 = 5+ interventions in last hour (calming music)
        """
        now = time.time()
        escalation_config = self.config.get('escalation', {})
        window_minutes = escalation_config.get('window_minutes', 60)
        max_level = escalation_config.get('max_level', 4)

        # Clean old events outside window
        window_seconds = window_minutes * 60
        self.escalation_events = [t for t in self.escalation_events if now - t < window_seconds]

        # Calculate level based on count
        count = len(self.escalation_events)
        if count <= 2:
            level = 1
        elif count == 3:
            level = 2
        elif count == 4:
            level = 3
        else:
            level = 4

        # Fast-escalation: when sustained barking exceeds the BPM threshold,
        # bypass the climb-one-level-at-a-time path and jump to max (L4 = music).
        # The user-tunable "punishment level" slider in the app writes to this
        # value via /sg/config. 0 means disabled.
        fast_bpm = int(self.config.get('bark_detection', {}).get('fast_escalation_bpm', 0) or 0)
        if fast_bpm > 0:
            try:
                bpm = self.bark_tracker.get_barks_per_minute('unknown')
                if bpm >= fast_bpm and level < max_level:
                    logger.info(
                        f"SG_FAST_ESCALATION: BPM {bpm} >= {fast_bpm} threshold — "
                        f"jumping level {level} -> {max_level}"
                    )
                    level = max_level
            except Exception as e:
                logger.debug(f"Fast-escalation BPM check failed: {e}")

        return min(level, max_level)

    def _check_escalation_reset(self):
        """
        Reset escalation after 15 minutes of continuous quiet.
        Continuous quiet = no bark events AND not in an active intervention.
        """
        # Nothing to reset if no interventions ever happened
        if self.last_intervention_time == 0.0:
            return

        # Don't check during active intervention
        if self.fsm_state != SGState.LISTENING:
            self.quiet_since = 0.0
            return

        now = time.time()

        # Start quiet timer if not already running
        if self.quiet_since == 0.0:
            self.quiet_since = now
            return

        quiet_minutes = (now - self.quiet_since) / 60.0

        if quiet_minutes >= self.quiet_reset_minutes and self.escalation_events:
            # === ESCALATION RESET CEREMONY ===
            old_level = self.current_escalation_level
            self.escalation_events = []
            self.current_escalation_level = 1
            self.consecutive_interventions = 0
            self.total_escalation_resets += 1
            self.quiet_since = 0.0  # Reset timer

            logger.info(
                f"SG_RESET: Dogs quiet for {quiet_minutes:.0f}min, "
                f"resetting escalation (was level {old_level} → 0, "
                f"reset #{self.total_escalation_resets} this session)"
            )

            # Green LED pulse (good dog signal)
            if self.led:
                self.led.set_pattern('success', duration=3.0)

            # Send event to app
            publish_system_event('sg_reset', {
                'reason': 'quiet_period',
                'quiet_minutes': round(quiet_minutes, 1),
                'previous_level': old_level,
                'resets_this_session': self.total_escalation_resets,
                'session_id': self.session_id,
            }, 'silent_guardian')

            # Log progressive quiet reset
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
            # Spec store session (dual-write; WIMZ_Data_Architecture_Spec §4)
            self.wimz = get_wimz_store()
            self.wimz_session_id = self.wimz.start_session(
                'monitor', 'autonomous',
                model_versions={'audio': self.wimz.model_id_for('dog_bark_classifier')})
            self._last_cue_event_id = None
            self.session_start_time = time.time()
            self.treats_dispensed = 0
            self.interventions_triggered = 0
            self.total_barks = 0
            self.successful_quiets = 0
            self.max_escalation_level = 0
            self.escalation_events = []
            self.quiet_since = 0.0
            self.total_escalation_resets = 0
            self._reset_bark_and_panic_state()

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

        # If a panic episode is in progress, record it before teardown
        if self.panic_active:
            self._log_panic_episode_event(time.time())
            self.panic_active = False

        # Stop calming music if playing
        self._stop_calming_music()

        # End database session
        if self.session_id:
            self.store.end_silent_guardian_session(
                self.session_id,
                total_barks=self.total_barks,
                interventions=self.interventions_triggered,
                successful_quiets=self.successful_quiets,
                treats_dispensed=self.treats_dispensed,
                max_escalation=self.max_escalation_level,
            )
        if getattr(self, 'wimz_session_id', None):
            self.wimz.end_session(self.wimz_session_id)
            self.wimz_session_id = None

        # Wait for thread
        if self.mode_thread and self.mode_thread.is_alive():
            self.mode_thread.join(timeout=2.0)

        # Re-enable AGC
        set_agc(True)
        logger.info("AGC re-enabled")

        # Publish stop event (bark-type breakdown + tags from the session log)
        stop_counts = self._bark_type_counts()
        aggressive_min = self.config.get('bark_classification', {}) \
            .get('aggressive_session_tag_min', 5)
        publish_system_event('silent_guardian_stopped', {
            'session_id': self.session_id,
            'interventions': self.interventions_triggered,
            'treats': self.treats_dispensed,
            'bark_types': stop_counts,
            'headline': self._headline(),
            'aggressive_tag': stop_counts.get('aggressive', 0) >= aggressive_min,
            'panic_episodes': self.panic_episodes_this_session,
        }, 'silent_guardian')

        logger.info("Silent Guardian stopped")

    def _log_bark_event(self, event, gate: str):
        """Write a spec `bark` event row (spec §5; rule 5: keep hard negatives).

        gate: 'passed' | 'below_loudness' | 'veto_rejected' — recorded in the
        payload so rejected/low barks are kept as training hard negatives.
        """
        try:
            data = event.data
            dog_id = data.get('dog_id')
            dog_name = data.get('dog_name')
            wimz_dog = None
            if (dog_id and dog_id != 'unknown') or dog_name:
                wimz_dog = self.wimz.get_or_create_dog(
                    legacy_id=dog_id if dog_id and dog_id != 'unknown' else None,
                    name=dog_name)
            payload = {
                'db': data.get('loudness_db'),
                'duration_ms': data.get('duration_ms'),
                'class': 'notbark' if gate == 'veto_rejected' else 'bark',
                'emotion': data.get('emotion'),
                'gate': gate,
            }
            self.wimz.log_event(
                self.wimz_session_id, 'bark', payload,
                dog_id=wimz_dog,
                confidence=data.get('confidence'),
                model_id=self.wimz.model_id_for('dog_bark_classifier'))
        except Exception as e:
            logger.debug(f"wimz bark event log failed: {e}")

    def _on_audio_event(self, event):
        """Handle bark events"""
        # Don't process events if not running
        if not self.running:
            return

        if event.subtype == 'bark_false_positive':
            self._log_bark_event(event, gate='veto_rejected')  # hard negative, kept
            logger.info(f"ML says not a bark (notbark={event.data.get('confidence', 0):.2f}) — "
                       f"cancelling any pending intervention")
            # Reset bark tracker count so false positive doesn't accumulate toward threshold
            try:
                bark_config = self.config.get('bark_detection', {})
                threshold = bark_config.get('threshold', 2)
                dog_id = event.data.get('dog_id', 'unknown')
                if hasattr(self, 'bark_tracker'):
                    self.bark_tracker.reset_dog(dog_id)
            except Exception:
                pass
            return

        if event.subtype == 'loud_noise':
            # High-energy sound the classifier vetoed as notbark — remember it
            # as environmental context for panic alerts ("heard loud noises").
            try:
                db = float(event.data.get('loudness_db', -99))
                min_db = self.config.get('panic', {}).get('loud_noise_min_db', -15)
                if db >= min_db:
                    now = time.time()
                    self.loud_noises.append((now, db))
                    # Keep a bounded window (2x lookback is plenty)
                    lookback = self.config.get('panic', {}).get(
                        'loud_noise_lookback_seconds', 120)
                    cutoff = now - 2 * lookback
                    self.loud_noises = [x for x in self.loud_noises if x[0] >= cutoff]
                    logger.info(f"SG: loud non-bark noise noted ({db:.1f}dB)")
            except Exception as e:
                logger.debug(f"loud_noise handling failed: {e}")
            return

        if event.subtype != 'bark_detected':
            return

        # Double-check speaker suppression: ignore bark events that slip through
        # during speaker playback (belt-and-suspenders with detection-loop suppression)
        try:
            bark_svc = get_bark_detector_service()
            if bark_svc.is_suppressed():
                logger.debug("Ignoring bark event during speaker suppression")
                return
        except Exception:
            pass

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
            self._log_bark_event(event, gate='below_loudness')  # hard negative, kept
            logger.info(f"BARK_BELOW_THRESHOLD: {loudness_db:.1f}dB < {loudness_threshold}dB — ignoring")
            return

        # Emotion classifier enriches bark events for logging/analytics only.
        # It must NOT gate Silent Guardian responses — all barks passing the
        # loudness threshold should trigger intervention regardless of emotion confidence.
        if confidence > 0:
            logger.info(f"Bark emotion: conf={confidence:.2f} (logged, not gating)")

        logger.info(f"Bark detected: {dog_name or dog_id or 'unknown'} (conf: {confidence:.2f}, loud: {loudness_db:.1f}dB)")
        self._log_bark_event(event, gate='passed')
        now = time.time()
        self.last_bark_time = now
        self.total_barks += 1  # confirmed bark (passed loudness threshold)
        self.quiet_since = 0.0  # Any bark resets the continuous quiet timer

        # Bark-type classification: attach classifier label + reporting group
        # to the session log (below confidence threshold => 'unclassified').
        emotion = event.data.get('emotion')
        group, label = self._classify_bark(emotion, confidence)
        self.session_bark_log.append({
            'ts': now,
            'label': label,
            'confidence': confidence,
            'group': group,
            'loudness_db': loudness_db,
            'dog_id': dog_id,
            'dog_name': dog_name,
        })

        # Panic signal 2 (futility): did barking resume within the window
        # after the last intervention ended?
        panic_cfg = self.config.get('panic', {})
        if self.futility_pending_since > 0:
            resume_window = panic_cfg.get('futility_resume_window_seconds', 30)
            if now - self.futility_pending_since <= resume_window:
                self.futility_count += 1
                logger.info(f"SG_FUTILITY: barking resumed within {resume_window}s "
                            f"of intervention end ({self.futility_count} consecutive)")
            else:
                self.futility_count = 0
            self.futility_pending_since = 0.0

        # Panic signal 1a (burst): lots of barks + bad emotion together
        if not self.panic_active and panic_cfg.get('enabled', True):
            if self._check_panic_burst(now):
                self._enter_panic('burst', dog_id, dog_name)
            elif self.futility_count >= panic_cfg.get('futility_cycles', 3):
                self._enter_panic('futility', dog_id, dog_name)

        # During panic (or post-panic cooldown) interventions are suppressed —
        # barks are still logged above for the episode record and calm detection.
        if self.panic_active:
            return

        # Handle based on current state
        if self.fsm_state == SGState.LISTENING:
            if now < self.post_panic_cooldown_until:
                logger.debug("SG: post-panic cooldown — not intervening")
                return
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
            self.max_escalation_level = max(self.max_escalation_level,
                                            self.current_escalation_level)

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

            # Spec cue_issued event — the intervention IS the 'quiet' cue.
            # Event id kept for the training_attempt row on quiet-achieved.
            try:
                wimz_dog = self.wimz.get_or_create_dog(
                    legacy_id=dog_id if dog_id and dog_id != 'unknown' else None,
                    name=dog_name)
                self._last_cue_event_id = self.wimz.log_event(
                    self.wimz_session_id, 'cue_issued',
                    {'trick': 'quiet', 'cue_type': 'voice',
                     'text': f'escalation_level_{self.current_escalation_level}'},
                    dog_id=wimz_dog, label_source='auto_rule')
            except Exception as e:
                logger.debug(f"wimz cue event log failed: {e}")

            # Publish to bus so DogEventLogger persists it in dog_events
            # and main_treatbot forwards it over the relay WebSocket to the app.
            publish_system_event('sg_escalation', {
                'dog_id': dog_id,
                'dog_name': dog_name,
                'escalation_level': self.current_escalation_level,
                'session_id': self.session_id,
                'action': 'intervention_started',
                'interventions_in_hour': len(self.escalation_events),
            }, 'silent_guardian')

            logger.info(f"Starting Level {self.current_escalation_level} intervention for {dog_name or dog_id or 'unknown'}")

        # First Level-4 hit this session -> one summary notification (never
        # repeated on later Level-4 hits; guard persisted in the session row).
        if self.current_escalation_level >= 4:
            self._maybe_send_level4_summary()

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

                # Panic detection (sustained rate + futility expiry)
                self._check_panic(time.time())

                # Chain calming music tracks (Level 4 / panic continuous play)
                self._tick_calming_music(time.time())

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
                total_barks=self.total_barks,
                interventions=self.interventions_triggered,
                successful_quiets=self.successful_quiets,
                treats_dispensed=self.treats_dispensed,
                max_escalation=self.max_escalation_level,
            )

        self.session_id = self.store.start_silent_guardian_session()
        self.session_start_time = time.time()
        self.treats_dispensed = 0
        self.interventions_triggered = 0
        self.total_barks = 0
        self.successful_quiets = 0
        self.max_escalation_level = 0
        self.escalation_events = []
        self.quiet_since = 0.0
        self.total_escalation_resets = 0
        self._reset_bark_and_panic_state()

        logger.info(f"New session: {self.session_id}")

    def _reset_bark_and_panic_state(self):
        """Reset the session bark log and panic tracking (new session)."""
        self.session_bark_log = []
        self.loud_noises = []
        self.summary_sent = False
        self.panic_active = False
        self.panic_start_time = 0.0
        self.panic_trigger = None
        self.panic_loud_noise_prior = False
        self.panic_episodes_this_session = 0
        self.panic_songs_played = 0
        self.post_panic_cooldown_until = 0.0
        self.sustained_rate_since = 0.0
        self.futility_count = 0
        self.futility_pending_since = 0.0
        self.calming_last_track = None
        self.calming_track_started_at = 0.0

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

        elif self.fsm_state == SGState.PANIC:
            self._process_panic()

    def _process_intervention(self):
        """Process the intervention based on escalation level"""
        now = time.time()

        # Check intervention timeout (90 seconds)
        if now - self.intervention_start_time >= self.intervention_timeout:
            logger.warning(f"Intervention timed out after {self.intervention_timeout}s - giving up")
            self._stop_calming_music()
            self.fsm_state = SGState.GAVE_UP
            self._cooldown_start = now
            # Intervention cycle ended (unsuccessfully) — arm the futility
            # resume check (panic signal 2)
            self.futility_pending_since = now

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

        # Bark during intervention — reset quiet timer but DON'T replay commands.
        # The "quiet" command was already given; replaying it on every bark creates
        # a spam loop. Just reset the quiet countdown so the dog must be quiet again.
        if self.bark_during_intervention:
            logger.info(f"Bark during Level {self.current_escalation_level} — resetting quiet timer (not restarting sequence)")
            self.bark_during_intervention = False
            self.quiet_start_time = time.time()
            self.quiet_periods_achieved = 0

        # Route to level-specific handler
        if self.current_escalation_level == 1:
            self._process_level_1(now)
        elif self.current_escalation_level == 2:
            self._process_level_2(now)
        elif self.current_escalation_level == 3:
            self._process_level_movement(now)
        else:
            self._process_level_music(now)

    def _process_level_1(self, now: float):
        """
        Level 1: Gentle reminder
        Step 0: Play "quiet" command (simple and clear)
        Step 1: Wait for quiet period (progressive) → reward
        """
        level_config = self.config.get('intervention_sequences', {}).get('level_1', {})
        base_quiet = level_config.get('quiet_required_seconds', 20)
        quiet_required = self._get_progressive_quiet_duration(base_quiet)

        if self.intervention_step == 0:
            if self.last_step_time == 0.0:
                logger.info("Level 1 - Step 0: Playing quiet command")
                if self.led:
                    self.led.set_pattern('attention', duration=2.0)

                # Just play "quiet" - simple and clear
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
        Simplified sequence: quiet → no → quiet (clear, non-overlapping)
        Step 0: Play "quiet" (waits for completion)
        Step 1: Play "no" (waits for completion)
        Step 2: Play "quiet" again (waits for completion)
        Step 3: Wait for quiet period (progressive) → reward
        """
        level_config = self.config.get('intervention_sequences', {}).get('level_2', {})
        base_quiet = level_config.get('quiet_required_seconds', 30)
        quiet_required = self._get_progressive_quiet_duration(base_quiet)

        if self.intervention_step == 0:
            if self.last_step_time == 0.0:
                logger.info("Level 2 - Step 0: quiet")
                if self.led:
                    self.led.set_pattern('attention', duration=5.0)
                self._play_audio('quiet.mp3')  # Waits for completion
                self.last_step_time = time.time()  # Update after audio finishes
                self.intervention_step = 1

        elif self.intervention_step == 1:
            # Small pause between commands (0.5s after audio finished)
            if now - self.last_step_time >= 0.5:
                logger.info("Level 2 - Step 1: no")
                self._play_audio('no.mp3')  # Waits for completion
                self.last_step_time = time.time()
                self.intervention_step = 2

        elif self.intervention_step == 2:
            # Small pause, then final quiet
            if now - self.last_step_time >= 0.5:
                logger.info("Level 2 - Step 2: quiet (final)")
                self._play_audio('quiet.mp3')  # Waits for completion
                self.last_step_time = time.time()
                self.quiet_start_time = time.time()
                self.intervention_step = 3

        elif self.intervention_step == 3:
            # Check if quiet period achieved
            if self.quiet_start_time > 0:
                quiet_duration = now - self.quiet_start_time
                if quiet_duration >= quiet_required:
                    logger.info(f"Level 2 - SUCCESS: {quiet_duration:.1f}s quiet achieved")
                    self._give_reward()
                    self.fsm_state = SGState.COOLDOWN
                    self._cooldown_start = now

    def _process_level_movement(self, now: float):
        """
        Level 3: Physical attention sequence (runs BEFORE the calming-music level).
        Step 0: play "quiet" + run the 3-cycle in-place movement sequence (blocking)
        Step 1: Wait for quiet period (progressive) → reward
        """
        level_config = self.config.get('intervention_sequences', {}).get('level_3', {})
        base_quiet = level_config.get('quiet_required_seconds', 30)
        quiet_required = self._get_progressive_quiet_duration(base_quiet)

        if self.intervention_step == 0:
            if self.last_step_time == 0.0:
                logger.info("Level 3 (movement) - Step 0: quiet command + movement sequence")
                if self.led:
                    self.led.set_pattern('attention', duration=8.0)
                self._play_audio('quiet.mp3')  # Waits for completion
                self._run_movement_sequence()  # Blocking ~21s; always finishes
                self.last_step_time = time.time()
                self.quiet_start_time = time.time()
                self.intervention_step = 1

        elif self.intervention_step == 1:
            # Check if quiet period achieved
            if self.quiet_start_time > 0:
                quiet_duration = now - self.quiet_start_time
                if quiet_duration >= quiet_required:
                    logger.info(f"Level 3 (movement) - SUCCESS: {quiet_duration:.1f}s quiet achieved")
                    self._give_reward()
                    self.fsm_state = SGState.COOLDOWN
                    self._cooldown_start = now

    def _run_movement_sequence(self):
        """Run the quiet movement sequence: 3 cycles of forward/back/left/right.

        Each move is a quick in-place burst (~300-500ms), motors halted between
        moves, with a pause between cycles. Drives the main wheels via the motor
        command bus (CommandSource.AUTONOMOUS); the bus applies its own +/-70
        safety clamp. The treat-carousel anti-jam is a separate stepper and is
        not touched here.

        Always completes all configured cycles (no bark abort) but bails out
        immediately on mode shutdown (self.running == False), guaranteeing the
        motors are halted via the finally block.
        """
        cfg = self.config.get('intervention_sequences', {}).get('level_3', {})
        cycles = int(cfg.get('movement_cycles', 3))
        move_dur = float(cfg.get('move_duration_seconds', 0.4))
        speed = int(cfg.get('move_speed_pct', 50))
        pause = float(cfg.get('cycle_pause_seconds', 5.0))

        try:
            from core.motor_command_bus import (
                get_motor_bus, create_motor_command, CommandSource
            )
            bus = get_motor_bus()
        except Exception as e:
            logger.error(f"SG_MOVEMENT: motor bus unavailable, skipping sequence: {e}")
            return

        if not (bus and bus.running):
            logger.warning("SG_MOVEMENT: motor bus not running, skipping sequence")
            return

        # In-place moves as (label, left_pct, right_pct)
        moves = [
            ('forward', speed, speed),
            ('back', -speed, -speed),
            ('left', -speed, speed),    # left wheel back, right wheel fwd -> pivot left
            ('right', speed, -speed),   # left wheel fwd, right wheel back -> pivot right
        ]

        def _drive(left: int, right: int):
            bus.send_command(create_motor_command(left, right, CommandSource.AUTONOMOUS))

        logger.info(f"SG_MOVEMENT: starting {cycles}-cycle quiet sequence "
                    f"(move={move_dur}s, speed={speed}%, pause={pause}s)")
        try:
            for c in range(cycles):
                if not self.running:
                    break
                for label, left, right in moves:
                    if not self.running:
                        break
                    logger.debug(f"SG_MOVEMENT: cycle {c + 1}/{cycles} - {label}")
                    _drive(left, right)
                    time.sleep(move_dur)
                    _drive(0, 0)          # halt between every move
                    time.sleep(0.08)      # settle before reversing direction
                # Pause between cycles, but not after the final one
                if c < cycles - 1 and self.running:
                    time.sleep(pause)
        finally:
            _drive(0, 0)  # guarantee motors are stopped
        logger.info("SG_MOVEMENT: quiet sequence complete")

    def _process_level_music(self, now: float):
        """
        Level 4: Calming music mode
        Step 0: quiet + start calming music
        Step 1: Wait for 2x quiet periods (progressive duration) → reward
        """
        level_config = self.config.get('intervention_sequences', {}).get('level_4', {})
        base_period = level_config.get('quiet_period_duration', 20)
        quiet_period_duration = self._get_progressive_quiet_duration(base_period)
        quiet_periods_required = level_config.get('quiet_periods_required', 2)

        if self.intervention_step == 0:
            if self.last_step_time == 0.0:
                logger.info("Level 4 - Step 0: quiet + starting calming music")
                if self.led:
                    self.led.set_pattern('calm', duration=60.0)
                self._play_audio('quiet.mp3')  # Waits for completion
                self._start_calming_music()
                self.last_step_time = time.time()
                self.quiet_start_time = time.time()
                self.quiet_periods_achieved = 0
                self.intervention_step = 1

        elif self.intervention_step == 1:
            # Check if quiet period achieved
            if self.quiet_start_time > 0:
                quiet_duration = now - self.quiet_start_time
                if quiet_duration >= quiet_period_duration:
                    self.quiet_periods_achieved += 1
                    logger.info(f"Level 4 - Quiet period {self.quiet_periods_achieved}/{quiet_periods_required} achieved")

                    if self.quiet_periods_achieved >= quiet_periods_required:
                        logger.info("Level 4 - SUCCESS: All quiet periods achieved")
                        self._stop_calming_music()
                        self._give_reward()
                        self.fsm_state = SGState.COOLDOWN
                        self._cooldown_start = now
                    else:
                        # Reset for next quiet period
                        self.quiet_start_time = now

    def _get_calming_playlist(self):
        """Return (playlist, base_path) for calming music.

        Playlist source: audio_paths.calming_music_playlist in the SG rules
        YAML (curated classical tracks). Falls back to the legacy single-file
        calming_music key if the playlist is missing.
        """
        audio_config = self.config.get('audio_paths', {})
        playlist = audio_config.get('calming_music_playlist', [])
        if not playlist:
            old_file = audio_config.get('calming_music', 'songs/default/mozart_piano.mp3')
            playlist = [old_file]
        base = audio_config.get('base', '/home/morgan/dogbot/VOICEMP3')
        return playlist, base

    def _play_random_calming_track(self) -> bool:
        """Play one random calming track (never the same track back-to-back)."""
        playlist, base = self._get_calming_playlist()
        candidates = [t for t in playlist if t != self.calming_last_track] or list(playlist)
        random.shuffle(candidates)
        for track in candidates:
            full_path = os.path.join(base, track)
            if not os.path.exists(full_path):
                logger.warning(f"Calming track missing, skipping: {full_path}")
                continue
            if self.audio:
                self.audio.play_file(full_path, loop=False)
                self.calming_last_track = track
                self.calming_track_started_at = time.time()
                logger.info(f"Calming music: {os.path.basename(track)}")
                return True
        logger.warning("No playable calming music track found")
        return False

    def _start_calming_music(self):
        """Start calming music (random track; chaining handled by _tick_calming_music)"""
        if self.calming_music_playing:
            return
        try:
            if self._play_random_calming_track():
                self.calming_music_playing = True
        except Exception as e:
            logger.error(f"Failed to start calming music: {e}")

    def _tick_calming_music(self, now: float):
        """Chain to the next random track when the current one ends.

        Level 4: continuous music for the whole music phase (a single track
        no longer dies mid-intervention). Panic: play exactly
        panic.calming_routine_songs tracks in a row, then stop.
        Called from the mode run loop (~10Hz).
        """
        if not self.calming_music_playing or not self.audio:
            return
        # Startup grace: play() takes a moment before is_busy() reports True
        if now - self.calming_track_started_at < 3.0:
            return
        try:
            if self.audio.is_busy():
                return
        except Exception:
            return

        # Current track ended naturally
        if self.panic_active:
            max_songs = self.config.get('panic', {}).get('calming_routine_songs', 3)
            if self.panic_songs_played >= max_songs:
                self.calming_music_playing = False
                logger.info(f"Panic calming routine complete ({max_songs} songs)")
                return
            if self._play_random_calming_track():
                self.panic_songs_played += 1
            else:
                self.calming_music_playing = False
        else:
            if not self._play_random_calming_track():
                self.calming_music_playing = False

    def _stop_calming_music(self):
        """Stop calming music"""
        if not self.calming_music_playing:
            return

        try:
            if self.audio:
                self.audio.stop()
            self.calming_music_playing = False
            self.calming_music_index = 0
            self.calming_track_started_at = 0.0
            logger.info("Calming music stopped")
        except Exception as e:
            logger.error(f"Failed to stop calming music: {e}")

    # ===== Bark type classification & session aggregation =====
    # Model label set (ai/models/emotion_mapping.json): aggressive, alert,
    # anxious, attention, playful, scared (+ notbark veto). Grouping and all
    # owner-facing phrasing live in bark_classification config — never here.

    def _classify_bark(self, emotion: Optional[str], confidence: float) -> tuple:
        """Map a classifier output to (reporting_group, label).
        Below the confidence threshold -> ('unclassified', 'unclassified')."""
        cfg = self.config.get('bark_classification', {})
        threshold = cfg.get('confidence_threshold', 0.40)
        if not emotion or emotion in ('unknown', 'notbark') or (confidence or 0.0) < threshold:
            return 'unclassified', 'unclassified'
        group = cfg.get('groups', {}).get(emotion, 'unclassified')
        return group, emotion

    def classify_bark_type(self, emotion: Optional[str], confidence: float) -> tuple:
        """Public boundary: same (reporting_group, label) mapping the session
        log uses. bark_detector calls this to stamp live bark events so the
        app can label every bark row (app contract 2026-09-01)."""
        return self._classify_bark(emotion, confidence)

    def _owner_phrase(self, group: str) -> str:
        cfg = self.config.get('bark_classification', {})
        return cfg.get('owner_phrases', {}).get(group, 'barking')

    def _bark_type_counts(self, since_ts: float = 0.0,
                          until_ts: Optional[float] = None) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for b in self.session_bark_log:
            if b['ts'] < since_ts:
                continue
            if until_ts is not None and b['ts'] > until_ts:
                continue
            counts[b['group']] = counts.get(b['group'], 0) + 1
        return counts

    def _bark_timeline(self, bucket_minutes: int = 30) -> List[Dict[str, Any]]:
        """Bark-type mix over time within the session (30-min buckets)."""
        if not self.session_start_time:
            return []
        buckets = []
        now = time.time()
        n = int((now - self.session_start_time) // (bucket_minutes * 60)) + 1
        for i in range(n):
            b_start = self.session_start_time + i * bucket_minutes * 60
            b_end = b_start + bucket_minutes * 60
            buckets.append({
                'offset_min': i * bucket_minutes,
                'counts': self._bark_type_counts(since_ts=b_start, until_ts=b_end),
            })
        return buckets

    def _bark_trend(self, now: Optional[float] = None) -> tuple:
        """Bark rate over the last trend window vs session average.
        Returns (trend_key, recent_rate_per_min, session_rate_per_min, window_min)."""
        now = now or time.time()
        sum_cfg = self.config.get('summary', {})
        window_min = sum_cfg.get('trend_window_minutes', 30)
        tolerance = sum_cfg.get('trend_tolerance', 0.25)
        session_min = max((now - self.session_start_time) / 60.0, 1e-6) \
            if self.session_start_time else 1e-6
        total = len(self.session_bark_log)
        session_rate = total / session_min
        effective_window = min(window_min, session_min)
        recent = sum(1 for b in self.session_bark_log
                     if b['ts'] >= now - window_min * 60)
        recent_rate = recent / max(effective_window, 1e-6)
        if session_rate <= 0:
            trend = 'flat'
        elif recent_rate < session_rate * (1 - tolerance):
            trend = 'improving'
        elif recent_rate > session_rate * (1 + tolerance):
            trend = 'worsening'
        else:
            trend = 'flat'
        return trend, round(recent_rate, 2), round(session_rate, 2), window_min

    def _headline(self, now: Optional[float] = None) -> str:
        """Owner-friendly one-liner: 'Mostly demand barking, tapering off'."""
        counts = self._bark_type_counts()
        if not counts:
            return "No barking so far"
        dominant = max(counts, key=lambda g: counts[g])
        trend, _, _, _ = self._bark_trend(now)
        trend_phrase = self.config.get('bark_classification', {}) \
            .get('trend_phrases', {}).get(trend, '')
        phrase = self._owner_phrase(dominant)
        return f"Mostly {phrase}, {trend_phrase}" if trend_phrase else f"Mostly {phrase}"

    def _current_action(self) -> str:
        if self.panic_active:
            return 'playing calming audio' if self.calming_music_playing \
                else 'monitoring quietly (interventions paused)'
        if self.fsm_state == SGState.INTERVENTION:
            if self.calming_music_playing:
                return 'playing calming audio'
            return f'running level {self.current_escalation_level} intervention'
        if self.fsm_state == SGState.LISTENING:
            return 'listening'
        return 'cooling down'

    def build_summary_payload(self, reason: str = 'status_pull') -> Dict[str, Any]:
        """Session summary — used by BOTH the Level-4 escalation notification
        and the on-demand app status pull (same payload, computed live).
        All numbers come from the session bark log (session-scoped) — never
        the cumulative counters (known persistence bug)."""
        now = time.time()
        counts = self._bark_type_counts()
        total = sum(counts.values())
        percentages = {g: round(100.0 * c / total, 1) for g, c in counts.items()} \
            if total else {}
        trend, recent_rate, session_rate, window_min = self._bark_trend(now)
        aggressive_min = self.config.get('bark_classification', {}) \
            .get('aggressive_session_tag_min', 5)
        return {
            'action': reason,
            'session_id': self.session_id,
            'session_duration_sec': int(now - self.session_start_time)
            if self.session_start_time else 0,
            'total_barks': total,
            'bark_types': counts,
            'bark_type_percentages': percentages,
            'bark_timeline': self._bark_timeline(),
            'treats_dispensed': self.treats_dispensed,
            'interventions_triggered': self.interventions_triggered,
            'current_escalation_level': self.current_escalation_level,
            'fsm_state': self.fsm_state.value if self.fsm_state else None,
            'trend': trend,
            'trend_detail': {
                'recent_rate_per_min': recent_rate,
                'session_rate_per_min': session_rate,
                'window_minutes': window_min,
            },
            'current_action': self._current_action(),
            'headline': self._headline(now),
            'aggressive_tag': counts.get('aggressive', 0) >= aggressive_min,
            'panic_active': self.panic_active,
            'panic_episodes': self.panic_episodes_this_session,
        }

    def _maybe_send_level4_summary(self):
        """Send ONE summary notification the first time this session hits
        Level 4. Guard persisted in the session row (summary_sent)."""
        if not self.config.get('summary', {}).get('enabled', True):
            return
        if self.summary_sent:
            return
        if self.session_id and self.session_id > 0 and \
                self.store.get_sg_summary_sent(self.session_id):
            self.summary_sent = True
            return
        self.summary_sent = True
        if self.session_id and self.session_id > 0:
            self.store.mark_sg_summary_sent(self.session_id)
        payload = self.build_summary_payload(reason='level4_escalation')
        publish_system_event('sg_summary', payload, 'silent_guardian')
        logger.info(f"SG_SUMMARY sent (first Level-4 hit): {payload['headline']}")

    # ===== Panic detection (do-not-intervene branch) =====

    def _check_panic_burst(self, now: float) -> bool:
        """Signal 1a: >= burst_barks in burst_window_seconds, with at least
        burst_bad_emotion_min of them distress/aggressive-class."""
        cfg = self.config.get('panic', {})
        window = cfg.get('burst_window_seconds', 5)
        need = cfg.get('burst_barks', 5)
        bad_min = cfg.get('burst_bad_emotion_min', 2)
        weighted_groups = set(cfg.get('panic_weighted_groups',
                                      ['distress', 'aggressive']))
        recent = [b for b in self.session_bark_log if b['ts'] >= now - window]
        if len(recent) < need:
            return False
        bad = sum(1 for b in recent if b['group'] in weighted_groups)
        return bad >= bad_min

    def _weighted_bark_rate(self, window_sec: float = 60.0,
                            now: Optional[float] = None) -> float:
        """Barks/min over the trailing window; distress/aggressive-class barks
        count extra (panic.distress_weight)."""
        now = now or time.time()
        cfg = self.config.get('panic', {})
        weight = cfg.get('distress_weight', 1.5)
        weighted_groups = set(cfg.get('panic_weighted_groups',
                                      ['distress', 'aggressive']))
        total = 0.0
        for b in self.session_bark_log:
            if b['ts'] >= now - window_sec:
                total += weight if b['group'] in weighted_groups else 1.0
        return total / (window_sec / 60.0)

    def _check_panic(self, now: float):
        """Run-loop panic checks: sustained-rate signal + futility expiry."""
        cfg = self.config.get('panic', {})

        # Futility window expiry: dog stayed quiet past the resume window,
        # so intervention cycles are no longer 'consecutive'
        if self.futility_pending_since > 0 and \
                now - self.futility_pending_since > \
                cfg.get('futility_resume_window_seconds', 30):
            self.futility_pending_since = 0.0
            self.futility_count = 0

        if not cfg.get('enabled', True) or self.panic_active:
            return

        # Signal 1b: sustained weighted rate with no self-termination
        rate = self._weighted_bark_rate(60.0, now)
        if rate >= cfg.get('panic_bark_rate', 10):
            if self.sustained_rate_since == 0.0:
                self.sustained_rate_since = now
            elif now - self.sustained_rate_since >= \
                    cfg.get('panic_duration_seconds', 180):
                last = self.session_bark_log[-1] if self.session_bark_log else {}
                self._enter_panic('sustained_rate',
                                  last.get('dog_id'), last.get('dog_name'))
        else:
            self.sustained_rate_since = 0.0

    def _recent_loud_noise(self, now: float) -> bool:
        lookback = self.config.get('panic', {}).get('loud_noise_lookback_seconds', 120)
        return any(ts >= now - lookback for ts, _db in self.loud_noises)

    def _panic_message(self, name: Optional[str]) -> str:
        who = name or 'Your dog'
        noise = ' after hearing loud noises' if self.panic_loud_noise_prior else ''
        if self.panic_trigger == 'futility':
            return (f"Possible distress episode — {who} isn't responding to "
                    f"interventions{noise}. Playing calming music. Live view available.")
        return (f"Possible distress episode — {who} had very intense barking{noise}. "
                f"Interventions paused; playing calming music. Live view available.")

    def _enter_panic(self, trigger: str, dog_id: Optional[str] = None,
                     dog_name: Optional[str] = None):
        """Enter the do-not-intervene panic branch."""
        with self._state_lock:
            if self.panic_active:
                return
            now = time.time()
            self.panic_active = True
            self.panic_start_time = now
            self.panic_trigger = trigger
            self.panic_loud_noise_prior = self._recent_loud_noise(now)
            self.panic_episodes_this_session += 1
            self.panic_songs_played = 0
            self.sustained_rate_since = 0.0
            self.futility_count = 0
            self.futility_pending_since = 0.0
            self.fsm_state = SGState.PANIC
            self._cooldown_start = None

        if self.session_id and self.session_id > 0:
            self.store.increment_sg_panic_episodes(self.session_id)

        cfg = self.config.get('panic', {})
        severity = 'high' if self.panic_episodes_this_session > \
            cfg.get('max_episodes_before_high_severity', 2) else 'warning'
        name = dog_name or self.intervention_dog_name

        # Calming routine: N random classical songs in a row. If Level-4 music
        # is already playing, the current track counts as song 1.
        if cfg.get('play_calming_routine', True):
            if self.calming_music_playing:
                self.panic_songs_played = 1
            elif self._play_random_calming_track():
                self.calming_music_playing = True
                self.panic_songs_played = 1

        if self.led:
            self.led.set_pattern('calm', duration=120.0)

        publish_system_event('panic_alert', {
            'action': 'started',
            'severity': severity,
            'trigger': trigger,
            'dog_id': dog_id or self.intervention_dog_id,
            'dog_name': name,
            'message': self._panic_message(name),
            'episode_num': self.panic_episodes_this_session,
            'loud_noise_prior': self.panic_loud_noise_prior,
            'bark_type_mix': self._bark_type_counts(since_ts=time.time() - 300),
            'session_id': self.session_id,
        }, 'silent_guardian')

        logger.warning(
            f"SG_PANIC: entered (trigger={trigger}, episode "
            f"#{self.panic_episodes_this_session}, severity={severity}, "
            f"loud_noise_prior={self.panic_loud_noise_prior})")

    def _process_panic(self):
        """PANIC state: no interventions; exit via hysteresis (quiet for
        calm_exit_seconds AND at least min_panic_duration_seconds elapsed)."""
        now = time.time()
        cfg = self.config.get('panic', {})
        calm_exit = cfg.get('calm_exit_seconds', 300)
        min_duration = cfg.get('min_panic_duration_seconds', 300)
        quiet_for = now - self.last_bark_time if self.last_bark_time > 0 \
            else now - self.panic_start_time
        if now - self.panic_start_time >= min_duration and quiet_for >= calm_exit:
            self._exit_panic(now)

    def _log_panic_episode_event(self, now: float):
        """Spec §5 `panic_episode` event (schema v0.5) — full episode record."""
        try:
            mix = self._bark_type_counts(since_ts=self.panic_start_time - 300)
            last = next((b for b in reversed(self.session_bark_log)
                         if b['ts'] >= self.panic_start_time - 300), {})
            wimz_dog = None
            dog_id = last.get('dog_id')
            dog_name = last.get('dog_name')
            if (dog_id and dog_id != 'unknown') or dog_name:
                wimz_dog = self.wimz.get_or_create_dog(
                    legacy_id=dog_id if dog_id and dog_id != 'unknown' else None,
                    name=dog_name)
            self.wimz.log_event(
                self.wimz_session_id, 'panic_episode', {
                    'trigger': self.panic_trigger,
                    'bark_type_mix': mix,
                    'barks_in_window': sum(mix.values()),
                    'loud_noise_prior': self.panic_loud_noise_prior,
                    'episode_num': self.panic_episodes_this_session,
                    'duration_sec': int(now - self.panic_start_time),
                },
                dog_id=wimz_dog, label_source='auto_rule')
        except Exception as e:
            logger.debug(f"wimz panic_episode log failed: {e}")

    def _exit_panic(self, now: float):
        """Exit panic: log the episode, notify, start intervention cooldown."""
        duration = int(now - self.panic_start_time)
        cfg = self.config.get('panic', {})
        self.post_panic_cooldown_until = now + \
            cfg.get('post_panic_cooldown_seconds', 900)

        self._log_panic_episode_event(now)
        self._stop_calming_music()

        last = next((b for b in reversed(self.session_bark_log)
                     if b['ts'] >= self.panic_start_time - 300), {})
        name = last.get('dog_name') or self.intervention_dog_name
        who = name or 'Your dog'
        publish_system_event('panic_alert', {
            'action': 'ended',
            'severity': 'info',
            'trigger': self.panic_trigger,
            'dog_id': last.get('dog_id') or self.intervention_dog_id,
            'dog_name': name,
            'message': f"{who} has calmed down. Episode lasted "
                       f"{max(duration // 60, 1)} min.",
            'episode_num': self.panic_episodes_this_session,
            'duration_sec': duration,
            'session_id': self.session_id,
        }, 'silent_guardian')

        with self._state_lock:
            self.panic_active = False
            self.panic_trigger = None
            self.panic_start_time = 0.0
            self.panic_songs_played = 0
            self.fsm_state = SGState.LISTENING
            self._cooldown_start = None

        logger.info(f"SG_PANIC: exited after {duration}s — intervention "
                    f"cooldown {cfg.get('post_panic_cooldown_seconds', 900)}s")

    def _give_reward(self):
        """Give reward for successful quiet — with bark-state safety check"""
        # SAFETY: Verify dog is STILL quiet at the moment of dispensing.
        # Prevents race condition where bark arrives between loop tick and reward.
        if self.bark_during_intervention:
            logger.info("SG_REWARD_CANCELLED: Dog barking at reward time — not dispensing")
            self.bark_during_intervention = False
            self.quiet_start_time = time.time()
            self.quiet_periods_achieved = 0
            return

        # Double-check: reject reward if bark detected in last 5 seconds
        if self.last_bark_time > 0 and (time.time() - self.last_bark_time) < 5.0:
            logger.info(
                f"SG_REWARD_CANCELLED: Last bark {time.time() - self.last_bark_time:.1f}s ago "
                f"(< 5s safety window) — restarting quiet timer"
            )
            self.quiet_start_time = time.time()
            self.quiet_periods_achieved = 0
            return

        # Stop calming music if playing
        self._stop_calming_music()

        # Track consecutive interventions for progressive quiet
        self.consecutive_interventions += 1
        # A quiet period was genuinely achieved (we're past the bark-state safety
        # checks). Counts whether the outcome is a treat or praise-only.
        self.successful_quiets += 1
        # Intervention cycle ended (successfully) — arm the futility resume
        # check: if barking restarts within the window it counts toward panic.
        self.futility_pending_since = time.time()

        # Check treat limit — after the session cap, KEEP INTERVENING with
        # verbal praise (prevents behavior extinction over a long session) but
        # dispense no more treats. Treat it as a normal praise-only outcome:
        # LED + audio + logged, so post-cap quiets still show up in history.
        max_treats = self.config.get('session_limits', {}).get('max_treats', 11)
        if self.treats_dispensed >= max_treats:
            logger.info(f"Treat limit reached ({self.treats_dispensed}/{max_treats}) - praise only, no treat")
            if self.led:
                self.led.set_pattern('success', duration=2.0)
            self._play_audio('good.mp3')

            self.store.log_sg_intervention(
                session_id=self.session_id,
                dog_id=self.intervention_dog_id,
                dog_name=self.intervention_dog_name,
                escalation_level=self.current_escalation_level,
                quiet_achieved=True,
                treat_given=False,  # Session treat cap reached
                music_played=(self.current_escalation_level == 4)
            )
            self._log_quiet_attempt(success=True, reward_dispensed=0, dispense_id=None)
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
            self._play_audio('good.mp3')

            # Log intervention without treat
            self.store.log_sg_intervention(
                session_id=self.session_id,
                dog_id=self.intervention_dog_id,
                dog_name=self.intervention_dog_name,
                escalation_level=self.current_escalation_level,
                quiet_achieved=True,
                treat_given=False,  # No treat during cooldown
                music_played=(self.current_escalation_level == 4)
            )
            self._log_quiet_attempt(success=True, reward_dispensed=0, dispense_id=None)
            return

        # Full reward: LED celebration + audio + treat
        if self.led:
            self.led.celebration_sequence(3.0)

        # Play good.mp3 for reward
        logger.info("SG_REWARD: Playing good.mp3")
        self._play_audio('good.mp3')
        time.sleep(1.0)

        # Dispense treat
        if self.dispenser:
            self.dispenser.dispense_treat(
                dog_id=self.intervention_dog_id,
                dog_name=self.intervention_dog_name,
                reason='silent_guardian_reward',
                wimz_session_id=getattr(self, 'wimz_session_id', None)
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
            music_played=(self.current_escalation_level == 4)
        )
        self._log_quiet_attempt(
            success=True,
            reward_dispensed=getattr(self.dispenser, 'last_dispense_confirmed', 0)
            if self.dispenser else 0,
            dispense_id=getattr(self.dispenser, 'last_wimz_dispense_id', None)
            if self.dispenser else None)

    def _log_quiet_attempt(self, success: bool, reward_dispensed: int,
                           dispense_id: Optional[str]):
        """Spec training_attempt for the SG quiet loop — cue (escalation audio)
        -> response (quiet achieved) -> reward. trick_label='quiet' per spec §4."""
        try:
            now_ms = int(time.time() * 1000)
            cue_ts_ms = int(self.intervention_start_time * 1000) \
                if self.intervention_start_time else now_ms
            wimz_dog = self.wimz.get_or_create_dog(
                legacy_id=self.intervention_dog_id
                if self.intervention_dog_id and self.intervention_dog_id != 'unknown'
                else None,
                name=self.intervention_dog_name)
            self.wimz.log_training_attempt(
                self.wimz_session_id,
                trick_label='quiet',
                dog_id=wimz_dog,
                cue_ts_ms=cue_ts_ms,
                cue_event_id=self._last_cue_event_id,
                cue_type='voice',
                detected_response='quiet',
                response_ts_ms=now_ms,
                latency_ms=now_ms - cue_ts_ms,
                success=1 if success else 0,
                reward_dispensed=reward_dispensed,
                dispense_id=dispense_id)
        except Exception as e:
            logger.debug(f"wimz quiet attempt failed: {e}")

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

    def _play_audio(self, filename: str, wait: bool = True):
        """Play audio file from talks directory with speaker echo suppression.

        Uses C3.2 voice resolution: intervention_dog_id > ArUco > session > select_dog > default

        Suppresses bark detection during playback + 2s buffer to prevent
        the robot's own speaker output from being classified as a bark.

        Args:
            filename: Audio filename (relative to talks dir or absolute)
            wait: If True, wait for audio to finish before returning
        """
        try:
            if filename.startswith('/'):
                full_path = filename
            else:
                from services.media.voice_lookup import resolve_voice_file
                command_id = filename.replace('.mp3', '').replace('.wav', '')
                full_path = resolve_voice_file(command_id, dog_id_override=self.intervention_dog_id)

            if full_path and os.path.exists(full_path):
                if self.audio:
                    # Suppress bark detection during playback + buffer
                    # This prevents the mic from picking up the speaker output
                    try:
                        bark_svc = get_bark_detector_service()
                        bark_svc.suppress_detection(3.0)  # Short audio (~2s) + echo buffer
                    except Exception:
                        pass  # Don't let suppression failure block audio

                    logger.info(f"SG_AUDIO: Playing {filename} from {full_path}")
                    self.audio.play_file(full_path)
                    if wait:
                        # Wait for audio to finish (max 5s timeout)
                        self.audio.wait_for_completion(timeout=5.0)
                        # Extend suppression for 2s after audio completes
                        # to catch any mic echo/reverb
                        try:
                            bark_svc = get_bark_detector_service()
                            bark_svc.suppress_detection(2.0)
                        except Exception:
                            pass
                    logger.info(f"SG_AUDIO: Playback complete: {filename}")
                else:
                    logger.error(f"SG_AUDIO: No audio service available for {filename}")
            else:
                logger.error(f"SG_AUDIO: File not found: {full_path}")

        except Exception as e:
            logger.error(f"SG_AUDIO: Playback error for {filename}: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get current status including escalation telemetry for app display"""
        now = time.time()

        # Calculate time until treat eligibility
        if self.last_treat_time > 0:
            time_since_treat = now - self.last_treat_time
            treat_cooldown_remaining = max(0, self.treat_eligibility_cooldown - time_since_treat)
        else:
            treat_cooldown_remaining = 0

        # Quiet timer: seconds since last bark/intervention (for app display)
        quiet_timer = now - self.quiet_since if self.quiet_since > 0 else 0.0

        max_level = self.config.get('escalation', {}).get('max_level', 3)

        return {
            'running': self.running,
            'fsm_state': self.fsm_state.value if self.fsm_state else None,
            'session_id': self.session_id,
            'session_duration': now - self.session_start_time if self.session_start_time else 0,
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
            'consecutive_interventions': self.consecutive_interventions,
            # Escalation telemetry for app display
            # e.g. "Guardian Level 2/3 — Quiet for 7 min"
            'sg_level': self.current_escalation_level,
            'sg_max': max_level,
            'quiet_timer': round(quiet_timer),
            'total_escalation_resets': self.total_escalation_resets,
            # Bark-type + panic telemetry
            'bark_types': self._bark_type_counts(),
            'panic_active': self.panic_active,
            'panic_episodes': self.panic_episodes_this_session,
            'post_panic_cooldown_remaining': max(
                0, round(self.post_panic_cooldown_until - now)),
            'summary_sent': self.summary_sent,
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

    print("\n=== SILENT GUARDIAN MODE TEST (4-Level Escalation) ===")
    print("Level 1: [name] + quiet → 20s quiet → reward")
    print("Level 2: quiet, quiet, no, come, quiet → 30s quiet → reward")
    print("Level 3: quiet + MOVEMENT SEQUENCE (3x fwd/back/left/right) → 30s quiet → reward")
    print("Level 4: quiet + CALMING MUSIC → 2x 20s quiet → reward")
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
