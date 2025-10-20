#!/usr/bin/env python3
"""
RewardSystem - Unified treat dispensing and reward logic
Consolidates reward logic from multiple implementations
"""

import time
import threading
import logging
from typing import Dict, Any, Optional
from ..utils.event_bus import EventBus

class RewardSystem:
    """
    Unified reward system that manages treat dispensing
    Handles cooldowns, behavior validation, and feedback
    """

    def __init__(self, behavior_config: Dict[str, Any], audio_controller,
                 led_controller, servo_controller, event_bus: EventBus):
        self.logger = logging.getLogger('RewardSystem')
        self.behavior_config = behavior_config
        self.audio = audio_controller
        self.leds = led_controller
        self.servos = servo_controller
        self.event_bus = event_bus

        # Reward tracking
        self.last_treat_times = {}  # Per-behavior cooldowns
        self.total_treats_dispensed = 0
        self.session_treats = 0

        # Configuration
        self.treat_cooldown = behavior_config.get('treat_cooldown_seconds', 30)
        self.max_treats_per_session = behavior_config.get('max_treats_per_session', 10)
        self.reward_behaviors = behavior_config.get('reward_behaviors', {})

        # State
        self.dispensing_in_progress = False

        self.logger.info(f"Reward system initialized - cooldown: {self.treat_cooldown}s, max treats: {self.max_treats_per_session}")

    def process_behavior(self, behavior: str, confidence: float) -> bool:
        """
        Process detected behavior and potentially dispense reward

        Args:
            behavior: Detected behavior name
            confidence: Detection confidence

        Returns:
            True if reward was dispensed, False otherwise
        """
        if self.dispensing_in_progress:
            return False

        # Check if behavior is eligible for reward
        if not self._is_behavior_eligible(behavior, confidence):
            return False

        # Check cooldowns
        if not self._check_cooldowns(behavior):
            return False

        # Check session limits
        if not self._check_session_limits():
            return False

        # Dispense reward
        return self._dispense_reward(behavior, confidence)

    def _is_behavior_eligible(self, behavior: str, confidence: float) -> bool:
        """Check if behavior meets reward criteria"""
        if behavior not in self.reward_behaviors:
            return False

        behavior_config = self.reward_behaviors[behavior]
        required_confidence = behavior_config.get('confidence_required', 0.8)

        return confidence >= required_confidence

    def _check_cooldowns(self, behavior: str) -> bool:
        """Check if enough time has passed since last treat for this behavior"""
        current_time = time.time()

        # Global cooldown
        if hasattr(self, 'last_treat_time') and (current_time - self.last_treat_time) < self.treat_cooldown:
            return False

        # Per-behavior cooldown
        if behavior in self.last_treat_times:
            if (current_time - self.last_treat_times[behavior]) < self.treat_cooldown:
                return False

        return True

    def _check_session_limits(self) -> bool:
        """Check if session treat limit has been reached"""
        return self.session_treats < self.max_treats_per_session

    def _dispense_reward(self, behavior: str, confidence: float) -> bool:
        """Actually dispense the treat with full feedback"""
        try:
            self.dispensing_in_progress = True
            current_time = time.time()

            self.logger.info(f"🎉 Dispensing treat for behavior: {behavior} (confidence: {confidence:.2f})")

            # Update tracking
            self.last_treat_time = current_time
            self.last_treat_times[behavior] = current_time
            self.total_treats_dispensed += 1
            self.session_treats += 1

            # Start feedback sequence in separate thread
            feedback_thread = threading.Thread(
                target=self._execute_reward_sequence,
                args=(behavior, confidence),
                daemon=True
            )
            feedback_thread.start()

            # Emit reward event
            self.event_bus.publish('reward_given', {
                'behavior': behavior,
                'confidence': confidence,
                'total_treats': self.total_treats_dispensed,
                'session_treats': self.session_treats,
                'timestamp': current_time
            })

            return True

        except Exception as e:
            self.logger.error(f"Reward dispensing failed: {e}")
            self.dispensing_in_progress = False
            return False

    def _execute_reward_sequence(self, behavior: str, confidence: float):
        """Execute the complete reward sequence with timing"""
        try:
            # Step 1: Audio feedback
            if self.audio and hasattr(self.audio, 'play_sound'):
                self.audio.play_sound("good_dog")

            # Step 2: Visual feedback
            if self.leds and hasattr(self.leds, 'set_mode'):
                from ..hardware.led_controller import LEDMode
                self.leds.set_mode(LEDMode.TREAT_LAUNCHING)

            # Step 3: Mechanical treat dispense
            time.sleep(0.5)  # Wait for audio/visual feedback
            if self.servos and hasattr(self.servos, 'rotate_winch'):
                self.servos.rotate_winch(direction='forward', duration=0.5)

            # Step 4: Return to normal state
            time.sleep(2.0)
            if self.leds and hasattr(self.leds, 'set_mode'):
                from ..hardware.led_controller import LEDMode
                self.leds.set_mode(LEDMode.IDLE)

            self.logger.info(f"Reward sequence completed for {behavior}")

        except Exception as e:
            self.logger.error(f"Reward sequence error: {e}")

        finally:
            self.dispensing_in_progress = False

    def manual_dispense(self, reason: str = "manual") -> bool:
        """Manually dispense a treat (for testing/training)"""
        if self.dispensing_in_progress:
            self.logger.warning("Cannot dispense - already in progress")
            return False

        self.logger.info(f"Manual treat dispense: {reason}")
        return self._dispense_reward(reason, 1.0)

    def reset_session(self):
        """Reset session counters"""
        old_session_treats = self.session_treats
        self.session_treats = 0
        self.last_treat_times.clear()

        self.logger.info(f"Session reset - previous session: {old_session_treats} treats")

        self.event_bus.publish('session_reset', {
            'previous_treats': old_session_treats,
            'timestamp': time.time()
        })

    def get_statistics(self) -> Dict[str, Any]:
        """Get reward system statistics"""
        current_time = time.time()

        # Calculate time since last treat
        time_since_last = None
        if hasattr(self, 'last_treat_time'):
            time_since_last = current_time - self.last_treat_time

        # Per-behavior cooldown status
        behavior_cooldowns = {}
        for behavior, last_time in self.last_treat_times.items():
            remaining = max(0, self.treat_cooldown - (current_time - last_time))
            behavior_cooldowns[behavior] = {
                'last_treat': last_time,
                'cooldown_remaining': remaining,
                'ready': remaining == 0
            }

        return {
            'total_treats_dispensed': self.total_treats_dispensed,
            'session_treats': self.session_treats,
            'max_treats_per_session': self.max_treats_per_session,
            'treats_remaining': self.max_treats_per_session - self.session_treats,
            'global_cooldown_seconds': self.treat_cooldown,
            'time_since_last_treat': time_since_last,
            'dispensing_in_progress': self.dispensing_in_progress,
            'behavior_cooldowns': behavior_cooldowns,
            'eligible_behaviors': list(self.reward_behaviors.keys())
        }

    def is_ready_for_treat(self, behavior: str = None) -> bool:
        """Check if system is ready to dispense a treat"""
        if self.dispensing_in_progress:
            return False

        if not self._check_session_limits():
            return False

        if behavior:
            return self._check_cooldowns(behavior)
        else:
            # Check global cooldown
            current_time = time.time()
            if hasattr(self, 'last_treat_time'):
                return (current_time - self.last_treat_time) >= self.treat_cooldown
            return True

    def get_next_available_treat_time(self, behavior: str = None) -> Optional[float]:
        """Get timestamp when next treat will be available"""
        if not behavior:
            if hasattr(self, 'last_treat_time'):
                return self.last_treat_time + self.treat_cooldown
            return time.time()

        if behavior in self.last_treat_times:
            return self.last_treat_times[behavior] + self.treat_cooldown

        return time.time()  # Available now