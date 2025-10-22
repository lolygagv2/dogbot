#!/usr/bin/env python3
"""
Reward decision engine
Implements reward policies, cooldowns, and variable ratio schedules
"""

import time
import random
import threading
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from core.bus import get_bus, publish_reward_event
from core.state import get_state
from core.store import get_store
from orchestrators.sequence_engine import get_sequence_engine


@dataclass
class RewardPolicy:
    """Reward policy for a specific behavior"""
    behavior: str
    min_duration: float = 10.0           # Minimum behavior duration (seconds)
    require_quiet: bool = True            # Require no motion/barking
    cooldown: float = 20.0               # Cooldown between rewards (seconds)
    treat_probability: float = 0.6        # Probability of dispensing treat
    max_daily_rewards: int = 10          # Max rewards per day for this behavior
    sounds: List[str] = field(default_factory=lambda: ["good_dog"])
    led_pattern: str = "celebration"
    sequence_name: str = "celebrate"


class RewardLogic:
    """
    Reward decision engine
    Manages reward policies, cooldowns, and variable ratio schedules
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.store = get_store()
        self.sequence_engine = get_sequence_engine()
        self.logger = logging.getLogger('RewardLogic')

        # Default policies
        self.policies = {
            'sit': RewardPolicy(
                behavior='sit',
                min_duration=10.0,
                require_quiet=True,
                cooldown=20.0,
                treat_probability=0.6,
                max_daily_rewards=8,
                sounds=['good_dog', 'excellent'],
                led_pattern='celebration',
                sequence_name='celebrate'
            ),
            'down': RewardPolicy(
                behavior='down',
                min_duration=15.0,
                require_quiet=True,
                cooldown=30.0,
                treat_probability=0.5,
                max_daily_rewards=6,
                sounds=['excellent', 'well_done'],
                led_pattern='pulse_green',
                sequence_name='celebrate'
            ),
            'stay': RewardPolicy(
                behavior='stay',
                min_duration=20.0,
                require_quiet=True,
                cooldown=45.0,
                treat_probability=0.7,
                max_daily_rewards=5,
                sounds=['great_job'],
                led_pattern='rainbow',
                sequence_name='celebrate'
            ),
            'spin': RewardPolicy(
                behavior='spin',
                min_duration=1.0,
                require_quiet=False,
                cooldown=60.0,
                treat_probability=0.8,
                max_daily_rewards=3,
                sounds=['excellent'],
                led_pattern='spinning_dot',
                sequence_name='celebrate'
            )
        }

        # Behavior tracking
        self.behavior_states = {}      # dog_id -> behavior_state
        self.last_rewards = {}         # dog_id -> last_reward_time
        self.daily_reward_counts = {}  # dog_id -> {behavior -> count}

        # Thread safety
        self._lock = threading.RLock()

        # Subscribe to vision events
        self.bus.subscribe('vision', self._on_vision_event)

        # Behavior detection state
        self.detection_start_times = {}  # dog_id -> {behavior -> start_time}
        self.stable_behaviors = {}       # dog_id -> {behavior -> is_stable}

    def _on_vision_event(self, event) -> None:
        """Handle vision events"""
        if event.subtype == 'behavior_detected':
            self._process_behavior_detection(event.data)

    def _process_behavior_detection(self, data: Dict[str, Any]) -> None:
        """Process behavior detection and evaluate for rewards"""
        dog_id = data.get('dog_id', 'unknown')
        behavior = data.get('behavior', '')
        confidence = data.get('confidence', 0.0)
        timestamp = data.get('timestamp', time.time())

        if behavior not in self.policies:
            return  # No policy for this behavior

        policy = self.policies[behavior]

        with self._lock:
            # Initialize tracking for this dog
            if dog_id not in self.detection_start_times:
                self.detection_start_times[dog_id] = {}
                self.stable_behaviors[dog_id] = {}
                self.daily_reward_counts[dog_id] = {}

            dog_start_times = self.detection_start_times[dog_id]
            dog_stable = self.stable_behaviors[dog_id]
            dog_rewards = self.daily_reward_counts[dog_id]

            # Check if behavior just started
            if behavior not in dog_start_times:
                # Behavior started
                dog_start_times[behavior] = timestamp
                dog_stable[behavior] = False
                self.logger.debug(f"Behavior started: {dog_id} {behavior}")
                return

            # Calculate behavior duration
            duration = timestamp - dog_start_times[behavior]

            # Check if behavior is stable (meets minimum duration and confidence)
            required_confidence = 0.7  # Minimum confidence for stable behavior
            if duration >= policy.min_duration and confidence >= required_confidence:
                if not dog_stable[behavior]:
                    # Behavior just became stable
                    dog_stable[behavior] = True
                    self.logger.info(f"Stable behavior detected: {dog_id} {behavior} ({duration:.1f}s, conf: {confidence:.2f})")

                    # Evaluate for reward
                    self._evaluate_reward(dog_id, behavior, confidence, duration, policy)

            elif confidence < required_confidence:
                # Behavior became unstable, reset
                if behavior in dog_start_times:
                    del dog_start_times[behavior]
                if behavior in dog_stable:
                    del dog_stable[behavior]

    def _evaluate_reward(self, dog_id: str, behavior: str, confidence: float,
                        duration: float, policy: RewardPolicy) -> None:
        """Evaluate whether to give a reward"""

        # Check cooldown
        if not self._check_cooldown(dog_id, policy.cooldown):
            self.logger.debug(f"Reward blocked by cooldown: {dog_id}")
            return

        # Check daily limit
        if not self._check_daily_limit(dog_id, behavior, policy.max_daily_rewards):
            self.logger.debug(f"Reward blocked by daily limit: {dog_id} {behavior}")
            return

        # Check quiet requirement (placeholder - would need motion/audio detection)
        if policy.require_quiet and not self._check_quiet_requirement():
            self.logger.debug(f"Reward blocked by noise: {dog_id}")
            return

        # Variable ratio reward (probability-based)
        if random.random() > policy.treat_probability:
            self.logger.info(f"Reward denied by probability: {dog_id} {behavior} (p={policy.treat_probability})")

            # Log non-reward to store
            self.store.log_reward(
                dog_id=dog_id,
                behavior=behavior,
                confidence=confidence,
                success=False,
                treats_dispensed=0,
                mission_name=self.state.mission.name
            )
            return

        # Grant reward!
        self._grant_reward(dog_id, behavior, confidence, duration, policy)

    def _check_cooldown(self, dog_id: str, cooldown: float) -> bool:
        """Check if dog is off cooldown"""
        now = time.time()

        if dog_id in self.last_rewards:
            time_since_reward = now - self.last_rewards[dog_id]
            return time_since_reward >= cooldown

        return True

    def _check_daily_limit(self, dog_id: str, behavior: str, max_daily: int) -> bool:
        """Check if daily limit is reached"""
        if dog_id not in self.daily_reward_counts:
            return True

        dog_rewards = self.daily_reward_counts[dog_id]
        current_count = dog_rewards.get(behavior, 0)

        return current_count < max_daily

    def _check_quiet_requirement(self) -> bool:
        """Check if environment is quiet (placeholder)"""
        # This would integrate with motion detection and audio analysis
        # For now, always return True
        return True

    def _grant_reward(self, dog_id: str, behavior: str, confidence: float,
                     duration: float, policy: RewardPolicy) -> None:
        """Grant a reward to the dog"""
        now = time.time()

        # Update tracking
        self.last_rewards[dog_id] = now

        if dog_id not in self.daily_reward_counts:
            self.daily_reward_counts[dog_id] = {}

        dog_rewards = self.daily_reward_counts[dog_id]
        dog_rewards[behavior] = dog_rewards.get(behavior, 0) + 1

        # Choose random sound
        sound = random.choice(policy.sounds)

        # Execute reward sequence
        sequence_context = {
            'dog_id': dog_id,
            'behavior': behavior,
            'confidence': confidence,
            'duration': duration,
            'sound': sound,
            'led_pattern': policy.led_pattern
        }

        sequence_id = self.sequence_engine.execute_sequence(
            policy.sequence_name,
            sequence_context,
            interrupt=True
        )

        # Publish reward event
        publish_reward_event('reward_granted', {
            'dog_id': dog_id,
            'behavior': behavior,
            'confidence': confidence,
            'duration': duration,
            'sequence_id': sequence_id,
            'sound': sound,
            'policy': policy.behavior,
            'daily_count': dog_rewards[behavior],
            'timestamp': now
        }, 'reward_logic')

        self.logger.info(f"🎉 REWARD GRANTED: {dog_id} for {behavior} (conf: {confidence:.2f}, dur: {duration:.1f}s)")

    def force_reward(self, dog_id: str, behavior: str = "manual",
                    confidence: float = 1.0) -> bool:
        """Force a reward (bypass policies)"""
        policy = self.policies.get(behavior, self.policies['sit'])

        sequence_context = {
            'dog_id': dog_id,
            'behavior': behavior,
            'confidence': confidence,
            'sound': random.choice(policy.sounds)
        }

        sequence_id = self.sequence_engine.execute_sequence(
            'celebrate',
            sequence_context,
            interrupt=True
        )

        self.logger.info(f"🎉 FORCED REWARD: {dog_id} for {behavior}")
        return sequence_id is not None

    def update_policy(self, behavior: str, **kwargs) -> bool:
        """Update reward policy for a behavior"""
        if behavior not in self.policies:
            return False

        policy = self.policies[behavior]

        for key, value in kwargs.items():
            if hasattr(policy, key):
                setattr(policy, key, value)
                self.logger.info(f"Updated policy {behavior}.{key} = {value}")

        return True

    def reset_daily_counters(self) -> None:
        """Reset daily reward counters"""
        with self._lock:
            self.daily_reward_counts.clear()
            self.logger.info("Daily reward counters reset")

    def get_dog_reward_stats(self, dog_id: str) -> Dict[str, Any]:
        """Get reward statistics for a dog"""
        with self._lock:
            now = time.time()

            stats = {
                'dog_id': dog_id,
                'last_reward': self.last_rewards.get(dog_id, 0),
                'time_since_reward': now - self.last_rewards.get(dog_id, 0) if dog_id in self.last_rewards else 999,
                'daily_rewards': self.daily_reward_counts.get(dog_id, {}).copy(),
                'total_daily_rewards': sum(self.daily_reward_counts.get(dog_id, {}).values()),
                'active_behaviors': []
            }

            # Check for active behaviors
            if dog_id in self.detection_start_times:
                for behavior, start_time in self.detection_start_times[dog_id].items():
                    duration = now - start_time
                    stable = self.stable_behaviors[dog_id].get(behavior, False)

                    stats['active_behaviors'].append({
                        'behavior': behavior,
                        'duration': duration,
                        'stable': stable
                    })

            return stats

    def get_policies(self) -> Dict[str, Dict[str, Any]]:
        """Get all reward policies"""
        return {
            behavior: {
                'min_duration': policy.min_duration,
                'cooldown': policy.cooldown,
                'treat_probability': policy.treat_probability,
                'max_daily_rewards': policy.max_daily_rewards,
                'sounds': policy.sounds,
                'led_pattern': policy.led_pattern
            }
            for behavior, policy in self.policies.items()
        }

    def get_status(self) -> Dict[str, Any]:
        """Get reward logic status"""
        with self._lock:
            return {
                'policies_loaded': len(self.policies),
                'tracked_dogs': len(self.detection_start_times),
                'active_behaviors': sum(len(behaviors) for behaviors in self.detection_start_times.values()),
                'total_daily_rewards': sum(sum(rewards.values()) for rewards in self.daily_reward_counts.values()),
                'dogs_with_rewards': len([dog for dog, rewards in self.daily_reward_counts.items() if sum(rewards.values()) > 0])
            }

    def cleanup(self) -> None:
        """Clean shutdown"""
        # No ongoing operations to stop
        self.logger.info("Reward logic cleaned up")


# Global reward logic instance
_reward_logic_instance = None
_reward_logic_lock = threading.Lock()

def get_reward_logic() -> RewardLogic:
    """Get the global reward logic instance (singleton)"""
    global _reward_logic_instance
    if _reward_logic_instance is None:
        with _reward_logic_lock:
            if _reward_logic_instance is None:
                _reward_logic_instance = RewardLogic()
    return _reward_logic_instance