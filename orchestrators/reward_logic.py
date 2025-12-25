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


@dataclass
class RewardDecision:
    """Result of reward decision logic"""
    should_dispense: bool
    reason: str
    dog_id: Optional[str] = None
    behavior: Optional[str] = None
    confidence: Optional[float] = None


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

        # Default policies - require_quiet now mission-context aware
        self.policies = {
            'sit': RewardPolicy(
                behavior='sit',
                min_duration=10.0,
                require_quiet=False,  # Made flexible - check mission context
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
                require_quiet=False,  # Made flexible - check mission context
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
                require_quiet=False,  # Made flexible - check mission context
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

        # Subscribe to vision and audio events
        self.bus.subscribe('vision', self._on_vision_event)
        self.bus.subscribe('audio', self._on_audio_event)

        # Behavior detection state
        self.detection_start_times = {}  # dog_id -> {behavior -> start_time}
        self.stable_behaviors = {}       # dog_id -> {behavior -> is_stable}

    def _on_vision_event(self, event) -> None:
        """Handle vision events"""
        if event.subtype == 'behavior_detected':
            self._process_behavior_detection(event.data)

    def _on_audio_event(self, event) -> None:
        """Handle audio events for bark-based rewards"""
        if event.subtype == 'bark_detected':
            self._process_bark_detection(event.data)

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

    def _process_bark_detection(self, data: Dict[str, Any]) -> None:
        """Process bark detection and evaluate for bark-based rewards"""
        emotion = data.get('emotion', '')
        confidence = data.get('confidence', 0.0)
        timestamp = data.get('timestamp', time.time())

        # Get dog from bark event (vision-audio fusion)
        dog_id = data.get('dog_id') or 'unknown'
        dog_name = data.get('dog_name') or 'unknown'

        self.logger.info(f"Bark detected: {dog_name} ({dog_id}) - {emotion} (conf: {confidence:.2f})")

        # Define bark reward policy
        bark_policy = RewardPolicy(
            behavior=f'bark_{emotion}',
            min_duration=0.0,  # Immediate bark reward
            require_quiet=False,  # Bark rewards don't require quiet!
            cooldown=5.0,  # 5-second cooldown between bark rewards
            treat_probability=0.4,  # Lower probability than behavior rewards
            max_daily_rewards=3,  # Limit bark-only rewards
            sounds=['good_dog'],
            led_pattern='pulse_blue',
            sequence_name='celebrate'
        )

        # Check if this emotion should trigger reward
        reward_emotions = ['alert', 'attention']  # From config
        if emotion in reward_emotions and confidence >= 0.55:
            self._evaluate_bark_reward(dog_id, emotion, confidence, bark_policy)

    def _evaluate_bark_reward(self, dog_id: str, emotion: str, confidence: float,
                             policy: RewardPolicy) -> None:
        """Evaluate whether to give a bark-based reward"""

        # Use same cooldown/limit checking but for bark rewards
        if not self._check_cooldown(dog_id, policy.cooldown):
            self.logger.debug(f"Bark reward blocked by cooldown: {dog_id}")
            return

        # Check daily limit for bark rewards specifically
        behavior_name = f'bark_{emotion}'
        if not self._check_daily_limit(dog_id, behavior_name, policy.max_daily_rewards):
            self.logger.debug(f"Bark reward blocked by daily limit: {dog_id} {behavior_name}")
            return

        # Bark rewards don't check mission quiet requirement - they're independent

        # Variable ratio reward
        if random.random() > policy.treat_probability:
            self.logger.info(f"Bark reward denied by probability: {dog_id} {behavior_name} (p={policy.treat_probability})")
            return

        # Grant bark reward!
        self.logger.info(f"Granting bark reward: {dog_id} {behavior_name}")
        self._grant_reward(dog_id, behavior_name, confidence, 0.0, policy)

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

        # Check mission-specific quiet requirement
        if not self._check_mission_quiet_requirement(behavior):
            self.logger.debug(f"Reward blocked by mission noise policy: {dog_id} {behavior}")
            return

        # Variable ratio reward (probability-based)
        if random.random() > policy.treat_probability:
            self.logger.info(f"Reward denied by probability: {dog_id} {behavior} (p={policy.treat_probability})")

            # Log non-reward to store
            mission_name = getattr(getattr(self.state, 'mission', None), 'name', 'unknown')
            self.store.log_reward(
                dog_id=dog_id,
                behavior=behavior,
                confidence=confidence,
                success=False,
                treats_dispensed=0,
                mission_name=mission_name
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

    def _check_mission_quiet_requirement(self, behavior: str) -> bool:
        """Check if mission allows rewards during barking/noise for this behavior"""
        try:
            # Get current mission from state
            if not hasattr(self.state, 'mission') or not self.state.mission:
                # No active mission - allow all rewards (flexible default)
                return True

            mission = self.state.mission

            # Check if mission has specific noise policy
            if hasattr(mission, 'config') and mission.config:
                config = mission.config

                # Check for behavior-specific quiet requirements
                quiet_behaviors = config.get('require_quiet_behaviors', [])
                if behavior in quiet_behaviors:
                    # This mission requires quiet for this specific behavior
                    return not self._is_environment_noisy()

                # Check for mission-wide quiet policy
                if config.get('require_quiet_always', False):
                    return not self._is_environment_noisy()

                # Check for bark-friendly missions
                if config.get('allow_bark_rewards', True):
                    # Mission explicitly allows bark + behavior rewards
                    return True

            # Default: allow rewards (flexible for bark + behavior combinations)
            return True

        except Exception as e:
            self.logger.warning(f"Error checking mission quiet requirement: {e}")
            # On error, be permissive
            return True

    def _is_environment_noisy(self) -> bool:
        """Check if environment currently has noise/barking"""
        # Check for recent bark detection events
        # This could be enhanced to check motion/audio sensors
        try:
            # For now, check if bark detector is actively classifying
            bark_state = self.state.get_service_state('bark_detector')
            if bark_state and bark_state.get('is_processing_bark', False):
                return True
            return False
        except:
            # If can't determine, assume quiet
            return False

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

        # Log successful reward to store
        mission_name = getattr(getattr(self.state, 'mission', None), 'name', 'unknown')
        self.store.log_reward(
            dog_id=dog_id,
            behavior=behavior,
            confidence=confidence,
            success=True,
            treats_dispensed=1,
            mission_name=mission_name
        )

        self.logger.info(f"🎉 REWARD GRANTED: {dog_id} for {behavior} (conf: {confidence:.2f}, dur: {duration:.1f}s)")

    def should_dispense(self, dog_id: str, behavior: str, confidence: float) -> RewardDecision:
        """
        Evaluate if a reward should be dispensed
        Public API method for checking reward eligibility
        """
        # Map common behavior names
        behavior_map = {
            'sitting': 'sit',
            'lying_down': 'down',
            'standing': 'stay'
        }
        behavior = behavior_map.get(behavior, behavior)

        # Check if we have a policy for this behavior
        if behavior not in self.policies:
            return RewardDecision(
                should_dispense=False,
                reason=f"No reward policy for behavior: {behavior}",
                dog_id=dog_id,
                behavior=behavior,
                confidence=confidence
            )

        policy = self.policies[behavior]

        # Check confidence threshold
        if confidence < 0.7:
            return RewardDecision(
                should_dispense=False,
                reason=f"Confidence too low: {confidence:.2f} < 0.7",
                dog_id=dog_id,
                behavior=behavior,
                confidence=confidence
            )

        # Check cooldown
        if not self._check_cooldown(dog_id, policy.cooldown):
            time_remaining = policy.cooldown - (time.time() - self.last_rewards.get(dog_id, 0))
            return RewardDecision(
                should_dispense=False,
                reason=f"Cooldown active: {time_remaining:.1f}s remaining",
                dog_id=dog_id,
                behavior=behavior,
                confidence=confidence
            )

        # Check daily limit
        if not self._check_daily_limit(dog_id, behavior, policy.max_daily_rewards):
            return RewardDecision(
                should_dispense=False,
                reason=f"Daily limit reached for {behavior}",
                dog_id=dog_id,
                behavior=behavior,
                confidence=confidence
            )

        # Check mission-specific quiet requirement
        if not self._check_mission_quiet_requirement(behavior):
            return RewardDecision(
                should_dispense=False,
                reason="Dog not quiet per mission policy (barking or motion detected)",
                dog_id=dog_id,
                behavior=behavior,
                confidence=confidence
            )

        # Variable ratio schedule
        if random.random() > policy.treat_probability:
            return RewardDecision(
                should_dispense=False,
                reason=f"Variable ratio schedule (probability {policy.treat_probability:.2f})",
                dog_id=dog_id,
                behavior=behavior,
                confidence=confidence
            )

        # All checks passed!
        return RewardDecision(
            should_dispense=True,
            reason=f"Reward approved for {behavior}",
            dog_id=dog_id,
            behavior=behavior,
            confidence=confidence
        )

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

    def evaluate_reward(self, behavior: str, confidence: float, dog_id: str = None) -> bool:
        """
        Consumer-focused reward evaluation for missions

        Args:
            behavior: Behavior detected (e.g., 'sit', 'down')
            confidence: Detection confidence (0.0-1.0)
            dog_id: Dog identifier

        Returns:
            True if reward should be given
        """
        if behavior not in self.policies:
            self.logger.warning(f"No policy for behavior: {behavior}")
            return False

        policy = self.policies[behavior]

        # Check confidence threshold
        if confidence < 0.7:  # Consumer-friendly threshold
            self.logger.info(f"Confidence too low for {behavior}: {confidence}")
            return False

        # Check cooldown
        dog_id = dog_id or 'default_dog'
        last_reward_key = f"{dog_id}_{behavior}"
        current_time = time.time()

        if last_reward_key in self.last_rewards:
            time_since_last = current_time - self.last_rewards[last_reward_key]
            if time_since_last < policy.cooldown:
                self.logger.info(f"Cooldown active for {behavior}: {time_since_last:.1f}s < {policy.cooldown}s")
                return False

        # Check daily limits
        if dog_id not in self.daily_reward_counts:
            self.daily_reward_counts[dog_id] = {}

        today_count = self.daily_reward_counts[dog_id].get(behavior, 0)
        if today_count >= policy.max_daily_rewards:
            self.logger.info(f"Daily limit reached for {behavior}: {today_count}/{policy.max_daily_rewards}")
            return False

        # Probability check (consumer-friendly - always reward good behavior)
        if random.random() > policy.treat_probability:
            self.logger.info(f"Probability check failed for {behavior}")
            return False

        # All checks passed - give reward!
        self.last_rewards[last_reward_key] = current_time
        self.daily_reward_counts[dog_id][behavior] = today_count + 1

        # Trigger reward sequence
        try:
            sequence_engine = get_sequence_engine()
            if sequence_engine:
                sequence_engine.execute_sequence(policy.sequence_name, {
                    'behavior': behavior,
                    'dog_id': dog_id,
                    'confidence': confidence
                })
        except Exception as e:
            self.logger.error(f"Failed to execute reward sequence: {e}")

        # Log successful reward to store
        mission_name = getattr(getattr(self.state, 'mission', None), 'name', 'mission_evaluate')
        self.store.log_reward(
            dog_id=dog_id,
            behavior=behavior,
            confidence=confidence,
            success=True,
            treats_dispensed=1,
            mission_name=mission_name
        )

        self.logger.info(f"✅ Reward approved for {dog_id}: {behavior} (confidence: {confidence:.2f})")
        return True

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