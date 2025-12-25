#!/usr/bin/env python3
"""
Rules Engine for WIM-Z Reward Logic
YAML-driven configuration for behavior rewards, cooldowns, and escalation

Example YAML rule:
```yaml
rules:
  sit:
    type: behavior
    min_duration: 2.0
    min_confidence: 0.7
    cooldown: 30
    daily_limit: 10
    treat_probability: 0.8
    sounds: ["good_dog", "excellent"]
    led_pattern: "celebration"
    sequence: "celebrate"

  excessive_barking:
    type: bark_frequency
    threshold: 3
    window_minutes: 1
    response: "stop_barking"
    escalation:
      after_events: 5
      increase_quiet_duration: true
```
"""

import yaml
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RuleType(Enum):
    """Types of reward rules"""
    BEHAVIOR = "behavior"           # Pose-based (sit, lie, spin, etc.)
    BARK = "bark"                   # Single bark emotion
    BARK_FREQUENCY = "bark_frequency"  # Bark count threshold
    QUIET = "quiet"                 # Silence duration
    COMBO = "combo"                 # Multiple conditions (sit + bark)
    TIME_BASED = "time_based"       # Scheduled rewards


@dataclass
class Rule:
    """Single reward rule definition"""
    name: str
    rule_type: RuleType
    enabled: bool = True

    # Behavior conditions
    min_duration: float = 0.0       # Seconds behavior must be held
    min_confidence: float = 0.5     # Minimum detection confidence

    # Bark conditions
    bark_emotions: List[str] = field(default_factory=list)  # Required emotions
    bark_threshold: int = 1         # Number of barks
    bark_window: int = 60           # Window in seconds

    # Quiet conditions
    quiet_duration: float = 5.0     # Required silence seconds

    # Reward settings
    cooldown: float = 30.0          # Seconds between rewards
    daily_limit: int = 10           # Max rewards per day
    hourly_limit: int = 5           # Max rewards per hour
    treat_probability: float = 0.8  # Chance of treat (0-1)
    treat_count: int = 1            # Treats per reward

    # Feedback
    sounds: List[str] = field(default_factory=lambda: ["good_dog"])
    led_pattern: str = "celebration"
    sequence: str = "celebrate"

    # Escalation
    escalation: Dict[str, Any] = field(default_factory=dict)

    # Per-dog overrides (dog_id -> override dict)
    dog_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RuleMatch:
    """Result of evaluating a rule"""
    matched: bool
    rule: Optional[Rule] = None
    reason: str = ""
    reward_allowed: bool = False
    blocked_by: str = ""  # "cooldown", "daily_limit", etc.
    treat_probability: float = 0.0
    sequence: str = ""
    sounds: List[str] = field(default_factory=list)


class RulesEngine:
    """
    YAML-driven rules engine for reward logic

    Features:
    - Load rules from YAML files
    - Per-behavior reward settings
    - Cooldown and daily limit tracking
    - Per-dog rule overrides
    - Escalation logic
    - Hot-reload of rules
    """

    DEFAULT_RULES_PATH = Path("/home/morgan/dogbot/configs/rules/default_rules.yaml")

    def __init__(self, rules_path: str = None):
        self.rules_path = Path(rules_path) if rules_path else self.DEFAULT_RULES_PATH
        self.rules: Dict[str, Rule] = {}
        self._lock = threading.Lock()

        # Tracking state
        self.last_reward_time: Dict[str, Dict[str, float]] = {}  # dog_id -> rule_name -> timestamp
        self.daily_counts: Dict[str, Dict[str, int]] = {}  # dog_id -> rule_name -> count
        self.hourly_counts: Dict[str, Dict[str, int]] = {}
        self.escalation_counts: Dict[str, Dict[str, int]] = {}  # dog_id -> rule_name -> event count

        self._last_hour = time.localtime().tm_hour
        self._last_day = time.localtime().tm_yday

        # Load rules
        self._load_rules()

        logger.info(f"RulesEngine initialized with {len(self.rules)} rules")

    def _load_rules(self) -> bool:
        """Load rules from YAML file"""
        if not self.rules_path.exists():
            logger.warning(f"Rules file not found: {self.rules_path}")
            self._load_default_rules()
            return False

        try:
            with open(self.rules_path, 'r') as f:
                data = yaml.safe_load(f)

            rules_data = data.get('rules', {})
            self.rules = {}

            for name, rule_data in rules_data.items():
                rule = self._parse_rule(name, rule_data)
                self.rules[name] = rule
                logger.debug(f"Loaded rule: {name} ({rule.rule_type.value})")

            # Load global settings
            self.global_settings = data.get('global', {})

            logger.info(f"Loaded {len(self.rules)} rules from {self.rules_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load rules: {e}")
            self._load_default_rules()
            return False

    def _parse_rule(self, name: str, data: Dict[str, Any]) -> Rule:
        """Parse rule from YAML data"""
        rule_type = RuleType(data.get('type', 'behavior'))

        return Rule(
            name=name,
            rule_type=rule_type,
            enabled=data.get('enabled', True),
            min_duration=data.get('min_duration', 0.0),
            min_confidence=data.get('min_confidence', 0.5),
            bark_emotions=data.get('bark_emotions', []),
            bark_threshold=data.get('threshold', 1),
            bark_window=data.get('window_minutes', 1) * 60,
            quiet_duration=data.get('quiet_duration', 5.0),
            cooldown=data.get('cooldown', 30.0),
            daily_limit=data.get('daily_limit', 10),
            hourly_limit=data.get('hourly_limit', 5),
            treat_probability=data.get('treat_probability', 0.8),
            treat_count=data.get('treat_count', 1),
            sounds=data.get('sounds', ['good_dog']),
            led_pattern=data.get('led_pattern', 'celebration'),
            sequence=data.get('sequence', 'celebrate'),
            escalation=data.get('escalation', {}),
            dog_overrides=data.get('dog_overrides', {})
        )

    def _load_default_rules(self):
        """Load hardcoded default rules"""
        self.rules = {
            'sit': Rule(
                name='sit',
                rule_type=RuleType.BEHAVIOR,
                min_duration=2.0,
                min_confidence=0.7,
                cooldown=30.0,
                daily_limit=10,
                treat_probability=0.8
            ),
            'lie': Rule(
                name='lie',
                rule_type=RuleType.BEHAVIOR,
                min_duration=3.0,
                min_confidence=0.7,
                cooldown=30.0,
                daily_limit=8,
                treat_probability=0.8
            ),
            'spin': Rule(
                name='spin',
                rule_type=RuleType.BEHAVIOR,
                min_duration=0.5,
                min_confidence=0.6,
                cooldown=45.0,
                daily_limit=5,
                treat_probability=0.9
            ),
            'cross': Rule(
                name='cross',
                rule_type=RuleType.BEHAVIOR,
                min_duration=2.0,
                min_confidence=0.7,
                cooldown=60.0,
                daily_limit=5,
                treat_probability=0.85
            ),
            'alert_bark': Rule(
                name='alert_bark',
                rule_type=RuleType.BARK,
                bark_emotions=['alert', 'attention'],
                min_confidence=0.6,
                cooldown=60.0,
                daily_limit=3,
                treat_probability=0.4
            ),
            'excessive_barking': Rule(
                name='excessive_barking',
                rule_type=RuleType.BARK_FREQUENCY,
                bark_threshold=3,
                bark_window=60,
                cooldown=120.0,
                daily_limit=10,
                sequence='stop_barking',
                escalation={
                    'after_events': 5,
                    'increase_quiet_duration': True,
                    'base_quiet': 5,
                    'max_quiet': 20
                }
            ),
            'quiet_reward': Rule(
                name='quiet_reward',
                rule_type=RuleType.QUIET,
                quiet_duration=30.0,
                cooldown=300.0,
                daily_limit=5,
                treat_probability=0.6
            )
        }
        logger.info("Loaded default hardcoded rules")

    def reload_rules(self) -> bool:
        """Hot-reload rules from file"""
        with self._lock:
            return self._load_rules()

    def _check_time_reset(self):
        """Reset hourly/daily counters if needed"""
        current_hour = time.localtime().tm_hour
        current_day = time.localtime().tm_yday

        if current_hour != self._last_hour:
            self.hourly_counts.clear()
            self._last_hour = current_hour

        if current_day != self._last_day:
            self.daily_counts.clear()
            self.escalation_counts.clear()
            self._last_day = current_day

    def evaluate(self, rule_name: str, dog_id: str = 'unknown',
                 confidence: float = 1.0, duration: float = 0.0,
                 context: Dict[str, Any] = None) -> RuleMatch:
        """
        Evaluate a rule for a given event

        Args:
            rule_name: Name of the rule to evaluate (e.g., 'sit', 'alert_bark')
            dog_id: Dog identifier
            confidence: Detection confidence
            duration: How long behavior has been held
            context: Additional context (emotion, loudness, etc.)

        Returns:
            RuleMatch with evaluation result
        """
        with self._lock:
            self._check_time_reset()

            if rule_name not in self.rules:
                return RuleMatch(matched=False, reason=f"Rule not found: {rule_name}")

            rule = self.rules[rule_name]

            if not rule.enabled:
                return RuleMatch(matched=False, rule=rule, reason="Rule disabled")

            # Apply per-dog overrides
            effective_rule = self._apply_dog_overrides(rule, dog_id)

            # Check conditions based on rule type
            if not self._check_conditions(effective_rule, confidence, duration, context):
                return RuleMatch(
                    matched=False,
                    rule=effective_rule,
                    reason="Conditions not met"
                )

            # Check cooldown
            if not self._check_cooldown(effective_rule, dog_id):
                return RuleMatch(
                    matched=True,
                    rule=effective_rule,
                    reward_allowed=False,
                    blocked_by="cooldown",
                    reason=f"Cooldown active ({effective_rule.cooldown}s)"
                )

            # Check hourly limit
            if not self._check_hourly_limit(effective_rule, dog_id):
                return RuleMatch(
                    matched=True,
                    rule=effective_rule,
                    reward_allowed=False,
                    blocked_by="hourly_limit",
                    reason=f"Hourly limit reached ({effective_rule.hourly_limit})"
                )

            # Check daily limit
            if not self._check_daily_limit(effective_rule, dog_id):
                return RuleMatch(
                    matched=True,
                    rule=effective_rule,
                    reward_allowed=False,
                    blocked_by="daily_limit",
                    reason=f"Daily limit reached ({effective_rule.daily_limit})"
                )

            # All checks passed - reward allowed
            return RuleMatch(
                matched=True,
                rule=effective_rule,
                reward_allowed=True,
                treat_probability=effective_rule.treat_probability,
                sequence=effective_rule.sequence,
                sounds=effective_rule.sounds,
                reason="All conditions met"
            )

    def _apply_dog_overrides(self, rule: Rule, dog_id: str) -> Rule:
        """Apply per-dog overrides to a rule"""
        if dog_id not in rule.dog_overrides:
            return rule

        overrides = rule.dog_overrides[dog_id]

        # Create a copy with overrides
        # (In practice, would create a new Rule with merged values)
        # For now, just log and return original
        logger.debug(f"Applying overrides for {dog_id}: {overrides}")
        return rule

    def _check_conditions(self, rule: Rule, confidence: float,
                          duration: float, context: Dict[str, Any]) -> bool:
        """Check if rule conditions are met"""
        context = context or {}

        # Check confidence
        if confidence < rule.min_confidence:
            return False

        # Check duration for behavior rules
        if rule.rule_type == RuleType.BEHAVIOR:
            if duration < rule.min_duration:
                return False

        # Check bark emotion for bark rules
        if rule.rule_type == RuleType.BARK:
            emotion = context.get('emotion', '')
            if rule.bark_emotions and emotion not in rule.bark_emotions:
                return False

        return True

    def _check_cooldown(self, rule: Rule, dog_id: str) -> bool:
        """Check if cooldown has expired"""
        if dog_id not in self.last_reward_time:
            return True

        if rule.name not in self.last_reward_time[dog_id]:
            return True

        elapsed = time.time() - self.last_reward_time[dog_id][rule.name]
        return elapsed >= rule.cooldown

    def _check_hourly_limit(self, rule: Rule, dog_id: str) -> bool:
        """Check if hourly limit is reached"""
        if dog_id not in self.hourly_counts:
            return True

        count = self.hourly_counts[dog_id].get(rule.name, 0)
        return count < rule.hourly_limit

    def _check_daily_limit(self, rule: Rule, dog_id: str) -> bool:
        """Check if daily limit is reached"""
        if dog_id not in self.daily_counts:
            return True

        count = self.daily_counts[dog_id].get(rule.name, 0)
        return count < rule.daily_limit

    def record_reward(self, rule_name: str, dog_id: str = 'unknown'):
        """Record that a reward was given"""
        with self._lock:
            # Update last reward time
            if dog_id not in self.last_reward_time:
                self.last_reward_time[dog_id] = {}
            self.last_reward_time[dog_id][rule_name] = time.time()

            # Update hourly count
            if dog_id not in self.hourly_counts:
                self.hourly_counts[dog_id] = {}
            self.hourly_counts[dog_id][rule_name] = \
                self.hourly_counts[dog_id].get(rule_name, 0) + 1

            # Update daily count
            if dog_id not in self.daily_counts:
                self.daily_counts[dog_id] = {}
            self.daily_counts[dog_id][rule_name] = \
                self.daily_counts[dog_id].get(rule_name, 0) + 1

            logger.debug(f"Recorded reward: {rule_name} for {dog_id}")

    def record_escalation_event(self, rule_name: str, dog_id: str = 'unknown'):
        """Record an escalation event (e.g., bark frequency exceeded)"""
        with self._lock:
            if dog_id not in self.escalation_counts:
                self.escalation_counts[dog_id] = {}
            self.escalation_counts[dog_id][rule_name] = \
                self.escalation_counts[dog_id].get(rule_name, 0) + 1

    def get_escalation_level(self, rule_name: str, dog_id: str = 'unknown') -> int:
        """Get current escalation level for a rule"""
        with self._lock:
            if rule_name not in self.rules:
                return 0

            rule = self.rules[rule_name]
            if not rule.escalation:
                return 0

            after_events = rule.escalation.get('after_events', 5)
            event_count = self.escalation_counts.get(dog_id, {}).get(rule_name, 0)

            if event_count >= after_events:
                return 1 + (event_count - after_events) // after_events
            return 0

    def get_quiet_duration_requirement(self, rule_name: str, dog_id: str = 'unknown') -> float:
        """Get required quiet duration based on escalation"""
        with self._lock:
            if rule_name not in self.rules:
                return 5.0

            rule = self.rules[rule_name]
            escalation = rule.escalation

            if not escalation or not escalation.get('increase_quiet_duration'):
                return rule.quiet_duration

            base = escalation.get('base_quiet', 5)
            max_quiet = escalation.get('max_quiet', 20)
            level = self.get_escalation_level(rule_name, dog_id)

            # Linear increase: 5 -> 10 -> 15 -> 20
            increment = (max_quiet - base) / 3  # 3 levels to max
            required = min(base + (level * increment), max_quiet)

            return required

    def get_rule(self, rule_name: str) -> Optional[Rule]:
        """Get a rule by name"""
        return self.rules.get(rule_name)

    def list_rules(self) -> List[Dict[str, Any]]:
        """List all rules with their current settings"""
        rules_list = []
        for name, rule in self.rules.items():
            rules_list.append({
                'name': name,
                'type': rule.rule_type.value,
                'enabled': rule.enabled,
                'cooldown': rule.cooldown,
                'daily_limit': rule.daily_limit,
                'treat_probability': rule.treat_probability
            })
        return rules_list

    def get_stats(self, dog_id: str = None) -> Dict[str, Any]:
        """Get reward statistics"""
        with self._lock:
            if dog_id:
                return {
                    'dog_id': dog_id,
                    'daily_counts': self.daily_counts.get(dog_id, {}),
                    'hourly_counts': self.hourly_counts.get(dog_id, {}),
                    'escalation_counts': self.escalation_counts.get(dog_id, {})
                }
            else:
                return {
                    'all_daily': dict(self.daily_counts),
                    'all_hourly': dict(self.hourly_counts),
                    'all_escalation': dict(self.escalation_counts)
                }


# Singleton
_rules_engine = None


def get_rules_engine() -> RulesEngine:
    """Get or create rules engine singleton"""
    global _rules_engine
    if _rules_engine is None:
        _rules_engine = RulesEngine()
    return _rules_engine
