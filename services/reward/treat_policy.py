"""TreatPolicy — the ONE treat-rate mechanism (relay protocol v2, 2026-11-06).

Replaces the scattered, mostly dead limits that existed before:
  * dispenser.daily_limit=300 and set_daily_limit() (never called)
  * dispenser.min_dispense_interval=0 (never armed)
  * SG session_limits.min_time_between_treats=120 (yaml, SG-only)
  * reward.yaml global_settings / time_modifiers (never loaded)

Inputs come from the relay settings (services/cloud/remote_settings),
key treat_schedule {daily_cap, default_multiplier, max_per_hour,
                    windows: [{start:"HH:MM", end:"HH:MM", multiplier}]}:
  daily_cap          -> dispenser.set_daily_limit (everyone, incl. owner taps)
  default_multiplier -> multiplier outside every window (default 1.0)
  max_per_hour       -> rolling-hour cap on protocol/mission/connector treats
  windows            -> per-window multiplier

Semantics
  * The current window's multiplier m scales the PROTOCOL treat rate:
    cooldown between protocol treats = PROTOCOL_BASE_COOLDOWN_S / m.
    m == 0 blocks protocol/mission/connector treats in that window.
    Outside every window m = 1.0. Windows may wrap midnight.
  * Owner taps (manual / app / cloud_command) are ALWAYS allowed, except
    by the daily cap — owner intent beats the schedule; the cap is the
    safety ceiling for everyone.
  * Midnight: dispenser.reset_daily_counters() (nothing called it before).

The dispenser consults check() in dispense_treat(); that is the single
choke point for all callers (coach, mission, SG, sequences, app, Xbox).
"""
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger('TreatPolicy')

PROTOCOL_BASE_COOLDOWN_S = 120.0

# dispense_treat(reason=...) strings seen in the codebase -> wire reason.
_REASON_MAP = {
    'manual': 'owner', 'app': 'owner', 'cloud_command': 'owner',
    'websocket_command': 'owner', 'xbox': 'owner', 'test': 'owner',
    'manual_pilot': 'owner', 'owner': 'owner',
    'mission': 'mission', 'mission_reward': 'mission',
    'coaching_reward': 'protocol', 'silent_guardian_reward': 'protocol',
    'silent_guardian_unprompted_quiet': 'protocol', 'celebration': 'protocol',
    'sequence': 'protocol', 'bark_reward': 'protocol', 'schedule': 'protocol',
    'protocol': 'protocol',
    'connector': 'connector', 'connector_trick': 'connector', 'agent': 'connector',
}


def classify_reason(reason: Optional[str]) -> str:
    """protocol | owner | mission | connector (unknown -> owner)."""
    r = (reason or '').lower()
    if r in _REASON_MAP:
        return _REASON_MAP[r]
    for prefix, cls in (('connector', 'connector'), ('ifttt', 'connector'),
                        ('webhook', 'connector'), ('mission', 'mission'),
                        ('coach', 'protocol'), ('silent_guardian', 'protocol'),
                        ('sg_', 'protocol'), ('multi', 'mission')):
        if r.startswith(prefix):
            return cls
    return 'owner'


@dataclass
class Decision:
    allowed: bool
    code: str                 # ok | daily_cap | window_blocked | cooldown
    multiplier: float
    cooldown_s: float
    reason_class: str


def _parse_hhmm(s: str) -> int:
    hh, mm = str(s).strip().split(':')
    hh, mm = int(hh), int(mm)
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        raise ValueError(f"bad time {s!r}")
    return hh * 60 + mm


def validate_schedule(windows) -> List[Dict]:
    """Normalise [{start,end,multiplier}]; raises ValueError on bad input."""
    if windows is None:
        return []
    if not isinstance(windows, list):
        raise ValueError("treat_schedule must be a list")
    out = []
    for w in windows:
        if not isinstance(w, dict):
            raise ValueError("treat_schedule entries must be objects")
        start, end = _parse_hhmm(w['start']), _parse_hhmm(w['end'])
        m = float(w.get('multiplier', 1.0))
        if m < 0:
            raise ValueError("multiplier must be >= 0")
        out.append({'start': f"{start // 60:02d}:{start % 60:02d}",
                    'end': f"{end // 60:02d}:{end % 60:02d}",
                    'multiplier': m, '_s': start, '_e': end})
    return out


class TreatPolicy:
    def __init__(self):
        self._lock = threading.RLock()
        self._daily_cap: Optional[int] = None
        self._default_multiplier: float = 1.0
        self._max_per_hour: Optional[int] = None
        self._schedule: List[Dict] = []
        self._last_protocol_time = 0.0
        self._protocol_times: List[float] = []     # rolling hour, non-owner dispenses
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._wake = threading.Event()
        self._dispenser = None

    # ------------------------------------------------------------- wiring

    def _get_dispenser(self):
        if self._dispenser is None:
            try:
                from services.reward.dispenser import get_dispenser_service
                self._dispenser = get_dispenser_service()
            except Exception as e:
                logger.debug(f"dispenser unavailable: {e}")
        return self._dispenser

    # ------------------------------------------------------------- settings

    def set_daily_cap(self, cap: Optional[int]) -> None:
        with self._lock:
            if cap is None:
                self._daily_cap = None
                return
            cap = int(cap)
            if cap < 1:
                raise ValueError("daily_cap must be >= 1")
            self._daily_cap = cap
        d = self._get_dispenser()
        if d is not None:
            try:
                d.set_daily_limit(cap)   # clamps 1..300
            except Exception as e:
                logger.error(f"set_daily_limit failed: {e}")
        logger.info(f"[TREAT-POLICY] daily_cap={cap}")

    def set_default_multiplier(self, m) -> None:
        m = 1.0 if m is None else float(m)
        if m < 0:
            raise ValueError("default_multiplier must be >= 0")
        with self._lock:
            self._default_multiplier = m
        logger.info(f"[TREAT-POLICY] default_multiplier={m}")

    def set_max_per_hour(self, n) -> None:
        with self._lock:
            if n is None:
                self._max_per_hour = None
            else:
                n = int(n)
                if n < 0:
                    raise ValueError("max_per_hour must be >= 0")
                self._max_per_hour = n
        logger.info(f"[TREAT-POLICY] max_per_hour={self._max_per_hour}")

    def set_schedule(self, windows) -> None:
        normalised = validate_schedule(windows)
        with self._lock:
            self._schedule = normalised
        logger.info(f"[TREAT-POLICY] treat_schedule={[(w['start'], w['end'], w['multiplier']) for w in normalised]}")

    def get_settings(self) -> Dict:
        with self._lock:
            return {
                'daily_cap': self._daily_cap,
                'default_multiplier': self._default_multiplier,
                'max_per_hour': self._max_per_hour,
                'windows': [{'start': w['start'], 'end': w['end'],
                             'multiplier': w['multiplier']} for w in self._schedule],
                'current_multiplier': self.current_multiplier(),
                'protocol_cooldown_s': self.protocol_cooldown_s(),
                'protocol_last_hour': len(self._recent_protocol(time.time())),
            }

    # ------------------------------------------------------------- queries

    def current_multiplier(self, now: Optional[datetime] = None) -> float:
        now = now or datetime.now()
        mins = now.hour * 60 + now.minute
        with self._lock:
            for w in self._schedule:
                s, e = w['_s'], w['_e']
                inside = (s <= mins < e) if s <= e else (mins >= s or mins < e)
                if inside:
                    return w['multiplier']
            return self._default_multiplier

    def protocol_cooldown_s(self, default: Optional[float] = None,
                            now: Optional[datetime] = None) -> float:
        base = PROTOCOL_BASE_COOLDOWN_S if default is None else float(default)
        m = self.current_multiplier(now)
        if m <= 0:
            return float('inf')
        return base / m

    def check(self, reason: Optional[str], now: Optional[float] = None,
              treats_today: Optional[int] = None) -> Decision:
        now = now if now is not None else time.time()
        cls = classify_reason(reason)
        dt = datetime.fromtimestamp(now)
        m = self.current_multiplier(dt)
        cooldown = self.protocol_cooldown_s(now=dt)
        with self._lock:
            cap = self._daily_cap
            last = self._last_protocol_time
        if cap is not None and treats_today is not None and treats_today >= cap:
            return Decision(False, 'daily_cap', m, cooldown, cls)
        if cls == 'owner':
            return Decision(True, 'ok', m, cooldown, cls)
        if m <= 0:
            return Decision(False, 'window_blocked', m, cooldown, cls)
        if last and (now - last) < cooldown:
            return Decision(False, 'cooldown', m, cooldown, cls)
        with self._lock:
            mph = self._max_per_hour
        if mph is not None and len(self._recent_protocol(now)) >= mph:
            return Decision(False, 'max_per_hour', m, cooldown, cls)
        return Decision(True, 'ok', m, cooldown, cls)

    def _recent_protocol(self, now: float) -> List[float]:
        with self._lock:
            self._protocol_times = [t for t in self._protocol_times if now - t < 3600]
            return list(self._protocol_times)

    def record_dispense(self, reason: Optional[str], now: Optional[float] = None) -> None:
        if classify_reason(reason) != 'owner':
            now = now if now is not None else time.time()
            with self._lock:
                self._last_protocol_time = now
                self._protocol_times.append(now)

    # ------------------------------------------------------------- midnight

    @staticmethod
    def seconds_until_midnight(now: Optional[datetime] = None) -> float:
        now = now or datetime.now()
        nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return max(1.0, (nxt - now).total_seconds())

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._midnight_loop, daemon=True,
                                            name='TreatPolicyMidnight')
            self._thread.start()
        logger.info("[TREAT-POLICY] midnight reset thread started")

    def stop(self) -> None:
        self._running = False
        self._wake.set()

    def _midnight_loop(self) -> None:
        while self._running:
            self._wake.wait(self.seconds_until_midnight())
            self._wake.clear()
            if not self._running:
                break
            if datetime.now().hour == 0:
                self.reset_daily()

    def reset_daily(self) -> None:
        d = self._get_dispenser()
        if d is not None:
            try:
                d.reset_daily_counters()
            except Exception as e:
                logger.error(f"reset_daily_counters failed: {e}")
        with self._lock:
            self._last_protocol_time = 0.0
            self._protocol_times = []
        logger.info("[TREAT-POLICY] daily counters reset (midnight)")


_instance: Optional[TreatPolicy] = None
_instance_lock = threading.Lock()


def get_treat_policy() -> TreatPolicy:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = TreatPolicy()
        return _instance
