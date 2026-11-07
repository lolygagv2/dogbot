#!/usr/bin/env python3
"""
services/motion/relay_move.py - Relay `move` command: short, bounded, open-loop base moves.

Contract (2026-11-07, Morgan):
  move {action: forward|back(ward)|turn_left|turn_right|stop, amount, unit?}
    forward/back  -> drive at the slowest safe motor speed for `amount` cm (cap 50)
    turn_left/right -> spin for `amount` degrees (cap 90)
    stop          -> motors off now
  Before and during any FORWARD move the camera gimbal is centered. The move
  stops at once if the obstacle sensor reads under 30 cm or a cliff/edge is
  detected. Hard cap 5 s per command. Motors are always off afterwards.
  Reply: {ok, stopped_reason: completed|obstacle|edge|timeout}.
  Nothing here runs while Silent Guardian is mid-intervention (ok:false busy).

Open loop: this platform has no encoders, so cm and degrees are converted to
seconds with the calibration constants below (robot_config `relay_move:` may
override them). No obstacle or cliff sensor is fitted on current units
(hardware_specs: both still "planned"); the checks are provider hooks that
stay inert until a sensor registers via set_obstacle_provider /
set_cliff_provider, and the reply carries `sensors` so the caller can see
what was actually protecting the move.
"""
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger('RelayMove')

ACTIONS = ('forward', 'back', 'turn_left', 'turn_right', 'stop')

# Defaults (overridable via robot_config `relay_move:` section).
DEFAULTS: Dict[str, float] = {
    'speed_pct': 40,            # hardware-safety safe range is 40-70 %; 40 is the slowest
    'cm_per_s': 15.0,           # CALIBRATE: cm/s at speed_pct on the floor
    'deg_per_s': 60.0,          # CALIBRATE: deg/s spinning in place at speed_pct
    'max_cm': 50,
    'max_deg': 90,
    'hard_cap_s': 5.0,
    'obstacle_stop_cm': 30.0,
    'ramp_s': 0.05,             # one command tick of ramp up/down: hardware-safety asks for a gradual
                                # transition and sets 50 ms as the minimum command interval; anything
                                # longer starves the short (0.3-0.5 s) steps the relay sends
    'tick_s': 0.05,             # 50 ms command cadence (rate-limit floor, watchdog is 500 ms)
    'recenter_every_s': 1.0,    # re-center gimbal during a forward move
}

# Sensor provider hooks. None => no sensor fitted => check is inert.
_obstacle_provider: Optional[Callable[[], Optional[float]]] = None   # -> distance cm or None
_cliff_provider: Optional[Callable[[], Optional[bool]]] = None       # -> True if edge/cliff
_bump_provider: Optional[Callable[[], Optional[bool]]] = None        # -> True if bumper pressed
_warned_no_sensors = False

# Relay bd3290a wire shape for the reply (2026-11-07): result
# {amount, unit, blocked: bool, reason}; a reflex stop is ok:false blocked:true
# reason cliff|bump|too_close. Internal stopped_reason -> (ok, blocked, reason).
REPLY_MAP = {
    'completed': (True, False, 'completed'),
    'timeout':   (True, False, 'timeout'),      # move happened, capped at 5 s — not a reflex
    'obstacle':  (False, True, 'too_close'),
    'edge':      (False, True, 'cliff'),
    'bump':      (False, True, 'bump'),
    'busy':      (False, False, 'busy'),
}


def set_obstacle_provider(fn: Optional[Callable[[], Optional[float]]]) -> None:
    global _obstacle_provider
    _obstacle_provider = fn


def set_cliff_provider(fn: Optional[Callable[[], Optional[bool]]]) -> None:
    global _cliff_provider
    _cliff_provider = fn


def set_bump_provider(fn: Optional[Callable[[], Optional[bool]]]) -> None:
    global _bump_provider
    _bump_provider = fn


def bump_detected() -> Optional[bool]:
    if _bump_provider is None:
        return None
    try:
        v = _bump_provider()
        return None if v is None else bool(v)
    except Exception as e:
        logger.warning(f"bump provider failed: {e}")
        return None


def read_obstacle_cm() -> Optional[float]:
    if _obstacle_provider is None:
        return None
    try:
        v = _obstacle_provider()
        return None if v is None else float(v)
    except Exception as e:
        logger.warning(f"obstacle provider failed: {e}")
        return None


def cliff_detected() -> Optional[bool]:
    if _cliff_provider is None:
        return None
    try:
        v = _cliff_provider()
        return None if v is None else bool(v)
    except Exception as e:
        logger.warning(f"cliff provider failed: {e}")
        return None


def load_config() -> Dict[str, float]:
    cfg = dict(DEFAULTS)
    try:
        from config.config_loader import get_config
        raw = getattr(get_config(), '_config', None) or {}
        over = raw.get('relay_move') or {}
        for k in DEFAULTS:
            if k in over:
                cfg[k] = type(DEFAULTS[k])(over[k])
    except Exception:
        pass
    return cfg


def sg_intervention_active() -> bool:
    """True while Silent Guardian is mid-intervention. Never constructs SG."""
    try:
        import modes.silent_guardian as sg
        inst = getattr(sg, '_silent_guardian_instance', None)
        if inst is None:
            return False
        return getattr(inst, 'fsm_state', None) == sg.SGState.INTERVENTION
    except Exception:
        return False


def _wheels(action: str, s: int):
    return {'forward': (s, s), 'back': (-s, -s),
            'turn_left': (-s, s), 'turn_right': (s, -s)}[action]


class RelayMove:
    """One bounded move at a time. Inject bus/pantilt/clock for tests."""

    def __init__(self, motor_bus=None, pantilt=None, config: Optional[Dict[str, float]] = None,
                 sg_busy: Callable[[], bool] = sg_intervention_active,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self._bus = motor_bus
        self._pantilt = pantilt
        self.cfg = config or load_config()
        self._sg_busy = sg_busy
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()

    # --- lazy hardware
    def _get_bus(self):
        if self._bus is None:
            from core.motor_command_bus import get_motor_bus
            self._bus = get_motor_bus()
        return self._bus

    def _get_pantilt(self):
        if self._pantilt is None:
            try:
                from services.motion.pan_tilt import get_pantilt_service
                self._pantilt = get_pantilt_service()
            except Exception as e:
                logger.warning(f"pan/tilt unavailable: {e}")
        return self._pantilt

    def _send(self, left: int, right: int) -> bool:
        from core.motor_command_bus import create_motor_command, CommandSource
        bus = self._get_bus()
        if bus is None or not getattr(bus, 'running', False):
            return False
        return bool(bus.send_command(create_motor_command(int(left), int(right), CommandSource.RELAY)))

    def _motors_off(self) -> None:
        try:
            self._send(0, 0)
        except Exception as e:
            logger.error(f"motors-off failed: {e}")

    def _center_gimbal(self) -> None:
        pt = self._get_pantilt()
        if pt is None:
            return
        try:
            pt.center_camera('relay_move')
        except Exception as e:
            logger.warning(f"gimbal center failed: {e}")

    def _sensors_summary(self) -> Dict[str, Any]:
        return {'obstacle': _obstacle_provider is not None, 'cliff': _cliff_provider is not None,
                'bump': _bump_provider is not None}

    # --- public
    def execute(self, action: str, amount: Any, cancel: Optional[threading.Event] = None,
                unit: Optional[str] = None) -> Dict[str, Any]:
        global _warned_no_sensors
        c = self.cfg
        action = {'backward': 'back', 'left': 'turn_left', 'right': 'turn_right'}.get(action, action)
        if action not in ACTIONS:
            return {'ok': False, 'error': {'code': 'bad_action', 'message': f"action must be one of {ACTIONS}"}}
        if self._sg_busy():
            return {'ok': False, 'error': {'code': 'busy', 'message': 'silent guardian intervention in progress'}}

        bus = self._get_bus()
        if bus is None or not getattr(bus, 'running', False):
            return {'ok': False, 'error': {'code': 'motors_unavailable', 'message': 'motor bus not running'}}

        if action == 'stop':
            self._motors_off()
            return {'ok': True, 'action': 'stop', 'amount': 0, 'unit': None, 'blocked': False,
                    'reason': 'completed', 'stopped_reason': 'completed'}

        try:
            amt = float(amount)
        except (TypeError, ValueError):
            return {'ok': False, 'error': {'code': 'bad_amount', 'message': 'amount must be a number'}}
        if amt <= 0:
            return {'ok': False, 'error': {'code': 'bad_amount', 'message': 'amount must be > 0'}}

        if unit and str(unit).lower() not in ('cm', 'deg', 'degrees'):
            return {'ok': False, 'error': {'code': 'bad_unit', 'message': 'unit must be cm or deg'}}
        if action in ('forward', 'back'):
            amt = min(amt, c['max_cm'])
            planned_s = amt / max(c['cm_per_s'], 0.01)
            unit = 'cm'
        else:
            amt = min(amt, c['max_deg'])
            planned_s = amt / max(c['deg_per_s'], 0.01)
            unit = 'deg'
        duration = min(planned_s, c['hard_cap_s'])
        speed = int(c['speed_pct'])
        guard = action == 'forward'

        if guard and not _warned_no_sensors and _obstacle_provider is None and _cliff_provider is None:
            logger.warning("[MOVE] no obstacle/cliff sensor registered — forward moves are unprotected")
            _warned_no_sensors = True

        if not self._lock.acquire(blocking=False):
            return {'ok': False, 'error': {'code': 'busy', 'message': 'a move is already running'}}

        left_t, right_t = _wheels(action, speed)
        stopped = 'completed'
        t0 = self._clock()
        try:
            pt = self._get_pantilt()
            if guard and pt is not None:
                try:
                    pt.set_manual_camera(True)
                except Exception:
                    pass
                self._center_gimbal()
            last_center = self._clock()

            logger.info(f"[MOVE] {action} {amt:g}{unit} -> {duration:.2f}s @ {speed}% "
                        f"(hard cap {c['hard_cap_s']:.0f}s, sensors={self._sensors_summary()})")
            while True:
                now = self._clock()
                elapsed = now - t0
                if cancel is not None and cancel.is_set():
                    stopped = 'timeout'   # cancelled by a preemptive tool; motors off below
                    break
                if elapsed >= c['hard_cap_s']:
                    stopped = 'timeout'
                    break
                if elapsed >= duration:
                    stopped = 'completed'
                    break
                if self._sg_busy():
                    stopped = 'busy'
                    break
                if guard:
                    d = read_obstacle_cm()
                    if d is not None and d < c['obstacle_stop_cm']:
                        stopped = 'obstacle'
                        break
                    if cliff_detected():
                        stopped = 'edge'
                        break
                    if bump_detected():
                        stopped = 'bump'
                        break
                    if now - last_center >= c['recenter_every_s']:
                        self._center_gimbal()
                        last_center = now

                # gradual ramp up at the start and down at the end
                remaining = duration - elapsed
                ramp = c['ramp_s'] if c['ramp_s'] > 0 else 1e-6
                scale = min(1.0, elapsed / ramp, max(remaining, 0.0) / ramp)
                if not self._send(int(round(left_t * scale)), int(round(right_t * scale))):
                    stopped = 'timeout'
                    logger.error("[MOVE] motor bus rejected command — aborting")
                    break
                self._sleep(c['tick_s'])
        finally:
            self._motors_off()
            if guard and pt is not None:
                try:
                    pt.set_manual_camera(False)
                except Exception:
                    pass
            self._lock.release()

        elapsed = self._clock() - t0
        ok, blocked, reason = REPLY_MAP[stopped]
        out = {'ok': ok, 'action': action, 'amount': amt, 'unit': unit, 'blocked': blocked, 'reason': reason,
               'stopped_reason': stopped, 'duration_ms': int(elapsed * 1000),
               'planned_s': round(duration, 2), 'speed_pct': speed, 'sensors': self._sensors_summary()}
        if stopped == 'busy':
            out['error'] = {'code': 'busy', 'message': 'silent guardian intervention started mid-move'}
        elif blocked:
            out['error'] = {'code': reason, 'message': f"safety reflex stopped the move: {reason}"}
        logger.info(f"[MOVE] {action} done: {stopped} after {elapsed:.2f}s")
        return out


_instance: Optional[RelayMove] = None
_instance_lock = threading.Lock()


def get_relay_move() -> RelayMove:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = RelayMove()
        return _instance
