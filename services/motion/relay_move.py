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

Heading (2026-11-18): no IMU, gyro, compass or encoders on tb3/4/5, so
forward/back are open loop (cm -> seconds via cm_per_s) and TURNS are closed
loop on the camera (services/motion/visual_heading.py): burst, stop, measure
the yaw between two still frames, repeat. The reply's `amount` is the degrees
actually turned (`requested` is what was asked; `measured` false = the camera
could not track and the figure is a timed estimate). Every reply is sent only
after the robot has stopped (`stopped_confirmed`). No obstacle or cliff sensor is fitted on current units
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
    'cm_per_s': 35.0,           # measured 2026-11-07 tb5: 2.0 s @ 40 % -> ~70 cm on floor
    'deg_per_s': 103.0,         # measured 2026-11-07 tb5: 1.02 s @ 40 % -> ~105 deg
    'max_cm': 50,
    'max_deg': 90,
    'hard_cap_s': 5.0,
    'obstacle_stop_cm': 30.0,
    'ramp_s': 0.05,             # one command tick of ramp up/down: hardware-safety asks for a gradual
                                # transition and sets 50 ms as the minimum command interval; anything
                                # longer starves the short (0.3-0.5 s) steps the relay sends
    'tick_s': 0.05,             # 50 ms command cadence (rate-limit floor, watchdog is 500 ms)
    'recenter_every_s': 1.0,    # re-center gimbal during a forward move
    # --- closed-loop turns on the camera (2026-11-18) ---
    'closed_loop_turns': True,
    'turn_tolerance_deg': 4.0,  # stop correcting inside this band
    'max_step_deg': 35.0,       # one burst never aims past this: the two frames must overlap
    'min_pulse_s': 0.15,
    'max_pulse_s': 1.5,
    'turn_budget_s': 8.0,       # whole turn incl. measuring; the relay waits 10 s for command_result
    'stall_deg': 2.0,           # a burst that turned less than this did not break free
    'speed_step_pct': 10,
    'max_speed_pct': 60,        # hardware-safety safe range is 40-70 %
    'slow_rate_deg_s': 45.0,    # slower than this (carpet) -> one more power step
    # --- reply only when the robot has physically stopped ---
    'settle_s': 0.35,           # coast time when the camera cannot confirm
    'settle_max_s': 1.2,        # longest wait for two matching frames
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
    'busy':      (False, False, 'robot_busy'),
    'cancelled': (True, False, 'timeout'),      # stop / standby cut it short; amount says how far it got
    'heading_lost': (True, False, 'heading_lost'),   # turned, but the camera lost track part-way
    'stalled':   (False, False, 'no_rotation'),      # wheels driven, camera saw no rotation
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


ROBOT_CONFIG_PATH = '/home/morgan/dogbot/config/robot_config.yaml'


def load_config(robot_config_path: str = ROBOT_CONFIG_PATH) -> Dict[str, float]:
    """DEFAULTS <- robot_config.yaml `relay_move:` (fleet default)
    <- unit profile `relay_move:` (config/robot_profiles/<unit>.yaml, via get_config())."""
    cfg = dict(DEFAULTS)
    sources = []

    def _apply(over: Any, name: str) -> None:
        if not isinstance(over, dict):
            return
        for k in DEFAULTS:
            if k in over:
                try:
                    cfg[k] = bool(over[k]) if isinstance(DEFAULTS[k], bool) else type(DEFAULTS[k])(over[k])
                except (TypeError, ValueError):
                    logger.warning(f"[MOVE] ignoring bad {name} relay_move.{k}={over[k]!r}")
        sources.append(name)

    try:
        import yaml
        with open(robot_config_path, 'r') as f:
            _apply((yaml.safe_load(f) or {}).get('relay_move'), 'robot_config')
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"[MOVE] robot_config relay_move unreadable: {e}")
    try:
        from config.config_loader import get_config
        raw = getattr(get_config(), '_config', None) or {}
        if raw.get('relay_move'):
            _apply(raw.get('relay_move'), 'unit_profile')
    except Exception:
        pass
    logger.info(f"[MOVE] config from {sources or ['code defaults']}: "
                f"speed={cfg['speed_pct']}% cm_per_s={cfg['cm_per_s']} deg_per_s={cfg['deg_per_s']}")
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


class _Drive:
    __slots__ = ('reason', 'elapsed', 't_off_wall')

    def __init__(self, reason: str, elapsed: float, t_off_wall: float):
        self.reason, self.elapsed, self.t_off_wall = reason, elapsed, t_off_wall


class RelayMove:
    """One bounded move at a time. Inject bus/pantilt/heading/clock for tests.

    heading: a VisualHeading (camera yaw measurement) or None. With it, turns
    are closed-loop and every move replies only after the picture has stopped
    moving; without it turns are timed (open loop) and the reply waits a fixed
    settle time.
    """

    def __init__(self, motor_bus=None, pantilt=None, config: Optional[Dict[str, float]] = None,
                 sg_busy: Callable[[], bool] = sg_intervention_active,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 heading=None, wall: Callable[[], float] = time.time):
        self._bus = motor_bus
        self._pantilt = pantilt
        self.cfg = config or load_config()
        for k, v in DEFAULTS.items():          # configs written before 2026-11-18 lack the new keys
            self.cfg.setdefault(k, v)
        self._sg_busy = sg_busy
        self._clock = clock
        self._sleep = sleep
        self._wall = wall
        self._heading = heading
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

    # --- one timed burst of wheel power; motors are ALWAYS off when it returns
    def _drive(self, action: str, speed: int, duration: float, cancel: Optional[threading.Event],
               guard: bool, cap_s: float) -> _Drive:
        c = self.cfg
        left_t, right_t = _wheels(action, speed)
        stopped = 'completed'
        t0 = self._clock()
        last_center = t0
        try:
            while True:
                now = self._clock()
                elapsed = now - t0
                if cancel is not None and cancel.is_set():
                    stopped = 'cancelled'
                    break
                if elapsed >= cap_s:
                    stopped = 'timeout'
                    break
                if elapsed >= duration:
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
        return _Drive(stopped, self._clock() - t0, self._wall())

    def _settle(self, t_off_wall: float):
        """Wait until the robot has physically stopped. Camera when available
        (two consecutive frames agree), else a fixed coast time.
        Returns (snapshot or None, confirmed_by_camera: bool)."""
        c = self.cfg
        h = self._heading
        if h is not None:
            try:
                snap, still = h.wait_still(after=t_off_wall + 0.05, timeout_s=c['settle_max_s'])
                if snap is not None:
                    return snap, bool(still)
            except Exception as e:
                logger.warning(f"[MOVE] camera settle check failed: {e}")
        self._sleep(c['settle_s'])
        return None, False

    # --- turns
    def _turn(self, action: str, target: float, cancel: Optional[threading.Event],
              budget_s: Optional[float] = None, drive_cap_s: Optional[float] = None,
              on_pulse: Optional[Callable[[Dict[str, Any]], bool]] = None,
              max_step_deg: Optional[float] = None) -> Dict[str, Any]:
        """Rotate `target` degrees. Closed loop on the camera: a short burst,
        stop, measure the yaw between two still frames, repeat until within
        tolerance. Each burst is sized from the rate the previous one actually
        achieved, so the floor surface calibrates itself inside the command
        (carpet ~10-25 deg/s at 40 % vs ~100 deg/s on hard floor). Power is
        raised a step at a time, inside the safe range, when the robot does not
        break free. Falls back to one timed burst when the camera cannot
        measure (no frame, dark, blank wall).

        Returns {turned, measured, stopped, pulses, speed_pct, deg_per_s, drive_s, confirmed}.
        """
        c = self.cfg
        h = self._heading if c['closed_loop_turns'] else None
        budget_s = c['turn_budget_s'] if budget_s is None else budget_s
        drive_cap_s = c['hard_cap_s'] if drive_cap_s is None else drive_cap_s
        sign = 1.0 if action == 'turn_right' else -1.0
        other = 'turn_left' if action == 'turn_right' else 'turn_right'
        t0 = self._clock()
        speed = int(c['speed_pct'])
        rate = float(c['deg_per_s'])

        key = None
        if h is not None:
            try:
                key = h.grab(newer_than=self._wall() - 0.05, timeout_s=0.8)
            except Exception as e:
                logger.warning(f"[MOVE] heading reference failed: {e}")
            if key is not None and not key.usable:
                key = None

        if key is None:
            # Open loop: no heading source available right now.
            dur = min(target / max(rate, 0.01), drive_cap_s)
            logger.info(f"[MOVE] {action} {target:g}deg OPEN LOOP -> {dur:.2f}s @ {speed}% "
                        f"({'camera could not measure' if h is not None else 'no heading source'})")
            r = self._drive(action, speed, dur, cancel, False, drive_cap_s)
            _snap, confirmed = self._settle(r.t_off_wall)
            return {'turned': rate * r.elapsed, 'measured': False, 'stopped': r.reason, 'pulses': 1,
                    'speed_pct': speed, 'deg_per_s': rate, 'drive_s': r.elapsed, 'confirmed': confirmed}

        turned = 0.0
        drive_used = 0.0
        pulses = stalls = 0
        last_pulse = None
        stopped = 'completed'
        measured = True
        confirmed = True
        logger.info(f"[MOVE] {action} {target:g}deg closed loop (camera), start {speed}% rate {rate:.0f}deg/s")
        while True:
            remaining = target - turned
            if abs(remaining) <= c['turn_tolerance_deg']:
                break
            wall_left = budget_s - (self._clock() - t0)
            drive_left = drive_cap_s - drive_used
            if wall_left < c['settle_max_s'] + c['min_pulse_s'] or drive_left < c['min_pulse_s']:
                stopped = 'timeout'
                break
            forward = remaining > 0
            step = min(abs(remaining), max_step_deg or c['max_step_deg'])
            pulse = step / max(rate, 5.0)
            if last_pulse is not None:
                pulse = min(pulse, 2.0 * last_pulse)        # never more than double: the rate is not linear
            pulse = max(c['min_pulse_s'], min(pulse, c['max_pulse_s'], drive_left,
                                              wall_left - c['settle_max_s']))
            r = self._drive(action if forward else other, speed, pulse, cancel, False, pulse + 0.2)
            drive_used += r.elapsed
            pulses += 1
            cur, still = self._settle(r.t_off_wall)
            confirmed = bool(still)
            y = None
            if cur is not None:
                try:
                    y = h.yaw_between(key, cur)
                except Exception as e:
                    logger.warning(f"[MOVE] yaw measurement failed: {e}")
            if y is None:
                # Lost the picture: count this burst at the last known rate and stop.
                turned += (1.0 if forward else -1.0) * rate * r.elapsed
                measured = False
                stopped = r.reason if r.reason != 'completed' else 'heading_lost'
                break
            delta = y['yaw_deg'] * sign                       # progress in the requested direction
            turned += delta
            key = cur
            moved = delta if forward else -delta              # progress in this burst's direction
            logger.info(f"[MOVE]   burst {pulses}: {r.elapsed:.2f}s @ {speed}% -> {moved:+.1f}deg "
                        f"(total {turned:.1f}/{target:g}, {y['inliers']} inliers)")
            info = {'turned': turned, 'pulse_s': r.elapsed, 'moved': moved, 'speed_pct': speed, 'snapshot': cur}
            if r.reason != 'completed':
                stopped = r.reason
                break
            if on_pulse is not None and on_pulse(info):
                break
            last_pulse = r.elapsed
            if moved < c['stall_deg']:
                stalls += 1
                if speed >= c['max_speed_pct'] and stalls >= 3:
                    stopped = 'stalled'
                    break
                speed = int(min(speed + c['speed_step_pct'], c['max_speed_pct']))
                rate = max(rate / 2.0, 5.0)                   # next burst longer as well as stronger
            else:
                stalls = 0
                rate = max(5.0, min(400.0, moved / max(r.elapsed, 0.05)))
                if rate < c['slow_rate_deg_s'] and speed < c['max_speed_pct']:
                    speed = int(min(speed + c['speed_step_pct'], c['max_speed_pct']))
        return {'turned': turned, 'measured': measured, 'stopped': stopped, 'pulses': pulses,
                'speed_pct': speed, 'deg_per_s': rate, 'drive_s': drive_used, 'confirmed': confirmed}

    # --- public
    def execute(self, action: str, amount: Any, cancel: Optional[threading.Event] = None,
                unit: Optional[str] = None) -> Dict[str, Any]:
        global _warned_no_sensors
        c = self.cfg
        action = {'backward': 'back', 'left': 'turn_left', 'right': 'turn_right'}.get(action, action)
        if action not in ACTIONS:
            return {'ok': False, 'error': {'code': 'bad_action', 'message': f"action must be one of {ACTIONS}"}}
        if self._sg_busy():
            return {'ok': False, 'error': {'code': 'robot_busy',
                                           'message': 'silent guardian intervention in progress'}}

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
        turning = action in ('turn_left', 'turn_right')
        if turning:
            amt = min(amt, c['max_deg'])
            planned_s = amt / max(c['deg_per_s'], 0.01)
            unit = 'deg'
        else:
            amt = min(amt, c['max_cm'])
            planned_s = amt / max(c['cm_per_s'], 0.01)
            unit = 'cm'
        duration = min(planned_s, c['hard_cap_s'])
        speed = int(c['speed_pct'])
        guard = action == 'forward'

        if guard and not _warned_no_sensors and _obstacle_provider is None and _cliff_provider is None:
            logger.warning("[MOVE] no obstacle/cliff sensor registered — forward moves are unprotected")
            _warned_no_sensors = True

        if not self._lock.acquire(blocking=False):
            return {'ok': False, 'error': {'code': 'robot_busy', 'message': 'a move is already running'}}

        t0 = self._clock()
        extra: Dict[str, Any] = {}
        pt = None
        try:
            if turning:
                res = self._turn(action, amt, cancel)
                stopped = res['stopped']
                confirmed = res['confirmed']
                speed = res['speed_pct']
                # The reply carries what the robot actually did, not what was asked.
                extra = {'requested': amt, 'turned_deg': round(res['turned'], 1),
                         'measured': bool(res['measured']),
                         'heading_source': 'camera' if res['measured'] else 'none',
                         'bursts': res['pulses'], 'drive_s': round(res['drive_s'], 2),
                         'deg_per_s': round(res['deg_per_s'], 1)}
                if stopped in ('completed', 'timeout', 'heading_lost') and res['measured'] \
                        and abs(res['turned']) < c['stall_deg']:
                    stopped = 'stalled'
                amt = round(res['turned'], 1)
            else:
                pt = self._get_pantilt()
                if guard and pt is not None:
                    try:
                        pt.set_manual_camera(True)
                    except Exception:
                        pass
                    self._center_gimbal()
                logger.info(f"[MOVE] {action} {amt:g}{unit} -> {duration:.2f}s @ {speed}% "
                            f"(hard cap {c['hard_cap_s']:.0f}s, sensors={self._sensors_summary()})")
                r = self._drive(action, speed, duration, cancel, guard, c['hard_cap_s'])
                stopped = r.reason
                _snap, confirmed = self._settle(r.t_off_wall)
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
               'planned_s': round(duration, 2), 'speed_pct': speed, 'sensors': self._sensors_summary(),
               # True: two consecutive camera frames agreed after motors-off.
               # False: replied after the fixed coast time (camera could not confirm).
               'stopped_confirmed': bool(confirmed)}
        out.update(extra)
        if stopped == 'busy':
            out['error'] = {'code': 'robot_busy', 'message': 'silent guardian intervention started mid-move'}
        elif stopped == 'stalled':
            out['error'] = {'code': 'no_rotation',
                            'message': f"wheels were driven up to {speed}% but the camera saw "
                                       f"{amt:g} deg of rotation"}
        elif blocked:
            out['error'] = {'code': reason, 'message': f"safety reflex stopped the move: {reason}"}
        logger.info(f"[MOVE] {action} done: {stopped} after {elapsed:.2f}s"
                    + (f", turned {amt:g} of {extra['requested']:g} deg "
                       f"({'measured' if extra['measured'] else 'estimated'})" if turning else ''))
        return out

    # --- calibration
    def calibrate_turn(self, cancel: Optional[threading.Event] = None, surface: str = 'default',
                       direction: str = 'turn_right') -> Dict[str, Any]:
        """Measure the camera yaw scale with a full spin: turn in measured
        bursts until the starting view comes back; whatever the camera summed
        over that loop is exactly 360 degrees. Saves yaw_scale (lens model
        trim) and, per surface, the rate the wheels achieved, to
        state/turn_calibration.json. Needs ~1 m of clear floor all round."""
        from services.motion import visual_heading as vh
        h = self._heading
        if h is None:
            return {'ok': False, 'error': {'code': 'unavailable', 'message': 'no camera heading source'}}
        if self._sg_busy():
            return {'ok': False, 'error': {'code': 'robot_busy', 'message': 'silent guardian intervention in progress'}}
        if not self._lock.acquire(blocking=False):
            return {'ok': False, 'error': {'code': 'robot_busy', 'message': 'a move is already running'}}
        try:
            ref = h.grab(newer_than=self._wall() - 0.05, timeout_s=1.0)
            if ref is None or not ref.usable:
                return {'ok': False, 'error': {'code': 'unavailable',
                                               'message': 'camera view has too little detail to track'}}
            sign = 1.0 if direction == 'turn_right' else -1.0
            old_scale = h.scale
            box: Dict[str, Any] = {'rates': []}

            def on_pulse(info: Dict[str, Any]) -> bool:
                if info['moved'] > self.cfg['stall_deg']:
                    box['rates'].append((info['speed_pct'], info['moved'] / max(info['pulse_s'], 0.05)))
                raw = info['turned'] / old_scale
                if raw < 200.0:                      # start view cannot be back yet
                    return False
                y = h.yaw_between(info['snapshot'], ref)
                if y is None or y['inliers'] < 60:
                    return False
                box['raw_full'] = raw + (y['yaw_deg'] * sign) / old_scale
                box['closure_inliers'] = y['inliers']
                return True

            # 25-degree aims: an uncalibrated camera may under-read, and the frames must still overlap
            res = self._turn(direction, 620.0, cancel, budget_s=75.0, drive_cap_s=40.0, on_pulse=on_pulse,
                             max_step_deg=25.0)
        finally:
            self._motors_off()
            self._lock.release()
        if 'raw_full' not in box:
            return {'ok': False, 'turned_uncalibrated_deg': round(res['turned'], 1), 'bursts': res['pulses'],
                    'error': {'code': 'no_closure',
                              'message': f"start view never came back ({res['stopped']}); nothing saved"}}
        scale = 360.0 / box['raw_full']
        if not 0.5 <= scale <= 2.0:
            return {'ok': False, 'error': {'code': 'implausible', 'message': f"yaw scale {scale:.2f} rejected"}}
        cal = vh.load_calibration()
        cal['yaw_scale'] = round(scale, 4)
        cal['hfov_deg'] = h.hfov_deg
        cal['calibrated_at'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
        surf = {'speed_pct': res['speed_pct'], 'bursts': res['pulses']}
        if box['rates']:
            spd = box['rates'][-1][0]
            same = [r for s, r in box['rates'] if s == spd]
            surf['deg_per_s'] = round(float(sum(same) / len(same)) * scale / old_scale, 1)
            surf['speed_pct'] = spd
        cal.setdefault('surfaces', {})[surface] = surf
        saved = vh.save_calibration(cal)
        h.scale = scale
        h.calibrated = True
        logger.info(f"[MOVE] turn calibration: camera summed {box['raw_full']:.1f} raw deg over one full spin "
                    f"-> yaw_scale {scale:.3f} (was {old_scale:.3f}); surface {surface!r}: {surf}")
        return {'ok': True, 'yaw_scale': round(scale, 4), 'previous_yaw_scale': round(old_scale, 4),
                'raw_degrees_per_revolution': round(box['raw_full'], 1), 'surface': surface, **surf,
                'closure_inliers': box['closure_inliers'], 'saved': saved}


_instance: Optional[RelayMove] = None
_instance_lock = threading.Lock()


def get_relay_move() -> RelayMove:
    global _instance
    with _instance_lock:
        if _instance is None:
            heading = None
            try:
                from services.motion.visual_heading import get_visual_heading
                heading = get_visual_heading()
            except Exception as e:
                logger.warning(f"[MOVE] camera heading unavailable, turns are open loop: {e}")
            _instance = RelayMove(heading=heading)
        return _instance
