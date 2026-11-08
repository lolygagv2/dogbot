"""TrickRunner — synchronous `run_trick(cue)` for the agent tool (2026-11-06).

Tool takes over (Morgan's decision): enter COACH so pose runs, HOLD the
autonomous coaching engine, play the owner's recorded cue, wait for the
pose, reward once, and restore the prior mode — always, in `finally`.

Why not coaching_engine.set_forced_trick(): that path only consumes the
trick after the engine's own eligibility/cooldown gates, dispenses in its
success state (double reward if the tool also rewarded), has no result
callback, and leaves mode ownership with the engine. This runner reuses the
same primitives the engine uses (resolve_voice_file, behavior_interpreter
check_trick, dispenser.dispense_treat, LED celebration) but owns the loop.

Result codes: success | no_dog | wrong_pose | timeout | busy | cancelled |
invalid_cue | error.
"""
import logging
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger('TrickRunner')

DEFAULT_TIMEOUT_S = 45.0
MAX_TIMEOUT_S = 60.0
POSE_WARMUP_S = 6.0          # camera reconfigure + first detections after IDLE->COACH
ANNOUNCE_WAIT_S = 4.0
CUE_WAIT_S = 5.0
REFUSE_MODES = ('silent_guardian', 'mission', 'night_sentry', 'emergency', 'shutdown')


class TrickRunner:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._dog_seen_at = 0.0
        self._bark_times: list = []
        self._bus = None

    # ------------------------------------------------------------- bus taps

    def _on_vision(self, event):
        if event.subtype in ('dog_detected', 'behavior_detected', 'pose_detected'):
            self._dog_seen_at = time.time()

    def _on_audio(self, event):
        if event.subtype == 'bark_detected':
            self._bark_times.append(time.time())

    # ------------------------------------------------------------- run

    def run(self, args: Dict[str, Any], cancel: Optional[threading.Event] = None) -> Dict[str, Any]:
        cancel = cancel or threading.Event()
        t0 = time.monotonic()
        cue = str(args.get('cue') or args.get('trick') or '').strip().lower()
        dog_id = args.get('dog_id')
        retry = bool(args.get('retry', True))
        force = bool(args.get('force', False))
        try:
            timeout_s = min(float(args.get('timeout_s') or DEFAULT_TIMEOUT_S), MAX_TIMEOUT_S)
        except (TypeError, ValueError):
            timeout_s = DEFAULT_TIMEOUT_S
        deadline = t0 + timeout_s

        from core.behavior_interpreter import get_behavior_interpreter
        from core.state import SystemMode, get_state
        from orchestrators.mode_fsm import get_mode_fsm
        interp = get_behavior_interpreter()
        state = get_state()
        fsm = get_mode_fsm()

        base = {'ok': False, 'cue': cue, 'dog_id': dog_id, 'treat_dispensed': False,
                'attempts': 0, 'behavior_detected': None, 'confidence': None, 'hold_s': None}

        tricks = interp.get_all_tricks()
        if cue not in tricks:
            return {**base, 'result': 'invalid_cue',
                    'error': {'code': 'invalid_cue', 'message': f"cue must be one of {tricks}"},
                    'duration_ms': int((time.monotonic() - t0) * 1000)}

        with self._lock:
            if self._active:
                return {**base, 'result': 'busy',
                        'error': {'code': 'busy', 'message': 'run_trick already active'},
                        'duration_ms': int((time.monotonic() - t0) * 1000)}
            self._active = True

        prev_mode = state.get_mode()
        prev_override = getattr(fsm, 'override_mode', None)
        # core.state exposes is_mode_locked as a @property (calling it raised
        # "'bool' object is not callable" and failed every live run_trick).
        locked = getattr(state, 'is_mode_locked', False)
        if callable(locked):
            locked = locked()
        if (prev_mode.value in REFUSE_MODES or bool(locked)) and not force:
            with self._lock:
                self._active = False
            return {**base, 'result': 'busy', 'prev_mode': prev_mode.value,
                    'error': {'code': 'busy', 'message': f"robot is in {prev_mode.value}; pass force=true"},
                    'duration_ms': int((time.monotonic() - t0) * 1000)}

        engine = None
        audio = dispenser = led = None
        mode_changed = False
        mode_restored = None
        result_code = 'error'
        err_msg = None
        try:
            try:
                from orchestrators.coaching_engine import get_coaching_engine
                engine = get_coaching_engine()
                engine.hold('tool run_trick')
            except Exception as e:
                logger.warning(f"[TRICK] coaching engine hold unavailable: {e}")
            from services.media.usb_audio import get_usb_audio_service
            from services.reward.dispenser import get_dispenser_service
            audio = get_usb_audio_service()
            dispenser = get_dispenser_service()
            try:
                from services.media.led import get_led_service
                led = get_led_service()
            except Exception:
                led = None

            from core.bus import get_bus
            self._bus = get_bus()
            self._dog_seen_at = 0.0
            self._bark_times = []
            self._bus.subscribe('vision', self._on_vision)
            self._bus.subscribe('audio', self._on_audio)

            # 1. Mode: COACH so pose/behavior inference runs
            if prev_mode != SystemMode.COACH:
                if not fsm.set_mode_override(SystemMode.COACH):
                    raise RuntimeError('could not enter COACH mode')
                mode_changed = True
                logger.info(f"[TRICK] {prev_mode.value} -> coach for cue={cue}")
                audio.wait_for_completion(ANNOUNCE_WAIT_S)   # mode announcement
                warm_until = time.monotonic() + POSE_WARMUP_S
                while time.monotonic() < warm_until and not self._dog_seen_at and not cancel.is_set():
                    time.sleep(0.1)
            else:
                # Already coaching: make sure the engine's current session can't
                # race us for the dispenser.
                try:
                    if engine and getattr(engine, 'current_session', None):
                        engine.reset_session_cooldown()
                except Exception:
                    pass

            rules = interp.get_trick_rules(cue) or {}
            required = rules.get('required_behavior', cue)
            alternatives = rules.get('alternative_behaviors') or []
            window = float(rules.get('detection_window_sec', 10))
            min_barks = int(rules.get('min_barks', 1))
            max_barks = int(rules.get('max_barks', 3))

            from services.media.voice_lookup import resolve_voice_file
            attempts = 2 if retry else 1
            seen_behavior = None
            seen_conf = None
            seen_hold = None
            for attempt in range(1, attempts + 1):
                if cancel.is_set():
                    result_code = 'cancelled'
                    break
                base['attempts'] = attempt
                remaining = deadline - time.monotonic()
                if remaining < 4:
                    result_code = 'timeout'
                    break
                path = resolve_voice_file(cue, dog_id_override=dog_id)
                if path:
                    audio.play_file(path)
                    audio.wait_for_completion(CUE_WAIT_S)
                else:
                    logger.warning(f"[TRICK] no cue audio for {cue}")
                interp.reset_tracking()
                if cue != 'speak' and not alternatives:
                    interp.set_target_behavior(required)
                bark_count_at_cue = len(self._bark_times)

                watch_until = time.monotonic() + min(window, max(1.0, deadline - time.monotonic() - 3))
                success = False
                while time.monotonic() < watch_until and not cancel.is_set():
                    if cue == 'speak':
                        barks = len(self._bark_times) - bark_count_at_cue
                        if barks >= min_barks:
                            success = barks <= max_barks or True
                            seen_behavior, seen_conf = 'bark', 1.0
                            break
                    else:
                        chk = interp.check_trick(cue, dog_id)
                        if chk.behavior_detected:
                            seen_behavior = chk.behavior_detected
                            seen_conf = chk.confidence
                            seen_hold = chk.hold_duration
                        if chk.completed:
                            success = True
                            break
                    time.sleep(0.1)

                if success:
                    result_code = 'success'
                    break
                if attempt < attempts and not cancel.is_set():
                    logger.info(f"[TRICK] attempt {attempt} failed for {cue}; retrying")
            else:
                pass

            if result_code == 'success':
                good = resolve_voice_file('good', dog_id_override=dog_id)
                if good:
                    audio.play_file(good)
                if led:
                    try:
                        led.celebration_sequence(3.0)
                    except Exception:
                        pass
                try:
                    dog_name = None
                    treat_ok = dispenser.dispense_treat(
                        dog_id=dog_id, dog_name=dog_name, reason='connector_trick',
                        behavior=seen_behavior or required, confidence=float(seen_conf or 0.0))
                    base['treat_dispensed'] = bool(treat_ok)
                except Exception as e:
                    err_msg = f"dispense failed: {e}"
                    logger.error(f"[TRICK] {err_msg}")
            elif result_code not in ('cancelled', 'timeout'):
                # classify the failure
                if not self._dog_seen_at:
                    result_code = 'no_dog'
                elif seen_behavior:
                    result_code = 'wrong_pose'
                else:
                    result_code = 'timeout'
                no = resolve_voice_file('no', dog_id_override=dog_id)
                if no and not cancel.is_set():
                    audio.play_file(no)

            base.update({'behavior_detected': seen_behavior, 'confidence': seen_conf,
                         'hold_s': seen_hold})
        except Exception as e:
            result_code = 'error'
            err_msg = str(e)
            logger.error(f"[TRICK] error: {e}")
        finally:
            # Relay `trick_attempt` row (2026-11-08) — only when a cue was actually
            # given; busy/invalid_cue/cancelled-before-cue/error produce none.
            # Published BEFORE the mode restore so the coaching engine's
            # coaching_stopped (which closes the coach_session window) sees it.
            if base['attempts'] >= 1 and result_code in ('success', 'no_dog', 'wrong_pose', 'timeout'):
                try:
                    from services.cloud.behavior_events import publish_trick_attempt
                    publish_trick_attempt('run_trick', cue, result_code, confidence=base.get('confidence'),
                                          dog_id=dog_id, attempt=base['attempts'],
                                          behavior=base.get('behavior_detected'))
                except Exception as e:
                    logger.debug(f"[TRICK] trick_attempt publish failed: {e}")
            try:
                interp.set_target_behavior(None)
            except Exception:
                pass
            if self._bus is not None:
                try:
                    self._bus.unsubscribe('vision', self._on_vision)
                    self._bus.unsubscribe('audio', self._on_audio)
                except Exception:
                    pass
            if engine is not None:
                try:
                    engine.release()
                except Exception:
                    pass
            if mode_changed:
                try:
                    if prev_override is not None and prev_override == prev_mode:
                        mode_restored = bool(fsm.set_mode_override(prev_mode))
                    else:
                        fsm.clear_override('run_trick restore')
                        mode_restored = bool(fsm.force_mode(prev_mode, 'run_trick restore'))
                    logger.info(f"[TRICK] restored mode -> {prev_mode.value} ok={mode_restored}")
                except Exception as e:
                    mode_restored = False
                    logger.error(f"[TRICK] mode restore failed: {e}")
            with self._lock:
                self._active = False

        out = {**base, 'ok': result_code == 'success', 'result': result_code,
               'prev_mode': prev_mode.value, 'mode_restored': mode_restored,
               'duration_ms': int((time.monotonic() - t0) * 1000)}
        if err_msg:
            out['error'] = {'code': result_code if result_code != 'success' else 'dispense_failed',
                            'message': err_msg}
        elif result_code != 'success':
            out['error'] = {'code': result_code, 'message': result_code}
        return out


_instance: Optional[TrickRunner] = None
_instance_lock = threading.Lock()


def get_trick_runner() -> TrickRunner:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = TrickRunner()
        return _instance


def register_run_trick(dispatcher) -> None:
    from services.cloud.tool_dispatcher import ToolSpec, MOTION
    dispatcher.register(ToolSpec(
        'run_trick', lambda args, cancel: get_trick_runner().run(args, cancel),
        timeout_s=MAX_TIMEOUT_S + 10.0,
        description='Play the owner cue, wait for the pose, reward; restores prior mode',
        group=MOTION))
