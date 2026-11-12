"""ToolDispatcher — relay commands with command_id (CONTRACT v2 FINAL, 2026-11-06).

Commands arrive as the existing WS command frame plus a `command_id`:
    {"type":"command", "command":"run_trick", "command_id":"c-123",
     "cue":"sit", "force":false, "timestamp":...}
and EVERY such command is answered exactly once with
    {"type":"command_result", "command_id", "ok", "result", "error"}
(error = {"code","message"} or null). Unknown command, busy, timeout and
exceptions all produce a command_result.

Registered commands: dispense_treat, set_mode, play_sound{url|sound_id},
start_mission, go_to (not supported on this platform), standby, run_trick
{cue, force?}, take_photo{with_hud}, agent_utterance (record one utterance
and POST it to /agent/utterance).

Rules
  * Handlers run on a worker thread, never on the relay asyncio loop.
  * Single-flight: one command at a time (a second one answers busy). The
    slot is released only when the handler actually returns, so a timed-out
    run_trick can never overlap a new one. `standby` is preemptive.
  * Every call is logged: `[TOOL] ...` in journald + a wimz `tool_call`
    event row (spec v0.8) via log_tool_call().

The same dispatcher serves /ws/local so the AP-mode app has parity.
"""
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger('ToolDispatcher')

Handler = Callable[[Dict[str, Any], threading.Event], Dict[str, Any]]


@dataclass
class ToolSpec:
    name: str
    handler: Handler
    timeout_s: float = 30.0
    preemptive: bool = False
    description: str = ''
    # Exclusivity group. Tools in the same group are single-flight with each
    # other (busy); tools with group=None run concurrently with everything.
    # 2026-11-07: audio playback (play_sound) held the one global slot and
    # take_photo got busy:"command play_sound is running". Only physical
    # motion (move, go_to, run_trick) is exclusive now.
    group: Optional[str] = None


MOTION = 'motion'


def _err(code: str, message: str, **extra) -> Dict[str, Any]:
    out = {'ok': False, 'error': {'code': code, 'message': message}}
    out.update(extra)
    return out


# ------------------------------------------------------------------ logging

def log_tool_call(name: str, args: Dict[str, Any], call_id: Optional[str],
                  duration_ms: int, ok: bool, error: Optional[str],
                  result_summary: Optional[str] = None) -> None:
    """journald line + wimz tool_call row. Never raises."""
    try:
        logger.info(f"[TOOL] name={name} command_id={call_id} ok={ok} duration_ms={duration_ms} "
                    f"args={json.dumps(args, default=str)[:300]} "
                    f"error={error!r}" + (f" result={result_summary[:600]}" if result_summary else ''))
    except Exception:
        pass
    try:
        from core.data.wimz_store import get_wimz_store
        get_wimz_store().log_event(None, 'tool_call', {
            'name': name, 'args': args, 'id': call_id, 'duration_ms': duration_ms,
            'ok': ok, 'error': error}, label_source='human')
    except Exception as e:
        logger.debug(f"tool_call wimz row failed: {e}")


SPEECH_WAIT_MAX_S = 15.0   # longest a treat waits for a relay voice clip to finish


def _relay_speech_active() -> bool:
    """True while a relay voice clip (play_sound url / PTT queue) is playing or queued."""
    try:
        from services.media.push_to_talk import get_push_to_talk_service
        ptt = get_push_to_talk_service()
        return bool(ptt.is_playing() or getattr(ptt, '_play_queue', None))
    except Exception:
        return False


def _wait_for_relay_speech(max_s: float, cancel: Optional[threading.Event] = None) -> float:
    """Block until relay speech has finished (or max_s / cancel). Returns seconds waited."""
    t0 = time.time()
    while _relay_speech_active() and time.time() - t0 < max_s:
        if cancel is not None and cancel.is_set():
            break
        time.sleep(0.1)
    return time.time() - t0


# ------------------------------------------------------------------ dispatcher

class ToolDispatcher:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._lock = threading.RLock()
        self._group_locks: Dict[str, threading.Lock] = {}
        self._running: Dict[int, Dict[str, Any]] = {}   # run_no -> {name, id, group, cancel, started}
        self._run_no = 0
        self._relay = None
        self._attached = False
        self._register_builtins()

    # ------------------------------------------------------------- registry

    def register(self, spec: ToolSpec) -> None:
        with self._lock:
            self._tools[spec.name] = spec
        logger.debug(f"[TOOL] registered {spec.name} (timeout {spec.timeout_s}s)")

    def register_tool(self, name: str, handler: Handler, timeout_s: float = 30.0,
                      preemptive: bool = False, description: str = '',
                      group: Optional[str] = None) -> None:
        self.register(ToolSpec(name, handler, timeout_s, preemptive, description, group))

    def _group_lock(self, group: str) -> threading.Lock:
        with self._lock:
            lk = self._group_locks.get(group)
            if lk is None:
                lk = self._group_locks[group] = threading.Lock()
            return lk

    def running(self, group: Optional[str] = None) -> list:
        """Snapshot of running commands (optionally one group), oldest first."""
        with self._lock:
            rows = [dict(r) for r in self._running.values() if group is None or r['group'] == group]
        for r in rows:
            r.pop('cancel', None)
        return sorted(rows, key=lambda r: r['started'])

    def has_tool(self, name: Optional[str]) -> bool:
        return bool(name) and name in self._tools

    def list_tools(self) -> Dict[str, Any]:
        with self._lock:
            tools = [{'name': s.name, 'timeout_s': s.timeout_s, 'preemptive': s.preemptive,
                      'group': s.group, 'description': s.description} for s in self._tools.values()]
        running = self.running()
        motion = [r for r in running if r['group'] == MOTION]
        return {'tools': sorted(tools, key=lambda t: t['name']),
                'busy': bool(motion),                       # motion slot held
                'current': motion[0] if motion else None,   # wire-compat
                'running': running}

    def attach(self, relay_client=None) -> None:
        """Wire late-bound tools (run_trick) and remember the relay."""
        self._relay = relay_client
        if self._attached:
            return
        self._attached = True
        try:
            from orchestrators.trick_runner import register_run_trick
            register_run_trick(self)
        except Exception as e:
            logger.error(f"[TOOL] run_trick not registered: {e}")

    # ------------------------------------------------------------- dispatch

    def dispatch(self, msg: Dict[str, Any], reply: Callable[[Dict[str, Any]], None]) -> None:
        """Non-blocking: validates, then runs the command on its own thread.

        msg: {"name"|"command": str, "id"|"command_id": str, "args": {...}}
        """
        name = msg.get('name') or msg.get('command') or msg.get('tool')
        call_id = msg.get('id') or msg.get('command_id')
        args = msg.get('args') or msg.get('arguments') or msg.get('params') or {}
        if not isinstance(args, dict):
            args = {'_raw': args}
        threading.Thread(target=self._run, args=(name, call_id, args, reply),
                         daemon=True, name=f"ToolCall-{name}").start()

    def call(self, name: str, args: Optional[Dict[str, Any]] = None,
             call_id: Optional[str] = None, wait_s: Optional[float] = None) -> Dict[str, Any]:
        """Synchronous variant for REST / tests; returns the command_result frame."""
        box: Dict[str, Any] = {}
        done = threading.Event()

        def _reply(frame):
            box.update(frame)
            done.set()

        self._run(name, call_id, args or {}, _reply)
        done.wait(wait_s or 1.0)
        return box

    def cancel_current(self, wait_s: float = 3.0) -> bool:
        """Cancel every running command (preemptive standby) and wait for the
        motion group to drain. Ungrouped commands get the cancel event but
        are not waited on."""
        with self._lock:
            rows = list(self._running.values())
        if not rows:
            return True
        for r in rows:
            r['cancel'].set()
            logger.info(f"[TOOL] cancel requested for {r['name']} command_id={r['id']}")
        groups = {r['group'] for r in rows if r['group']}
        ok = True
        deadline = time.monotonic() + wait_s
        for g in groups:
            lk = self._group_lock(g)
            got = lk.acquire(timeout=max(0.0, deadline - time.monotonic()))
            if got:
                lk.release()
            ok = ok and got
        return ok

    def _run(self, name: str, call_id: Optional[str], args: Dict[str, Any],
             reply: Callable[[Dict[str, Any]], None]) -> None:
        t0 = time.monotonic()
        spec = self._tools.get(name) if name else None
        result: Dict[str, Any]
        err: Optional[str] = None
        if call_id and isinstance(args, dict):
            args.setdefault('_command_id', call_id)
        if spec is None:
            result = _err('unknown_command', f"unknown command {name!r}",
                          available=sorted(self._tools))
            self._finish(name or '?', call_id, args, t0, result, result['error']['message'], reply)
            return

        if spec.preemptive:
            self.cancel_current(wait_s=3.0)

        lock = self._group_lock(spec.group) if spec.group else None
        if lock is not None and not lock.acquire(blocking=False):
            others = self.running(spec.group)
            cur = others[0]['name'] if others else '?'
            result = _err('busy', f"command {cur} is running",
                          result='busy', running=cur, group=spec.group)
            self._finish(name, call_id, args, t0, result, result['error']['message'], reply)
            return

        cancel = threading.Event()
        with self._lock:
            self._run_no += 1
            run_no = self._run_no
            self._running[run_no] = {'name': name, 'id': call_id, 'group': spec.group,
                                     'cancel': cancel, 'started': time.time()}
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"Tool-{name}")
        released = threading.Event()

        def _release_when_done(fut):
            try:
                fut.result()
            except Exception:
                pass
            with self._lock:
                self._running.pop(run_no, None)
            if not released.is_set():
                released.set()
                if lock is not None:
                    lock.release()
            pool.shutdown(wait=False)

        future = pool.submit(spec.handler, args, cancel)
        future.add_done_callback(_release_when_done)
        try:
            out = future.result(timeout=spec.timeout_s)
            if not isinstance(out, dict):
                out = {'ok': True, 'value': out}
            out.setdefault('ok', True)
            result = out
            if not result.get('ok'):
                e = result.get('error')
                err = e.get('message') if isinstance(e, dict) else (str(e) if e else 'failed')
        except FutureTimeout:
            cancel.set()
            result = _err('timeout', f"{name} exceeded {spec.timeout_s:.0f}s", result='timeout')
            err = result['error']['message']
        except Exception as e:
            result = _err('exception', str(e))
            err = str(e)
        self._finish(name, call_id, args, t0, result, err, reply)

    def _finish(self, name, call_id, args, t0, result, err, reply) -> None:
        duration_ms = int((time.monotonic() - t0) * 1000)
        summary = None
        try:
            summary = json.dumps({k: v for k, v in result.items() if k not in ('jpeg_b64', 'data', 'image')},
                                 default=str)
        except Exception:
            pass
        ok = bool(result.get('ok'))
        log_tool_call(name, args, call_id, duration_ms, ok, err, summary)
        error = result.get('error') if not ok else None
        if error is not None and not isinstance(error, dict):
            error = {'code': 'error', 'message': str(error)}
        payload = {k: v for k, v in result.items() if k not in ('ok', 'error')}
        frame = {'type': 'command_result', 'command_id': call_id, 'command': name,
                 'ok': ok, 'result': payload, 'error': error, 'duration_ms': duration_ms}
        try:
            reply(frame)
        except Exception as e:
            logger.error(f"[TOOL] reply failed for {name}: {e}")

    # ------------------------------------------------------------- built-ins

    def _register_builtins(self) -> None:
        self.register(ToolSpec('take_photo', self._tool_take_photo, self.PHOTO_REPLY_BUDGET_S + 2.0,
                               description='JPEG of the current camera frame (any mode); '
                                           'deferred via media upload if capture is slow'))
        self.register(ToolSpec('standby', self._tool_standby, 10.0, preemptive=True,
                               description='Stop audio and return to IDLE'))
        self.register(ToolSpec('dispense_treat', self._tool_dispense_treat, 75.0,   # retry ladder + unjam ~45 s, + speech wait
                               description='Dispense one treat (source app|agent|connector)'))
        self.register(ToolSpec('set_mode', self._tool_set_mode, 10.0,
                               description='Switch mode: idle|silent_guardian|coach|manual|night_sentry'))
        self.register(ToolSpec('play_sound', self._tool_play_sound, 60.0,
                               description='Queue a relay-hosted url or a registry sound_id on the audio queue; '
                                           'replies at once (wait:true to block until played)'))
        self.register(ToolSpec('start_mission', self._tool_start_mission, 10.0,
                               description='Start a mission by id/name'))
        self.register(ToolSpec('schedule_mission', self._tool_schedule_mission, 10.0,
                               description='Persist a timer: {mission|mission_name, dog_id, fire_at ISO|epoch '
                                           '(one-shot) or start_time HH:MM + type once|daily|weekly [+days], '
                                           'mission_id? (relay id), end_time?, grace_s?}; survives restart'))
        self.register(ToolSpec('schedule_sync', self._tool_schedule_sync, 10.0,
                               description='Every scheduled mission with next_fire_at, last_run_at, last_result '
                                           '({dog_id?, enabled_only?})'))
        self.register(ToolSpec('cancel_mission', self._tool_cancel_mission, 15.0,
                               description='{mission_id}: delete that schedule (schedule_id | relay mission_id | '
                                           'mission_name) and stop its run if executing; "active" stops the '
                                           'running mission'))
        self.register(ToolSpec('move', self._tool_move, 8.0,   # 5 s hard cap + ramp/center slack
                               description='Bounded base move: {action: forward|back|turn_left|turn_right|stop, '
                                           'amount cm (cap 50) or deg (cap 90)}; 5 s hard cap, motors off after',
                               group=MOTION))
        self.register(ToolSpec('look', self._tool_look, 6.0,
                               description='One gentle camera step: {action: pan_left|pan_right|tilt_up|tilt_down|'
                                           'center, degrees 5-45 (default 15)} -> {pan, tilt, limit_reached, '
                                           'moved_degrees}; gimbal_busy while the owner steers or a move runs'))
        self.register(ToolSpec('go_to', self._tool_go_to, 5.0,
                               description='Navigate to a named place (not supported yet)', group=MOTION))
        self.register(ToolSpec('listen', self._tool_listen, self.LISTEN_MAX_S + 15.0,
                               description='Record {seconds<=15} from the mic (16 kHz mono WAV), tag with '
                                           'YAMNet, upload WAV; bark detection keeps running'))
        self.register(ToolSpec('agent_utterance', self._tool_agent_utterance, 60.0,
                               description='Record one utterance (WAV <=30 s), POST /agent/utterance, play reply'))
        self.register(ToolSpec('ultrasonic', self._tool_ultrasonic, 20.0,
                               description='Ultrasonic tweeter: {style: before_quiet|before_music|nonstop_barking} '
                                           'or {freq_hz, level_pct, duration_s, pulse_s?, gap_s?, alt_freq_hz?}; '
                                           '{action: status}. Needs ultrasonic_enabled on the robot; '
                                           '>=20 s between bursts (error cooldown + retry_after_s)'))
        self.register(ToolSpec('smart_plug', self._tool_smart_plug, 15.0,
                               description='WiFi smart plug (Adaptil diffuser): {action: on|off|toggle|status} — '
                                           'on/off/toggle is a manual override (the robot then never '
                                           'auto-switches it off); setup: {action: scan} lists Kasa plugs on '
                                           'the robot LAN, {action: configure, device_id, alias?, '
                                           'trigger_level?, auto_off_minutes?}, {action: remove}'))

    # The relay waits 10 s for a command_result. Capture normally takes
    # <1 s; if it has not finished inside this budget we answer at once with
    # ok:true + photo_id (pending:true), finish the capture in the background,
    # upload the JPEG to the relay media store as <photo_id>.jpg and announce
    # it with a WS `photo_ready` {command_id, photo_id, image_url} frame.
    PHOTO_REPLY_BUDGET_S = 8.0

    def _tool_take_photo(self, args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        from core.data.ids import uuid7
        photo_id = uuid7()
        command_id = args.get('_command_id')
        with_hud = bool(args.get('with_hud', args.get('hud', True)))
        max_w = args.get('max_width')
        box: Dict[str, Any] = {}
        done = threading.Event()

        def _capture():
            try:
                box['res'] = self._capture_photo_blocking(with_hud)
            except Exception as e:
                box['res'] = {'success': False, 'error': str(e)}
            finally:
                done.set()

        t_start = time.monotonic()
        threading.Thread(target=_capture, daemon=True, name=f"Photo-{photo_id[:8]}").start()
        if done.wait(self.PHOTO_REPLY_BUDGET_S):
            res = box.get('res') or {'success': False, 'error': 'capture returned no result'}
            if not res.get('success'):
                return _err('capture_failed', str(res.get('error', 'capture failed')))
            out = self._photo_result(res, max_w, with_hud)
            out['photo_id'] = photo_id
            out['media_filename'] = f"{photo_id}.jpg"
            # Also park the JPEG in the relay media store as <photo_id>.jpg so
            # the relay can fetch by photo_id instead of the inline b64
            # (2026-11-07: relay answered no_image_returned to an ok:true
            # result that carried a 147 KB jpeg_b64).
            out['image_url'] = None
            if self._relay is not None and res.get('filepath'):
                try:
                    # Spend only what is left of the reply budget on the upload
                    # so capture + upload together stay inside the relay's wait.
                    remaining = self.PHOTO_REPLY_BUDGET_S - (time.monotonic() - t_start)
                    out['image_url'] = self._relay.upload_media_sync(
                        res['filepath'], 'image/jpeg', filename=f"{photo_id}.jpg",
                        timeout=max(1.0, remaining))
                except Exception as e:
                    logger.warning(f"[TOOL] take_photo media upload failed: {e}")
            return out

        # Slow capture: reply now, deliver later.
        logger.warning(f"[TOOL] take_photo slower than {self.PHOTO_REPLY_BUDGET_S:.0f}s; "
                       f"deferring photo_id={photo_id} command_id={command_id}")

        def _deliver():
            done.wait(60.0)
            res = box.get('res') or {'success': False, 'error': 'capture did not finish'}
            url = None
            if res.get('success') and self._relay is not None and res.get('filepath'):
                url = self._relay.upload_media_sync(res['filepath'], 'image/jpeg',
                                                    filename=f"{photo_id}.jpg")
            # Relay contract (2026-11-07, relay 2422577): photo_ready
            # {type, command_id, photo_id, image_url} is the FINAL result of
            # that command_id; nothing else may follow it. On failure send a
            # command_result error instead (no photo_ready, so it is still
            # the single final answer).
            if url:
                frame = {'type': 'photo_ready', 'command_id': command_id,
                         'photo_id': photo_id, 'image_url': url}
            else:
                err = str(res.get('error') or 'upload failed')
                frame = {'type': 'command_result', 'command_id': command_id, 'command': 'take_photo',
                         'ok': False, 'result': {'photo_id': photo_id},
                         'error': {'code': 'capture_failed' if not res.get('success') else 'upload_failed',
                                   'message': err}}
            if self._relay is not None:
                try:
                    self._relay.send_message(frame)
                except Exception as e:
                    logger.error(f"[TOOL] {frame['type']} send failed: {e}")
            logger.info(f"[TOOL] deferred take_photo {photo_id}: "
                        f"{'photo_ready ' + url if url else 'failed: ' + frame['error']['message']}")

        threading.Thread(target=_deliver, daemon=True, name=f"PhotoDeliver-{photo_id[:8]}").start()
        # Interim reply: result.deferred=true tells the relay's waiters to
        # ignore this and keep waiting for photo_ready.
        return {'ok': True, 'deferred': True, 'pending': True, 'photo_id': photo_id,
                'with_hud': with_hud, 'media_filename': f"{photo_id}.jpg"}

    # Hard cap on the capture itself (frame copy -> JPEG -> SD write). The
    # safe wrapper returns success:false on timeout/exception; it never
    # raises and never touches the picamera2 handle (reads the detector's
    # locked last-frame copy only). Upload time is budgeted separately.
    PHOTO_CAPTURE_TIMEOUT_S = 5.0

    @classmethod
    def _capture_photo_blocking(cls, with_hud: bool) -> Dict[str, Any]:
        from services.media.photo_capture import get_photo_capture_service
        detector = None
        try:
            from services.perception.detector import get_detector_service
            detector = get_detector_service()
        except Exception:
            pass
        return get_photo_capture_service().capture_photo_safe(
            with_hud=with_hud, detector=detector, timeout_s=cls.PHOTO_CAPTURE_TIMEOUT_S)

    @staticmethod
    def _photo_result(res: Dict[str, Any], max_w, with_hud: bool) -> Dict[str, Any]:
        jpeg_b64 = res.get('data')
        width = height = None
        try:
            w, h = str(res.get('resolution', '')).lower().split('x')
            width, height = int(w), int(h)
        except Exception:
            pass
        if max_w and width and width > int(max_w) and jpeg_b64:
            try:
                import base64
                import cv2
                import numpy as np
                buf = np.frombuffer(base64.b64decode(jpeg_b64), np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                scale = int(max_w) / img.shape[1]
                img = cv2.resize(img, (int(max_w), int(img.shape[0] * scale)))
                ok, enc = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    jpeg_b64 = base64.b64encode(enc.tobytes()).decode()
                    height, width = img.shape[0], img.shape[1]
            except Exception as e:
                logger.debug(f"take_photo resize skipped: {e}")
        dog_id = None
        try:
            from core.state import get_state
            st = get_state()
            dog_id = st.get_aruco_dog_within(5) or st.get_current_dog_id()
        except Exception:
            pass
        return {'ok': True, 'jpeg_b64': jpeg_b64, 'filename': res.get('filename'),
                'filepath': res.get('filepath'), 'width': width, 'height': height,
                'size_bytes': (len(jpeg_b64) * 3 // 4) if jpeg_b64 else 0,
                'with_hud': with_hud, 'timestamp': res.get('timestamp'), 'dog_id': dog_id}

    def _tool_standby(self, args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        stopped: Dict[str, Any] = {}
        try:
            from services.media.push_to_talk import get_push_to_talk_service
            stopped['queued_clips'] = get_push_to_talk_service().clear_queue()
        except Exception:
            pass
        try:
            from services.media.usb_audio import get_usb_audio_service
            audio = get_usb_audio_service()
            if audio.is_busy():
                audio.stop()
                stopped['audio'] = True
        except Exception:
            pass
        from core.state import SystemMode, get_state
        from orchestrators.mode_fsm import get_mode_fsm
        fsm = get_mode_fsm()
        ok = fsm.set_mode_override(SystemMode.IDLE)
        if not ok:
            ok = fsm.force_mode(SystemMode.IDLE, 'standby command')
        if args.get('announce'):
            try:
                from services.media.usb_audio import get_usb_audio_service
                from services.media.sound_registry import get_sound_registry
                path = get_sound_registry().resolve('wimz/Wimz_standby')
                if path:
                    audio = get_usb_audio_service()
                    audio.wait_for_completion(4.0)
                    audio.play_file(path)
            except Exception as e:
                logger.debug(f"standby announce: {e}")
        return {'ok': bool(ok), 'mode': get_state().get_mode().value, 'stopped': stopped}

    @staticmethod
    def _tool_dispense_treat(args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        """source app (default) | agent | connector -> treat event data.source."""
        from services.reward.dispenser import get_dispenser_service
        source = str(args.get('source') or 'app').lower()
        reason = {'agent': 'agent', 'connector': 'connector', 'schedule': 'schedule'}.get(source, 'cloud_command')
        d = get_dispenser_service()
        # Let a relay voice clip finish first, so the carousel does not run over
        # the sentence that announces the treat (bounded; music is not waited for).
        waited = _wait_for_relay_speech(SPEECH_WAIT_MAX_S, cancel)
        if waited >= 0.2:
            logger.info(f"[TOOL] dispense_treat waited {waited:.1f}s for speech to finish")
        ok = d.dispense_treat(dog_id=args.get('dog_id'), dog_name=args.get('dog_name'),
                              reason=reason, command_id=args.get('_command_id'))
        if not ok:
            ref = getattr(d, 'last_refusal', None)
            if ref:
                # treat_cap_reached {used, cap, resets_at} | treat_rate | treat_window_closed
                return _err(ref['code'], ref['message'], used=ref['used'], cap=ref['cap'],
                            resets_at=ref['resets_at'], treats_remaining=d.treats_remaining)
            if getattr(d, 'empty_or_jammed', False):
                return _err('dispenser_empty', 'carousel empty or jammed',
                            treats_remaining=d.treats_remaining)
            return _err('dispense_blocked', 'dispense refused (busy or hardware)',
                        treats_remaining=d.treats_remaining)
        return {'ok': True, 'treats_remaining': d.treats_remaining,
                'treats_today': d.treats_dispensed_today}

    def _tool_look(self, args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        """look {action, degrees}: one bounded pan/tilt step (photo -> look -> photo).
        Never starts or touches video; the camera stays where it ends up."""
        action = str(args.get('action') or '').lower()
        if action not in ('pan_left', 'pan_right', 'tilt_up', 'tilt_down', 'center'):
            return _err('invalid_action', 'action must be pan_left|pan_right|tilt_up|tilt_down|center')
        degrees = 15.0
        if action != 'center' and args.get('degrees') is not None:
            try:
                degrees = float(args['degrees'])
            except (TypeError, ValueError):
                return _err('invalid_value', 'degrees must be a number from 5 to 45')
            if not 5.0 <= degrees <= 45.0:
                return _err('invalid_value', 'degrees must be from 5 to 45')
        if self.running(MOTION):
            return _err('gimbal_busy', 'a base move is running (it holds the camera centred)')
        try:
            from services.motion.pan_tilt import get_pantilt_service
            pt = get_pantilt_service()
        except Exception as e:
            return _err('unavailable', f'pan/tilt service not available: {e}')
        res = pt.look_step(action, degrees)
        if not res.get('ok'):
            code = res.get('code') or 'unavailable'
            msg = {'gimbal_busy': 'the camera is being steered right now',
                   'unavailable': 'camera servos are not initialised'}.get(code, code)
            return _err(code, msg)
        return res

    @staticmethod
    def _tool_ultrasonic(args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        """Ultrasonic tweeter (ultrasonic.py). {action: status} | {style} | {freq_hz, ...}.

        Refused unless the robot profile sets ultrasonic_enabled. Explicit bursts are
        capped at relay_max_level_pct (default 80, the highest level tested) and 5 s.
        The robot enforces >= 20 s between bursts: error 'cooldown' + retry_after_s.
        """
        import ultrasonic as u
        cfg = u.config()
        max_level = int(cfg.get('relay_max_level_pct', 80))
        if str(args.get('action') or '').lower() in ('status', 'get'):
            return {'ok': True, 'enabled': cfg['enabled'],
                    'available': Path(f'/proc/asound/{u.ULTRA_CARD}').exists(),
                    'styles': sorted(u.STYLES), 'freqs_hz': list(u.BURST_FREQS),
                    'max_level_pct': max_level, 'max_duration_s': u.MAX_DURATION_S,
                    'cooldown_s': max(u.MIN_COOLDOWN_S, float(cfg.get('cooldown_s', u.MIN_COOLDOWN_S)))}
        if not cfg['enabled']:
            return _err('ultrasonic_disabled', 'ultrasonic is switched off on this robot')
        source = str(args.get('source') or 'relay')
        who = {'dog_id': args.get('dog_id'), 'dog_name': args.get('dog_name')}
        try:
            if args.get('style'):
                res = u.play_style(str(args['style']), source=source, **who)
            elif args.get('freq_hz') is not None:
                res = u.burst(freq_hz=int(args['freq_hz']),
                              level_pct=min(max_level, int(args.get('level_pct', 40))),
                              duration_s=float(args.get('duration_s', 1.0)),
                              pulse_s=float(args.get('pulse_s') or 0.0),
                              gap_s=float(args.get('gap_s') or 0.2),
                              alt_freq_hz=args.get('alt_freq_hz'), source=source,
                              cooldown_s=cfg.get('cooldown_s', u.MIN_COOLDOWN_S), **who)
            else:
                return _err('bad_request', 'ultrasonic needs style, freq_hz or action=status',
                            styles=sorted(u.STYLES))
        except (TypeError, ValueError) as e:
            return _err('bad_request', str(e))
        if res.get('ok'):
            res.pop('settings', None)
            return res
        extra = {k: res[k] for k in ('retry_after_s', 'style') if k in res}
        code = {'disabled': 'ultrasonic_disabled'}.get(res.get('code'), res.get('code') or 'play_failed')
        return _err(code, str(res.get('reason')), **extra)

    @staticmethod
    def _tool_smart_plug(args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        """{action: on|off|toggle|status} (also {on: true|false} / {state: ...})."""
        from services.network.smart_plug import get_smart_plug_service
        svc = get_smart_plug_service()
        action = args.get('action', args.get('state'))
        if action is None and 'on' in args:
            action = 'on' if args.get('on') in (True, 1, 'on', 'true', '1') else 'off'
        action = str(action if action is not None else 'status').lower()
        if action in ('status', 'get'):
            return svc.status()          # {configured: false} when no plug is set up
        if action in ('scan', 'discover'):
            return svc.scan()
        if action == 'configure':
            return svc.configure(device_id=args.get('device_id'), alias=args.get('alias'),
                                 trigger_level=args.get('trigger_level'),
                                 auto_off_minutes=args.get('auto_off_minutes'),
                                 source=str(args.get('source') or 'relay'))
        if action == 'remove':
            return svc.remove(source=str(args.get('source') or 'relay'))
        return svc.set_manual(action, source=str(args.get('source') or 'relay'))

    @staticmethod
    def _tool_set_mode(args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        from core.state import SystemMode, get_state
        from orchestrators.mode_fsm import get_mode_fsm
        raw = str(args.get('mode') or '').lower()
        aliases = {'guardian': 'silent_guardian', 'sg': 'silent_guardian', 'training': 'coach',
                   'detection': 'coach', 'standby': 'idle'}
        raw = aliases.get(raw, raw)
        try:
            mode = SystemMode(raw)
        except ValueError:
            return _err('invalid_mode', f"mode {raw!r} not one of {[m.value for m in SystemMode]}")
        fsm = get_mode_fsm()
        duration = args.get('duration')
        ok = fsm.set_mode_override(mode, float(duration)) if duration else fsm.set_mode_override(mode)
        if not ok:
            ok = fsm.force_mode(mode, 'relay set_mode')
        return {'ok': bool(ok), 'mode': get_state().get_mode().value}

    def _tool_play_sound(self, args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        """{url} (relay asset, e.g. agent TTS: kind="tts") or {sound_id}."""
        from services.media.usb_audio import get_usb_audio_service
        audio = get_usb_audio_service()
        url, sound_id = args.get('url'), args.get('sound_id')
        # Default: enqueue and return (the PTT queue / pygame play in the
        # background). agent_utterance passes wait=True explicitly.
        wait = bool(args.get('wait', False))
        if url:
            data = None
            if self._relay is not None:
                data = self._relay.download_sync(url)
            if not data:
                return _err('download_failed', f"could not fetch {url}")
            fmt = (args.get('format') or url.rsplit('.', 1)[-1].split('?')[0] or 'mp3').lower()
            if fmt not in ('mp3', 'wav', 'aac', 'm4a', 'opus', 'ogg'):
                fmt = 'mp3'
            from services.media.push_to_talk import get_push_to_talk_service
            ptt = get_push_to_talk_service()
            # A TTS clip stops music / other audio, but never another relay clip
            # that is still speaking: it queues behind it (2026-11-11: a second
            # clip cut the first one off 6 s into a 15 s sentence). Only an
            # explicit interrupt=true cuts speech.
            interrupt = args.get('interrupt')
            if interrupt is None:
                interrupt = args.get('kind') == 'tts' and not _relay_speech_active()
            if interrupt:
                ptt.clear_queue()
                if audio.is_busy():
                    audio.stop()
            res = ptt.play_audio(data, fmt)
            if not res.get('success'):
                return _err('play_failed', str(res.get('error')))
            if wait:
                deadline = time.time() + 45
                time.sleep(0.5)
                while (ptt.is_playing() or audio.is_busy()) and time.time() < deadline and not cancel.is_set():
                    time.sleep(0.1)
            return {'ok': True, 'played': 'url', 'bytes': len(data), 'kind': args.get('kind'),
                    'queued': not wait}
        if sound_id:
            from services.media.sound_registry import get_sound_registry
            path = get_sound_registry().resolve(sound_id)
            if not path:
                # Cue names without a group: try the default cue folder
                from services.media.voice_lookup import resolve_voice_file
                path = resolve_voice_file(str(sound_id), dog_id_override=args.get('dog_id'))
            if not path:
                return _err('unknown_sound', f"no sound for {sound_id!r}")
            res = audio.play_file(path)
            if not res.get('success'):
                return _err('play_failed', str(res.get('error')))
            if wait:
                audio.wait_for_completion(30.0)
            return {'ok': True, 'played': 'sound_id', 'sound_id': sound_id, 'path': path, 'queued': not wait}
        return _err('bad_request', 'play_sound needs url or sound_id')

    @staticmethod
    def _tool_start_mission(args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        from orchestrators.mission_engine import get_mission_engine
        me = get_mission_engine()
        name = args.get('mission_id') or args.get('mission_name') or args.get('mission')
        if not name:
            return _err('bad_request', 'mission_id required', available=sorted(me.missions))
        if me.active_session:
            return _err('busy', 'a mission is already running')
        ok = me.start_mission(str(name), dog_id=args.get('dog_id'))
        if not ok:
            return _err('start_failed', f"could not start {name!r}", available=sorted(me.missions))
        return {'ok': True, 'mission_id': name}

    # ---- mission scheduler (persisted timers, 2026-11-07) ------------------

    @staticmethod
    def _tool_schedule_mission(args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        from core.schedule_manager import get_schedule_manager
        from core.mission_scheduler import get_mission_scheduler
        from orchestrators.mission_engine import get_mission_engine, MISSION_ALIASES
        me = get_mission_engine()
        name = args.get('mission') or args.get('mission_name') or args.get('type_name')
        if not name and args.get('mission_id') and str(args['mission_id']) in me.missions:
            name = args['mission_id']          # relay sent our mission name as the id
        if not name:
            return _err('bad_request', 'mission (mission name) required', available=sorted(me.missions))
        name = MISSION_ALIASES.get(str(name), str(name))
        if name not in me.missions:
            return _err('unknown_mission', f"mission {name!r} not loaded", available=sorted(me.missions))

        fire_at = args.get('fire_at') or args.get('at')
        start_time = args.get('start_time')
        times = args.get('times')
        if not fire_at and not start_time and isinstance(times, list) and times:
            start_time = str(times[0])
        if not fire_at and not start_time:
            return _err('bad_request', 'fire_at (ISO/epoch) or start_time HH:MM required')
        stype = args.get('schedule_type') or args.get('type')
        if fire_at:
            stype = 'once'
        elif not stype:
            stype = 'weekly' if args.get('days') or args.get('days_of_week') else 'daily'
        data = {
            'mission_name': name,
            'dog_id': str(args.get('dog_id') or 'unknown'),
            'name': args.get('name') or args.get('title') or f"{name} ({stype})",
            'type': stype,
            'start_time': start_time,
            'end_time': args.get('end_time'),
            'days_of_week': args.get('days_of_week') or args.get('days') or [],
            'enabled': args.get('enabled', True),
            'mission_id': args.get('mission_id'),
            'grace_s': args.get('grace_s'),
        }
        if fire_at:
            data['fire_at'] = fire_at
        res = get_schedule_manager().create_schedule(data)
        if not res.get('success'):
            return _err('bad_request', res.get('error', 'create failed'))
        sched = get_mission_scheduler()
        sched.poke()
        row = sched._sync_row(res['schedule'], __import__('core.mission_scheduler', fromlist=['_now'])._now())
        return {'ok': True, 'scheduled': True, 'scheduler_enabled': sched.enabled, **row}

    @staticmethod
    def _tool_schedule_sync(args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        from datetime import datetime
        from core.mission_scheduler import get_mission_scheduler, _now
        sched = get_mission_scheduler()
        rows = sched.get_scheduled_missions(dog_id=args.get('dog_id'),
                                            include_disabled=not args.get('enabled_only', False))
        me = sched.mission_engine
        sess = me.active_session
        active = None
        if sess:
            meta = sched.run_meta(sess.mission_id) or {}
            active = {'mission_name': sess.mission.name, 'run_id': sess.mission_id, 'dog_id': sess.dog_id,
                      'kind': 'mission', 'id': meta.get('mission_id'), 'schedule_id': meta.get('schedule_id')}
        else:
            mr = sched.active_mode_run()
            if mr:
                active = {'mission_name': mr['mode'], 'run_id': mr['run_id'], 'dog_id': mr.get('dog_id'),
                          'kind': 'mode', 'id': mr.get('mission_id'), 'schedule_id': mr.get('schedule_id'),
                          'ends_at': datetime.fromtimestamp(mr['ends_at']).astimezone().isoformat(),
                          'remaining_s': max(0, int(mr['ends_at'] - time.time()))}
        return {'ok': True, 'missions': rows, 'count': len(rows),
                'scheduler': {'enabled': sched.enabled, 'now': _now().isoformat(),
                              'check_interval_s': sched.check_interval},
                'active': active}

    @staticmethod
    def _tool_cancel_mission(args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        from core.mission_scheduler import get_mission_scheduler
        mid = args.get('mission_id') or args.get('schedule_id') or args.get('mission')
        if not mid:
            return _err('bad_request', 'mission_id required')
        return get_mission_scheduler().cancel(str(mid), stop_running=args.get('stop_running', True))

    LISTEN_MAX_S = 15.0

    def _tool_listen(self, args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        """listen {seconds}: PTT capture (16 kHz mono WAV, bark detector NOT
        paused) -> YAMNet top-5 + mean dBFS -> WAV to relay media store.
        Reply {labels:[{label, confidence}], mean_db, audio_url, seconds, bytes}."""
        import base64
        import shutil
        try:
            seconds = float(args.get('seconds') or args.get('duration_s') or 5)
        except (TypeError, ValueError):
            return _err('bad_args', 'seconds must be a number')
        seconds = max(1.0, min(seconds, self.LISTEN_MAX_S))

        from services.media.push_to_talk import get_push_to_talk_service
        ptt = get_push_to_talk_service()
        cap = ptt.capture_audio(seconds, max_duration=self.LISTEN_MAX_S, pause_bark=False)
        if not cap.get('success'):
            return _err('capture_failed', str(cap.get('error')))
        wav = base64.b64decode(cap['data'])

        # Park the WAV in the spec media tree + media_asset row
        filepath = None
        try:
            from core.data import get_wimz_store
            store = get_wimz_store()
            filepath = store.media_path_for(ext='wav')
            with open(filepath, 'wb') as f:
                f.write(wav)
            store.register_media(str(filepath), 'audio', codec='pcm_s16le',
                                 duration_ms=int(seconds * 1000),
                                 start_ts_ms=int((time.time() - seconds) * 1000),
                                 retention_class='ephemeral')
        except Exception as e:
            logger.warning(f"[TOOL] listen: media store write failed: {e}")
            import tempfile
            fd, tmp = tempfile.mkstemp(prefix='listen_', suffix='.wav')
            with open(fd, 'wb') as f:
                f.write(wav)
            filepath = tmp

        out: Dict[str, Any] = {'ok': True, 'seconds': seconds, 'bytes': len(wav),
                               'filepath': str(filepath), 'labels': [], 'mean_db': None, 'audio_url': None}
        try:
            from ai.yamnet_classifier import get_yamnet
            y = get_yamnet()
            if y.available():
                res = y.classify_wav(str(filepath), top_k=5)
                out['labels'], out['mean_db'] = res['labels'], res['mean_db']
            else:
                out['classifier_error'] = 'yamnet model missing'
        except Exception as e:
            logger.warning(f"[TOOL] listen: YAMNet failed: {e}")
            out['classifier_error'] = str(e)
            try:
                from ai.yamnet_classifier import read_wav_mono16k, mean_dbfs
                out['mean_db'] = mean_dbfs(read_wav_mono16k(str(filepath)))
            except Exception:
                pass

        if self._relay is not None:
            try:
                out['audio_url'] = self._relay.upload_media_sync(
                    str(filepath), 'audio/wav', filename=f"{Path(str(filepath)).stem}.wav", timeout=20.0)
            except Exception as e:
                logger.warning(f"[TOOL] listen: upload failed: {e}")
                out['upload_error'] = str(e)
        return out

    @staticmethod
    def _tool_move(args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        from services.motion.relay_move import get_relay_move
        action = str(args.get('action') or '').strip().lower()
        amount = args.get('amount', args.get('cm', args.get('degrees')))
        # relay bd3290a also sends unit ("cm"|"deg"), speed ("creep") and source;
        # speed is always the slowest safe setting here regardless.
        return get_relay_move().execute(action, amount, cancel, unit=args.get('unit'))

    @staticmethod
    def _tool_go_to(args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        return _err('not_supported', 'go_to: no navigation/docking on this platform yet')

    def _tool_agent_utterance(self, args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        """Record one utterance (WAV <= 30 s) via the PTT capture path, POST
        multipart to /agent/utterance with HMAC + dog_id, play the returned
        audio_url (play_sound). The relay owns ASR/LLM/TTS."""
        from services.media.push_to_talk import get_push_to_talk_service, UTTERANCE_MAX_S
        ptt = get_push_to_talk_service()
        duration = min(float(args.get('duration_s') or 8), UTTERANCE_MAX_S)
        cap = ptt.capture_audio(duration, max_duration=UTTERANCE_MAX_S)
        if not cap.get('success'):
            return _err('capture_failed', str(cap.get('error')))
        import base64
        wav = base64.b64decode(cap['data'])
        if self._relay is None:
            return _err('no_relay', 'relay client not attached')
        dog_id = args.get('dog_id')
        if not dog_id:
            try:
                from core.dog_profile_manager import resolve_relay_dog
                dog_id, _ = resolve_relay_dog()
            except Exception:
                dog_id = None
        import aiohttp
        form = aiohttp.FormData()
        form.add_field('dog_id', str(dog_id or ''))
        form.add_field('device_id', str(self._relay.config.device_id))
        form.add_field('file', wav, filename='utterance.wav', content_type='audio/wav')
        try:
            status, body = self._relay.http_sync('POST', '/agent/utterance', data=form, timeout=45.0)
        except Exception as e:
            return _err('upload_failed', str(e))
        if not (200 <= status < 300) or not isinstance(body, dict):
            return _err('upload_failed', f"/agent/utterance -> {status}", status=status)
        out = {'ok': True, 'duration_s': duration, 'bytes': len(wav), 'dog_id': dog_id,
               'transcript': body.get('transcript'), 'reply_text': body.get('text') or body.get('reply_text')}
        audio_url = body.get('audio_url')
        if audio_url:
            play = self._tool_play_sound({'url': audio_url, 'kind': 'tts', 'wait': True}, cancel)
            out['played'] = play.get('ok', False)
            if not play.get('ok'):
                out['play_error'] = play.get('error')
        return out


_instance: Optional[ToolDispatcher] = None
_instance_lock = threading.Lock()


def get_tool_dispatcher() -> ToolDispatcher:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = ToolDispatcher()
        return _instance
