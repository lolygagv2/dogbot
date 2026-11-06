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
                    f"error={error!r}" + (f" result={result_summary[:200]}" if result_summary else ''))
    except Exception:
        pass
    try:
        from core.data.wimz_store import get_wimz_store
        get_wimz_store().log_event(None, 'tool_call', {
            'name': name, 'args': args, 'id': call_id, 'duration_ms': duration_ms,
            'ok': ok, 'error': error}, label_source='human')
    except Exception as e:
        logger.debug(f"tool_call wimz row failed: {e}")


# ------------------------------------------------------------------ dispatcher

class ToolDispatcher:
    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._lock = threading.RLock()
        self._slot = threading.Lock()
        self._current: Optional[Dict[str, Any]] = None   # {name, id, cancel, started}
        self._relay = None
        self._attached = False
        self._register_builtins()

    # ------------------------------------------------------------- registry

    def register(self, spec: ToolSpec) -> None:
        with self._lock:
            self._tools[spec.name] = spec
        logger.debug(f"[TOOL] registered {spec.name} (timeout {spec.timeout_s}s)")

    def register_tool(self, name: str, handler: Handler, timeout_s: float = 30.0,
                      preemptive: bool = False, description: str = '') -> None:
        self.register(ToolSpec(name, handler, timeout_s, preemptive, description))

    def has_tool(self, name: Optional[str]) -> bool:
        return bool(name) and name in self._tools

    def list_tools(self) -> Dict[str, Any]:
        with self._lock:
            tools = [{'name': s.name, 'timeout_s': s.timeout_s, 'preemptive': s.preemptive,
                      'description': s.description} for s in self._tools.values()]
            cur = dict(self._current) if self._current else None
        if cur:
            cur.pop('cancel', None)
        return {'tools': sorted(tools, key=lambda t: t['name']), 'busy': cur is not None,
                'current': cur}

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
        with self._lock:
            cur = self._current
        if not cur:
            return True
        cur['cancel'].set()
        logger.info(f"[TOOL] cancel requested for {cur['name']} command_id={cur['id']}")
        got = self._slot.acquire(timeout=wait_s)
        if got:
            self._slot.release()
        return got

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

        if not self._slot.acquire(blocking=False):
            with self._lock:
                cur = self._current
            result = _err('busy', f"command {cur['name'] if cur else '?'} is running",
                          result='busy', running=cur['name'] if cur else None)
            self._finish(name, call_id, args, t0, result, result['error']['message'], reply)
            return

        cancel = threading.Event()
        with self._lock:
            self._current = {'name': name, 'id': call_id, 'cancel': cancel, 'started': time.time()}
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"Tool-{name}")
        released = threading.Event()

        def _release_when_done(fut):
            try:
                fut.result()
            except Exception:
                pass
            with self._lock:
                self._current = None
            if not released.is_set():
                released.set()
                self._slot.release()
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
        self.register(ToolSpec('take_photo', self._tool_take_photo, 15.0,
                               description='JPEG of the current camera frame (any mode)'))
        self.register(ToolSpec('standby', self._tool_standby, 10.0, preemptive=True,
                               description='Stop audio and return to IDLE'))
        self.register(ToolSpec('dispense_treat', self._tool_dispense_treat, 15.0,
                               description='Dispense one treat (source app|agent|connector)'))
        self.register(ToolSpec('set_mode', self._tool_set_mode, 10.0,
                               description='Switch mode: idle|silent_guardian|coach|manual|night_sentry'))
        self.register(ToolSpec('play_sound', self._tool_play_sound, 60.0,
                               description='Play a relay-hosted url or a registry sound_id'))
        self.register(ToolSpec('start_mission', self._tool_start_mission, 10.0,
                               description='Start a mission by id/name'))
        self.register(ToolSpec('go_to', self._tool_go_to, 5.0,
                               description='Navigate to a named place (not supported yet)'))
        self.register(ToolSpec('agent_utterance', self._tool_agent_utterance, 60.0,
                               description='Record one utterance (WAV <=30 s), POST /agent/utterance, play reply'))

    @staticmethod
    def _tool_take_photo(args: Dict[str, Any], cancel: threading.Event) -> Dict[str, Any]:
        from services.media.photo_capture import get_photo_capture_service
        detector = None
        try:
            from services.perception.detector import get_detector_service
            detector = get_detector_service()
        except Exception:
            pass
        with_hud = bool(args.get('with_hud', args.get('hud', True)))
        res = get_photo_capture_service().capture_photo(with_hud=with_hud, detector=detector)
        if not res.get('success'):
            return _err('capture_failed', str(res.get('error', 'capture failed')))
        jpeg_b64 = res.get('data')
        max_w = args.get('max_width')
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
        ok = d.dispense_treat(dog_id=args.get('dog_id'), dog_name=args.get('dog_name'),
                              reason=reason, command_id=args.get('_command_id'))
        if not ok:
            return _err('dispense_blocked', 'dispense refused (policy, cap, empty or jam)',
                        treats_remaining=d.treats_remaining)
        return {'ok': True, 'treats_remaining': d.treats_remaining,
                'treats_today': d.treats_dispensed_today}

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
        wait = bool(args.get('wait', True))
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
            if args.get('interrupt', args.get('kind') == 'tts'):
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
            return {'ok': True, 'played': 'url', 'bytes': len(data), 'kind': args.get('kind')}
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
            return {'ok': True, 'played': 'sound_id', 'sound_id': sound_id, 'path': path}
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
        form.add_field('audio', wav, filename='utterance.wav', content_type='audio/wav')
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
