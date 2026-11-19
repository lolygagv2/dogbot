#!/usr/bin/env python3
"""
Sleep mode: everything that moves, sees, hears or makes sound is stopped;
WiFi, the relay connection (heartbeat), the command listeners and a periodic
status report keep running. The process itself never stops.

Sleep is NOT shutdown. Nothing in this module touches the power latch board
(GPIO26 kill pulse / GPIO21 relay / GPIO20 sense): those belong to
wimz_power_button.py and the systemd shutdown hook. The only path from here to
a real power-off is `battery_shutdown()`, which runs the ordinary
`sudo shutdown -h now` (the same thing a button press does) when the battery
reaches the safety monitor's emergency threshold while asleep.

Entry order (sleep):
    1 sessions   mission / Silent Guardian / Coach / Night Sentry stopped, mode -> sleep
    2 motors     drive stopped; dispenser stepper driver disabled (EN high)
    3 gimbal     tracking loop stopped, pan/tilt PWM released
    4 camera     WebRTC sessions closed, capture loop stopped, Picamera2 closed
    5 hailo      network group + VDevice released (driver drops the chip to D3)
    6 audio_in   bark detection + WebRTC mic capture stopped
    7 amp        playback stopped, speaker mixer muted, ultrasonic amp relay off
    8 leds       slow dim pulse, blue tube off
    9 governor   CPU governor -> powersave
Wake runs the same list in reverse, checks camera / Hailo / audio, sets mode
idle, then runs the boot greeting hook.

Triggers: relay / local commands `sleep` and `wake` (ToolDispatcher), REST
POST /system/sleep and /system/wake, and `ensure_awake()` which every action
command path calls before it runs.
"""
import glob
import logging
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger('SleepManager')

AWAKE, ENTERING, ASLEEP, WAKING = 'awake', 'entering_sleep', 'asleep', 'waking'

STATUS_INTERVAL_S = 180.0        # status report while asleep
WAKE_TARGET_S = 5.0
CAMERA_HEALTH_TIMEOUT_S = 3.0
SLEEP_GOVERNOR = 'powersave'
HAILO_PCI_VENDOR = '0x1e60'

# Commands that never wake the robot: they are not actions, or they act on
# something that keeps running in sleep (schedules, WiFi, the smart plug).
# Anything not listed here is an action command: wake first, then run.
PASSIVE_COMMANDS = frozenset({
    'sleep', 'wake', 'ping', 'standby', 'stop', 'audio_stop', 'stop_mission', 'cancel_mission',
    'stop_recording', 'move_stop',
    'schedule_mission', 'schedule_sync', 'create_schedule', 'update_schedule', 'delete_schedule',
    'get_schedules', 'get_scheduler_status', 'get_mission_status', 'mission_status', 'list_missions',
    'mission_config', 'list_songs', 'list_voices', 'delete_song', 'delete_voice', 'upload_song',
    'upload_voice', 'download_song', 'select_dog', 'reload_dogs', 'dog_weekly_summary_pull',
    'sg_config', 'sg_status_pull', 'audio_volume', 'set_volume', 'set_video_quality', 'audio_loop',
    'treat_counter_reset', 'treat_counter_set', 'smart_plug', 'local_mode', 'cloud_mode',
    'start_update',
})


SLEEP_MODE_NAMES = ('sleep', 'sleeping', 'asleep')


def sleep_intent(command: Optional[str], args: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """'sleep' / 'wake' when the command asks for one of them, else None.

    The app (build 191) sends `set_mode {mode: sleep}` to sleep and
    `set_mode {mode: idle}` to wake; `sleep` / `wake` also exist as commands.
    `set_mode idle` only means wake while the robot is asleep."""
    command = str(command or '').lower()
    if command in ('sleep', 'wake'):
        return command
    if command in ('set_mode', 'mode'):
        mode = str((args or {}).get('mode') or '').lower()
        if mode in SLEEP_MODE_NAMES:
            return 'sleep'
        if mode in ('idle', 'standby') and is_asleep():
            return 'wake'
    return None


def run_intent(intent: str, source: str) -> Dict[str, Any]:
    mgr = get_sleep_manager()
    return mgr.sleep(source=source) if intent == 'sleep' else mgr.wake(source=source)


def is_action_command(command: Optional[str], args: Optional[Dict[str, Any]] = None) -> bool:
    """True when `command` must wake a sleeping robot before it runs."""
    if not command:
        return False
    command = str(command).lower()
    args = args or {}
    if command in PASSIVE_COMMANDS:
        return False
    if command in ('set_mode', 'mode'):
        # sleep / idle are handled as sleep_intent(), not as wake-then-run.
        return str(args.get('mode') or '').lower() not in ('', 'idle', 'standby') + SLEEP_MODE_NAMES
    if command == 'move' and str(args.get('action') or '').lower() == 'stop':
        return False
    if command in ('ultrasonic',) and str(args.get('action') or '').lower() == 'status':
        return False
    return True


class SleepManager:
    SETTLE_S = 2.0                   # wait before the "after" power reading
    def __init__(self):
        self._lock = threading.RLock()          # serialises sleep()/wake()
        self._state = AWAKE
        self._state_changed = threading.Condition()
        self._greeting_hook: Optional[Callable[[], None]] = None
        self._app = None                        # TreatBotMain (optional)
        self._saved: Dict[str, Any] = {}        # what sleep changed, for wake
        self._status_thread: Optional[threading.Thread] = None
        self._status_stop = threading.Event()
        self._subscribed = False
        self._owner: Optional[int] = None       # thread running sleep()/wake()
        self.last_sleep: Dict[str, Any] = {}
        self.last_wake: Dict[str, Any] = {}
        self.asleep_since: Optional[float] = None

    # ------------------------------------------------------------ wiring
    def attach(self, app=None, greeting_hook: Optional[Callable[[], None]] = None) -> None:
        self._app = app
        if greeting_hook:
            self._greeting_hook = greeting_hook
        if not self._subscribed:
            try:
                from core.state import get_state
                get_state().subscribe('mode_change', self._on_mode_change)
                self._subscribed = True
            except Exception as e:
                logger.error(f"[SLEEP] mode_change subscribe failed: {e}")

    # ------------------------------------------------------------ queries
    @property
    def state(self) -> str:
        return self._state

    def is_asleep(self) -> bool:
        """True from the moment sleep starts until wake has finished."""
        return self._state != AWAKE

    def status(self) -> Dict[str, Any]:
        return {
            'state': self._state,
            'asleep': self._state in (ENTERING, ASLEEP),
            'asleep_since': self.asleep_since,
            'asleep_for_s': round(time.time() - self.asleep_since, 1) if self.asleep_since else None,
            'last_sleep': self.last_sleep,
            'last_wake': self.last_wake,
            'power': self.read_power(),
        }

    def _set_state(self, s: str) -> None:
        with self._state_changed:
            self._state = s
            self._state_changed.notify_all()

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _step(steps: List[Dict[str, Any]], name: str, fn: Callable[[], Any]) -> Any:
        """Run one step; never raises. fn returns a detail (str/dict) or raises."""
        t0 = time.monotonic()
        row: Dict[str, Any] = {'step': name, 'ok': True}
        out = None
        try:
            out = fn()
            if isinstance(out, dict):
                row.update(out)
            elif out is not None:
                row['detail'] = out
        except Exception as e:
            row['ok'] = False
            row['error'] = f"{type(e).__name__}: {e}"
        row['ms'] = int((time.monotonic() - t0) * 1000)
        steps.append(row)
        (logger.info if row['ok'] else logger.error)(f"[SLEEP] {name}: {row}")
        return out

    @staticmethod
    def read_power() -> Dict[str, Any]:
        """What can be measured. There is no current sensor on the battery
        rail (I2C carries only the PCA9685 and the ADS1115 voltage divider),
        so total draw is unknown. The Pi 5 PMIC reports its own rails; their
        sum covers the SoC/RAM/WiFi side of the Pi only — not the Hailo, the
        camera, USB devices, servos, motors, LEDs or amplifiers."""
        out: Dict[str, Any] = {'battery_current_sensor': False, 'battery_v': None,
                               'pi_pmic_w': None, 'hailo_pci_state': hailo_pci_state()}
        try:
            from core.state import get_state
            v = float(get_state().hardware.battery_voltage or 0)
            out['battery_v'] = round(v, 2) if v > 0 else None
        except Exception:
            pass
        try:
            txt = subprocess.run(['vcgencmd', 'pmic_read_adc'], capture_output=True,
                                 text=True, timeout=3).stdout
            amps, volts = {}, {}
            for line in txt.splitlines():
                parts = line.split()
                if len(parts) < 2 or '=' not in parts[1]:
                    continue
                name, val = parts[0], parts[1].split('=')[1].rstrip('AV')
                if name.endswith('_A'):
                    amps[name[:-2]] = float(val)
                elif name.endswith('_V'):
                    volts[name[:-2]] = float(val)
            if amps:
                out['pi_pmic_w'] = round(sum(a * volts.get(n, 0.0) for n, a in amps.items()), 2)
        except Exception:
            pass
        return out

    # ------------------------------------------------------------ SLEEP
    def sleep(self, source: str = 'unknown') -> Dict[str, Any]:
        with self._lock:
            if self._state == ASLEEP:
                return {'ok': True, 'already': True, 'mode': 'sleep'}
            t0 = time.monotonic()
            self._owner = threading.get_ident()
            self._set_state(ENTERING)
            steps: List[Dict[str, Any]] = []
            power_before = self.read_power()
            logger.info(f"[SLEEP] entering sleep (source={source}) power_before={power_before}")
            self._saved = {}

            self._step(steps, 'sessions', lambda: self._sleep_sessions(source))
            self._step(steps, 'motors', self._sleep_motors)
            self._step(steps, 'gimbal', self._sleep_gimbal)
            self._step(steps, 'camera', self._sleep_camera)
            self._step(steps, 'hailo', self._sleep_hailo)
            self._step(steps, 'audio_in', self._sleep_audio_in)
            self._step(steps, 'amp', self._sleep_amp)
            self._step(steps, 'leds', self._sleep_leds)
            self._step(steps, 'governor', self._sleep_governor)

            self.asleep_since = time.time()
            self._owner = None
            self._set_state(ASLEEP)
            self._start_status_loop()
            time.sleep(self.SETTLE_S)
            power_after = self.read_power()
            self.last_sleep = {
                'at': self.asleep_since, 'source': source,
                'enter_s': round(time.monotonic() - t0 - self.SETTLE_S, 2),
                'steps': steps, 'failed': [s['step'] for s in steps if not s['ok']],
                'power_before': power_before, 'power_after': power_after,
            }
            logger.info(f"[SLEEP] asleep after {self.last_sleep['enter_s']}s; failed={self.last_sleep['failed']} "
                        f"power_after={power_after} (no battery current sensor on this robot)")
            self._emit_status()
            return {'ok': True, 'mode': 'sleep', **self.last_sleep}

    def _sleep_sessions(self, source: str) -> Dict[str, Any]:
        from core.state import SystemMode, get_state
        st = get_state()
        prev = st.get_mode().value
        self._saved['previous_mode'] = prev
        out: Dict[str, Any] = {'previous_mode': prev}
        try:
            from services.cloud.tool_dispatcher import get_tool_dispatcher
            get_tool_dispatcher().cancel_current(wait_s=2.0)
        except Exception as e:
            logger.debug(f"[SLEEP] dispatcher cancel: {e}")
        try:
            from orchestrators.mission_engine import get_mission_engine
            me = get_mission_engine()
            if me and me.active_session:
                out['mission_stopped'] = bool(me.stop_mission(reason='sleep'))
        except Exception as e:
            out['mission_error'] = str(e)
        try:
            from orchestrators.mode_fsm import get_mode_fsm
            fsm = get_mode_fsm()
            fsm.clear_override('sleep')
        except Exception:
            fsm = None
        # The mode_change listeners (main_treatbot._on_mode_change) stop Silent
        # Guardian / Coach / Night Sentry and bark detection synchronously, and
        # BehaviorEventService emits the relay `mode_change` event.
        ok = st.set_mode(SystemMode.SLEEP, f"user sleep ({source})", force=True)
        if fsm is not None and ok:
            fsm.current_mode = SystemMode.SLEEP
            fsm.mode_start_time = time.time()
        # Belt and braces: the handlers above swallow their own errors.
        app = self._app
        for name in ('silent_guardian_mode', 'coaching_engine', 'night_sentry_mode'):
            h = getattr(app, name, None) if app else None
            try:
                if h is not None and getattr(h, 'running', False):
                    h.stop()
                    out[f'{name}_stopped'] = True
            except Exception as e:
                out[f'{name}_error'] = str(e)
        if not ok:
            raise RuntimeError(f"mode change to sleep refused (mode={st.get_mode().value})")
        return out

    def _sleep_motors(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        from core.motor_command_bus import get_motor_bus, create_motor_command, CommandSource
        bus = get_motor_bus()
        if bus:
            bus.send_command(create_motor_command(0, 0, CommandSource.API))
            mc = getattr(bus, 'motor_controller', None)
            if mc is not None and hasattr(mc, 'stop_all'):
                out['drive_stopped'] = bool(mc.stop_all())
            out['drive_enable_line'] = bool(mc is not None and hasattr(mc, 'disable'))
            if out['drive_enable_line']:
                mc.disable()
        try:
            from services.reward.dispenser import get_dispenser_service
            d = get_dispenser_service()
            if getattr(d, 'gpio_chip', None) is not None:
                if d.is_dispensing():
                    d.cancel_dispense()
                d._disable_motor()               # TMC2209 EN high: coils unpowered
                out['dispenser_driver_disabled'] = True
        except Exception as e:
            out['dispenser_error'] = str(e)
        return out

    def _sleep_gimbal(self) -> Dict[str, Any]:
        from services.motion.pan_tilt import get_pantilt_service
        pt = get_pantilt_service()
        self._saved['gimbal_running'] = bool(getattr(pt, 'running', False))
        pt.stop_tracking()
        servo = getattr(pt, 'servo', None)
        if servo is None or not getattr(servo, 'pca', None):
            return {'released': False, 'detail': 'servo controller not initialised'}
        # Pan/tilt only. The third PCA9685 channel is the ultrasonic amp relay
        # on this hardware; the amp step handles it.
        servo.pan_channel.duty_cycle = 0
        servo.pitch_channel.duty_cycle = 0
        return {'released': True}

    def _run_on_loop(self, coro_fn, timeout: float = 5.0):
        import asyncio
        app = self._app
        svc = getattr(app, 'webrtc_service', None) if app else None
        loop = getattr(svc, '_loop', None) if svc else None
        if svc is None or loop is None or not loop.is_running():
            return None
        return asyncio.run_coroutine_threadsafe(coro_fn(svc), loop).result(timeout=timeout)

    def _sleep_camera(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            async def _close(svc):
                ids = list(svc.connections.keys())
                for sid in ids:
                    await svc.close_connection(sid)
                return len(ids)
            n = self._run_on_loop(_close)
            if n:
                out['webrtc_closed'] = n
        except Exception as e:
            out['webrtc_error'] = str(e)
        app = self._app
        nm = getattr(app, 'night_mode', None) if app else None
        try:
            if nm is not None:
                nm.stop()
                self._saved['night_mode'] = True
        except Exception as e:
            out['night_mode_error'] = str(e)
        from services.perception.detector import get_detector_service
        det = get_detector_service()
        out.update(det.suspend_for_sleep())
        self._saved['camera'] = out.get('camera_was_initialized', False)
        self._saved['detection'] = out.get('detection_was_running', False)
        return out

    def _sleep_hailo(self) -> Dict[str, Any]:
        from services.perception.detector import get_detector_service
        det = get_detector_service()
        before = hailo_pci_state()
        if not det.ai or not det.ai_initialized:
            return {'unloaded': False, 'detail': 'AI controller was not initialised', 'pci_state': before}
        self._saved['hailo'] = True
        ok = det.ai.unload_hailo()
        time.sleep(0.3)
        return {'unloaded': bool(ok), 'pci_state_before': before, 'pci_state_after': hailo_pci_state()}

    def _sleep_audio_in(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            from services.perception.bark_detector import get_bark_detector_service
            bd = get_bark_detector_service()
            if bd and getattr(bd, 'is_running', False):
                bd.stop()
                out['bark_stopped'] = True
        except Exception as e:
            out['bark_error'] = str(e)
        try:
            from services.streaming.audio_track import get_audio_track
            at = get_audio_track()
            if at is not None:
                self._saved['mic_capture'] = at._stream is not None
                at.pause_capture()
                out['webrtc_mic_released'] = True
        except Exception as e:
            out['mic_error'] = str(e)
        return out

    def _sleep_amp(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            from services.media.push_to_talk import get_push_to_talk_service
            get_push_to_talk_service().clear_queue()
        except Exception:
            pass
        try:
            from services.media.usb_audio import get_usb_audio_service
            audio = get_usb_audio_service()
            if audio.is_busy():
                audio.stop()
                out['playback_stopped'] = True
        except Exception as e:
            out['playback_error'] = str(e)
        out['speaker_muted'] = _set_speaker_mute(True)
        try:
            import ultrasonic
            ultrasonic._set_amp(False)
            out['ultrasonic_amp_off'] = True
        except Exception as e:
            out['ultrasonic_amp'] = f"not switched: {e}"
        return out

    def _sleep_leds(self) -> Dict[str, Any]:
        from services.media.led import get_led_service
        led = get_led_service()
        self._saved['led_pattern'] = getattr(led, 'current_pattern', None)
        ok = led.set_pattern('sleep')
        try:
            ctrl = getattr(led, 'led', None)
            if ctrl is not None and getattr(ctrl, 'blue_is_on', False):
                ctrl.blue_off()
                self._saved['blue_tube'] = True
        except Exception:
            pass
        if not ok:
            raise RuntimeError('LED service refused the sleep pattern')
        return {'pattern': 'sleep'}

    def _sleep_governor(self) -> Dict[str, Any]:
        prev = read_governor()
        self._saved['governor'] = prev
        ok = set_governor(SLEEP_GOVERNOR)
        if not ok:
            raise RuntimeError(f'could not set governor (still {read_governor()})')
        return {'from': prev, 'to': read_governor()}

    # ------------------------------------------------------------ WAKE
    def wake(self, source: str = 'unknown', greet: bool = True) -> Dict[str, Any]:
        with self._lock:
            if self._state == AWAKE:
                return {'ok': True, 'already': True, 'mode': self._mode_value()}
            t0 = time.monotonic()
            self._owner = threading.get_ident()
            self._set_state(WAKING)
            self._stop_status_loop()
            steps: List[Dict[str, Any]] = []
            logger.info(f"[SLEEP] waking (source={source})")

            self._step(steps, 'governor', self._wake_governor)
            self._step(steps, 'leds', self._wake_leds)
            self._step(steps, 'amp', self._wake_amp)
            self._step(steps, 'audio_in', self._wake_audio_in)
            # Hailo firmware/HEF load and the camera open do not depend on each
            # other; running them side by side is what keeps wake under target.
            hailo_steps: List[Dict[str, Any]] = []
            th = threading.Thread(target=self._step, args=(hailo_steps, 'hailo', self._wake_hailo),
                                  daemon=True, name='SleepWakeHailo')
            th.start()
            self._step(steps, 'camera', self._wake_camera)
            th.join(timeout=15.0)
            steps.extend(hailo_steps or [{'step': 'hailo', 'ok': False, 'error': 'timeout'}])
            self._step(steps, 'gimbal', self._wake_gimbal)
            self._step(steps, 'motors', lambda: {'detail': 'drive bus was never stopped; '
                                                 'dispenser driver re-enables per dispense'})
            health = self._step(steps, 'health', self._health_check) or {}
            self._step(steps, 'sessions', lambda: self._wake_mode(source))

            ready_s = round(time.monotonic() - t0, 2)
            healthy = bool(health.get('camera', {}).get('ok') and health.get('hailo', {}).get('ok')
                           and health.get('audio', {}).get('ok'))
            self.last_wake = {
                'at': time.time(), 'source': source, 'ready_s': ready_s,
                'within_target': ready_s <= WAKE_TARGET_S, 'target_s': WAKE_TARGET_S,
                'healthy': healthy, 'health': health,
                'slept_s': round(time.time() - self.asleep_since, 1) if self.asleep_since else None,
                'steps': steps, 'failed': [s['step'] for s in steps if not s['ok']],
            }
            self.asleep_since = None
            self._owner = None
            self._set_state(AWAKE)
            logger.info(f"[SLEEP] READY in {ready_s}s (target {WAKE_TARGET_S}s) healthy={healthy} "
                        f"failed={self.last_wake['failed']} health={health}")
            self._emit_status()
            if greet and self._greeting_hook:
                threading.Thread(target=self._run_greeting, daemon=True, name='SleepWakeGreeting').start()
            return {'ok': True, 'mode': self._mode_value(), **self.last_wake}

    def _run_greeting(self) -> None:
        try:
            self._greeting_hook()
        except Exception as e:
            logger.error(f"[SLEEP] greeting hook failed: {e}")

    @staticmethod
    def _mode_value() -> Optional[str]:
        try:
            from core.state import get_state
            return get_state().get_mode().value
        except Exception:
            return None

    def _wake_governor(self) -> Dict[str, Any]:
        target = self._saved.get('governor') or 'ondemand'
        if target == SLEEP_GOVERNOR:
            target = 'ondemand'
        if not set_governor(target):
            raise RuntimeError(f'could not restore governor {target} (now {read_governor()})')
        return {'to': read_governor()}

    def _wake_leds(self) -> Dict[str, Any]:
        from services.media.led import get_led_service
        led = get_led_service()
        if not led.set_pattern('idle'):
            raise RuntimeError('LED service refused the idle pattern')
        if self._saved.get('blue_tube'):
            try:
                led.led.blue_on()
            except Exception:
                pass
        return {'pattern': 'idle'}

    def _wake_amp(self) -> Dict[str, Any]:
        # The ultrasonic amp is only ever powered for the length of a burst.
        ok = _set_speaker_mute(False)
        if not ok:
            raise RuntimeError('could not unmute the speaker mixer')
        return {'speaker_unmuted': True}

    def _wake_audio_in(self) -> Dict[str, Any]:
        # Bark detection belongs to Silent Guardian / Coach and starts with
        # those modes; wake lands in idle, so only the WebRTC mic comes back.
        if self._saved.get('mic_capture'):
            from services.streaming.audio_track import get_audio_track
            at = get_audio_track()
            if at is not None:
                at.resume_capture()
                return {'webrtc_mic_resumed': True}
        return {'detail': 'nothing was capturing'}

    def _wake_hailo(self) -> Dict[str, Any]:
        from services.perception.detector import get_detector_service
        det = get_detector_service()
        if not self._saved.get('hailo') or not det.ai:
            return {'detail': 'AI controller was not loaded before sleep'}
        if not det.ai.reload_hailo():
            raise RuntimeError('Hailo model reload failed')
        return {'loaded': True, 'pci_state': hailo_pci_state()}

    def _wake_camera(self) -> Dict[str, Any]:
        from services.perception.detector import get_detector_service
        det = get_detector_service()
        out = det.resume_from_sleep(start_detection=self._saved.get('detection', True))
        nm = getattr(self._app, 'night_mode', None) if self._app else None
        if nm is not None and self._saved.get('night_mode'):
            try:
                nm.start()
            except Exception as e:
                out['night_mode_error'] = str(e)
        if not out.get('camera'):
            raise RuntimeError(f"camera did not come back: {out}")
        return out

    def _wake_gimbal(self) -> Dict[str, Any]:
        from services.motion.pan_tilt import get_pantilt_service
        pt = get_pantilt_service()
        if not getattr(pt, 'servo_initialized', False):
            return {'detail': 'servos were not initialised'}
        if self._saved.get('gimbal_running', True):
            pt.start_tracking()
        pt.center_camera(reason='wake')
        return {'centered': True}

    def _wake_mode(self, source: str) -> Dict[str, Any]:
        from core.state import SystemMode, get_state
        st = get_state()
        if st.get_mode() != SystemMode.SLEEP:
            return {'mode': st.get_mode().value, 'detail': 'mode already changed'}
        ok = st.set_mode(SystemMode.IDLE, f"user wake ({source})", force=True)
        try:
            from orchestrators.mode_fsm import get_mode_fsm
            fsm = get_mode_fsm()
            fsm.current_mode = SystemMode.IDLE
            fsm.mode_start_time = time.time()
        except Exception:
            pass
        if not ok:
            raise RuntimeError('mode change to idle refused')
        return {'mode': 'idle', 'previous_session': self._saved.get('previous_mode')}

    def _health_check(self) -> Dict[str, Any]:
        """Camera delivers a new frame, Hailo runs one inference on it, audio
        output and microphone are present and unmuted."""
        from services.perception.detector import get_detector_service
        det = get_detector_service()
        out: Dict[str, Any] = {}

        t0 = time.time()
        frame, ts = None, 0.0
        deadline = time.monotonic() + CAMERA_HEALTH_TIMEOUT_S
        while time.monotonic() < deadline:
            frame, ts = det.get_last_frame_with_timestamp()
            if frame is not None and ts >= t0 - 0.5:
                break
            time.sleep(0.03)
        cam_ok = bool(det.camera_initialized and frame is not None and ts >= t0 - 0.5)
        out['camera'] = {'ok': cam_ok, 'frame_age_s': round(time.time() - ts, 2) if ts else None,
                         'shape': list(frame.shape) if frame is not None else None}

        h: Dict[str, Any] = {'ok': False, 'pci_state': hailo_pci_state()}
        try:
            ai = det.ai
            if ai is None or not self._saved.get('hailo'):
                h['detail'] = 'AI controller was not loaded before sleep'
            elif not ai.model_loaded:
                h['detail'] = 'model not loaded'
            elif frame is None:
                h['detail'] = 'model loaded; no frame to run'
            else:
                i0 = time.monotonic()
                h['ok'] = bool(ai.self_test(frame))
                h['inference_ms'] = int((time.monotonic() - i0) * 1000)
        except Exception as e:
            h['error'] = str(e)
        out['hailo'] = h

        a: Dict[str, Any] = {'ok': False}
        try:
            from services.media.usb_audio import get_usb_audio_service
            audio = get_usb_audio_service()
            a['playback_initialized'] = bool(audio.initialized)
            a['speaker_muted'] = _speaker_muted()
            card = _speaker_card()
            a['card'] = card
            rec = subprocess.run(['arecord', '-l'], capture_output=True, text=True, timeout=3).stdout
            a['mic_present'] = bool(card is not None and f'card {card}:' in rec)
            try:
                from services.perception.bark_detector import get_bark_detector_service
                a['bark_detector_enabled'] = bool(get_bark_detector_service().enabled)
            except Exception:
                a['bark_detector_enabled'] = None
            a['ok'] = bool(a['playback_initialized'] and a['mic_present'] and a['speaker_muted'] is False)
        except Exception as e:
            a['error'] = str(e)
        out['audio'] = a
        return out

    # ------------------------------------------------------------ wake gates
    def ensure_awake(self, source: str = 'command', timeout_s: float = 20.0) -> bool:
        """Called by every action path before it acts. Returns once the robot
        is awake (immediately when it already is)."""
        if self._state == AWAKE or self._owner == threading.get_ident():
            return True                          # awake, or we ARE the sleep/wake in progress
        if self._state in (ENTERING, ASLEEP):
            logger.info(f"[SLEEP] action while asleep -> waking first (source={source})")
            try:
                self.wake(source=source)         # blocks behind a sleep in progress
            except Exception as e:
                logger.error(f"[SLEEP] wake for {source} failed: {e}")
        deadline = time.monotonic() + timeout_s
        with self._state_changed:
            while self._state != AWAKE and time.monotonic() < deadline:
                self._state_changed.wait(0.2)
        return self._state == AWAKE

    def wake_async(self, source: str) -> None:
        if self._state in (ENTERING, ASLEEP):
            threading.Thread(target=self.ensure_awake, args=(source,), daemon=True,
                             name='SleepWakeAsync').start()

    def _on_mode_change(self, data: Dict[str, Any]) -> None:
        """Safety net: something left sleep mode without going through wake()
        (a direct state.set_mode). Restore the hardware anyway."""
        try:
            if data.get('previous_mode') == 'sleep' and self._state == ASLEEP:
                logger.warning(f"[SLEEP] mode left sleep outside wake() ({data.get('reason')}) — restoring hardware")
                threading.Thread(target=self.wake, kwargs={'source': 'mode_change', 'greet': False},
                                 daemon=True, name='SleepWakeNet').start()
        except Exception as e:
            logger.error(f"[SLEEP] mode_change net: {e}")

    # ------------------------------------------------------------ status loop
    def _start_status_loop(self) -> None:
        self._status_stop.clear()
        self._status_thread = threading.Thread(target=self._status_loop, daemon=True, name='SleepStatus')
        self._status_thread.start()

    def _stop_status_loop(self) -> None:
        self._status_stop.set()

    def _status_loop(self) -> None:
        stop = self._status_stop
        while not stop.wait(STATUS_INTERVAL_S):
            if self._state != ASLEEP:
                return
            self._emit_status()

    def _emit_status(self) -> None:
        try:
            from services.cloud.behavior_events import get_behavior_event_service
            st = get_behavior_event_service().emit_status_now()
            logger.info(f"[SLEEP] status report: {st} power={self.read_power()}")
        except Exception as e:
            logger.warning(f"[SLEEP] status report failed: {e}")

    # ------------------------------------------------------------ battery
    def battery_shutdown(self, reason: str, data: Optional[Dict[str, Any]] = None) -> bool:
        """Battery reached the shutdown threshold while asleep: do the normal
        real shutdown (OS halt; the systemd shutdown hook then cuts power the
        way it does after a button press). Returns True when it was started."""
        if self._state not in (ENTERING, ASLEEP):
            return False
        logger.critical(f"[SLEEP] battery at shutdown threshold while asleep ({reason} {data}) — real shutdown")
        try:
            subprocess.Popen(['sudo', '-n', 'shutdown', '-h', 'now'])
            return True
        except Exception as e:
            logger.error(f"[SLEEP] shutdown command failed: {e}")
            return False


# ---------------------------------------------------------------- module helpers

def hailo_pci_state() -> Optional[str]:
    """PCI power state of the Hailo chip (D0 = on, D3hot = powered down by the driver)."""
    try:
        for dev in glob.glob('/sys/bus/pci/devices/*'):
            with open(f'{dev}/vendor') as f:
                if f.read().strip() != HAILO_PCI_VENDOR:
                    continue
            with open(f'{dev}/power_state') as f:
                return f.read().strip()
    except Exception:
        pass
    return None


def _governor_paths() -> List[str]:
    return sorted(glob.glob('/sys/devices/system/cpu/cpufreq/policy*/scaling_governor'))


def read_governor() -> Optional[str]:
    try:
        with open(_governor_paths()[0]) as f:
            return f.read().strip()
    except Exception:
        return None


def set_governor(name: str) -> bool:
    paths = _governor_paths()
    if not paths:
        return False
    try:
        r = subprocess.run(['sudo', '-n', 'tee'] + paths, input=name + '\n', text=True,
                           capture_output=True, timeout=5)
        if r.returncode != 0:
            logger.error(f"[SLEEP] governor write failed: {r.stderr.strip()}")
    except Exception as e:
        logger.error(f"[SLEEP] governor write failed: {e}")
    return read_governor() == name


_MUTE_CONTROLS = ('Speaker', 'PCM', 'Master', 'Headphone')


def _speaker_card() -> Optional[str]:
    try:
        from services.media.volume_manager import _detect_usb_audio_card
        return _detect_usb_audio_card()
    except Exception:
        return None


def _set_speaker_mute(mute: bool) -> bool:
    """Mute/unmute the speaker at the ALSA hardware mixer; the volume level
    itself (and its saved value) is not touched."""
    card = _speaker_card()
    if card is None:
        return False
    for ctl in _MUTE_CONTROLS:
        try:
            r = subprocess.run(['amixer', '-c', str(card), 'sset', ctl, 'mute' if mute else 'unmute'],
                               capture_output=True, text=True, timeout=3)
            if r.returncode == 0 and ('[off]' in r.stdout) == mute:
                return True
        except Exception:
            continue
    return False


def _speaker_muted() -> Optional[bool]:
    card = _speaker_card()
    if card is None:
        return None
    for ctl in _MUTE_CONTROLS:
        try:
            r = subprocess.run(['amixer', '-c', str(card), 'sget', ctl],
                               capture_output=True, text=True, timeout=3)
            if r.returncode == 0 and ('[on]' in r.stdout or '[off]' in r.stdout):
                return '[off]' in r.stdout
        except Exception:
            continue
    return None


_instance: Optional[SleepManager] = None
_instance_lock = threading.Lock()


def get_sleep_manager() -> SleepManager:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = SleepManager()
        return _instance


def ensure_awake(source: str = 'command') -> bool:
    """Cheap module-level gate for action paths (no-op unless asleep)."""
    mgr = _instance
    if mgr is None or mgr.state == AWAKE:
        return True
    return mgr.ensure_awake(source)


def is_asleep() -> bool:
    mgr = _instance
    return bool(mgr is not None and mgr.state in (ENTERING, ASLEEP))
