"""BehaviorEventService — derives the CONTRACT v2 event stream from the bus.

CONTRACT v2 FINAL (2026-11-06). Subscribes to the internal bus (audio /
system / reward / vision) and turns the robot's existing signals into the
structured event rows, emitted through RelayEventEmitter (-> outbox ->
POST /events). It only LISTENS; it never changes perception or Silent
Guardian (SG) behaviour.

Event `data` per contract:
  bark_episode  {start, end, bark_count, peak_db, severity: low|med|high}
                severity: low <3 barks, med 3-9, high >=10 barks or >60 s
  intervention  {time, sg_level, kind: sound|voice|treat|music|approach,
                 episode_id, result: quiet|no_quiet}
  settle        {time, seconds_since_last_intervention, calm_seconds}
  pacing        {start, end, reversals}   (best-effort heuristic, kill switch)
  treat         {time, source: auto|app|connector|agent|schedule, command_id}
  music         {start, end, track_class}
  mission_run   {mission_id, start, end, result: success|fail|aborted}
  status        {battery, mode, treats_remaining, docked,
                 light_level: mean camera frame brightness 0-255}
  departure /   {time, method: "app"} — only on app or relay instruction
  arrival       (mark_departure / mark_arrival), never from sensing.
"""
import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from core.bus import get_bus, publish_system_event

logger = logging.getLogger('BehaviorEvents')

CONFIG: Dict[str, Any] = {
    'bark_gap_s': 30.0,
    'bark_episode_min_s': 0.0,          # sg_sensitivity.min_episode_s
    'settle_calm_s': 300.0,
    'intervention_outcome_timeout_s': 900.0,
    'pacing_enabled': True,
    'pacing_window_s': 60.0,
    'pacing_min_reversals': 6,
    'pacing_min_travel_frames': 3.0,
    'pacing_reversal_min_dx': 0.10,     # fraction of frame width
    'pacing_end_quiet_s': 20.0,
    'status_period_s': 60.0,
    'status_battery_delta': 5,
    'tick_s': 1.0,
}

# dispense_treat(reason=...) -> treat.data.source
_SOURCE_MAP = {
    'manual': 'app', 'app': 'app', 'cloud_command': 'app', 'websocket_command': 'app',
    'xbox': 'app', 'test': 'app', 'manual_pilot': 'app', 'owner': 'app',
    'agent': 'agent', 'connector_trick': 'agent',
    'connector': 'connector', 'ifttt': 'connector', 'webhook': 'connector',
    'schedule': 'schedule', 'mission': 'schedule',
}


def treat_source(reason: Optional[str]) -> str:
    r = (reason or '').lower()
    if r in _SOURCE_MAP:
        return _SOURCE_MAP[r]
    for prefix, src in (('connector', 'connector'), ('agent', 'agent'), ('schedule', 'schedule')):
        if r.startswith(prefix):
            return src
    return 'auto'


def bark_severity(bark_count: int, duration_s: float) -> str:
    if bark_count >= 10 or duration_s > 60:
        return 'high'
    if bark_count >= 3:
        return 'med'
    return 'low'


def _iso(ts: float) -> str:
    from services.cloud.event_emitter import iso_z
    return iso_z(ts)


class BehaviorEventService:
    def __init__(self, emitter=None, config: Optional[Dict[str, Any]] = None):
        self.config = dict(CONFIG)
        if config:
            self.config.update(config)
        self._emitter = emitter
        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.bus = get_bus()

        # bark episode (id pre-generated so interventions can reference it)
        self._ep_id: Optional[str] = None
        self._ep_start: Optional[float] = None
        self._ep_last: float = 0.0
        self._ep_peak: float = -120.0
        self._ep_barks: int = 0
        self._ep_dogs: Dict[str, int] = {}
        self._ep_dog_names: Dict[str, Optional[str]] = {}
        self._last_episode_id: Optional[str] = None
        self._last_gated_bark: float = 0.0

        # intervention / settle
        self._iv_open: Optional[Dict[str, Any]] = None
        self._iv_last_outcome_time: Optional[float] = None
        self._settle_emitted = True
        self._settle_dog: Optional[str] = None

        # pacing: dog_id -> deque[(t, cx)]
        self._pace_hist: Dict[str, Deque[Tuple[float, float]]] = {}
        self._pace_open: Dict[str, Dict[str, Any]] = {}
        self._pace_last_seen: Dict[str, float] = {}
        self._frame_w: float = 1280.0

        # music
        self._music_start: Optional[float] = None
        self._music_track: Optional[str] = None
        self._music_class: str = 'other'
        self._music_owner_hint_until: float = 0.0

        # missions
        self._mission_start: Dict[Any, Tuple[float, Optional[str]]] = {}

        # status
        self._last_status_t: float = 0.0
        self._last_status: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------- lifecycle

    @property
    def emitter(self):
        if self._emitter is None:
            from services.cloud.event_emitter import get_event_emitter
            self._emitter = get_event_emitter()
        return self._emitter

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        self.bus.subscribe('audio', self._on_audio)
        self.bus.subscribe('system', self._on_system)
        self.bus.subscribe('reward', self._on_reward)
        self.bus.subscribe('vision', self._on_vision)
        self._thread = threading.Thread(target=self._loop, daemon=True, name='BehaviorEvents')
        self._thread.start()
        logger.info("[BEHAVIOR-EVENTS] started")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        for cat, cb in (('audio', self._on_audio), ('system', self._on_system),
                        ('reward', self._on_reward), ('vision', self._on_vision)):
            try:
                self.bus.unsubscribe(cat, cb)
            except Exception:
                pass
        now = time.time()
        try:
            self._close_bark_episode(now)
            for dog_id in list(self._pace_open):
                self._close_pacing(dog_id, now)
            if self._music_start is not None:
                self._close_music(now)
        except Exception as e:
            logger.debug(f"close-on-stop failed: {e}")
        logger.info("[BEHAVIOR-EVENTS] stopped")

    def _loop(self) -> None:
        while self._running:
            try:
                self.tick(time.time())
            except Exception as e:
                logger.error(f"[BEHAVIOR-EVENTS] tick error: {e}")
            time.sleep(self.config['tick_s'])

    def tick(self, now: float) -> None:
        """Time-driven closes (public so tests can drive a fake clock)."""
        with self._lock:
            if self._ep_start is not None and now - self._ep_last >= self.config['bark_gap_s']:
                self._close_bark_episode(now)
            if (self._iv_open is not None
                    and now - self._iv_open['time'] >= self.config['intervention_outcome_timeout_s']):
                self._emit_intervention(now, quiet=False, treat_given=False, music_played=False)
            if (not self._settle_emitted and self._iv_last_outcome_time is not None
                    and now - max(self._iv_last_outcome_time, self._last_gated_bark)
                    >= self.config['settle_calm_s']):
                self._settle_emitted = True
                self.emitter.emit('settle', {
                    'time': _iso(now),
                    'seconds_since_last_intervention': int(now - self._iv_last_outcome_time),
                    'calm_seconds': int(self.config['settle_calm_s']),
                }, dog_id=self._settle_dog)
            for dog_id in list(self._pace_open):
                if now - self._pace_last_seen.get(dog_id, 0) >= self.config['pacing_end_quiet_s']:
                    self._close_pacing(dog_id, now)
            if now - self._last_status_t >= self.config['status_period_s']:
                self._emit_status(now, force=True)

    # ------------------------------------------------------------- bus: audio

    def _sg_loudness_gate(self) -> float:
        try:
            from modes.silent_guardian import get_silent_guardian_mode
            return float(get_silent_guardian_mode().config.get('bark_detection', {})
                         .get('loudness_threshold_db', -28))
        except Exception:
            return -28.0

    def _on_audio(self, event) -> None:
        try:
            if event.subtype == 'bark_detected':
                self.on_bark(event.data or {}, getattr(event, 'timestamp', None) or time.time())
            elif event.subtype == 'audio_state':
                self.on_audio_state(event.data or {}, getattr(event, 'timestamp', None) or time.time())
        except Exception as e:
            logger.error(f"[BEHAVIOR-EVENTS] audio handler: {e}")

    def on_bark(self, data: Dict[str, Any], now: float) -> None:
        loud = data.get('loudness_db')
        try:
            loud = float(loud) if loud is not None else -120.0
        except (TypeError, ValueError):
            loud = -120.0
        if loud < self._sg_loudness_gate():
            return
        with self._lock:
            self._last_gated_bark = now
            if self._ep_start is None:
                from core.data.ids import uuid7
                self._ep_id = uuid7()
                self._ep_start, self._ep_peak, self._ep_barks = now, loud, 0
                self._ep_dogs, self._ep_dog_names = {}, {}
            self._ep_last = now
            self._ep_peak = max(self._ep_peak, loud)
            self._ep_barks += 1
            d = data.get('dog_id')
            if d:
                self._ep_dogs[d] = self._ep_dogs.get(d, 0) + 1
                self._ep_dog_names[d] = data.get('dog_name')

    def current_episode_id(self) -> Optional[str]:
        return self._ep_id if self._ep_start is not None else self._last_episode_id

    def _close_bark_episode(self, now: float) -> None:
        if self._ep_start is None:
            return
        duration = max(0.0, self._ep_last - self._ep_start)
        dog_id = max(self._ep_dogs, key=self._ep_dogs.get) if self._ep_dogs else None
        ep_id = self._ep_id
        data = {'start': _iso(self._ep_start), 'end': _iso(self._ep_last),
                'bark_count': self._ep_barks, 'peak_db': round(self._ep_peak, 1),
                'severity': bark_severity(self._ep_barks, duration),
                'duration_s': round(duration, 1)}
        self._ep_start = None
        self._last_episode_id = ep_id
        if duration < self.config.get('bark_episode_min_s', 0) and self._ep_barks < 3:
            logger.debug(f"[BEHAVIOR-EVENTS] bark episode below min_episode_s; not emitted")
            return
        self.emitter.emit('bark_episode', data, dog_id=dog_id,
                          dog_name=self._ep_dog_names.get(dog_id) if dog_id else None,
                          event_id=ep_id)

    def on_audio_state(self, data: Dict[str, Any], now: float) -> None:
        playing = bool(data.get('playing')) or data.get('state') == 'playing'
        track = data.get('track')
        with self._lock:
            if playing and self._music_start is None:
                self._music_start, self._music_track = now, track
                self._music_class = self._classify_track(track, now)
            elif playing and self._music_start is not None and track and track != self._music_track:
                self._close_music(now)
                self._music_start, self._music_track = now, track
                self._music_class = self._classify_track(track, now)
            elif not playing and self._music_start is not None:
                self._close_music(now)

    def note_owner_play(self) -> None:
        """Called by app/WS play commands so the next music start is 'owner'."""
        with self._lock:
            self._music_owner_hint_until = time.time() + 5.0

    def _classify_track(self, track: Optional[str], now: float) -> str:
        try:
            from modes.silent_guardian import get_silent_guardian_mode
            sg = get_silent_guardian_mode()
            if getattr(sg, 'calming_music_playing', False):
                return 'calming'
            playlist, _ = sg._get_calming_playlist()
            if track and any(str(track).endswith(str(t).split('/')[-1]) for t in playlist):
                return 'calming'
        except Exception:
            pass
        if now <= self._music_owner_hint_until:
            return 'owner'
        return 'other'

    def _close_music(self, now: float) -> None:
        if self._music_start is None:
            return
        data = {'start': _iso(self._music_start), 'end': _iso(now),
                'track_class': self._music_class, 'track': self._music_track}
        self._music_start = None
        self.emitter.emit('music', data)

    # ------------------------------------------------------------- bus: system

    def _on_system(self, event) -> None:
        try:
            st, d, now = event.subtype, event.data or {}, getattr(event, 'timestamp', None) or time.time()
            if st == 'sg_escalation':
                self.on_sg_escalation(d, now)
            elif st == 'sg_intervention_record':
                self.on_sg_intervention_record(d, now)
            elif st == 'mission.started':
                with self._lock:
                    self._mission_start[d.get('mission_id')] = (now, d.get('dog_id'))
            elif st in ('mission.completed', 'mission.stopped'):
                self.on_mission_end(st, d, now)
            elif st == 'owner_departure':
                self._on_owner_departure(d)
        except Exception as e:
            logger.error(f"[BEHAVIOR-EVENTS] system handler: {e}")

    def _open_iv(self, data: Dict[str, Any], now: float) -> None:
        self._iv_open = {'time': now, 'level': int(data.get('escalation_level') or 1),
                         'dog_id': data.get('dog_id'), 'dog_name': data.get('dog_name'),
                         'episode_id': self.current_episode_id()}

    def on_sg_escalation(self, data: Dict[str, Any], now: float) -> None:
        with self._lock:
            if self._iv_open is not None:
                self._emit_intervention(now, quiet=False, treat_given=False, music_played=False)
            self._open_iv(data, now)

    def on_sg_intervention_record(self, data: Dict[str, Any], now: float) -> None:
        phase = data.get('phase')
        with self._lock:
            if phase == 'triggered':
                if self._iv_open is None:
                    self._open_iv(data, now)
                return
            if phase == 'outcome':
                if self._iv_open is None:
                    self._open_iv(data, now)
                self._emit_intervention(now, quiet=bool(data.get('quiet_achieved')),
                                        treat_given=bool(data.get('treat_given')),
                                        music_played=bool(data.get('music_played')))

    @staticmethod
    def intervention_kind(level: int, treat_given: bool, music_played: bool) -> str:
        """SG levels: 1 = quiet/no sound cue, 2 = voice cue, 3 = movement toward
        the dog (approach), 4 = calming music; a treat on quiet = treat."""
        if treat_given:
            return 'treat'
        if music_played or level >= 4:
            return 'music'
        if level == 3:
            return 'approach'
        if level == 2:
            return 'voice'
        return 'sound'

    def _emit_intervention(self, now: float, quiet: bool, treat_given: bool, music_played: bool) -> None:
        iv = self._iv_open
        if iv is None:
            return
        level = iv['level']
        self._iv_open = None
        self._iv_last_outcome_time = now
        self._settle_emitted = False
        self._settle_dog = iv.get('dog_id')
        self.emitter.emit('intervention', {
            'time': _iso(iv['time']), 'sg_level': level,
            'kind': self.intervention_kind(level, treat_given, music_played),
            'episode_id': iv.get('episode_id'),
            'result': 'quiet' if quiet else 'no_quiet',
        }, dog_id=iv.get('dog_id'), dog_name=iv.get('dog_name'))

    def on_mission_end(self, subtype: str, data: Dict[str, Any], now: float) -> None:
        mid = data.get('mission_id')
        with self._lock:
            start, dog_id = self._mission_start.pop(mid, (None, data.get('dog_id')))
        if subtype == 'mission.stopped':
            result = 'aborted'
        else:
            result = 'success' if data.get('success', True) else 'fail'
        self.emitter.emit('mission_run', {
            'mission_id': data.get('mission_name') or mid,
            'run_id': mid,
            'start': _iso(start) if start else None,
            'end': _iso(now),
            'result': result,
            'rewards_given': data.get('rewards_given'),
            'reason': data.get('reason'),
        }, dog_id=dog_id)

    # ------------------------------------------------------------- bus: reward

    def _on_reward(self, event) -> None:
        try:
            if event.subtype == 'treat_dispensed':
                d = event.data or {}
                now = d.get('timestamp') or getattr(event, 'timestamp', None) or time.time()
                self.emitter.emit('treat', {
                    'time': _iso(float(now)),
                    'source': treat_source(d.get('reason')),
                    'command_id': d.get('command_id'),
                    'reason': d.get('reason'),
                    'behavior': d.get('behavior') or None,
                }, dog_id=d.get('dog_id'), dog_name=d.get('dog_name'), id_method=d.get('id_method'))
                self._emit_status(time.time(), force=False)
        except Exception as e:
            logger.error(f"[BEHAVIOR-EVENTS] reward handler: {e}")

    # ------------------------------------------------------------- bus: vision

    def _on_vision(self, event) -> None:
        if event.subtype != 'dog_detected' or not self.config.get('pacing_enabled', True):
            return
        try:
            d = event.data or {}
            bbox = d.get('bbox')
            if not bbox or len(bbox) < 4:
                return
            cx = (float(bbox[0]) + float(bbox[2])) / 2.0
            self.on_detection(d.get('dog_id') or 'dog', cx, float(d.get('timestamp') or time.time()),
                              dog_name=d.get('dog_name'))
        except Exception as e:
            logger.debug(f"[BEHAVIOR-EVENTS] vision handler: {e}")

    def _frame_width(self) -> float:
        try:
            from services.perception.detector import get_detector_service
            res = getattr(get_detector_service(), '_current_resolution', None)
            if res and res[0]:
                return float(res[0])
        except Exception:
            pass
        return self._frame_w

    def on_detection(self, dog_id: str, cx_px: float, now: float, dog_name: Optional[str] = None,
                     frame_w: Optional[float] = None) -> None:
        w = frame_w or self._frame_width()
        cx = cx_px / w if w else cx_px
        with self._lock:
            hist = self._pace_hist.setdefault(dog_id, deque())
            hist.append((now, cx))
            cutoff = now - self.config['pacing_window_s']
            while hist and hist[0][0] < cutoff:
                hist.popleft()
            self._pace_last_seen[dog_id] = now
            reversals, travel = self._pace_stats(hist)
            if dog_id not in self._pace_open:
                if (reversals >= self.config['pacing_min_reversals']
                        and travel >= self.config['pacing_min_travel_frames']):
                    self._pace_open[dog_id] = {'start': hist[0][0], 'reversals': reversals,
                                               'dog_name': dog_name}
                    logger.info(f"[BEHAVIOR-EVENTS] pacing open dog={dog_id} rev={reversals} travel={travel:.1f}")
            else:
                self._pace_open[dog_id]['reversals'] = max(self._pace_open[dog_id]['reversals'], reversals)
                if reversals < 2:
                    self._close_pacing(dog_id, now)

    def _pace_stats(self, hist) -> Tuple[int, float]:
        min_dx = self.config['pacing_reversal_min_dx']
        reversals, travel = 0, 0.0
        last_dir, seg = 0, 0.0
        pts = list(hist)
        for (t0, x0), (t1, x1) in zip(pts, pts[1:]):
            dx = x1 - x0
            travel += abs(dx)
            d = 1 if dx > 0 else (-1 if dx < 0 else 0)
            if d == 0:
                continue
            if d == last_dir or last_dir == 0:
                seg += abs(dx)
            else:
                if seg >= min_dx:
                    reversals += 1
                seg = abs(dx)
            last_dir = d
        return reversals, travel

    def _close_pacing(self, dog_id: str, now: float) -> None:
        info = self._pace_open.pop(dog_id, None)
        self._pace_hist.pop(dog_id, None)
        if not info:
            return
        self.emitter.emit('pacing', {'start': _iso(info['start']), 'end': _iso(now),
                                     'reversals': info['reversals']},
                          dog_id=dog_id, dog_name=info.get('dog_name'))

    # ------------------------------------------------------------- status

    @staticmethod
    def frame_brightness() -> Optional[int]:
        """Mean camera frame brightness 0-255 (contract light_level)."""
        try:
            from services.perception.detector import get_detector_service
            frame = get_detector_service().get_last_frame()
            if frame is None:
                return None
            import numpy as np
            if frame.ndim == 3:
                # BGR -> luma
                b, g, r = frame[..., 0], frame[..., 1], frame[..., 2]
                lum = 0.114 * b.mean() + 0.587 * g.mean() + 0.299 * r.mean()
            else:
                lum = float(np.mean(frame))
            return int(max(0, min(255, round(float(lum)))))
        except Exception:
            return None

    def read_status(self) -> Dict[str, Any]:
        battery, docked, treats, mode = None, None, None, None
        try:
            from services.power.battery_monitor import get_battery_monitor
            bm = get_battery_monitor()
            battery = int(bm.get_percentage())
            docked = bool(getattr(bm, 'charging_detected', False))
        except Exception:
            pass
        try:
            from services.reward.dispenser import get_dispenser_service
            treats = int(get_dispenser_service().treats_remaining)
        except Exception:
            pass
        try:
            from core.state import get_state
            mode = get_state().get_mode().value
        except Exception:
            pass
        return {'battery': battery, 'mode': mode, 'treats_remaining': treats, 'docked': docked,
                'light_level': self.frame_brightness()}

    def _emit_status(self, now: float, force: bool) -> None:
        st = self.read_status()
        prev = self._last_status or {}
        changed = (self._last_status is None
                   or st['docked'] != prev.get('docked')
                   or st['treats_remaining'] != prev.get('treats_remaining')
                   or st['mode'] != prev.get('mode')
                   or (st['battery'] is not None and prev.get('battery') is not None
                       and abs(st['battery'] - prev['battery']) >= self.config['status_battery_delta']))
        if not (force or changed):
            return
        self._last_status, self._last_status_t = st, now
        self.emitter.emit('status', st, resolve_dog=False)

    def emit_status_now(self) -> Dict[str, Any]:
        self._emit_status(time.time(), force=True)
        return dict(self._last_status or {})

    # ------------------------------------------------------------- departure / arrival

    def mark_departure(self, dog_id: Optional[str] = None, method: str = 'app') -> Dict[str, Any]:
        now = time.time()
        ev = self.emitter.emit('departure', {'time': _iso(now), 'method': method}, dog_id=dog_id)
        publish_system_event('owner_departure', {'time': now, 'method': method,
                                                 'dog_id': ev.get('dog_id')}, 'behavior_events')
        return ev

    def mark_arrival(self, dog_id: Optional[str] = None, method: str = 'app') -> Dict[str, Any]:
        now = time.time()
        ev = self.emitter.emit('arrival', {'time': _iso(now), 'method': method}, dog_id=dog_id)
        publish_system_event('owner_arrival', {'time': now, 'method': method,
                                               'dog_id': ev.get('dog_id')}, 'behavior_events')
        return ev

    def _on_owner_departure(self, data: Dict[str, Any]) -> None:
        """music_plan.on_departure {track, delay_s, duration_min}."""
        try:
            from services.cloud.remote_settings import get_remote_settings
            plan = get_remote_settings().get_music_plan() or {}
            od = plan.get('on_departure') if isinstance(plan, dict) else None
            if plan.get('enabled', True) and isinstance(od, dict) and od:
                threading.Thread(target=self._play_departure_music, args=(plan, od),
                                 daemon=True, name='DepartureMusic').start()
        except Exception as e:
            logger.debug(f"departure music: {e}")

    def _play_departure_music(self, plan: Dict[str, Any], od: Dict[str, Any]) -> None:
        try:
            delay = float(od.get('delay_s') or 0)
            if delay > 0:
                time.sleep(delay)
            if not self._running:
                return
            from modes.silent_guardian import get_silent_guardian_mode
            sg = get_silent_guardian_mode()
            if getattr(sg, 'calming_music_playing', False):
                return
            import os
            import random
            from services.media.usb_audio import get_usb_audio_service
            audio = get_usb_audio_service()
            playlist, base = sg._get_calming_playlist()
            tracks = [os.path.join(base, t) for t in playlist if os.path.exists(os.path.join(base, t))]
            track = od.get('track')
            if track:
                cand = track if os.path.isabs(str(track)) else os.path.join(base, str(track))
                if os.path.exists(cand):
                    tracks = [cand]
                else:
                    try:
                        from services.media.sound_registry import get_sound_registry
                        p = get_sound_registry().resolve(str(track))
                        if p:
                            tracks = [p]
                    except Exception:
                        pass
            prev_vol = None
            if plan.get('volume') is not None:
                from services.media.volume_manager import get_volume_manager
                vm = get_volume_manager()
                prev_vol = vm.get_volume()
                vm.set_volume(int(plan['volume']))
            minutes = float(od.get('duration_min') or 10)
            deadline = time.time() + minutes * 60
            while tracks and time.time() < deadline and self._running:
                audio.play_file(random.choice(tracks))
                time.sleep(3)
                while audio.is_busy() and time.time() < deadline and self._running:
                    time.sleep(1)
            if time.time() >= deadline and audio.is_busy():
                audio.stop()
            if prev_vol is not None:
                from services.media.volume_manager import get_volume_manager
                get_volume_manager().set_volume(prev_vol)
        except Exception as e:
            logger.error(f"[BEHAVIOR-EVENTS] departure music failed: {e}")


_instance: Optional[BehaviorEventService] = None
_instance_lock = threading.Lock()


def get_behavior_event_service() -> BehaviorEventService:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = BehaviorEventService()
        return _instance
