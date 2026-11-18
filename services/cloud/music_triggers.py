#!/usr/bin/env python3
"""
services/cloud/music_triggers.py - the robot runs every music_plan trigger
(relay brief 2026-11-18; the relay no longer starts music itself).

music_plan (settings key, fetched on connect + settings_updated):
  {"enabled": true, "volume": 40,
   "on_mode_start":   {"mode": "silent_guardian", "delay_s": 60, "track": "...", "duration_min": 30} | null,
   "after_treat":     {"every_n": 3, "minutes": 10, "track": "...", "duration_min": 20} | null,
   "on_bark_episode": {"min_severity": "med", "track": "...", "duration_min": 15} | null,
   "at_time":         [{"time": "13:00", "track": "...", "duration_min": 20}]}

Rules
  * enabled:false silences every trigger; volume applies to all of them (a
    temporary level: the owner's saved volume comes back afterwards).
  * on_mode_start: delay_s after the robot ENTERS mode; skipped if the mode
    ended before the delay was up.
  * after_treat: every_n treats inside a rolling `minutes` window, any source;
    then the count resets.
  * on_bark_episode: as soon as a running episode reaches min_severity
    (low < med < high), once per episode; never waits for the episode to end.
  * at_time: daily at HH:MM in the dog's timezone setting (<= 12 slots).
  * quiet hours silence music: nothing starts inside them and a running
    session stops when they begin.
  * music already playing (a trigger session, Silent Guardian's calming music,
    or anything else on the music channel): the new trigger is dropped, never
    stacked or restarted.
  * events: music {action:"start", track, trigger} and, on stop,
    music {action:"stop", track, trigger} with duration_s.
  * on_departure and schedule are gone and ignored.

Silent Guardian's own escalation music is not touched by this module.
"""
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any, Callable, Deque, Dict, Optional

logger = logging.getLogger('MusicTriggers')

SEVERITY_RANK = {'low': 0, 'med': 1, 'medium': 1, 'high': 2}
TRIGGERS = ('on_mode_start', 'after_treat', 'on_bark_episode', 'at_time')
BUSY_WAIT_S = 20.0          # a voice clip may be speaking when a trigger fires: wait this long for it
SONGS_DIR = '/home/morgan/dogbot/VOICEMP3/songs'


def _hhmm(value) -> Optional[int]:
    try:
        hh, mm = str(value).split(':')[:2]
        hh, mm = int(hh), int(mm)
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            return hh * 60 + mm
    except Exception:
        pass
    return None


def in_quiet_hours(quiet, minute_of_day: int) -> bool:
    """quiet_hours {enabled, start, end} (or a list of them); a window may cross midnight."""
    windows = quiet if isinstance(quiet, list) else [quiet]
    for w in windows:
        if not isinstance(w, dict) or not w.get('enabled', True):
            continue
        s, e = _hhmm(w.get('start')), _hhmm(w.get('end'))
        if s is None or e is None or s == e:
            continue
        if (s <= minute_of_day < e) if s < e else (minute_of_day >= s or minute_of_day < e):
            return True
    return False


class MusicTriggers:
    def __init__(self, plan_source: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
                 quiet_source: Optional[Callable[[], Any]] = None,
                 tz_source: Optional[Callable[[], Optional[str]]] = None,
                 audio=None, emit: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
                 mode_source: Optional[Callable[[], str]] = None,
                 resolve_track: Optional[Callable[[str], Optional[str]]] = None,
                 other_music_playing: Optional[Callable[[], bool]] = None,
                 volume=None, clock: Callable[[], float] = time.time):
        self._plan_source = plan_source or self._default_plan
        self._quiet_source = quiet_source or self._default_quiet
        self._tz_source = tz_source or self._default_tz
        self._audio = audio
        self._emit = emit or self._default_emit
        self._mode_source = mode_source or self._default_mode
        self._resolve = resolve_track or self._default_resolve
        self._other_music = other_music_playing or self._default_other_music
        self._volume = volume
        self._clock = clock
        self._lock = threading.Lock()
        self._running = False
        self._stop_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[Dict[str, Any]] = None     # the one music session a trigger started
        self._mode_gen = 0                                  # bumps on every mode switch
        self._treat_times: Deque[float] = deque(maxlen=64)
        self._bark_fired_episode: Optional[str] = None
        self._at_time_fired: Dict[str, str] = {}            # "HH:MM" -> local date it last fired

    # ---------------------------------------------------------- defaults
    @staticmethod
    def _default_plan():
        from services.cloud.remote_settings import get_remote_settings
        return get_remote_settings().get_music_plan()

    @staticmethod
    def _default_quiet():
        from services.cloud.remote_settings import get_remote_settings
        return get_remote_settings().get_quiet_hours()

    @staticmethod
    def _default_tz():
        from services.cloud.remote_settings import get_remote_settings
        return get_remote_settings().get_timezone()

    @staticmethod
    def _default_mode() -> str:
        from core.state import get_state
        return get_state().get_mode().value

    @staticmethod
    def _default_emit(event_type: str, data: Dict[str, Any]):
        from services.cloud.event_emitter import get_event_emitter
        return get_event_emitter().emit(event_type, data)

    @staticmethod
    def _default_resolve(track: str) -> Optional[str]:
        try:
            from services.media.sound_registry import get_sound_registry
            p = get_sound_registry().resolve(str(track))
            if p and os.path.exists(p):
                return p
        except Exception as e:
            logger.debug(f"sound registry lookup failed: {e}")
        for cand in (str(track), os.path.join(SONGS_DIR, str(track)), os.path.join(SONGS_DIR, f"{track}.mp3")):
            if os.path.isabs(cand) and os.path.exists(cand):
                return cand
        return None

    @staticmethod
    def _default_other_music() -> bool:
        try:
            import modes.silent_guardian as sg
            inst = getattr(sg, '_silent_guardian_instance', None)     # never constructs SG
            return bool(inst is not None and getattr(inst, 'calming_music_playing', False))
        except Exception:
            return False

    def _get_audio(self):
        if self._audio is None:
            from services.media.usb_audio import get_usb_audio_service
            self._audio = get_usb_audio_service()
        return self._audio

    def _get_volume(self):
        if self._volume is None:
            from services.media.volume_manager import get_volume_manager
            self._volume = get_volume_manager()
        return self._volume

    # ---------------------------------------------------------- lifecycle
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
        self._stop_evt.clear()
        try:
            from core.state import get_state
            get_state().subscribe('mode_change', self._on_state_mode_change)
        except Exception as e:
            logger.warning(f"[MUSIC] mode listener unavailable: {e}")
        try:
            from core.bus import get_bus
            get_bus().subscribe('reward', self._on_reward)
        except Exception as e:
            logger.warning(f"[MUSIC] reward listener unavailable: {e}")
        self._thread = threading.Thread(target=self._loop, daemon=True, name='MusicTriggers')
        self._thread.start()
        logger.info("[MUSIC] trigger service started")

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        self._stop_evt.set()
        try:
            from core.bus import get_bus
            get_bus().unsubscribe('reward', self._on_reward)
        except Exception:
            pass
        self._end_session('service_stop')

    def _loop(self) -> None:
        while not self._stop_evt.wait(5.0):
            try:
                self.tick()
            except Exception as e:
                logger.error(f"[MUSIC] tick failed: {e}")

    # ---------------------------------------------------------- plan helpers
    def _plan(self) -> Dict[str, Any]:
        try:
            p = self._plan_source()
        except Exception:
            p = None
        return p if isinstance(p, dict) else {}

    def _trigger_cfg(self, name: str):
        plan = self._plan()
        if not plan.get('enabled', True):
            return None
        cfg = plan.get(name)
        if name == 'at_time':
            return [s for s in cfg if isinstance(s, dict)] if isinstance(cfg, list) else []
        return cfg if isinstance(cfg, dict) and cfg else None

    def _local_now(self) -> datetime:
        tz = None
        try:
            name = self._tz_source()
            if name:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo(name)
        except Exception:
            tz = None
        return datetime.fromtimestamp(self._clock(), tz) if tz else datetime.fromtimestamp(self._clock())

    def _quiet_now(self) -> bool:
        try:
            now = self._local_now()
            return in_quiet_hours(self._quiet_source(), now.hour * 60 + now.minute)
        except Exception:
            return False

    # ---------------------------------------------------------- trigger inputs
    def _on_state_mode_change(self, data: Dict[str, Any]) -> None:
        try:
            self.on_mode_change(str(data.get('new_mode') or ''))
        except Exception as e:
            logger.error(f"[MUSIC] mode handler: {e}")

    def on_mode_change(self, new_mode: str) -> None:
        with self._lock:
            self._mode_gen += 1
            gen = self._mode_gen
        cfg = self._trigger_cfg('on_mode_start')
        if not cfg or str(cfg.get('mode') or '') != new_mode:
            return
        delay = max(0.0, min(float(cfg.get('delay_s') or 0), 3600.0))
        logger.info(f"[MUSIC] on_mode_start armed: {new_mode} +{delay:.0f}s")

        def _fire():
            if self._stop_evt.wait(delay):
                return
            with self._lock:
                still = gen == self._mode_gen
            if not still or self._mode_source() != new_mode:
                logger.info(f"[MUSIC] on_mode_start skipped: {new_mode} ended before the delay was up")
                return
            cur = self._trigger_cfg('on_mode_start')        # plan may have changed while waiting
            if cur and str(cur.get('mode') or '') == new_mode:
                self.fire('on_mode_start', cur)
        threading.Thread(target=_fire, daemon=True, name='MusicOnModeStart').start()

    def _on_reward(self, event) -> None:
        try:
            if event.subtype == 'treat_dispensed':
                d = event.data or {}
                self.on_treat(float(d.get('timestamp') or getattr(event, 'timestamp', None) or self._clock()))
        except Exception as e:
            logger.error(f"[MUSIC] reward handler: {e}")

    def on_treat(self, ts: Optional[float] = None) -> None:
        ts = self._clock() if ts is None else ts
        cfg = self._trigger_cfg('after_treat')
        with self._lock:
            self._treat_times.append(ts)
            if not cfg:
                return
            n = max(1, min(int(cfg.get('every_n') or 1), 50))
            window = max(1.0, min(float(cfg.get('minutes') or 1), 240.0)) * 60.0
            recent = [t for t in self._treat_times if ts - t <= window]
            if len(recent) < n:
                return
            self._treat_times.clear()                        # reset the count
        threading.Thread(target=self.fire, args=('after_treat', cfg), daemon=True,
                         name='MusicAfterTreat').start()

    def on_bark_progress(self, episode_id: Optional[str], severity: str) -> None:
        """Called on every gated bark with the running episode's severity so far."""
        cfg = self._trigger_cfg('on_bark_episode')
        if not cfg or not episode_id:
            return
        need = SEVERITY_RANK.get(str(cfg.get('min_severity') or 'low').lower(), 0)
        if SEVERITY_RANK.get(str(severity).lower(), 0) < need:
            return
        with self._lock:
            if self._bark_fired_episode == episode_id:
                return
            self._bark_fired_episode = episode_id
        threading.Thread(target=self.fire, args=('on_bark_episode', cfg), daemon=True,
                         name='MusicOnBark').start()

    def tick(self) -> None:
        """Every few seconds: at_time slots, session deadline, quiet hours, plan switched off."""
        now = self._local_now()
        today = now.strftime('%Y-%m-%d')
        minute = now.hour * 60 + now.minute
        for slot in (self._trigger_cfg('at_time') or [])[:12]:
            m = _hhmm(slot.get('time'))
            if m is None or m != minute:
                continue
            key = f"{m // 60:02d}:{m % 60:02d}"
            with self._lock:
                if self._at_time_fired.get(key) == today:
                    continue
                self._at_time_fired[key] = today
            threading.Thread(target=self.fire, args=('at_time', slot), daemon=True, name='MusicAtTime').start()
        sess = self._session
        if sess is None:
            return
        if self._clock() >= sess['deadline']:
            self._end_session('duration')
        elif self._quiet_now():
            self._end_session('quiet_hours')
        elif not self._plan().get('enabled', True):
            self._end_session('disabled')
        else:
            self._keep_playing(sess)

    # ---------------------------------------------------------- playback
    def fire(self, trigger: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Start one music session for `trigger`, or say why not."""
        plan = self._plan()
        track = str(cfg.get('track') or '')
        if not plan.get('enabled', True):
            return self._skip(trigger, track, 'plan_disabled')
        if self._quiet_now():
            return self._skip(trigger, track, 'quiet_hours')
        path = self._resolve(track) if track else None
        if not path:
            return self._skip(trigger, track, 'unknown_track')
        audio = self._get_audio()
        # Never stack or restart: music that is already playing finishes first.
        waited = 0.0
        while True:
            if self._session is not None or self._other_music():
                return self._skip(trigger, track, 'music_already_playing')
            try:
                busy = bool(audio.is_busy())
            except Exception:
                busy = False
            if not busy:
                break
            if waited >= BUSY_WAIT_S:
                return self._skip(trigger, track, 'music_already_playing')
            if self._stop_evt.wait(0.5):
                return self._skip(trigger, track, 'service_stop')
            waited += 0.5
        minutes = max(1.0, min(float(cfg.get('duration_min') or 1), 480.0))
        now = self._clock()
        sess = {'trigger': trigger, 'track': track, 'path': path, 'start': now,
                'deadline': now + minutes * 60.0, 'prev_volume': None, 'plan_volume': None}
        with self._lock:
            if self._session is not None:
                return self._skip(trigger, track, 'music_already_playing')
            self._session = sess
        vol = plan.get('volume')
        if vol is not None:
            try:
                vm = self._get_volume()
                sess['prev_volume'] = vm.get_volume()
                sess['plan_volume'] = max(0, min(int(vol), 100))
                vm.set_volume(sess['plan_volume'], persist=False)    # temporary: owner's level is kept on disk
            except Exception as e:
                logger.warning(f"[MUSIC] plan volume not applied: {e}")
        try:
            res = audio.play_file(path)
            ok = bool(res.get('success', True)) if isinstance(res, dict) else bool(res)
        except Exception as e:
            logger.error(f"[MUSIC] play failed: {e}")
            ok = False
        if not ok:
            self._restore_volume(sess)
            with self._lock:
                self._session = None
            return self._skip(trigger, track, 'play_failed')
        sess['last_play'] = self._clock()
        logger.info(f"[MUSIC] start trigger={trigger} track={track} for {minutes:g} min")
        try:
            self._emit('music', {'action': 'start', 'track': track, 'trigger': trigger})
        except Exception as e:
            logger.error(f"[MUSIC] start event failed: {e}")
        return {'ok': True, 'trigger': trigger, 'track': track, 'duration_min': minutes}

    @staticmethod
    def _skip(trigger: str, track: str, why: str) -> Dict[str, Any]:
        logger.info(f"[MUSIC] trigger {trigger} (track={track or '-'}) not started: {why}")
        return {'ok': False, 'trigger': trigger, 'track': track, 'reason': why}

    def _keep_playing(self, sess: Dict[str, Any]) -> None:
        """The track ended before duration_min was up: play it again."""
        try:
            audio = self._get_audio()
            if audio.is_busy() or self._clock() - sess.get('last_play', 0) < 3.0:
                return
            audio.play_file(sess['path'])
            sess['last_play'] = self._clock()
        except Exception as e:
            logger.debug(f"[MUSIC] replay failed: {e}")

    def _restore_volume(self, sess: Dict[str, Any]) -> None:
        if sess.get('prev_volume') is None:
            return
        try:
            vm = self._get_volume()
            # Only put the old level back if nobody changed the volume meanwhile.
            if vm.get_volume() == sess.get('plan_volume'):
                vm.set_volume(int(sess['prev_volume']), persist=False)
        except Exception as e:
            logger.warning(f"[MUSIC] volume not restored: {e}")

    def _end_session(self, why: str) -> None:
        with self._lock:
            sess, self._session = self._session, None
        if sess is None:
            return
        try:
            audio = self._get_audio()
            if audio.is_busy():
                audio.stop()
        except Exception as e:
            logger.debug(f"[MUSIC] stop failed: {e}")
        self._restore_volume(sess)
        dur = round(max(0.0, self._clock() - sess['start']), 1)
        logger.info(f"[MUSIC] stop trigger={sess['trigger']} track={sess['track']} after {dur:.0f}s ({why})")
        try:
            self._emit('music', {'action': 'stop', 'track': sess['track'], 'trigger': sess['trigger'],
                                 'duration_s': dur, 'reason': why})
        except Exception as e:
            logger.error(f"[MUSIC] stop event failed: {e}")

    def session_active(self) -> bool:
        return self._session is not None

    def status(self) -> Dict[str, Any]:
        s = self._session
        plan = self._plan()
        return {'running': self._running, 'enabled': bool(plan.get('enabled', True)),
                'armed': [t for t in TRIGGERS if self._trigger_cfg(t)],
                'quiet_now': self._quiet_now(),
                'session': None if s is None else {'trigger': s['trigger'], 'track': s['track'],
                                                   'remaining_s': max(0, int(s['deadline'] - self._clock()))}}


_instance: Optional[MusicTriggers] = None
_instance_lock = threading.Lock()


def get_music_triggers() -> MusicTriggers:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = MusicTriggers()
        return _instance
