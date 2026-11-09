"""RemoteSettings — consume relay settings (CONTRACT v2 FINAL, 2026-11-06).

Source of truth is the relay, per dog:
  * on connect (after the profiles push): GET /api/dogs/{dog_id}/settings
    (HMAC) and apply every key;
  * live: WS {type:"settings_updated", dog_id, key, value, version} applies
    that one key.
The last good document is cached in treatbot.db (settings table, key
'remote_settings') so an offline boot applies the same values.

Keys and shapes
  treat_schedule  {daily_cap, default_multiplier, max_per_hour,
                   windows: [{start:"HH:MM", end:"HH:MM", multiplier}]}
  sg_sensitivity  {level: low|medium|high, min_episode_s, escalate_after_s}
                  level -> SG profile (relaxed|normal|sensitive):
                  threshold / loudness / confidence presets written into
                  sg.config['bark_detection'] (same seam as sg_config);
                  escalate_after_s -> bark window (seconds of barking before
                  an intervention: bark_detection.window_seconds + tracker);
                  min_episode_s -> BehaviorEvents bark_episode_min_s.
  music_plan      {enabled, volume, on_departure{track, delay_s, duration_min},
                   schedule: [...]}  (stored; SG music helpers read volume,
                   BehaviorEvents plays on_departure; schedule stored only)
  quiet_hours     stored verbatim, exposed via get(); not enforced here.
  notify_rules    {rules: [...]} (relay shape) or a bare list; stored
                  verbatim (relay evaluates).
Unknown keys are kept for forward compatibility. Nothing here touches the
SG escalation state machine.
"""
import json
import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger('RemoteSettings')

STORE_KEY = 'remote_settings'

# SG profiles. 'normal' == configs/rules/silent_guardian_rules.yaml today.
SG_PROFILES: Dict[str, Dict[str, Any]] = {
    'relaxed':   {'threshold': 4, 'loudness_threshold_db': -21, 'confidence_minimum': 0.45},
    'normal':    {'threshold': 2, 'loudness_threshold_db': -28, 'confidence_minimum': 0.35},
    'sensitive': {'threshold': 1, 'loudness_threshold_db': -32, 'confidence_minimum': 0.30},
}
LEVEL_TO_PROFILE = {'low': 'relaxed', 'medium': 'normal', 'med': 'normal', 'high': 'sensitive',
                    'relaxed': 'relaxed', 'normal': 'normal', 'sensitive': 'sensitive'}
# Settings sources that come FROM the relay (authoritative once it has our
# write). Anything else is a local setter (app sg_config, local API, Xbox).
RELAY_SOURCES = ('relay', 'relay_get', 'relay_push', 'cache')
# How long a locally-set treat_schedule outranks a stale relay push while the
# upstream PUT is still in flight / retrying.
LOCAL_OVERRIDE_TTL_S = 600.0
UPSTREAM_BACKOFF_S = (2, 5, 15, 30, 60, 120, 180, 240)

KNOWN_KEYS = ('treat_schedule', 'sg_sensitivity', 'music_plan', 'quiet_hours', 'notify_rules',
              'missions', 'timezone')


class RemoteSettings:
    def __init__(self):
        self._lock = threading.RLock()
        self._settings: Dict[str, Any] = {}
        self._versions: Dict[str, Any] = {}
        self._meta: Dict[str, Any] = {'received_at': None, 'source': None, 'applied_at': None,
                                      'dog_id': None}
        self._store = None
        self._dog_ids: set = set()             # every dog the relay sent settings for
        # Local treat_schedule write awaiting the relay's ack (PUT 2xx or an
        # equal push). {'value': {...}, 'since': mono, 'source': str}
        self._pending_local: Optional[Dict[str, Any]] = None
        self._upstream_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------- persistence

    def _get_store(self):
        if self._store is None:
            try:
                from core.store import get_store
                self._store = get_store()
            except Exception as e:
                logger.debug(f"store unavailable: {e}")
        return self._store

    def load_cached(self) -> bool:
        store = self._get_store()
        if store is None:
            return False
        try:
            raw = store.get_setting(STORE_KEY)
        except Exception as e:
            logger.debug(f"get_setting failed: {e}")
            return False
        if not raw:
            logger.info("[SETTINGS] no cached remote settings")
            return False
        try:
            doc = json.loads(raw)
        except ValueError:
            logger.warning("[SETTINGS] cached settings unreadable; ignoring")
            return False
        settings = doc.get('settings', doc) if isinstance(doc, dict) else {}
        result = self.apply(settings, source='cache', persist=False,
                            dog_id=(doc.get('dog_id') if isinstance(doc, dict) else None))
        with self._lock:
            if isinstance(doc, dict):
                self._meta['received_at'] = doc.get('received_at')
                self._versions = dict(doc.get('versions') or {})
        logger.info(f"[SETTINGS] applied cached settings: {result}")
        return True

    def _persist(self) -> None:
        store = self._get_store()
        if store is None:
            return
        try:
            with self._lock:
                doc = {'settings': self._settings, 'versions': self._versions,
                       'received_at': self._meta['received_at'], 'source': self._meta['source'],
                       'dog_id': self._meta['dog_id']}
            store.set_setting(STORE_KEY, json.dumps(doc, default=str))
        except Exception as e:
            logger.error(f"[SETTINGS] persist failed: {e}")

    # ------------------------------------------------------------- apply

    def apply(self, settings: Dict[str, Any], source: str = 'relay', persist: bool = True,
              dog_id: Optional[str] = None, version: Any = None) -> Dict[str, Any]:
        """Validate + merge + fan out. Returns {'applied': [...], 'rejected': {key: reason}}."""
        if not isinstance(settings, dict):
            return {'applied': [], 'rejected': {'_': 'settings must be an object'}}
        applied: List[str] = []
        rejected: Dict[str, str] = {}
        now = time.time()

        # The relay's GET /api/dogs/{id}/settings wraps each key as
        # {"key","value","version","updated_at","updated_by"}; pushes send
        # the bare value. Unwrap so both paths apply the same document.
        unwrapped: Dict[str, Any] = {}
        wrapped_versions: Dict[str, Any] = {}
        for key, value in settings.items():
            if isinstance(value, dict) and 'value' in value and value.get('key') == key:
                if value.get('version') is not None:
                    wrapped_versions[key] = value.get('version')
                value = value.get('value')
            unwrapped[key] = value
        settings = unwrapped

        if dog_id:
            with self._lock:
                self._dog_ids.add(str(dog_id))

        # timezone first: the missions loader needs it for the dog's HH:MM
        ordered = sorted(settings.items(), key=lambda kv: 0 if kv[0] == 'timezone' else 1)
        for key, value in ordered:
            try:
                if key == 'treat_schedule' and source in RELAY_SOURCES:
                    verdict = self._reconcile_relay_treat_schedule(value, source)
                    if verdict == 'deferred':
                        rejected[key] = 'local treat_schedule pending upstream ack'
                        continue
                if key == 'treat_schedule':
                    self._apply_treat_schedule(value)
                elif key == 'daily_cap':            # legacy flat key
                    self._apply_treat_schedule({'daily_cap': value}, partial=True)
                elif key == 'sg_sensitivity':
                    self._apply_sg_sensitivity(value)
                elif key == 'music_plan':
                    self._validate_music_plan(value)
                elif key == 'quiet_hours':
                    self._validate_quiet_hours(value)
                elif key == 'notify_rules':
                    rules = value.get('rules') if isinstance(value, dict) else value
                    if value is not None and not isinstance(rules, list):
                        raise ValueError("notify_rules must be {rules: [...]} or a list")
                elif key == 'timezone':
                    self._validate_timezone(value)
                with self._lock:
                    self._settings[key] = value
                    if version is not None:
                        self._versions[key] = version
                    elif key in wrapped_versions:
                        self._versions[key] = wrapped_versions[key]
                if key == 'missions':
                    # Loaded synchronously: the scheduler set is replaced
                    # before this call returns, i.e. before the next command.
                    self._apply_missions(value, dog_id)
                applied.append(key)
            except Exception as e:
                rejected[key] = str(e)
                logger.warning(f"[SETTINGS] rejected {key}: {e}")

        with self._lock:
            self._meta.update({'received_at': now, 'source': source, 'applied_at': now})
            if dog_id:
                self._meta['dog_id'] = dog_id
        if persist and applied:
            self._persist()
        self._announce(source)
        return {'applied': applied, 'rejected': rejected}

    def _apply_treat_schedule(self, value, partial: bool = False) -> None:
        if value is None:
            value = {}
        if isinstance(value, list):           # tolerate bare windows list
            value = {'windows': value}
        if not isinstance(value, dict):
            raise ValueError("treat_schedule must be an object")
        from services.reward.treat_policy import get_treat_policy, validate_schedule
        # Validate everything BEFORE touching the live policy so a bad field
        # cannot half-apply (e.g. reset the cap and then fail on windows).
        if 'windows' in value and value['windows'] is not None:
            validate_schedule(value['windows'])
        if value.get('daily_cap') is not None and int(value['daily_cap']) < 0:
            raise ValueError("daily_cap must be >= 0")
        if value.get('default_multiplier') is not None and float(value['default_multiplier']) < 0:
            raise ValueError("default_multiplier must be >= 0")
        if value.get('max_per_hour') is not None and int(value['max_per_hour']) < 0:
            raise ValueError("max_per_hour must be >= 0")
        tp = get_treat_policy()
        if 'daily_cap' in value or 'daily_cap_enabled' in value or not partial:
            # daily_cap_enabled false = unlimited; daily_cap then only parks
            # the slider value for when the toggle comes back on.
            enabled = value.get('daily_cap_enabled', True)
            if enabled is None:
                enabled = True
            tp.set_daily_cap(value.get('daily_cap') if enabled else None)
        if 'default_multiplier' in value or not partial:
            tp.set_default_multiplier(value.get('default_multiplier', 1.0))
        if 'max_per_hour' in value or not partial:
            tp.set_max_per_hour(value.get('max_per_hour'))
        if 'windows' in value or not partial:
            tp.set_schedule(value.get('windows') or [])

    # ------------------------------------------------- treat limits: any source

    @staticmethod
    def _limits_equal(a, b) -> bool:
        a = a if isinstance(a, dict) else {}
        b = b if isinstance(b, dict) else {}
        def norm(d):
            en = d.get('daily_cap_enabled', True)
            return (None if d.get('daily_cap') is None else int(d['daily_cap']),
                    True if en is None else bool(en),
                    None if d.get('max_per_hour') is None else int(d['max_per_hour']))
        return norm(a) == norm(b)

    def _reconcile_relay_treat_schedule(self, value, source: str) -> str:
        """A relay-sourced treat_schedule arrives while a local write is pending.
        Equal limits -> the relay has our value: pending cleared ('acked').
        Different + pending fresh -> 'deferred' (local wins; relay is stale).
        Different + pending expired -> 'accept' (relay regains authority)."""
        with self._lock:
            pend = self._pending_local
            if not pend:
                return 'accept'
            if self._limits_equal(pend['value'], value):
                self._pending_local = None
                logger.info(f"[TREAT-SYNC] relay ({source}) now carries the local limits "
                            f"{self._limits_brief(value)} — pending cleared")
                return 'acked'
            if time.monotonic() - pend['since'] > LOCAL_OVERRIDE_TTL_S:
                self._pending_local = None
                logger.warning("[TREAT-SYNC] local treat limits never reached the relay "
                               "within the override window — accepting relay value "
                               f"{self._limits_brief(value)}")
                return 'accept'
            logger.warning(f"[TREAT-SYNC] ignoring stale relay ({source}) treat_schedule "
                           f"{self._limits_brief(value)}; local {self._limits_brief(pend['value'])} "
                           f"set by {pend['source']} is awaiting upstream ack")
            return 'deferred'

    @staticmethod
    def _limits_brief(v) -> str:
        v = v if isinstance(v, dict) else {}
        en = v.get('daily_cap_enabled', True)
        return (f"cap={v.get('daily_cap')}{'' if en in (True, None) else ' (disabled)'} "
                f"per_hour={v.get('max_per_hour')}")

    def set_treat_limits(self, daily_cap=..., max_per_hour=..., daily_cap_enabled=...,
                         source: str = 'local', push_upstream: bool = True) -> Dict[str, Any]:
        """THE setter for the treat cap from any non-relay path (app sg_config,
        local API, Xbox, voice). Merges into the current treat_schedule, applies
        it through the normal path (TreatPolicy + dispenser + persisted cache),
        then writes it back to the relay so the app / relay / other robots agree.
        Ellipsis = leave that field as it is. Returns the apply() result plus
        'treat_schedule' (merged) and 'upstream' (pending|skipped|no_change)."""
        with self._lock:
            cur = self._settings.get('treat_schedule')
            merged = dict(cur) if isinstance(cur, dict) else {}
        if daily_cap is not ...:
            merged['daily_cap'] = None if daily_cap is None else int(daily_cap)
        if max_per_hour is not ...:
            merged['max_per_hour'] = None if max_per_hour is None else int(max_per_hour)
        if daily_cap_enabled is not ...:
            merged['daily_cap_enabled'] = bool(daily_cap_enabled)
        merged.setdefault('daily_cap_enabled', True)
        merged.setdefault('default_multiplier', 1.0)
        merged.setdefault('windows', [])
        if self._limits_equal(cur, merged) and isinstance(cur, dict):
            return {'applied': [], 'rejected': {}, 'treat_schedule': merged,
                    'upstream': 'no_change'}
        result = self.apply({'treat_schedule': merged}, source=source)
        result['treat_schedule'] = merged
        if 'treat_schedule' not in result['applied']:
            result['upstream'] = 'skipped'
            return result
        logger.info(f"[TREAT-SYNC] treat limits set by {source}: {self._limits_brief(merged)}")
        if not push_upstream:
            result['upstream'] = 'skipped'
            return result
        with self._lock:
            self._pending_local = {'value': dict(merged), 'since': time.monotonic(),
                                   'source': source}
            dogs = sorted(self._dog_ids) or ([self._meta['dog_id']] if self._meta['dog_id'] else [])
        if not dogs:
            logger.warning("[TREAT-SYNC] no dog_id known yet — cannot write limits to relay; "
                           "local value holds until the relay pushes")
            result['upstream'] = 'no_dog'
            return result
        self._start_upstream_push(dict(merged), dogs)
        result['upstream'] = 'pending'
        return result

    def _start_upstream_push(self, value: Dict[str, Any], dogs: List[str]) -> None:
        th = threading.Thread(target=self._push_upstream, args=(value, dogs),
                              name='treat-limit-upstream', daemon=True)
        with self._lock:
            self._upstream_thread = th
        th.start()

    def _push_upstream(self, value: Dict[str, Any], dogs: List[str]) -> None:
        """PUT /api/dogs/{dog}/settings/treat_schedule once. treat_schedule is
        account-wide on the relay (relay brief 2026-11-09): one PUT updates every
        dog of the owner and the relay fans out a settings_updated per dog, so the
        first accepted write finishes the job; the other dogs are only fallbacks
        if that one fails. Retries with backoff until acked or the override TTL
        ends. A newer local write supersedes this one (pending value differs)."""
        try:
            from services.cloud.relay_client import get_relay_client
            relay = get_relay_client()
        except Exception as e:
            logger.warning(f"[TREAT-SYNC] relay client unavailable, limits stay local: {e}")
            return
        remaining = list(dogs)
        for attempt, delay in enumerate(UPSTREAM_BACKOFF_S, 1):
            with self._lock:
                pend = self._pending_local
            if not pend or not self._limits_equal(pend['value'], value):
                return  # acked by a relay push, or superseded by a newer local write
            still = []
            for dog in remaining:
                try:
                    status, body = relay.http_sync(
                        'PUT', f'/api/dogs/{dog}/settings/treat_schedule',
                        json_body={'value': value}, timeout=10.0)
                except Exception as e:
                    logger.warning(f"[TREAT-SYNC] PUT treat_schedule dog={dog} failed: {e}")
                    still.append(dog)
                    continue
                if 200 <= status < 300:
                    logger.info(f"[TREAT-SYNC] relay accepted treat limits via dog {dog} "
                                f"({self._limits_brief(value)}) — account-wide, fans out to all dogs")
                    still = []
                    break
                elif status in (404, 405):
                    logger.warning(f"[TREAT-SYNC] relay has no PUT treat_schedule yet "
                                   f"(HTTP {status}) dog={dog} — will retry")
                    still.append(dog)
                else:
                    logger.warning(f"[TREAT-SYNC] PUT treat_schedule dog={dog} -> HTTP {status} "
                                   f"{str(body)[:160]}")
                    still.append(dog)
            remaining = still
            if not remaining:
                with self._lock:
                    if self._pending_local and self._limits_equal(self._pending_local['value'], value):
                        self._pending_local = None
                return
            time.sleep(delay)
        logger.error(f"[TREAT-SYNC] gave up writing treat limits to relay for dogs {remaining}; "
                     "local value holds until the override window expires")

    def get_treat_sync_status(self) -> Dict[str, Any]:
        with self._lock:
            pend = self._pending_local
            return {'dog_ids': sorted(self._dog_ids),
                    'pending_local': None if not pend else {
                        'value': pend['value'], 'source': pend['source'],
                        'age_s': round(time.monotonic() - pend['since'], 1)}}

    def _apply_sg_sensitivity(self, value) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            raise ValueError("sg_sensitivity must be an object")
        level = value.get('level', value.get('profile'))
        profile = LEVEL_TO_PROFILE.get(str(level).lower()) if level is not None else None
        if level is not None and profile is None:
            raise ValueError(f"unknown level {level!r} (use low|medium|high)")
        esc = value.get('escalate_after_s', value.get('window'))
        min_ep = value.get('min_episode_s')
        threshold = value.get('threshold')
        if esc is not None and not (5 <= int(esc) <= 3600):
            raise ValueError("escalate_after_s must be 5..3600")
        if min_ep is not None and not (0 <= float(min_ep) <= 3600):
            raise ValueError("min_episode_s must be 0..3600")
        if threshold is not None and not (1 <= int(threshold) <= 50):
            raise ValueError("threshold must be 1..50")

        patch: Dict[str, Any] = {}
        if profile:
            patch.update(SG_PROFILES[profile])
            patch['profile'] = profile
        if threshold is not None:
            patch['threshold'] = int(threshold)
        if esc is not None:
            patch['window_seconds'] = int(esc)
        try:
            from modes.silent_guardian import get_silent_guardian_mode
            sg = get_silent_guardian_mode()
            sg.config.setdefault('bark_detection', {}).update(patch)
        except Exception as e:
            logger.warning(f"[SETTINGS] SG config not reachable ({e}); tracker/episode still applied")
        if esc is not None:
            try:
                from core.bark_frequency_tracker import get_bark_frequency_tracker
                get_bark_frequency_tracker().window_seconds = int(esc)
            except Exception as e:
                logger.warning(f"[SETTINGS] bark tracker window not applied: {e}")
        if min_ep is not None:
            try:
                from services.cloud.behavior_events import get_behavior_event_service
                get_behavior_event_service().config['bark_episode_min_s'] = float(min_ep)
            except Exception as e:
                logger.debug(f"bark_episode_min_s not applied: {e}")
        logger.info(f"[SETTINGS] sg_sensitivity level={level} -> {patch} min_episode_s={min_ep}")

    @staticmethod
    def _validate_music_plan(value) -> None:
        if value is None:
            return
        if not isinstance(value, dict):
            raise ValueError("music_plan must be an object")
        if value.get('volume') is not None and not (0 <= int(value['volume']) <= 100):
            raise ValueError("volume must be 0..100")
        od = value.get('on_departure')
        if od is not None:
            if not isinstance(od, dict):
                raise ValueError("on_departure must be an object")
            if od.get('delay_s') is not None and float(od['delay_s']) < 0:
                raise ValueError("on_departure.delay_s must be >= 0")
            if od.get('duration_min') is not None and float(od['duration_min']) <= 0:
                raise ValueError("on_departure.duration_min must be > 0")
        if value.get('schedule') is not None and not isinstance(value['schedule'], list):
            raise ValueError("schedule must be a list")

    @staticmethod
    def _validate_timezone(value) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            raise ValueError("timezone must be an IANA zone string")
        from zoneinfo import ZoneInfo
        ZoneInfo(value)              # raises on an unknown zone

    @staticmethod
    def _apply_missions(value, dog_id) -> None:
        """missions {missions:[{id, type, name, enabled, trigger{kind, times[], days[]},
        duration_s, reward_treats}]} -> MissionScheduler (replace that dog's set)."""
        if value is not None and not isinstance(value, (dict, list)):
            raise ValueError("missions must be an object or list")
        from core.mission_scheduler import get_mission_scheduler
        res = get_mission_scheduler().apply_missions_setting(dog_id, value)
        logger.info(f"[SETTINGS] missions for dog {dog_id}: loaded={len(res['loaded'])} "
                    f"skipped={res['skipped']} replaced={res['replaced']}")

    def get_timezone(self) -> Optional[str]:
        with self._lock:
            tz = self._settings.get('timezone')
            return tz if isinstance(tz, str) and tz else None

    @staticmethod
    def _validate_quiet_hours(value) -> None:
        if value is None:
            return
        if isinstance(value, dict):
            for k in ('start', 'end'):
                if k in value and value[k] is not None:
                    hh, mm = str(value[k]).split(':')
                    if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
                        raise ValueError(f"quiet_hours.{k} must be HH:MM")
        elif not isinstance(value, list):
            raise ValueError("quiet_hours must be an object or list")

    def _announce(self, source: str) -> None:
        snapshot = self.get()
        try:
            from api.ws import get_websocket_server
            get_websocket_server().broadcast_event_threadsafe(
                'settings_state', {**snapshot, 'reason': f'applied:{source}'})
        except Exception as e:
            logger.debug(f"settings_state broadcast failed: {e}")
        try:
            from core.bus import publish_system_event
            publish_system_event('remote_settings_applied',
                                 {'source': source, 'keys': list(self._settings.keys())},
                                 'remote_settings')
        except Exception as e:
            logger.debug(f"bus publish failed: {e}")

    # ------------------------------------------------------------- getters

    def get(self) -> Dict[str, Any]:
        with self._lock:
            out = {k: self._settings.get(k) for k in KNOWN_KEYS}
            for k, v in self._settings.items():
                out.setdefault(k, v)
            out['_versions'] = dict(self._versions)
            out['_meta'] = dict(self._meta)
        try:
            from services.reward.treat_policy import get_treat_policy
            out['_effective'] = get_treat_policy().get_settings()
        except Exception:
            pass
        return out

    def get_music_plan(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            mp = self._settings.get('music_plan')
            return dict(mp) if isinstance(mp, dict) else None

    def get_notify_rules(self) -> List[Any]:
        with self._lock:
            nr = self._settings.get('notify_rules')
            if isinstance(nr, dict):
                nr = nr.get('rules')
            return list(nr) if isinstance(nr, list) else []

    def get_quiet_hours(self):
        with self._lock:
            return self._settings.get('quiet_hours')

    def get_sg_sensitivity(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            v = self._settings.get('sg_sensitivity')
            return dict(v) if isinstance(v, dict) else None


def apply_settings_updated_frame(data: Dict[str, Any], source: str = 'relay_push') -> Dict[str, Any]:
    """{type:"settings_updated", dog_id, key, value, version} -> apply().
    The ONE code path for the relay WS push and the /ws/local replay of it
    (live test), so both behave identically. value is the bare setting."""
    key, value = data.get('key'), data.get('value')
    if not key:
        raise ValueError("settings_updated without key")
    return get_remote_settings().apply({key: value}, source=source, dog_id=data.get('dog_id'),
                                       version=data.get('version'))


_instance: Optional[RemoteSettings] = None
_instance_lock = threading.Lock()


def get_remote_settings() -> RemoteSettings:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = RemoteSettings()
        return _instance
