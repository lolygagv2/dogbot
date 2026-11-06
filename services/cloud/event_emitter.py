"""RelayEventEmitter — builds CONTRACT v2 event rows and fans them out.

CONTRACT v2 FINAL (2026-11-06). One row per event, POSTed by the outbox
pump to /events in batches:
    {"id": "<uuid7>", "type": "<event type>", "device_id": "...",
     "dog_id": "<app id|null>", "id_method": "...", "timestamp": "ISO Z",
     "data": {...type-specific fields...}}

For every emit():
  1. row built here (id = UUIDv7 so the relay can order/dedupe);
  2. dog_id resolved to the app's canonical id via the SAME rule the legacy
     relay boundary uses (core.dog_profile_manager.resolve_relay_dog);
  3. wimz.db row for the natively-stored types;
  4. durable outbox -> HTTP POST /events (services/cloud/relay_outbox via
     RelayClient.enqueue_event);
  5. local /ws/local broadcast so the AP-mode app sees the same thing.
"""
import json
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger('EventEmitter')

EVENT_TYPES = ('bark_episode', 'intervention', 'settle', 'pacing', 'departure',
               'arrival', 'treat', 'music', 'mission_run', 'status')

# Written to wimz.db as native rows. intervention/treat/music/status are
# already covered by sg_intervention / dispense_log / audio / telemetry.
STORED_NATIVE = frozenset({'bark_episode', 'settle', 'pacing', 'departure',
                           'arrival', 'mission_run'})


def iso_z(ts: Optional[float] = None) -> str:
    dt = datetime.fromtimestamp(ts if ts is not None else time.time(), tz=timezone.utc)
    return dt.isoformat(timespec='milliseconds').replace('+00:00', 'Z')


def parse_ts(value) -> float:
    """ISO8601 (Z/offset/naive=UTC) or epoch s/ms -> epoch seconds."""
    if value is None:
        return time.time()
    if isinstance(value, (int, float)):
        return value / 1000.0 if value > 1e11 else float(value)
    s = str(value).strip()
    if s.endswith('Z'):
        s = s[:-1] + '+00:00'
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class RelayEventEmitter:
    def __init__(self):
        self._lock = threading.RLock()
        self._relay = None
        self._wimz = None
        self._device_id: Optional[str] = None
        self.emitted = 0

    # ------------------------------------------------------------- deps

    def _relay_client(self):
        if self._relay is None:
            try:
                from services.cloud import relay_client as rc_mod
                self._relay = rc_mod._relay_client
            except Exception:
                self._relay = None
        return self._relay

    def _wimz_store(self):
        if self._wimz is None:
            try:
                from core.data.wimz_store import get_wimz_store
                self._wimz = get_wimz_store()
            except Exception as e:
                logger.debug(f"wimz store unavailable: {e}")
        return self._wimz

    def device_id(self) -> str:
        if self._device_id:
            return self._device_id
        rc = self._relay_client()
        if rc is not None and getattr(rc, 'config', None) and rc.config.device_id:
            self._device_id = rc.config.device_id
            return self._device_id
        import os
        return os.environ.get('DEVICE_ID', 'unknown')

    # ------------------------------------------------------------- emit

    def build(self, event_type: str, data: Dict[str, Any], *, dog_id=None,
              dog_name=None, aruco_id=None, id_method=None, resolve_dog=True,
              ts: Optional[float] = None, event_id: Optional[str] = None) -> Dict[str, Any]:
        from core.data.ids import uuid7
        if resolve_dog:
            try:
                from core.dog_profile_manager import resolve_relay_dog
                app_id, method = resolve_relay_dog(dog_id=dog_id, dog_name=dog_name,
                                                   aruco_id=aruco_id, id_method=id_method)
            except Exception as e:
                logger.debug(f"dog resolve failed: {e}")
                app_id, method = None, None
        else:
            app_id, method = dog_id, id_method
        return {
            'id': event_id or uuid7(),
            'type': event_type,
            'device_id': self.device_id(),
            'dog_id': app_id,
            'id_method': method,
            'timestamp': iso_z(ts),
            'data': dict(data or {}),
        }

    def emit(self, event_type: str, data: Dict[str, Any], *, dog_id=None,
             dog_name=None, aruco_id=None, id_method=None, resolve_dog: bool = True,
             ts: Optional[float] = None, event_id: Optional[str] = None,
             store: Optional[bool] = None) -> Dict[str, Any]:
        """Build + store + queue + broadcast. Never raises into the producer."""
        if event_type not in EVENT_TYPES:
            logger.warning(f"emit: unknown event type {event_type!r} (sending anyway)")
        ev = self.build(event_type, data, dog_id=dog_id, dog_name=dog_name, aruco_id=aruco_id,
                        id_method=id_method, resolve_dog=resolve_dog, ts=ts, event_id=event_id)

        should_store = store if store is not None else (event_type in STORED_NATIVE)
        if should_store:
            self._store_row(ev)

        rc = self._relay_client()
        if rc is not None:
            try:
                rc.enqueue_event(ev)
            except Exception as e:
                logger.error(f"outbox enqueue failed: {e}")

        self._broadcast_local(ev)
        with self._lock:
            self.emitted += 1
        logger.info(f"[EVENT] {event_type} dog={ev.get('dog_id')} "
                    f"{json.dumps(ev['data'], default=str)[:220]}")
        return ev

    def _store_row(self, ev: Dict[str, Any]) -> None:
        wimz = self._wimz_store()
        if wimz is None:
            return
        try:
            wimz_dog = wimz.dog_id_for_app_id(ev.get('dog_id')) if ev.get('dog_id') else None
            wimz.log_event(None, ev['type'], ev['data'], dog_id=wimz_dog,
                           ts_ms=int(parse_ts(ev['timestamp']) * 1000),
                           label_source='auto_rule', event_id=ev['id'])
        except Exception as e:
            logger.debug(f"wimz row failed for {ev.get('type')}: {e}")

    def _broadcast_local(self, ev: Dict[str, Any]) -> None:
        try:
            from api.ws import get_websocket_server
            srv = get_websocket_server()
            srv.broadcast_event_threadsafe(ev['type'], {k: v for k, v in ev.items() if k != 'type'})
        except Exception as e:
            logger.debug(f"local broadcast failed: {e}")


_instance: Optional[RelayEventEmitter] = None
_instance_lock = threading.Lock()


def get_event_emitter() -> RelayEventEmitter:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = RelayEventEmitter()
        return _instance
