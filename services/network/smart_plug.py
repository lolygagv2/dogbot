#!/usr/bin/env python3
"""
Smart plug control (TP-Link Kasa, local LAN protocol) + Silent Guardian rule.

Rule (Morgan, 2026-11-10):
  - Silent Guardian intervention at level >= trigger_level (2):
      plug OFF  -> robot turns it ON and OWNS it (auto-off armed)
      plug ON   -> left alone (if the robot already owns it, the timer restarts)
  - A robot-owned plug is turned OFF when the first of these happens:
      auto_off_minutes (120) after the last level>=2 trigger, or
      the robot has been out of Silent Guardian mode for exit_grace_s.
  - A plug the robot did NOT turn on is never turned off by the robot.
  - A manual request (relay / app / REST `smart_plug` tool) always wins and
    clears robot ownership: manual ON stays on until someone turns it off,
    manual OFF is off (a later level>=2 trigger may turn it on again).

Ownership + deadline are persisted (state/smart_plug.json) so a service
restart inside the window still turns the plug off.

Config: `smart_plug:` section of the per-unit robot profile. No section or
enabled: false -> the service is inert (fleet units without a plug).

Protocol: Kasa "IOT" local API, TCP 9999, XOR-autokey obfuscated JSON with a
4-byte big-endian length prefix. No cloud, no extra dependency. If the plug's
DHCP address changes it is re-found by deviceId via UDP broadcast.
"""

import json
import logging
import os
import queue
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

STATE_FILE = Path('/home/morgan/dogbot/state/smart_plug.json')
KASA_PORT = 9999

DEFAULTS = {
    'enabled': False,
    'host': None,
    'device_id': None,
    'alias': None,
    'port': KASA_PORT,
    'timeout_s': 3.0,
    'trigger_level': 2,
    'auto_off_minutes': 120,
    'exit_grace_s': 60,
    'retry_s': 30,
}


def _kasa_encrypt(text: str) -> bytes:
    key = 171
    out = bytearray()
    for c in text.encode():
        key ^= c
        out.append(key)
    return bytes(out)


def _kasa_decrypt(data: bytes) -> str:
    key = 171
    out = bytearray()
    for c in data:
        out.append(key ^ c)
        key = c
    return out.decode(errors='replace')


class PlugError(Exception):
    pass


class KasaPlug:
    """One Kasa plug on the LAN (blocking I/O; callers keep it off hot paths)."""

    def __init__(self, host: Optional[str], device_id: Optional[str] = None,
                 alias: Optional[str] = None, port: int = KASA_PORT, timeout_s: float = 3.0):
        self.host = host
        self.device_id = device_id
        self.alias = alias
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def _request_host(self, host: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = _kasa_encrypt(json.dumps(payload, separators=(',', ':')))
        with socket.create_connection((host, self.port), self.timeout_s) as s:
            s.settimeout(self.timeout_s)
            s.sendall(struct.pack('>I', len(body)) + body)
            head = self._recv_exact(s, 4)
            length = struct.unpack('>I', head)[0]
            if length > 65536:
                raise PlugError(f"implausible reply length {length}")
            return json.loads(_kasa_decrypt(self._recv_exact(s, length)))

    @staticmethod
    def _recv_exact(s: socket.socket, n: int) -> bytes:
        buf = b''
        while len(buf) < n:
            chunk = s.recv(n - len(buf))
            if not chunk:
                raise PlugError("connection closed mid-reply")
            buf += chunk
        return buf

    def _matches(self, info: Dict[str, Any]) -> bool:
        if self.device_id:
            return info.get('deviceId') == self.device_id
        if self.alias:
            return (info.get('alias') or '').strip().lower() == self.alias.strip().lower()
        return True

    def discover(self) -> Optional[str]:
        """Find this plug by deviceId/alias with a UDP broadcast; returns its IP."""
        if not (self.device_id or self.alias):
            return None
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            s.settimeout(self.timeout_s)
            s.sendto(_kasa_encrypt('{"system":{"get_sysinfo":{}}}'), ('255.255.255.255', KASA_PORT))
            deadline = time.time() + self.timeout_s
            while time.time() < deadline:
                try:
                    data, addr = s.recvfrom(4096)
                except socket.timeout:
                    break
                try:
                    info = json.loads(_kasa_decrypt(data))['system']['get_sysinfo']
                except Exception:
                    continue
                if self._matches(info):
                    return addr[0]
        except OSError as e:
            logger.debug(f"[PLUG] discovery failed: {e}")
        finally:
            try:
                s.close()
            except Exception:
                pass
        return None

    def request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Send one command; on a network failure or a different device at the
        configured address, re-find the plug by id once and retry."""
        first_err: Optional[Exception] = None
        if self.host:
            try:
                return self._request_host(self.host, payload)
            except (OSError, ValueError, PlugError) as e:
                first_err = e
        found = self.discover()
        if found:
            if found != self.host:
                logger.info(f"[PLUG] {self.alias or self.device_id} moved {self.host} -> {found}")
            self.host = found
            try:
                return self._request_host(found, payload)
            except (OSError, ValueError, PlugError) as e:
                first_err = e
        raise PlugError(f"plug unreachable ({self.host}): {first_err or 'not found on the LAN'}")

    def sysinfo(self) -> Dict[str, Any]:
        info = self.request({'system': {'get_sysinfo': {}}})['system']['get_sysinfo']
        if not self._matches(info):
            # Another device took the old DHCP address: look the plug up by id.
            found = self.discover()
            if not found or found == self.host:
                raise PlugError(f"device at {self.host} is not the configured plug")
            logger.info(f"[PLUG] {self.alias or self.device_id} moved {self.host} -> {found}")
            self.host = found
            info = self._request_host(found, {'system': {'get_sysinfo': {}}})['system']['get_sysinfo']
            if not self._matches(info):
                raise PlugError("configured plug not found")
        return info

    def is_on(self) -> bool:
        return bool(self.sysinfo().get('relay_state'))

    def set(self, on: bool) -> None:
        self.sysinfo()   # confirms identity (and corrects the address) before switching
        reply = self.request({'system': {'set_relay_state': {'state': 1 if on else 0}}})
        err = reply.get('system', {}).get('set_relay_state', {}).get('err_code', -1)
        if err != 0:
            raise PlugError(f"set_relay_state err_code {err}")


class SmartPlugService:
    """Owns the plug + the Silent Guardian rule. All switching goes through here."""

    TICK_S = 5.0

    def __init__(self, config: Optional[Dict[str, Any]] = None, plug: Optional[KasaPlug] = None,
                 state_file: Optional[Path] = None, clock: Callable[[], float] = time.time,
                 mode_getter: Optional[Callable[[], bool]] = None):
        cfg = dict(DEFAULTS)
        cfg.update(config if config is not None else self._profile_config())
        self.cfg = cfg
        self.enabled = bool(cfg.get('enabled')) and bool(cfg.get('host') or cfg.get('device_id') or plug)
        self.state_file = Path(state_file) if state_file else STATE_FILE
        self._clock = clock
        self._in_sg = mode_getter or self._default_in_sg
        self._lock = threading.RLock()
        self._queue: "queue.Queue" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Rule state (persisted)
        self.owner: Optional[str] = None        # 'robot' while the robot must turn it off
        self.auto_off_at: Optional[float] = None
        self.turned_on_at: Optional[float] = None
        self.last_change: Optional[Dict[str, Any]] = None
        # Runtime only
        self.last_known_on: Optional[bool] = None
        self.last_error: Optional[str] = None
        self._out_of_sg_since: Optional[float] = None
        self._next_off_attempt = 0.0

        self._load_state()
        self.plug = plug or (KasaPlug(self._saved_host or cfg.get('host'), cfg.get('device_id'),
                                      cfg.get('alias'), cfg.get('port') or KASA_PORT,
                                      cfg.get('timeout_s') or 3.0) if self.enabled else None)

    # ---------------- config / persistence ----------------
    @staticmethod
    def _profile_config() -> Dict[str, Any]:
        try:
            from config.config_loader import get_config
            return dict(get_config().raw.get('smart_plug') or {})
        except Exception as e:
            logger.warning(f"[PLUG] profile config unavailable: {e}")
            return {}

    @staticmethod
    def _default_in_sg() -> bool:
        from core.state import SystemMode, get_state
        return get_state().get_mode() == SystemMode.SILENT_GUARDIAN

    def _load_state(self) -> None:
        self._saved_host = None
        try:
            data = json.loads(self.state_file.read_text())
        except FileNotFoundError:
            return
        except Exception as e:
            logger.warning(f"[PLUG] state file unreadable ({e}); starting clean")
            return
        # Ownership only carries over for the same physical plug.
        if data.get('device_id') != self.cfg.get('device_id'):
            return
        self.owner = data.get('owner') if data.get('owner') == 'robot' else None
        self.auto_off_at = data.get('auto_off_at') if self.owner else None
        self.turned_on_at = data.get('turned_on_at') if self.owner else None
        self.last_change = data.get('last_change')
        self._saved_host = data.get('host')

    def _save_state(self) -> None:
        data = {'device_id': self.cfg.get('device_id'), 'host': getattr(self.plug, 'host', None),
                'owner': self.owner, 'auto_off_at': self.auto_off_at,
                'turned_on_at': self.turned_on_at, 'last_change': self.last_change}
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_file.with_suffix('.tmp')
            with open(tmp, 'w') as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.state_file)
        except Exception as e:
            logger.error(f"[PLUG] could not persist state: {e}")

    # ---------------- lifecycle ----------------
    def start(self) -> bool:
        if not self.enabled:
            logger.info("[PLUG] no smart_plug configured for this unit — service inert")
            return False
        if self._running:
            return True
        self._running = True
        from core.bus import get_bus
        get_bus().subscribe('system', self._on_system_event)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SmartPlug")
        self._thread.start()
        logger.info(f"[PLUG] started: {self.cfg.get('alias') or self.cfg.get('device_id')} "
                    f"@ {self.plug.host}, on at SG level >= {self.cfg['trigger_level']}, "
                    f"auto-off {self.cfg['auto_off_minutes']} min, owner={self.owner}")
        return True

    def stop(self) -> None:
        self._running = False
        self._queue.put(None)

    def _on_system_event(self, event) -> None:
        # Bus callback: never do network I/O here, hand over to the worker.
        try:
            if event.subtype == 'sg_escalation' and event.data.get('action') == 'intervention_started':
                self._queue.put(('sg_trigger', int(event.data.get('escalation_level') or 0)))
        except Exception as e:
            logger.debug(f"[PLUG] event ignored: {e}")

    def _loop(self) -> None:
        while self._running:
            try:
                item = self._queue.get(timeout=self.TICK_S)
            except queue.Empty:
                item = 'tick'
            if item is None:
                break
            try:
                if item == 'tick':
                    self.tick()
                elif item[0] == 'sg_trigger':
                    self.on_sg_trigger(item[1])
            except Exception as e:
                logger.error(f"[PLUG] worker error: {e}")

    # ---------------- the rule ----------------
    def on_sg_trigger(self, level: int) -> Optional[str]:
        """Silent Guardian started an intervention at `level`. Returns what was done."""
        if not self.enabled or level < int(self.cfg['trigger_level']):
            return None
        with self._lock:
            now = self._clock()
            try:
                is_on = self.plug.is_on()
            except PlugError as e:
                self._note_error(e)
                return 'error'
            self.last_known_on, self.last_error = is_on, None
            deadline = now + float(self.cfg['auto_off_minutes']) * 60.0
            if not is_on:
                try:
                    self.plug.set(True)
                except PlugError as e:
                    self._note_error(e)
                    return 'error'
                self.owner, self.auto_off_at, self.turned_on_at = 'robot', deadline, now
                self.last_known_on = True
                self._out_of_sg_since = None
                self._changed(True, 'sg_level', level=level)
                return 'turned_on'
            if self.owner == 'robot':
                self.auto_off_at = deadline          # still barking: restart the timer
                self._save_state()
                logger.info(f"[PLUG] SG level {level}: already on (robot-owned), auto-off timer restarted")
                return 'extended'
            logger.info(f"[PLUG] SG level {level}: plug already on (not turned on by the robot) — left alone")
            return 'left_on'

    def tick(self) -> Optional[str]:
        """Turn a robot-owned plug off when its timer ran out or SG mode ended."""
        if not self.enabled:
            return None
        with self._lock:
            if self.owner != 'robot':
                self._out_of_sg_since = None
                return None
            now = self._clock()
            reason = None
            if self.auto_off_at is not None and now >= self.auto_off_at:
                reason = 'timer'
            else:
                try:
                    in_sg = bool(self._in_sg())
                except Exception:
                    in_sg = True     # unknown mode: let the timer decide
                if in_sg:
                    self._out_of_sg_since = None
                else:
                    if self._out_of_sg_since is None:
                        self._out_of_sg_since = now
                    if now - self._out_of_sg_since >= float(self.cfg['exit_grace_s']):
                        reason = 'sg_ended'
            if reason is None or now < self._next_off_attempt:
                return None
            try:
                self.plug.set(False)
            except PlugError as e:
                self._next_off_attempt = now + float(self.cfg['retry_s'])
                self._note_error(e)
                return 'error'
            self.owner, self.auto_off_at, self.turned_on_at = None, None, None
            self.last_known_on, self.last_error = False, None
            self._out_of_sg_since = None
            self._changed(False, reason)
            return 'turned_off'

    def set_manual(self, action: str, source: str = 'manual') -> Dict[str, Any]:
        """Ad-hoc on/off/toggle. Always wins and clears robot ownership."""
        if not self.enabled:
            return {'ok': False, 'error': {'code': 'no_plug', 'message': 'no smart plug configured on this robot'}}
        action = str(action or '').strip().lower()
        action = {'1': 'on', 'true': 'on', '0': 'off', 'false': 'off'}.get(action, action)
        if action not in ('on', 'off', 'toggle'):
            return {'ok': False, 'error': {'code': 'invalid_action',
                                           'message': f"action {action!r} must be on|off|toggle|status"}}
        with self._lock:
            try:
                if action == 'toggle':
                    action = 'off' if self.plug.is_on() else 'on'
                want = action == 'on'
                self.plug.set(want)
            except PlugError as e:
                self._note_error(e)
                return {'ok': False, 'error': {'code': 'plug_unreachable', 'message': str(e)},
                        **self._status_fields()}
            was_robot = self.owner == 'robot'
            self.owner, self.auto_off_at, self.turned_on_at = None, None, None
            self.last_known_on, self.last_error = want, None
            self._out_of_sg_since = None
            self._changed(want, 'manual', source=source, overrode_robot=was_robot)
            return {'ok': True, **self._status_fields()}

    def status(self, refresh: bool = True) -> Dict[str, Any]:
        if not self.enabled:
            return {'ok': True, 'configured': False}
        with self._lock:
            if refresh:
                try:
                    self.last_known_on, self.last_error = self.plug.is_on(), None
                except PlugError as e:
                    self._note_error(e)
            return {'ok': True, **self._status_fields()}

    # ---------------- helpers ----------------
    def _status_fields(self) -> Dict[str, Any]:
        now = self._clock()
        return {
            'configured': True,
            'alias': self.cfg.get('alias'),
            'host': self.plug.host,
            'on': self.last_known_on,
            'reachable': self.last_error is None,
            'owner': self.owner or 'manual',     # 'robot' = robot will turn it off
            'auto_off_at': self.auto_off_at,
            'auto_off_in_s': max(0, int(self.auto_off_at - now)) if self.auto_off_at else None,
            'trigger_level': int(self.cfg['trigger_level']),
            'auto_off_minutes': self.cfg['auto_off_minutes'],
            'last_change': self.last_change,
            'last_error': self.last_error,
        }

    def _note_error(self, e: Exception) -> None:
        self.last_error = str(e)
        logger.warning(f"[PLUG] {e}")

    def _changed(self, on: bool, reason: str, **extra) -> None:
        self.last_change = {'on': on, 'reason': reason, 'time': self._clock(), **extra}
        self._save_state()
        logger.info(f"[PLUG] {'ON' if on else 'OFF'} ({reason}) {extra or ''} "
                    f"owner={self.owner or 'manual'}"
                    + (f" auto-off in {int(self.auto_off_at - self._clock())}s" if self.auto_off_at else ''))
        try:
            from core.bus import publish_system_event
            publish_system_event('smart_plug_state', {
                'on': on, 'reason': reason, 'alias': self.cfg.get('alias'),
                'owner': self.owner or 'manual', 'auto_off_at': self.auto_off_at, **extra,
            }, 'smart_plug')
        except Exception as e:
            logger.debug(f"[PLUG] bus publish failed: {e}")


_service: Optional[SmartPlugService] = None
_service_lock = threading.Lock()


def get_smart_plug_service() -> SmartPlugService:
    global _service
    with _service_lock:
        if _service is None:
            _service = SmartPlugService()
        return _service
