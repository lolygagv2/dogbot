#!/usr/bin/env python3
"""
ControllerManager — remote Bluetooth game-controller pairing for WIM-Z.

Lets the phone app drive the robot's Bluetooth stack (scan / pair / trust /
forget / reconnect) over the SAME WebSocket channel as motor/servo commands,
so a non-technical owner can re-pair a dropped Xbox controller from the couch
instead of SSH + bluetoothctl.

This module is the single authoritative brain. Both transports call into it:
  - cloud relay  : services/cloud/relay_client.py  (controller_* commands)
  - local AP / ws: api/ws.py                        (controller_* commands)

It is deliberately decoupled from the *input* path
(services/control/xbox_controller.py + xbox_hybrid_controller.py, which read
/dev/input/js0). This module only manages the BlueZ bond/connection; once a pad
is connected here it shows up as js0 and the existing input path takes over.

Design rules honored here:
  - NO hardware init at import time (CLAUDE.md / memory: import-time HW init has
    crashed the Pi). Nothing touches Bluetooth until start() is called.
  - bluetoothctl calls take seconds -> everything runs on a worker thread; the
    asyncio relay/ws loops are never blocked. Command handlers enqueue work and
    return an immediate ack.
  - Events carry BOTH 'type' and 'event' keys set to the same value, so the app
    works whether it keys on 'type' (per the app brief) or 'event' (the legacy
    relay field) — this de-risks the field-drift the brief explicitly flagged.

Persistence / root-cause fixes for "random unpairing":
  1. Trust + durable allowlist at ~/.wimz/trusted_controllers.json.
  2. Auto-reconnect daemon: periodically `connect` every trusted-but-dropped pad.
  3. NoInputNoOutput agent so pairing never blocks on a PIN prompt.
  (ERTM disable lives in /etc/modprobe.d/bluetooth.conf — system-level, applied
   separately; see docs.)
"""

import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Strip the ANSI color codes bluetoothctl 5.66 emits.
_ANSI = re.compile(r'\x1b\[[0-9;]*m')

# A BlueZ MAC, e.g. AC:8E:BD:4A:0F:97
_MAC = re.compile(r'([0-9A-F]{2}(?::[0-9A-F]{2}){5})', re.IGNORECASE)

# `[NEW] Device AC:8E:BD:4A:0F:97 Xbox Wireless Controller`
_NEW_DEVICE = re.compile(
    r'Device\s+([0-9A-F]{2}(?::[0-9A-F]{2}){5})\s*(.*)', re.IGNORECASE
)

# `Battery Percentage: 0x32 (50)` -> 50
_BATTERY = re.compile(r'Battery Percentage:\s*0x[0-9a-fA-F]+\s*\((\d+)\)')

# Pull hex fields out of bluetoothctl `info` / scan `[CHG]` lines.
_CLASS = re.compile(r'\bClass:\s*(0x[0-9a-fA-F]+)')
_APPEARANCE = re.compile(r'\bAppearance:\s*(0x[0-9a-fA-F]+)')
_MODALIAS = re.compile(r'\bModalias:\s*usb:v([0-9A-Fa-f]{4})p([0-9A-Fa-f]{4})')

# Name fallback only — primary classification is by BT device class / appearance
# (see _looks_like_controller). Kept loose so an odd-named pad still surfaces.
_CONTROLLER_NAME_HINTS = (
    'xbox', 'gamepad', 'controller', 'joystick', 'joypad',
    '8bitdo', 'dualshock', 'dualsense', 'playstation', 'joy-con', 'joycon',
    'stadia', 'nintendo switch pro',
)

# USB vendor id -> app `kind`. Used when a Modalias is available.
_VENDOR_KIND = {
    '045e': 'xbox',         # Microsoft
    '054c': 'playstation',  # Sony
    '2dc8': '8bitdo',       # 8BitDo
}

ALLOWLIST_PATH = os.path.expanduser('~/.wimz/trusted_controllers.json')


def _looks_like_controller(name: str = '', icon: str = '',
                           dev_class: Optional[int] = None,
                           appearance: Optional[int] = None) -> bool:
    """Brand-agnostic gamepad/joystick test (per the app brief: classify by
    BT device class, not by the Xbox name)."""
    # BR/EDR Class of Device: major device class (bits 8-12) == 0x05 peripheral,
    # minor (bits 2-7) joystick(0x01)/gamepad(0x02)/  (0x03 = both bits set).
    if dev_class is not None:
        major = (dev_class >> 8) & 0x1F
        minor = (dev_class >> 2) & 0x3F
        if major == 0x05 and (minor & 0x03):
            return True
    # BLE Appearance: HID category (0x03C0..0x03CF), subtype joystick(3)/gamepad(4).
    if appearance is not None:
        if (appearance >> 6) == 0x0F and (appearance & 0x3F) in (0x03, 0x04):
            return True
    if icon and 'input-gaming' in icon.lower():
        return True
    n = (name or '').lower()
    return any(h in n for h in _CONTROLLER_NAME_HINTS)


def _derive_kind(name: str = '', vendor: str = '') -> Optional[str]:
    """Map to the app's `kind` enum (xbox|playstation|8bitdo|generic). Returns
    None when unknown so we omit the field and the app defaults to generic."""
    if vendor:
        k = _VENDOR_KIND.get(vendor.lower())
        if k:
            return k
    n = (name or '').lower()
    if 'xbox' in n:
        return 'xbox'
    if any(h in n for h in ('dualsense', 'dualshock', 'playstation', 'sony')):
        return 'playstation'
    if '8bitdo' in n:
        return '8bitdo'
    return None


class ControllerManager:
    """Singleton. Created lazily via get_controller_manager()."""

    def __init__(self):
        self._lock = threading.RLock()
        self._work: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._reconnect_thread: Optional[threading.Thread] = None
        self._scan_thread: Optional[threading.Thread] = None
        self._scan_proc: Optional[subprocess.Popen] = None
        self._agent_proc: Optional[subprocess.Popen] = None
        self._running = False
        self._scanning = False
        self._emitters: List[Callable[[dict], None]] = []
        # Per-address backoff bookkeeping for the auto-reconnect daemon.
        self._next_attempt: Dict[str, float] = {}
        self._backoff: Dict[str, float] = {}
        # Cache last connected-set so the daemon can emit a spontaneous status
        # only when something actually changes.
        self._last_connected: set = set()
        self._allowlist: Dict[str, dict] = {}

    # ---- lifecycle -------------------------------------------------------

    def start(self):
        with self._lock:
            if self._running:
                return
            self._running = True
        self._allowlist = self._load_allowlist()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="ControllerMgrWorker")
        self._worker.start()
        self._reconnect_thread = threading.Thread(
            target=self._reconnect_loop, daemon=True, name="ControllerMgrReconnect")
        self._reconnect_thread.start()
        # Best-effort: a persistent NoInputNoOutput agent so pairing auto-accepts.
        self._start_agent()
        logger.info("ControllerManager started (trusted=%d)", len(self._allowlist))

    def stop(self):
        with self._lock:
            self._running = False
            self._scanning = False
        self._kill_proc('_scan_proc')
        self._kill_proc('_agent_proc')

    # ---- emitter plumbing ------------------------------------------------

    def register_emitter(self, fn: Callable[[dict], None]):
        """Register a transport sink. fn receives the full event dict (which
        already contains a 'type' key). Safe to call multiple times; dead
        transports should just no-op inside fn."""
        with self._lock:
            if fn not in self._emitters:
                self._emitters.append(fn)

    def _emit(self, payload: dict):
        # Carry both keys (see module docstring).
        payload.setdefault('event', payload.get('type'))
        for fn in list(self._emitters):
            try:
                fn(dict(payload))
            except Exception as e:
                logger.debug("controller emitter error: %s", e)

    # ---- public command surface -----------------------------------------
    # Each enqueues work and returns immediately so the asyncio caller never
    # blocks on bluetoothctl. Spontaneous results arrive via _emit().

    def handle_command(self, command: str, params: dict) -> dict:
        """Dispatch a controller_* command. Returns a small ack dict for the
        transport to relay synchronously; the real payload follows via _emit."""
        if not self._running:
            # Lazy autostart so a command works even if main() didn't start us.
            self.start()
        params = params or {}
        addr = self._norm_mac(params.get('address'))

        if command == 'controller_status':
            self._enqueue(self._do_status)
        elif command == 'controller_scan':
            enable = bool(params.get('enable', True))
            self._enqueue(lambda: self._do_scan(enable))
        elif command == 'controller_pair':
            if not addr:
                return {'success': False, 'error': 'address required'}
            self._enqueue(lambda: self._do_pair(addr))
        elif command == 'controller_trust':
            if not addr:
                return {'success': False, 'error': 'address required'}
            trusted = bool(params.get('trusted', True))
            self._enqueue(lambda: self._do_trust(addr, trusted))
        elif command == 'controller_forget':
            if not addr:
                return {'success': False, 'error': 'address required'}
            self._enqueue(lambda: self._do_forget(addr))
        elif command == 'controller_reconnect':
            if not addr:
                return {'success': False, 'error': 'address required'}
            self._enqueue(lambda: self._do_reconnect(addr))
        else:
            return {'success': False, 'error': f'unknown controller command {command}'}
        return {'success': True}

    # ---- worker thread ---------------------------------------------------

    def _enqueue(self, fn: Callable[[], None]):
        self._work.put(fn)

    def _worker_loop(self):
        while True:
            with self._lock:
                if not self._running and self._work.empty():
                    return
            try:
                fn = self._work.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                fn()
            except Exception as e:
                logger.error("controller work error: %s", e)
                self._emit({
                    'type': 'controller_error',
                    'code': 'INTERNAL',
                    'message': str(e),
                })

    # ---- command implementations (run on worker thread) ------------------

    def _do_status(self):
        self._emit(self.snapshot())

    def _do_pair(self, addr: str):
        self._emit({'type': 'controller_pair_progress', 'address': addr,
                    'stage': 'pairing', 'message': 'Pairing…'})
        info = self._info(addr)
        # Idempotent: if already bonded, skip straight to connect.
        if not info.get('paired'):
            rc, out = self._bctl(['pair', addr], timeout=30)
            if rc != 0 and 'already' not in out.lower() and \
               not self._info(addr).get('paired'):
                self._emit({'type': 'controller_error', 'code': 'PAIR_FAILED',
                            'message': self._reason(out) or
                            'Controller not in pairing mode'})
                self._emit(self.snapshot())
                return
        self._emit({'type': 'controller_pair_progress', 'address': addr,
                    'stage': 'connecting', 'message': 'Connecting…'})
        self._bctl(['connect', addr], timeout=20)
        # Auto-trust on successful pair so the pad survives reboot without a
        # second round-trip; the app's Trust toggle still reflects/edits this.
        if self._info(addr).get('connected'):
            self._do_trust(addr, True, emit_status=False)
            self._emit({'type': 'controller_pair_progress', 'address': addr,
                        'stage': 'done', 'message': 'Connected'})
        self._emit(self.snapshot())

    def _do_trust(self, addr: str, trusted: bool, emit_status: bool = True):
        self._bctl(['trust' if trusted else 'untrust', addr], timeout=10)
        if trusted:
            self._allowlist_add(addr)
        else:
            self._allowlist_remove(addr)
        if emit_status:
            self._emit(self.snapshot())

    def _do_forget(self, addr: str):
        self._bctl(['disconnect', addr], timeout=10)
        self._bctl(['remove', addr], timeout=10)
        self._allowlist_remove(addr)
        self._next_attempt.pop(addr, None)
        self._backoff.pop(addr, None)
        self._emit(self.snapshot())

    def _do_reconnect(self, addr: str):
        self._emit({'type': 'controller_pair_progress', 'address': addr,
                    'stage': 'connecting', 'message': 'Reconnecting…'})
        rc, out = self._bctl(['connect', addr], timeout=20)
        if not self._info(addr).get('connected'):
            self._emit({'type': 'controller_error', 'code': 'CONNECT_FAILED',
                        'message': self._reason(out) or
                        'Controller did not respond (is it awake?)'})
        else:
            self._emit({'type': 'controller_pair_progress', 'address': addr,
                        'stage': 'done', 'message': 'Connected'})
        self._emit(self.snapshot())

    # ---- scanning --------------------------------------------------------

    def _do_scan(self, enable: bool):
        if enable:
            if self._scanning:
                self._emit(self.snapshot())
                return
            self._scanning = True
            self._scan_thread = threading.Thread(
                target=self._scan_loop, daemon=True, name="ControllerMgrScan")
            self._scan_thread.start()
        else:
            self._scanning = False
            self._kill_proc('_scan_proc')
        self._emit(self.snapshot())

    def _scan_loop(self):
        """Run an interactive bluetoothctl session with discovery on and trickle
        a controller_scan_result per controller-like device as BlueZ reports it.
        Auto-stops after a safety window so we never leave discovery on forever."""
        deadline = time.time() + 60.0  # safety cap; app re-enables to extend
        try:
            self._scan_proc = subprocess.Popen(
                ['bluetoothctl'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1)
            self._scan_proc.stdin.write('scan on\n')
            self._scan_proc.stdin.flush()
            # Accumulate per-address attributes as BlueZ trickles [NEW]/[CHG]
            # lines, then classify by device class (not name) before emitting.
            attrs: Dict[str, dict] = {}
            while self._scanning and time.time() < deadline:
                line = self._scan_proc.stdout.readline()
                if not line:
                    break
                line = _ANSI.sub('', line).strip()
                if '[NEW]' not in line and '[CHG]' not in line:
                    continue
                m = _NEW_DEVICE.search(line)
                if not m:
                    continue
                addr = m.group(1).upper()
                rest = m.group(2).strip()
                a = attrs.setdefault(
                    addr, {'name': '', 'dev_class': None, 'appearance': None,
                           'icon': '', 'rssi': None, 'emitted': False})
                # The trailing text is either the device name (NEW) or a
                # "Field: value" change (CHG). Pull out the bits we classify on.
                if rest.startswith('RSSI:'):
                    a['rssi'] = self._rssi_from(line)
                elif rest.startswith('Class:'):
                    cm = _CLASS.search(line)
                    if cm:
                        a['dev_class'] = int(cm.group(1), 16)
                elif rest.startswith('Appearance:'):
                    am = _APPEARANCE.search(line)
                    if am:
                        a['appearance'] = int(am.group(1), 16)
                elif rest.startswith('Icon:'):
                    a['icon'] = rest.split(':', 1)[1].strip()
                elif rest.startswith('Name:'):
                    a['name'] = rest.split(':', 1)[1].strip()
                elif rest and ':' not in rest:
                    a['name'] = rest  # bare name on the [NEW] line
                if a['rssi'] is None:
                    a['rssi'] = self._rssi_from(line) or a['rssi']

                if a['emitted']:
                    continue
                if _looks_like_controller(a['name'], a['icon'],
                                          a['dev_class'], a['appearance']):
                    a['emitted'] = True
                    evt = {'type': 'controller_scan_result', 'address': addr,
                           'name': a['name'] or 'Game Controller',
                           'rssi': a['rssi']}
                    kind = _derive_kind(a['name'])
                    if kind:
                        evt['kind'] = kind
                    self._emit(evt)
        except Exception as e:
            logger.error("scan loop error: %s", e)
            self._emit({'type': 'controller_error', 'code': 'SCAN_FAILED',
                        'message': str(e)})
        finally:
            try:
                if self._scan_proc and self._scan_proc.stdin:
                    self._scan_proc.stdin.write('scan off\n')
                    self._scan_proc.stdin.write('quit\n')
                    self._scan_proc.stdin.flush()
            except Exception:
                pass
            self._kill_proc('_scan_proc')
            self._scanning = False
            self._emit(self.snapshot())

    @staticmethod
    def _rssi_from(line: str) -> Optional[int]:
        m = re.search(r'RSSI:\s*(0x[0-9a-fA-F]+|-?\d+)', line)
        if not m:
            return None
        try:
            val = int(m.group(1), 0)
            # BlueZ reports RSSI as a signed byte; normalize 0xNN -> negative.
            if val > 127:
                val -= 256
            return val
        except ValueError:
            return None

    # ---- auto-reconnect daemon ------------------------------------------

    def _reconnect_loop(self):
        """For every trusted-but-disconnected pad, periodically issue connect
        with per-address exponential backoff (10s -> 30s). Emits a spontaneous
        controller_status whenever the connected-set changes."""
        while True:
            with self._lock:
                if not self._running:
                    return
            try:
                now = time.time()
                connected_now = set(self._connected_addresses())

                # Spontaneous status if anything connected/dropped on its own.
                if connected_now != self._last_connected:
                    self._last_connected = connected_now
                    self._emit(self.snapshot())

                if not self._scanning:  # don't fight an active discovery session
                    for addr in list(self._allowlist.keys()):
                        if addr in connected_now:
                            self._backoff.pop(addr, None)
                            self._next_attempt.pop(addr, None)
                            continue
                        if now < self._next_attempt.get(addr, 0):
                            continue
                        self._bctl(['connect', addr], timeout=12)
                        delay = min(30.0, self._backoff.get(addr, 10.0))
                        self._backoff[addr] = min(30.0, delay * 1.5)
                        self._next_attempt[addr] = now + delay
            except Exception as e:
                logger.debug("reconnect loop error: %s", e)
            time.sleep(5.0)

    # ---- bluetoothctl helpers -------------------------------------------

    def _bctl(self, args: List[str], timeout: float = 10) -> (int, str):
        """Run a single-shot bluetoothctl subcommand. Returns (rc, clean_stdout)."""
        try:
            p = subprocess.run(
                ['bluetoothctl'] + args, capture_output=True, text=True,
                timeout=timeout)
            out = _ANSI.sub('', (p.stdout or '') + (p.stderr or ''))
            logger.debug("bctl %s -> rc=%s", ' '.join(args), p.returncode)
            return p.returncode, out
        except subprocess.TimeoutExpired:
            logger.warning("bluetoothctl %s timed out", ' '.join(args))
            return 124, 'timeout'
        except Exception as e:
            logger.error("bluetoothctl %s failed: %s", ' '.join(args), e)
            return 1, str(e)

    def _info(self, addr: str) -> dict:
        rc, out = self._bctl(['info', addr], timeout=8)
        d = {'address': addr, 'paired': False, 'trusted': False,
             'connected': False, 'name': None, 'icon': '',
             'dev_class': None, 'appearance': None, 'vendor': ''}
        if rc != 0 and 'Name:' not in out:
            return d
        for raw in out.splitlines():
            ln = raw.strip()
            if ln.startswith('Name:'):
                d['name'] = ln.split(':', 1)[1].strip()
            elif ln.startswith('Icon:'):
                d['icon'] = ln.split(':', 1)[1].strip()
            elif ln.startswith('Paired:'):
                d['paired'] = ln.endswith('yes')
            elif ln.startswith('Trusted:'):
                d['trusted'] = ln.endswith('yes')
            elif ln.startswith('Connected:'):
                d['connected'] = ln.endswith('yes')
            elif ln.startswith('Class:'):
                m = _CLASS.search(ln)
                if m:
                    d['dev_class'] = int(m.group(1), 16)
            elif ln.startswith('Appearance:'):
                m = _APPEARANCE.search(ln)
                if m:
                    d['appearance'] = int(m.group(1), 16)
            elif ln.startswith('Modalias:'):
                m = _MODALIAS.search(ln)
                if m:
                    d['vendor'] = m.group(1)
            elif ln.startswith('Battery Percentage:'):
                m = _BATTERY.search(ln)
                if m:
                    d['battery'] = int(m.group(1))
        d['kind'] = _derive_kind(d.get('name') or '', d.get('vendor') or '')
        return d

    def _known_devices(self) -> List[tuple]:
        rc, out = self._bctl(['devices'], timeout=8)
        devs = []
        for raw in out.splitlines():
            ln = _ANSI.sub('', raw).strip()
            m = _NEW_DEVICE.search(ln)
            if m:
                devs.append((m.group(1).upper(), m.group(2).strip()))
        return devs

    def _connected_addresses(self) -> List[str]:
        rc, out = self._bctl(['devices', 'Connected'], timeout=8)
        addrs = []
        for raw in out.splitlines():
            m = _NEW_DEVICE.search(_ANSI.sub('', raw))
            if m:
                addrs.append(m.group(1).upper())
        return addrs

    def snapshot(self) -> dict:
        """Build the authoritative controller_status payload."""
        # Union of currently-known devices and our durable allowlist, so a
        # trusted pad that BlueZ has dropped from `devices` still shows up.
        addrs = {}
        for addr, name in self._known_devices():
            addrs[addr] = name
        for addr, meta in self._allowlist.items():
            addrs.setdefault(addr, meta.get('name'))

        controllers = []
        active = None
        for addr, name in addrs.items():
            info = self._info(addr)
            disp = info.get('name') or name or ''
            is_ctrl = _looks_like_controller(
                disp, info.get('icon', ''), info.get('dev_class'),
                info.get('appearance'))
            if not is_ctrl and addr not in self._allowlist:
                # Keep allowlisted entries even if BlueZ can't classify them.
                continue
            entry = {
                'address': addr,
                'name': disp or 'Game Controller',
                'paired': bool(info.get('paired')),
                'trusted': bool(info.get('trusted')),
                'connected': bool(info.get('connected')),
            }
            kind = info.get('kind')
            if kind:
                entry['kind'] = kind
            if 'battery' in info:
                entry['battery'] = info['battery']
            controllers.append(entry)
            if entry['connected'] and active is None:
                active = addr

        return {
            'type': 'controller_status',
            'scanning': bool(self._scanning),
            'active_address': active,
            'controllers': controllers,
        }

    # ---- agent -----------------------------------------------------------

    def _start_agent(self):
        """Persistent NoInputNoOutput agent so pairing auto-accepts (no PIN)."""
        try:
            self._agent_proc = subprocess.Popen(
                ['bluetoothctl'], stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
            self._agent_proc.stdin.write('agent NoInputNoOutput\n')
            self._agent_proc.stdin.write('default-agent\n')
            self._agent_proc.stdin.flush()
        except Exception as e:
            logger.warning("could not start BT agent: %s", e)

    # ---- allowlist persistence ------------------------------------------

    def _load_allowlist(self) -> Dict[str, dict]:
        try:
            with open(ALLOWLIST_PATH) as f:
                data = json.load(f)
            ctrls = data.get('controllers', {})
            return {self._norm_mac(k): v for k, v in ctrls.items() if k}
        except FileNotFoundError:
            return {}
        except Exception as e:
            logger.warning("allowlist load failed: %s", e)
            return {}

    def _save_allowlist(self):
        try:
            os.makedirs(os.path.dirname(ALLOWLIST_PATH), exist_ok=True)
            tmp = ALLOWLIST_PATH + '.tmp'
            with open(tmp, 'w') as f:
                json.dump({'controllers': self._allowlist}, f, indent=2)
            os.replace(tmp, ALLOWLIST_PATH)
        except Exception as e:
            logger.error("allowlist save failed: %s", e)

    def _allowlist_add(self, addr: str):
        info = self._info(addr)
        self._allowlist[addr] = {'name': info.get('name'),
                                 'added': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                        time.gmtime())}
        self._save_allowlist()

    def _allowlist_remove(self, addr: str):
        if addr in self._allowlist:
            del self._allowlist[addr]
            self._save_allowlist()

    # ---- misc helpers ----------------------------------------------------

    @staticmethod
    def _norm_mac(addr) -> Optional[str]:
        if not addr:
            return None
        m = _MAC.search(str(addr))
        return m.group(1).upper() if m else None

    @staticmethod
    def _reason(out: str) -> str:
        """Pull a human-ish failure reason out of bluetoothctl chatter."""
        for ln in (out or '').splitlines():
            ln = ln.strip()
            if 'Failed' in ln or 'not available' in ln or 'org.bluez.Error' in ln:
                return ln
        return ''

    def _kill_proc(self, attr: str):
        p = getattr(self, attr, None)
        if p is not None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            setattr(self, attr, None)


# Singleton ------------------------------------------------------------------
_instance: Optional[ControllerManager] = None
_instance_lock = threading.Lock()


def get_controller_manager() -> ControllerManager:
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = ControllerManager()
    return _instance
