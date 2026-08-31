"""OTA update manager — the main-app slice of the OTA contract (2026-08-07).

The heavy lifting (download/verify/install/flip/rollback) is done by the
standalone stdlib-only wimz-updater (scripts/ota/wimz_updater.py, installed
at /home/morgan/wimz/updater/ by the bootstrap script). This module is the
main app's thin interface to it:

1. `handle_start_update()` — the app's `start_update` WS command lands here
   (via relay -> bus -> main_treatbot). Runs the safety gates; a refusal is
   emitted as `update_status {state: "failed", error: <reason>}` because
   the app has no separate refusal channel. On pass it writes the request
   file; the wimz-updater.path systemd unit takes it from there.
2. A watcher thread tails /home/morgan/wimz/update-status.json and forwards
   every state change to the relay as an `update_status` event (that's the
   ONLY progress channel the app renders).
3. The same watcher consumes update-result.json — written by the updater
   after the restart/health/rollback dance — and emits the terminal
   success/failed/rolled_back event. This is how "the NEW code must emit
   success" is satisfied: the old process dies at `restarting`; whichever
   process is alive afterwards (new code on success, old code on rollback)
   finds the result file and closes the loop. Consumed results are renamed
   so they are never re-emitted.

Update availability is app-side (relay latest.version vs telemetry
sw_version, plain string compare) — the robot only reports sw_version.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger('OTAManager')

WIMZ_ROOT = Path('/home/morgan/wimz')
REQUEST = WIMZ_ROOT / 'update-request.json'
STATUS = WIMZ_ROOT / 'update-status.json'
RESULT = WIMZ_ROOT / 'update-result.json'
UPDATER = WIMZ_ROOT / 'updater' / 'wimz_updater.py'

TERMINAL_STATES = {'success', 'failed', 'rolled_back'}
MIN_BATTERY_PCT = 30
# A non-terminal status younger than this means an update is in flight
IN_FLIGHT_STALENESS = 3600


class OTAManager:
    def __init__(self):
        self._relay_client = None
        self._webrtc_service = None
        self._watcher: Optional[threading.Thread] = None
        self._running = False
        # Status snapshot present at process start — never re-forwarded
        # (stale states from before a restart would confuse the app).
        self._last_status_raw: Optional[str] = None

    def start(self, relay_client=None, webrtc_service=None):
        """Wire service refs and start the status/result watcher thread."""
        self._relay_client = relay_client
        self._webrtc_service = webrtc_service
        if self._watcher and self._watcher.is_alive():
            return
        try:
            self._last_status_raw = STATUS.read_text() if STATUS.exists() else None
        except Exception:
            self._last_status_raw = None
        self._running = True
        self._watcher = threading.Thread(target=self._watch_loop, daemon=True,
                                         name='OTAWatcher')
        self._watcher.start()
        logger.info("OTA manager started (layout %s)",
                    'present' if UPDATER.exists() else 'NOT bootstrapped')

    def stop(self):
        self._running = False

    # ---- command handling -------------------------------------------------

    def handle_start_update(self, params: Dict[str, Any]):
        """Handle the app's start_update command. Gate, then hand off."""
        data = params.get('data') if isinstance(params.get('data'), dict) else {}
        version = str(params.get('version') or data.get('version') or '').strip()

        if not version:
            return self._refuse(version, 'no target version in start_update command')

        reason = self._gate_check()
        if reason:
            return self._refuse(version, reason)

        self._emit({'state': 'checking', 'version': version})
        try:
            tmp = REQUEST.with_suffix('.json.tmp')
            tmp.write_text(json.dumps({
                'version': version,
                'requested_at': time.time(),
            }))
            os.replace(tmp, REQUEST)
            logger.info(f"OTA: update to {version} requested — handed to wimz-updater")
        except Exception as e:
            self._refuse(version, f'could not write update request: {e}')

    def _gate_check(self) -> Optional[str]:
        """Return a human-readable refusal reason, or None if clear to update."""
        if not UPDATER.exists():
            return 'robot updater not installed yet (OTA bootstrap pending)'

        if REQUEST.exists():
            return 'an update is already queued'
        st = self._read_json(STATUS)
        if st and st.get('state') not in TERMINAL_STATES \
                and time.time() - st.get('ts', 0) < IN_FLIGHT_STALENESS:
            return f"an update is already in progress (state: {st.get('state')})"

        try:
            from core.state import get_state, SystemMode
            mode = get_state().get_mode()
            if mode != SystemMode.IDLE:
                return f'robot is not idle (mode: {mode.value})'
        except Exception as e:
            return f'cannot verify robot mode: {e}'

        try:
            from services.power.battery_monitor import get_battery_monitor
            batt = get_battery_monitor().get_status()
            pct = batt.get('percentage')
            if pct is not None and pct < MIN_BATTERY_PCT:
                return f'battery too low ({pct}% < {MIN_BATTERY_PCT}%)'
            if (batt.get('status') or '').upper() == 'CRITICAL':
                return 'battery critical'
        except Exception as e:
            logger.warning(f"OTA: battery gate unavailable ({e}) — allowing")

        try:
            if self._webrtc_service and getattr(self._webrtc_service,
                                                'connections', None):
                if len(self._webrtc_service.connections) > 0:
                    return 'live video session active — close it and retry'
        except Exception:
            pass

        return None

    def _refuse(self, version: str, reason: str):
        logger.warning(f"OTA: refusing start_update ({reason})")
        self._emit({'state': 'failed', 'version': version or 'unknown',
                    'error': reason})

    # ---- status/result forwarding -----------------------------------------

    def _watch_loop(self):
        while self._running:
            try:
                self._forward_status_change()
                self._consume_result()
            except Exception as e:
                logger.debug(f"OTA watcher error: {e}")
            time.sleep(1.0)

    def _forward_status_change(self):
        if not STATUS.exists():
            return
        raw = STATUS.read_text()
        if raw == self._last_status_raw:
            return
        self._last_status_raw = raw
        st = json.loads(raw)
        state = st.get('state')
        # Terminal states are emitted from the RESULT file (exactly once,
        # consumed); forwarding them from the status file too would duplicate.
        if state in TERMINAL_STATES:
            return
        payload = {'state': state, 'version': st.get('version')}
        if 'progress_pct' in st:
            payload['progress_pct'] = st['progress_pct']
        if 'error' in st:
            payload['error'] = st['error']
        self._emit(payload)

    def _consume_result(self):
        if not RESULT.exists():
            return
        res = self._read_json(RESULT)
        consumed = RESULT.with_name('update-result.consumed.json')
        try:
            os.replace(RESULT, consumed)
        except Exception as e:
            logger.error(f"OTA: could not consume result file: {e}")
            return
        if not res:
            return
        payload = {'state': res.get('state'), 'version': res.get('version')}
        if res.get('error'):
            payload['error'] = res['error']
        self._emit(payload)
        logger.info(f"OTA terminal: {payload}")

    @staticmethod
    def _read_json(path: Path) -> Optional[dict]:
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def _emit(self, payload: Dict[str, Any]):
        """Send an update_status event to the app via the relay.

        send_event buffers while the socket is down and replays on reconnect,
        which is exactly what the post-restart terminal event needs.
        """
        try:
            if self._relay_client:
                self._relay_client.send_event('update_status', payload)
            else:
                from services.cloud.relay_client import get_relay_client
                relay = get_relay_client()
                if relay:
                    relay.send_event('update_status', payload)
        except Exception as e:
            logger.error(f"OTA: could not emit update_status: {e}")


_ota_manager: Optional[OTAManager] = None


def get_ota_manager() -> OTAManager:
    global _ota_manager
    if _ota_manager is None:
        _ota_manager = OTAManager()
    return _ota_manager
