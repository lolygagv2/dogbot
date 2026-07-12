#!/usr/bin/env python3
"""
Night Mode Controller — automatic day/night switching for NoIR camera + IR illuminator.

The IMX708 Wide NoIR sensor (IR-cut filter removed) is paired with a 48-LED 940nm
IR illuminator that has its own ambient-light sensor and toggles its LEDs autonomously.
This controller detects ambient light via the camera's own AE metadata (Lux field) and
switches a separate "night mode" state in the robot — different camera controls, optional
chassis-LED dimming, and a flag exposed to the app.

The illuminator's IR-cut signal is NOT wired to the Pi.

Files:
  - state/night_mode.json — persists the override preference across restarts (NOT the
    auto-detected day/night state; we re-detect that on every startup).
  - logs/night_mode.log — transition log; informs the future IR retraining dataset.
"""

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from core.bus import publish_system_event

logger = logging.getLogger(__name__)


# --- Tunable constants (top of file per nightvisionrobo.md spec) ---
POLL_INTERVAL_SEC = 20
LOW_THRESHOLD_LUX = 5.0           # Below this -> dark (day->night entry, sensitive)
# Exit threshold is intentionally MUCH higher than the entry threshold (asymmetric
# hysteresis). The IR illuminator floods the scene with 940nm IR when its own LDR
# decides it's dark; the NoIR camera sees that IR and libcamera computes Lux from
# total photon counts INCLUDING IR contribution. Without a high exit threshold the
# illuminator turning on would push measured lux to ~30-60 and bounce us back to day
# mode, causing oscillation. Real daylight delivers 500-10000+ lux, so 100 is a safe
# margin that filters out illuminator-driven IR contamination but trips on sunrise.
HIGH_THRESHOLD_LUX = 100.0
CONFIRM_COUNT = 3                 # Consecutive readings before switching (60s at 20s poll)
HEARTBEAT_INTERVAL_SEC = 60       # State push to app even with no transition
AE_SETTLE_SEC = 3.0               # Wait for AE to settle on IR-lit scene before locking

STATE_FILE = Path('/home/morgan/dogbot/state/night_mode.json')
LOG_FILE = Path('/home/morgan/dogbot/logs/night_mode.log')

VALID_OVERRIDES = ('auto', 'force_day', 'force_night')


# --- Camera profiles ---
# DAY_PROFILE: leave camera as-is, just clear any night locks.
# The actual daytime controls live in the robot profile yaml (loaded at detector
# init in services/perception/detector.py::_apply_saved_calibration). Switching
# back to day re-applies that saved calibration so any AWB/ColourGains tuning
# from the yaml is restored.
NIGHT_PROFILE: Dict[str, Any] = {
    "AwbEnable": False,
    "ColourGains": (1.0, 1.0),       # monochrome IR — neutral gains
    "AeEnable": False,                # locked after AE settle pass below
    "ExposureTime": 20000,            # 20ms — starting point; may be re-locked from AE
    "AnalogueGain": 4.0,              # starting point; may be re-locked from AE
    "Saturation": 0.0,                # IR-lit scenes have no useful color info
}


class NightModeController:
    """Owns the day/night state, polls ambient light, transitions the camera profile."""

    def __init__(self) -> None:
        # Public state (read by API/relay)
        self.current_mode: str = 'day'                # 'day' | 'night'
        self.override_mode: str = 'auto'              # auto | force_day | force_night
        self.last_lux: Optional[float] = None
        self.last_changed_at: float = time.time()

        # Internal
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()  # reentrant: _transition can re-acquire under set_override
        self._callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._below_count = 0
        self._above_count = 0
        self._last_heartbeat = 0.0

        # Detector handle is acquired lazily — startup order matters
        self._detector = None

        self._load_state()
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ---- Persistence ----
    def _load_state(self) -> None:
        try:
            if STATE_FILE.exists():
                data = json.loads(STATE_FILE.read_text())
                override = data.get('override_mode', 'auto')
                if override in VALID_OVERRIDES:
                    self.override_mode = override
                    logger.info(f"NightMode: loaded override={override}")
        except Exception as e:
            logger.warning(f"NightMode: failed to load state ({e}); defaulting to auto")

    def _save_state(self) -> None:
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            STATE_FILE.write_text(json.dumps({'override_mode': self.override_mode}))
        except Exception as e:
            logger.warning(f"NightMode: failed to save state ({e})")

    # ---- Public API ----
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, name='NightMode', daemon=True)
        self._thread.start()
        logger.info("NightModeController started")

        # Reconcile mode with persisted override (e.g. boot with force_night previously set)
        target: Optional[str] = None
        if self.override_mode == 'force_night' and self.current_mode != 'night':
            target = 'night'
        elif self.override_mode == 'force_day' and self.current_mode != 'day':
            target = 'day'
        if target:
            threading.Thread(
                target=self._transition,
                args=(target,),
                kwargs={'reason': f'startup_reconcile override={self.override_mode}'},
                name='NightModeReconcile',
                daemon=True,
            ).start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        logger.info("NightModeController stopped")

    def set_override(self, override: str) -> bool:
        """Update the override preference. Returns immediately; any required mode
        transition runs on a worker thread (transitions can take ~3s due to AE settle)."""
        if override not in VALID_OVERRIDES:
            return False
        target_mode: Optional[str] = None
        with self._lock:
            if override == self.override_mode:
                return True
            prev = self.override_mode
            self.override_mode = override
            self._save_state()
            logger.info(f"NightMode override: {prev} -> {override}")
            if override == 'force_night' and self.current_mode != 'night':
                target_mode = 'night'
            elif override == 'force_day' and self.current_mode != 'day':
                target_mode = 'day'

        if target_mode is not None:
            threading.Thread(
                target=self._transition,
                args=(target_mode,),
                kwargs={'reason': f'override={override}'},
                name='NightModeTransition',
                daemon=True,
            ).start()
        else:
            # No mode change needed — just notify app of the override flip
            self._push_state(reason=f'override={override}')
        return True

    def add_callback(self, cb: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a callback(mode, info_dict) invoked on every mode transition."""
        self._callbacks.append(cb)

    def get_status(self) -> Dict[str, Any]:
        return {
            'mode': self.current_mode,
            'override': self.override_mode,
            'lux': self.last_lux,
            'last_changed_at': self.last_changed_at,
        }

    # ---- Monitor loop ----
    def _get_detector(self):
        if self._detector is None:
            try:
                from services.perception.detector import get_detector_service
                self._detector = get_detector_service()
            except Exception as e:
                logger.debug(f"NightMode: detector not yet available ({e})")
        return self._detector

    def _read_lux(self) -> Optional[float]:
        det = self._get_detector()
        if det is None or not getattr(det, 'camera_initialized', False) or det.camera is None:
            return None
        try:
            meta = det.camera.capture_metadata()
        except Exception as e:
            logger.debug(f"NightMode: capture_metadata failed ({e})")
            return None
        lux = meta.get('Lux')
        if lux is not None:
            return float(lux)
        # Fallback: heuristic from exposure*gain when Lux is unavailable
        et = meta.get('ExposureTime', 0)
        ag = meta.get('AnalogueGain', 0)
        if et and ag:
            # Higher product -> darker scene. Crude proxy returned as negative lux placeholder.
            # Treat any product > 200000 (~20ms * 10x gain) as "dark" via threshold below.
            return -float(et) * float(ag) / 10000.0
        return None

    def _monitor_loop(self) -> None:
        # Small initial delay so detector has time to initialize
        self._stop_event.wait(2.0)

        while self._running and not self._stop_event.is_set():
            lux = self._read_lux()
            if lux is not None:
                with self._lock:
                    self.last_lux = lux
                self._update_hysteresis(lux)

            self._maybe_heartbeat()

            # Sleep responsively so stop() returns quickly
            self._stop_event.wait(POLL_INTERVAL_SEC)

    def _update_hysteresis(self, lux: float) -> None:
        # Override modes do not transition automatically — they just log what auto would do
        if self.override_mode != 'auto':
            return

        # Negative lux placeholders (fallback heuristic) — magnitude determines dark/light
        is_dark_now = lux < LOW_THRESHOLD_LUX
        is_light_now = lux > HIGH_THRESHOLD_LUX

        if is_dark_now:
            self._below_count += 1
            self._above_count = 0
        elif is_light_now:
            self._above_count += 1
            self._below_count = 0
        else:
            # Inside dead band — don't change counts (preserve confirmation toward last edge)
            pass

        if self.current_mode == 'day' and self._below_count >= CONFIRM_COUNT:
            self._transition('night', reason=f'lux={lux:.1f} below {LOW_THRESHOLD_LUX} x{CONFIRM_COUNT}')
            self._below_count = 0
        elif self.current_mode == 'night' and self._above_count >= CONFIRM_COUNT:
            self._transition('day', reason=f'lux={lux:.1f} above {HIGH_THRESHOLD_LUX} x{CONFIRM_COUNT}')
            self._above_count = 0

    # ---- Transitions ----
    def _transition(self, new_mode: str, reason: str) -> None:
        with self._lock:
            if new_mode == self.current_mode:
                return
            prev = self.current_mode
            self.current_mode = new_mode
            self.last_changed_at = time.time()

        logger.info(f"NightMode transition: {prev} -> {new_mode} ({reason}, lux={self.last_lux})")
        self._write_transition_log(prev, new_mode, reason)

        applied: Dict[str, Any] = {}
        try:
            if new_mode == 'night':
                applied = self._switch_to_night()
            else:
                applied = self._switch_to_day()
        except Exception as e:
            logger.error(f"NightMode: camera transition failed ({e}); state flipped anyway")

        info = {'prev': prev, 'mode': new_mode, 'reason': reason, 'lux': self.last_lux, 'applied': applied}
        for cb in self._callbacks:
            try:
                cb(new_mode, info)
            except Exception as e:
                logger.warning(f"NightMode callback error: {e}")

        try:
            publish_system_event('night_mode_changed', info)
        except Exception as e:
            logger.debug(f"NightMode: publish_system_event failed ({e})")

        self._push_state(reason=reason)

    def _switch_to_night(self) -> Dict[str, Any]:
        """Switch camera to IR-illuminated profile.

        Sequence:
          1. Enable AE temporarily and apply night-style controls so AE meters the IR scene.
          2. Wait AE_SETTLE_SEC for AE to converge.
          3. Read settled ExposureTime + AnalogueGain from metadata.
          4. Apply final NIGHT_PROFILE with those values locked and AE off.

        Never recreates the camera — only set_controls(). The WebRTC stream stays live.
        """
        det = self._get_detector()
        if det is None or not det.camera_initialized or det.camera is None:
            logger.warning("NightMode: camera not available, skipping night profile apply")
            return {}

        cam = det.camera
        # Phase 1: AE on, NoIR-appropriate WB locked, saturation 0
        cam.set_controls({
            'AeEnable': True,
            'AwbEnable': False,
            'ColourGains': (1.0, 1.0),
            'Saturation': 0.0,
        })

        # Phase 2: settle
        time.sleep(AE_SETTLE_SEC)

        # Phase 3: read what AE picked
        try:
            meta = cam.capture_metadata()
            settled_et = int(meta.get('ExposureTime', NIGHT_PROFILE['ExposureTime']))
            settled_ag = float(meta.get('AnalogueGain', NIGHT_PROFILE['AnalogueGain']))
        except Exception as e:
            logger.warning(f"NightMode: AE settle read failed ({e}), using defaults")
            settled_et = NIGHT_PROFILE['ExposureTime']
            settled_ag = NIGHT_PROFILE['AnalogueGain']

        # Phase 4: lock
        final = {
            'AeEnable': False,
            'AwbEnable': False,
            'ColourGains': (1.0, 1.0),
            'ExposureTime': settled_et,
            'AnalogueGain': settled_ag,
            'Saturation': 0.0,
        }
        cam.set_controls(final)
        logger.info(f"NightMode applied: ExposureTime={settled_et}us AnalogueGain={settled_ag:.2f}")
        return final

    def _switch_to_day(self) -> Dict[str, Any]:
        """Switch back to daytime profile by re-applying the yaml-saved calibration."""
        det = self._get_detector()
        if det is None or not det.camera_initialized or det.camera is None:
            logger.warning("NightMode: camera not available, skipping day profile apply")
            return {}

        # Explicitly undo every control the night profile locked. The yaml
        # calibration below is optional per unit — if it has no color keys
        # (treatbot5), nothing else would clear Saturation=0/AwbEnable=False
        # and the "day" video stays black-and-white.
        day_reset = {
            'AeEnable': True,      # auto exposure (unlocks ExposureTime/AnalogueGain)
            'AwbEnable': True,     # auto white balance (supersedes locked ColourGains)
            'Saturation': 1.0,     # libcamera default — color back on
        }
        try:
            det.camera.set_controls(day_reset)
        except Exception as e:
            logger.warning(f"NightMode: day control reset failed ({e})")

        # Re-apply the saved daytime calibration from robot profile yaml on top
        try:
            det._apply_saved_calibration()
        except Exception as e:
            logger.warning(f"NightMode: failed to re-apply day calibration ({e})")
        return {**day_reset, 'restored': 'yaml_calibration'}

    # ---- Heartbeat + push ----
    def _maybe_heartbeat(self) -> None:
        now = time.time()
        if now - self._last_heartbeat >= HEARTBEAT_INTERVAL_SEC:
            self._last_heartbeat = now
            self._push_state(reason='heartbeat')

    def _push_state(self, reason: str) -> None:
        """Send night_mode_state to the app via the relay (if connected)."""
        try:
            from services.cloud.relay_client import get_relay_client
            relay = get_relay_client()
            if relay and relay.connected:
                relay.send_event('night_mode_state', {
                    **self.get_status(),
                    'reason': reason,
                })
        except Exception as e:
            logger.debug(f"NightMode: relay push skipped ({e})")

    def _write_transition_log(self, prev: str, new: str, reason: str) -> None:
        try:
            with LOG_FILE.open('a') as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{prev}->{new}\tlux={self.last_lux}\t{reason}\n")
        except Exception as e:
            logger.debug(f"NightMode: transition log write failed ({e})")


# ---- Singleton ----
_controller: Optional[NightModeController] = None
_controller_lock = threading.Lock()


def get_night_mode_controller() -> NightModeController:
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = NightModeController()
    return _controller
