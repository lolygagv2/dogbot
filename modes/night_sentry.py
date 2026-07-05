#!/usr/bin/env python3
"""
Night Sentry — WIM-Z demo watch mode.

Slowly sweeps the robot body through a small arc, pausing at waypoints to
"watch". When the Hailo pipeline reports a dog/animal above the configured
confidence during a watch dwell, it captures a still snapshot and raises a
'sentry_detection' system event (forwarded to the cloud by main_treatbot so
the phone app shows the photo), then cools down before alerting again.

Scope: detection is DOG-ONLY — the on-device model has no person/general-animal
class. Night vision is handled separately by night_mode_controller (NoIR camera
+ 940nm IR illuminator), which switches the low-light camera profile by Lux.

Safety: body motion uses the motor command bus (CommandSource.AUTONOMOUS); the
bus applies its own +/-70 clamp and a dead-man watchdog. Movement is short
bursts with halts between, guaranteed to stop via finally blocks and the moment
self.running goes False or the system leaves NIGHT_SENTRY. The sweep is gated
behind config sweep.enabled so it stays OFF until it has been stand-tested.
"""

import os
import sys
import time
import threading
import logging
import yaml
from typing import Dict, Any, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bus import get_bus, publish_system_event
from core.state import get_state, SystemMode

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = '/home/morgan/dogbot/configs/rules/night_sentry_rules.yaml'
CAPTURE_DIR = '/home/morgan/dogbot/captures'


class NightSentryMode:
    """Sweep → watch → detect → snapshot → alert, with a guaranteed motor halt."""

    def __init__(self, config_path: str = None):
        self.bus = get_bus()
        self.state = get_state()

        self.config = self._load_config(config_path or DEFAULT_CONFIG_PATH)

        # Mode state
        self.running = False
        self.mode_thread = None

        # Detection latch (set by the vision handler during a watch dwell)
        self._watching = False
        self._latched_detection: Optional[Dict[str, Any]] = None
        self._latch_lock = threading.Lock()
        self._subscribed = False

        # Alert tracking
        self._last_alert_time = 0.0
        self._alerts_sent = 0

        logger.info("Night Sentry mode initialized")

    # ---- config -------------------------------------------------------------
    def _load_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            logger.info(f"Loaded Night Sentry config from {path}")
            return cfg
        except Exception as e:
            logger.warning(f"Night Sentry config load failed ({e}); using defaults")
            return self._default_config()

    def _default_config(self) -> Dict[str, Any]:
        return {
            'detection': {'confidence_threshold': 0.60},
            'sweep': {'enabled': False, 'motor_speed_pct': 45, 'waypoint_count': 4,
                      'burst_seconds': 0.35, 'settle_seconds': 0.10},
            'watch': {'dwell_seconds': 3.0},
            'alert': {'cooldown_seconds': 60, 'upload_photo': True},
        }

    # ---- lifecycle ----------------------------------------------------------
    def start(self) -> bool:
        if self.running:
            logger.warning("Night Sentry already running")
            return True
        try:
            if not self._subscribed:
                self.bus.subscribe('vision', self._on_vision_event)
                self._subscribed = True

            # Mode announcement (WIMZ_nsentry.mp3) is handled centrally by
            # main_treatbot._announce_mode via the mode_change event.
            self.state.set_mode(SystemMode.NIGHT_SENTRY, "Night Sentry started")

            self._latched_detection = None
            self._watching = False
            self.running = True
            self.mode_thread = threading.Thread(
                target=self._run_loop, daemon=True, name="NightSentry")
            self.mode_thread.start()

            publish_system_event('night_sentry_started', {
                'sweep_enabled': bool(self.config.get('sweep', {}).get('enabled', False)),
            }, 'night_sentry')
            logger.info("Night Sentry started")
            return True
        except Exception as e:
            logger.error(f"Failed to start Night Sentry: {e}")
            self.running = False
            return False

    def stop(self):
        if not self.running:
            return
        logger.info("Stopping Night Sentry...")
        self.running = False
        self._watching = False
        # Guarantee motors are halted in case a sweep burst was in flight.
        self._halt_motors()
        if self.mode_thread and self.mode_thread.is_alive():
            self.mode_thread.join(timeout=2.0)
        publish_system_event('night_sentry_stopped', {
            'alerts_sent': self._alerts_sent,
        }, 'night_sentry')
        logger.info("Night Sentry stopped")

    # ---- main loop ----------------------------------------------------------
    def _run_loop(self):
        dwell = float(self.config.get('watch', {}).get('dwell_seconds', 3.0))
        sweep_cfg = self.config.get('sweep', {})
        sweep_enabled = bool(sweep_cfg.get('enabled', False))
        waypoints = max(1, int(sweep_cfg.get('waypoint_count', 4)))

        logger.info(f"Night Sentry loop running (sweep={'on' if sweep_enabled else 'off'}, "
                    f"dwell={dwell}s, waypoints={waypoints})")

        wp = 0
        while self.running and self.state.get_mode() == SystemMode.NIGHT_SENTRY:
            # 1) Pivot toward the next waypoint (gated OFF until stand-tested).
            if sweep_enabled:
                direction = 1 if (wp % 2 == 0) else -1
                self._sweep_step(direction)
                wp = (wp + 1) % (waypoints * 2)
                if not self.running:
                    break

            # 2) Watch dwell — open the detection window.
            self._latched_detection = None
            self._watching = True
            t_end = time.time() + dwell
            while self.running and time.time() < t_end:
                if self._latched_detection is not None:
                    break
                time.sleep(0.05)
            self._watching = False

            # 3) Act on a detection.
            det = self._latched_detection
            self._latched_detection = None
            if det is not None and self.running:
                self._halt_motors()          # stop moving while we capture/alert
                self._trigger_alert(det)

        self._halt_motors()
        logger.info("Night Sentry loop exited")

    # ---- detection ----------------------------------------------------------
    def _on_vision_event(self, event):
        if not (self.running and self._watching):
            return
        if getattr(event, 'subtype', None) != 'dog_detected':
            return
        try:
            conf = float(event.data.get('confidence', 0.0))
        except Exception:
            return
        threshold = float(self.config.get('detection', {}).get('confidence_threshold', 0.6))
        if conf < threshold:
            return
        # Respect the alert cooldown — don't latch if we just alerted.
        cooldown = float(self.config.get('alert', {}).get('cooldown_seconds', 60))
        if time.time() - self._last_alert_time < cooldown:
            return
        with self._latch_lock:
            if self._latched_detection is None:
                self._latched_detection = {
                    'confidence': conf,
                    'dog_id': event.data.get('dog_id'),
                    'dog_name': event.data.get('dog_name'),
                    'bbox': event.data.get('bbox'),
                    'timestamp': event.data.get('timestamp', time.time()),
                }
                logger.info(f"Night Sentry latched detection (conf={conf:.2f})")

    # ---- alert --------------------------------------------------------------
    def _trigger_alert(self, det: Dict[str, Any]):
        self._last_alert_time = time.time()
        snapshot_path, image_b64 = self._capture_snapshot()
        self._alerts_sent += 1
        payload = {
            'confidence': det.get('confidence'),
            'dog_id': det.get('dog_id'),
            'dog_name': det.get('dog_name'),
            'snapshot_path': snapshot_path,
            'timestamp': time.time(),
        }
        if self.config.get('alert', {}).get('upload_photo', True) and image_b64:
            payload['image_b64'] = image_b64
        publish_system_event('sentry_detection', payload, 'night_sentry')
        logger.info(f"Night Sentry ALERT #{self._alerts_sent} "
                    f"(conf={det.get('confidence')}, photo={'yes' if snapshot_path else 'no'})")

    def _capture_snapshot(self) -> Tuple[Optional[str], Optional[str]]:
        """Grab the last detector frame, save a JPEG, return (path, base64)."""
        try:
            import base64
            import cv2
            from services.perception.detector import get_detector_service
            frame = get_detector_service().get_last_frame()
            if frame is None:
                logger.warning("Night Sentry: no frame available for snapshot")
                return None, None
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                return None, None
            data = buf.tobytes()
            # Spec media tree + media_asset row (REC work item)
            from core.data import get_wimz_store
            wimz = get_wimz_store()
            p = wimz.media_path_for(ext='jpg')
            with open(p, 'wb') as f:
                f.write(data)
            wimz.register_media(str(p), 'image', codec='jpeg',
                                width=frame.shape[1], height=frame.shape[0],
                                retention_class='ephemeral')
            path = str(p)
            return path, base64.b64encode(data).decode('ascii')
        except Exception as e:
            logger.error(f"Night Sentry snapshot failed: {e}")
            return None, None

    # ---- motion (gated by sweep.enabled) ------------------------------------
    def _sweep_step(self, direction: int):
        """One in-place pivot burst toward the next waypoint (direction +1/-1).

        Short burst + guaranteed halt, mirroring the proven Silent Guardian
        movement pattern. Speed is capped 40-70 here; the motor bus also clamps
        to +/-70 and its dead-man watchdog halts if commands stop arriving.
        """
        cfg = self.config.get('sweep', {})
        speed = max(40, min(70, int(cfg.get('motor_speed_pct', 45))))
        burst = float(cfg.get('burst_seconds', 0.35))
        settle = max(0.05, float(cfg.get('settle_seconds', 0.10)))
        try:
            from core.motor_command_bus import (
                get_motor_bus, create_motor_command, CommandSource)
            bus = get_motor_bus()
        except Exception as e:
            logger.error(f"Night Sentry: motor bus unavailable: {e}")
            return
        if not (bus and bus.running):
            logger.warning("Night Sentry: motor bus not running, skipping sweep")
            return
        # pivot right = (speed, -speed); pivot left = (-speed, speed)
        left, right = (speed, -speed) if direction > 0 else (-speed, speed)
        try:
            if not self.running:
                return
            bus.send_command(create_motor_command(left, right, CommandSource.AUTONOMOUS))
            time.sleep(burst)
        finally:
            bus.send_command(create_motor_command(0, 0, CommandSource.AUTONOMOUS))
            time.sleep(settle)

    def _halt_motors(self):
        try:
            from core.motor_command_bus import (
                get_motor_bus, create_motor_command, CommandSource)
            bus = get_motor_bus()
            if bus and bus.running:
                bus.send_command(create_motor_command(0, 0, CommandSource.AUTONOMOUS))
        except Exception:
            pass

    # ---- status -------------------------------------------------------------
    def get_status(self) -> Dict[str, Any]:
        return {
            'running': self.running,
            'watching': self._watching,
            'alerts_sent': self._alerts_sent,
            'sweep_enabled': bool(self.config.get('sweep', {}).get('enabled', False)),
            'last_alert_time': self._last_alert_time,
        }


_night_sentry_instance = None
_night_sentry_lock = threading.Lock()


def get_night_sentry_mode() -> NightSentryMode:
    """Get or create the Night Sentry mode instance (singleton)."""
    global _night_sentry_instance
    if _night_sentry_instance is None:
        with _night_sentry_lock:
            if _night_sentry_instance is None:
                _night_sentry_instance = NightSentryMode()
    return _night_sentry_instance
