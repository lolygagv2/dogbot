#!/usr/bin/env python3
"""
services/motion/visual_heading.py - heading change from the camera (2026-11-18).

tb3/4/5 have no IMU, gyro, compass or wheel encoders (I2C carries only the
PCA9685 and the battery ADC), so the camera is the only heading source on
board. This module measures how far the robot has yawed between two STILL
frames, and tells when the picture has stopped moving.

How: ORB features in both frames, ratio-test matches, each matched point is
turned into a bearing (azimuth about the vertical axis, with the gimbal tilt
taken out), and the yaw is the consensus of the per-match azimuth differences.
A pure rotation about the vertical axis shifts every azimuth by the same
angle whatever the point's height or distance, so no depth is needed.

Limits (callers must respect them):
  * frames must be sharp: measure only while the robot is stationary. At the
    5 fps idle frame rate and indoor exposure a moving frame is a smear.
  * the two frames must overlap: keep one step under ~40 degrees.
  * the scale depends on the lens model. `scale` (turn calibration file) trims
    it; 1.0 assumes an ideal pinhole with the IMX708-wide 102 deg horizontal FOV.

Read-only on perception: frames come from DetectorService.get_last_frame_with
_timestamp() (a locked copy of the last captured frame); nothing in the
detector or the camera configuration is touched.
"""
import json
import logging
import math
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger('VisualHeading')

CALIBRATION_FILE = '/home/morgan/dogbot/state/turn_calibration.json'

DEFAULT_HFOV_DEG = 102.0     # Camera Module 3 Wide (IMX708 wide), full-width readout
WORK_WIDTH = 640             # features are found on a frame scaled to this width
MIN_INLIERS = 25
INLIER_BAND_DEG = 2.5
STILL_PX = 1.5               # median feature shift (at WORK_WIDTH) that counts as "not moving"


def load_calibration(path: str = CALIBRATION_FILE) -> Dict[str, Any]:
    try:
        with open(path, 'r') as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"turn calibration unreadable ({e})")
        return {}


def save_calibration(data: Dict[str, Any], path: str = CALIBRATION_FILE) -> bool:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        return True
    except Exception as e:
        logger.error(f"turn calibration not saved: {e}")
        return False


class Snapshot:
    """Features of one still frame."""
    __slots__ = ('pts', 'des', 'width', 'height', 'ts', 'tilt_deg')

    def __init__(self, pts, des, width, height, ts, tilt_deg):
        self.pts, self.des, self.width, self.height = pts, des, width, height
        self.ts, self.tilt_deg = ts, tilt_deg

    @property
    def usable(self) -> bool:
        return self.des is not None and len(self.pts) >= MIN_INLIERS


def _default_frame_source() -> Tuple[Optional[np.ndarray], float]:
    from services.perception.detector import get_detector_service
    return get_detector_service().get_last_frame_with_timestamp()


def _default_tilt() -> float:
    """Camera pitch above the horizon in degrees (0 = calibrated centre)."""
    try:
        from services.motion.pan_tilt import get_pantilt_service
        return float(get_pantilt_service().offsets().get('tilt', 0.0))
    except Exception:
        return 0.0


class VisualHeading:
    def __init__(self, frame_source: Callable[[], Tuple[Optional[np.ndarray], float]] = _default_frame_source,
                 tilt_source: Callable[[], float] = _default_tilt,
                 hfov_deg: Optional[float] = None, scale: Optional[float] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        cal = load_calibration() if (hfov_deg is None or scale is None) else {}
        self._frames = frame_source
        self._tilt = tilt_source
        self.hfov_deg = float(hfov_deg if hfov_deg is not None else cal.get('hfov_deg', DEFAULT_HFOV_DEG))
        self.scale = float(scale if scale is not None else cal.get('yaw_scale', 1.0))
        self.calibrated = bool(cal.get('yaw_scale')) if scale is None else True
        self._clock = clock
        self._sleep = sleep
        self._orb = None

    # ------------------------------------------------------------ features

    def snapshot_of(self, frame: np.ndarray, ts: float = 0.0, tilt_deg: Optional[float] = None) -> Snapshot:
        import cv2
        h, w = frame.shape[:2]
        s = WORK_WIDTH / float(w)
        small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if abs(s - 1.0) > 1e-3 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        if self._orb is None:
            self._orb = cv2.ORB_create(nfeatures=1200)
        kp, des = self._orb.detectAndCompute(gray, None)
        pts = (np.float32([k.pt for k in kp]) / s) if kp else np.zeros((0, 2), np.float32)
        return Snapshot(pts, des, w, h, ts, self._tilt() if tilt_deg is None else tilt_deg)

    def grab(self, newer_than: float = 0.0, timeout_s: float = 1.0) -> Optional[Snapshot]:
        """Features of the first frame captured after `newer_than` (epoch s)."""
        deadline = self._clock() + timeout_s
        while True:
            try:
                frame, ts = self._frames()
            except Exception as e:
                logger.debug(f"frame source failed: {e}")
                return None
            if frame is not None and ts > newer_than:
                return self.snapshot_of(frame, ts)
            if self._clock() >= deadline:
                return None
            self._sleep(0.03)

    def _match(self, a: Snapshot, b: Snapshot):
        import cv2
        if not (a.usable and b.usable):
            return None
        knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(a.des, b.des, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
        if len(good) < MIN_INLIERS:
            return None
        return a.pts[[g.queryIdx for g in good]], b.pts[[g.trainIdx for g in good]]

    def _azimuth(self, pts: np.ndarray, snap: Snapshot) -> np.ndarray:
        f = (snap.width / 2.0) / math.tan(math.radians(self.hfov_deg / 2.0))
        x = (pts[:, 0] - snap.width / 2.0) / f
        y = (pts[:, 1] - snap.height / 2.0) / f          # image y points down
        t = math.radians(snap.tilt_deg)                  # camera pitched up by t
        z_level = y * math.sin(t) + math.cos(t)
        return np.degrees(np.arctan2(x, z_level))

    # ------------------------------------------------------------- public

    def yaw_between(self, a: Snapshot, b: Snapshot) -> Optional[Dict[str, Any]]:
        """Yaw of the camera from frame a to frame b, degrees, + = turned right.
        None when the frames cannot be matched (dark, blank wall, no overlap)."""
        m = self._match(a, b)
        if m is None:
            return None
        d = self._azimuth(m[0], a) - self._azimuth(m[1], b)   # scene slides left => turned right
        hist, edges = np.histogram(d, bins=np.arange(-130.0, 131.0, 2.0))
        k = int(np.argmax(hist))
        centre = (edges[k] + edges[k + 1]) / 2.0
        inl = d[np.abs(d - centre) < INLIER_BAND_DEG]
        if len(inl) < MIN_INLIERS or len(inl) < 0.25 * len(d):
            return None
        return {'yaw_deg': float(np.median(inl)) * self.scale, 'inliers': int(len(inl)), 'matches': int(len(d))}

    def shift_px(self, a: Snapshot, b: Snapshot) -> Optional[float]:
        """Median feature displacement between two frames, in WORK_WIDTH pixels."""
        m = self._match(a, b)
        if m is None:
            return None
        s = WORK_WIDTH / float(a.width)
        return float(np.median(np.linalg.norm((m[1] - m[0]) * s, axis=1)))

    def wait_still(self, after: float = 0.0, timeout_s: float = 1.5) -> Tuple[Optional[Snapshot], bool]:
        """Block until two consecutive frames captured after `after` agree
        (the robot has physically stopped). Returns (last snapshot, still?).
        still is False when it timed out or the frames could not be compared."""
        deadline = self._clock() + timeout_s
        prev = self.grab(after, max(0.05, deadline - self._clock()))
        if prev is None:
            return None, False
        while True:
            left = deadline - self._clock()
            if left <= 0:
                return prev, False
            cur = self.grab(prev.ts, left)
            if cur is None:
                return prev, False
            shift = self.shift_px(prev, cur)
            prev_usable = prev.usable
            prev = cur
            if shift is None:
                if not (prev_usable and cur.usable):
                    return cur, False          # nothing to compare (dark / blank): cannot tell
                continue                       # frames differ too much to match: still moving
            if shift <= STILL_PX:
                return cur, True


_instance: Optional[VisualHeading] = None


def get_visual_heading() -> VisualHeading:
    global _instance
    if _instance is None:
        _instance = VisualHeading()
    return _instance


def reset_visual_heading() -> None:
    """Drop the singleton so a new calibration file is picked up."""
    global _instance
    _instance = None
