"""
Camera type detection.

WIM-Z ships in 5 robot variants — 2 with Raspberry Pi AI Camera (IMX500),
3 with Camera Module 3 Wide (IMX708). Behavior models must be selected
per camera because of FOV/lens differences (~78° vs ~120° FOV).

The physical module on a unit is NOT stable — Morgan swaps the standard
(IR-cut) and NoIR Camera Module 3 Wide on treatbot4 all the time. Nothing
may assume a unit's variant from its yaml or from history; read it at
runtime. libcamera reports the variant in the sensor Model string:
'imx708_wide' (standard) vs 'imx708_wide_noir' (IR-cut removed), and
picks the matching tuning file. The yaml carries a daytime profile for
each variant and select_day_profile() chooses by that string.
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CAMERA_IMX500 = "imx500"
CAMERA_IMX708 = "imx708"
CAMERA_UNKNOWN = "unknown"


def detect_camera_type() -> str:
    """Return detected camera sensor type or CAMERA_UNKNOWN.

    Reads sensor model from Picamera2.global_camera_info() and matches
    against known sensor strings.
    """
    try:
        from picamera2 import Picamera2
    except ImportError:
        logger.warning("Picamera2 not available - cannot detect camera type")
        return CAMERA_UNKNOWN

    try:
        cams = Picamera2.global_camera_info()
        if not cams:
            logger.warning("No cameras detected")
            return CAMERA_UNKNOWN

        model = (cams[0].get("Model") or "").lower()
        logger.info(f"Camera sensor model: {model!r}")

        if "imx500" in model:
            return CAMERA_IMX500
        if "imx708" in model:
            return CAMERA_IMX708

        logger.warning(f"Unrecognized camera model: {model!r}")
        return CAMERA_UNKNOWN
    except Exception as e:
        logger.error(f"Camera detection failed: {e}")
        return CAMERA_UNKNOWN


def detect_camera_model() -> str:
    """Return the raw libcamera sensor Model string ('' when unavailable).

    Examples: 'imx708_wide', 'imx708_wide_noir', 'imx500'. Prefer passing the
    live Picamera2 instance's camera_properties['Model'] where one exists;
    this helper is for callers that have no camera object.
    """
    try:
        from picamera2 import Picamera2
        cams = Picamera2.global_camera_info()
        if cams:
            return (cams[0].get("Model") or "")
    except Exception as e:
        logger.warning(f"Camera model detection failed: {e}")
    return ""


def is_noir_model(model: Optional[str]) -> bool:
    """True when the sensor Model string names a NoIR (IR-cut removed) module."""
    return "noir" in (model or "").lower()


def camera_variant(model: Optional[str]) -> str:
    """'noir' | 'standard' | 'unknown' from a sensor Model string."""
    if not model:
        return "unknown"
    return "noir" if is_noir_model(model) else "standard"


# Daytime calibration keys the detector knows how to apply (robot profile yaml).
DAY_PROFILE_KEYS = (
    "awb_mode", "awb_enable", "colour_gains", "brightness", "contrast",
    "saturation", "sharpness", "exposure_time", "analogue_gain",
)


def select_day_profile(cam_cfg: Dict[str, Any], model: Optional[str]) -> Dict[str, Any]:
    """Pick the daytime calibration block for the camera actually fitted.

    The yaml `camera:` section may carry per-variant blocks:
        noir_day_profile:      applied when Model contains 'noir'
        standard_day_profile:  applied otherwise
    Any calibration key written directly under `camera:` is the fallback
    used when the matching block is absent (legacy layout). Returns a dict
    of yaml keys (subset of DAY_PROFILE_KEYS); empty when nothing is set.
    """
    cam_cfg = cam_cfg or {}
    block_name = "noir_day_profile" if is_noir_model(model) else "standard_day_profile"
    block = cam_cfg.get(block_name)
    if isinstance(block, dict) and block:
        src = block
    else:
        src = cam_cfg
    return {k: src[k] for k in DAY_PROFILE_KEYS if src.get(k) is not None}


def behavior_model_for_camera(camera_type: Optional[str] = None) -> str:
    """Return the behavior model filename to load for the given camera.

    Falls back to behavior_shared.ts if camera-specific model is not set,
    then to behavior_14.ts (legacy) if shared model also missing.
    """
    import os
    from pathlib import Path

    if camera_type is None:
        camera_type = detect_camera_type()

    candidates = {
        CAMERA_IMX500: ["behavior_imx500.ts", "behavior_shared.ts", "behavior_14.ts"],
        CAMERA_IMX708: ["behavior_imx708.ts", "behavior_shared.ts", "behavior_14.ts"],
        CAMERA_UNKNOWN: ["behavior_shared.ts", "behavior_14.ts"],
    }[camera_type]

    models_dir = Path("ai/models")
    for name in candidates:
        if (models_dir / name).exists():
            return name

    return candidates[-1]
