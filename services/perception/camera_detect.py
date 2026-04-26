"""
Camera type detection.

WIM-Z ships in 5 robot variants — 2 with Raspberry Pi AI Camera (IMX500),
3 with Camera Module 3 Wide (IMX708). Behavior models must be selected
per camera because of FOV/lens differences (~78° vs ~120° FOV).
"""
import logging
from typing import Optional

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
