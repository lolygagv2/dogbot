"""
Hailo device architecture detection.

The fleet runs a mix of AI HAT+ variants: the 26 TOPS board is a Hailo-8,
the 13 TOPS board is a Hailo-8L. A HEF compiled for one arch is rejected
outright by the other:

    [HailoRT] HEF format is not compatible with device.
              Device arch: HAILO8L, HEF arch: HAILO8

So the HEF has to be picked at load time from the detected device, not
hardcoded per unit — the same image runs on both.

Note that lspci and the firmware's "Board Name" both report "Hailo-8" on an
8L; only the device architecture field distinguishes them.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

ARCH_HAILO8 = "hailo8"
ARCH_HAILO8L = "hailo8l"
ARCH_UNKNOWN = "unknown"


def detect_hailo_arch() -> str:
    """Return the detected Hailo device architecture, or ARCH_UNKNOWN.

    Queries the device through hailo_platform. Returns ARCH_UNKNOWN when the
    platform module is missing or no device is present, so callers degrade to
    the default HEF rather than failing hard.
    """
    try:
        import hailo_platform as hpf
    except ImportError:
        logger.warning("hailo_platform not available - cannot detect device arch")
        return ARCH_UNKNOWN

    try:
        device_ids = hpf.Device.scan()
        if not device_ids:
            logger.warning("No Hailo devices found")
            return ARCH_UNKNOWN

        with hpf.Device(device_ids[0]) as device:
            # identify() carries device_architecture ('HAILO8' / 'HAILO8L'); the
            # extended-info struct does not. Note board_name reads "Hailo-8" on
            # an 8L, so only this field distinguishes the two.
            arch_raw = device.control.identify().device_architecture
            arch = getattr(arch_raw, "name", str(arch_raw)).upper()

        logger.info(f"Hailo device architecture: {arch}")

        # Order matters: 'HAILO8L' contains 'HAILO8', so test the longer one first.
        if "HAILO8L" in arch:
            return ARCH_HAILO8L
        if "HAILO8" in arch:
            return ARCH_HAILO8

        logger.warning(f"Unrecognized Hailo architecture: {arch!r}")
        return ARCH_UNKNOWN
    except Exception as e:
        logger.error(f"Hailo arch detection failed: {e}")
        return ARCH_UNKNOWN


def hef_for_arch(base_name: str, arch: Optional[str] = None) -> str:
    """Return the HEF filename to load for the detected device architecture.

    On an 8L, prefers the '_8l' variant (e.g. dogpose_14.hef ->
    dogpose_14_8l.hef) and falls back to the base name if it is not present.
    On a Hailo-8 or an undetectable device, returns base_name unchanged so
    26 TOPS units keep loading exactly what they always have.
    """
    from pathlib import Path

    if arch is None:
        arch = detect_hailo_arch()

    if arch != ARCH_HAILO8L:
        return base_name

    stem, _, ext = base_name.rpartition(".")
    variant = f"{stem}_8l.{ext}" if stem else f"{base_name}_8l"

    if (Path("ai/models") / variant).exists():
        logger.info(f"Hailo-8L detected - using {variant} instead of {base_name}")
        return variant

    logger.error(
        f"Hailo-8L detected but {variant} is missing - falling back to {base_name}, "
        f"which is compiled for HAILO8 and will be REJECTED at load. "
        f"Recompile the model with hw_arch='hailo8l'."
    )
    return base_name
