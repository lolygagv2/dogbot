#!/usr/bin/env python3
"""
volume_manager.py - Single source of truth for WIM-Z system audio volume.

Problem this solves
-------------------
Volume changes used to be split across three code paths (HTTP -> amixer
hardware, WebSocket -> pygame software) with no persistence. On a hard power
cycle, alsa-restore never saves state, so volume reset every boot.

This module is the ONE place volume is changed. Every caller (app HTTP API,
WebSocket command, Xbox controller if mapped) routes through set_volume(),
which:
  1. Applies the volume to the USB audio hardware mixer via amixer.
  2. Atomically writes /etc/wimz/audio_state.json.

A boot-time systemd service (wimz-audio.service / scripts/apply_saved_volume.py)
reads that same JSON file and re-applies the volume before WIM-Z services start,
so the setting survives reboots and hard power cycles.

amixer (ALSA hardware mixer) is the source of truth because it is system-wide
and can be applied at boot when no Python process is running. pygame's
per-process software volume is pinned to 1.0 so it can never attenuate output.
"""

import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger('VolumeManager')

# Path constants -- shared contract with scripts/apply_saved_volume.py
STATE_DIR = "/etc/wimz"
STATE_FILE = os.path.join(STATE_DIR, "audio_state.json")
DEFAULT_VOLUME = 60

# amixer simple controls to try, in priority order. The WIM-Z USB audio
# dongle exposes "Speaker"; "PCM"/"Master" are fallbacks for other hardware.
_SPEAKER_CONTROLS = ("Speaker", "PCM", "Master")


def _detect_usb_audio_card() -> Optional[str]:
    """Return the card number of the USB Audio device as a string, or None."""
    try:
        result = subprocess.run(
            ['aplay', '-l'], capture_output=True, text=True, timeout=3
        )
        for line in result.stdout.split('\n'):
            if 'USB Audio' in line and 'card' in line:
                # Parse "card 2: Device [USB Audio Device], device 0: ..."
                return line.split('card ')[1].split(':')[0].strip()
    except Exception as e:
        logger.warning(f"Could not detect USB audio card: {e}")
    return None


class VolumeManager:
    """Singleton owning system volume state + the persistent JSON file."""

    def __init__(self):
        self._lock = threading.Lock()
        self._card: Optional[str] = _detect_usb_audio_card()
        self._control: Optional[str] = None  # resolved on first apply
        # Authoritative in-memory value, seeded from the persisted file.
        self._volume: int = self._load_state()

        logger.info(
            f"VolumeManager init: volume={self._volume}% "
            f"card={self._card if self._card is not None else 'UNKNOWN'} "
            f"file={STATE_FILE}"
        )

        # Apply the loaded value so a freshly-started process matches the file
        # (the boot service already applied it, but this covers a mid-life
        # service restart where the hardware mixer may have drifted).
        self._apply(self._volume)

        # Pin pygame's software volume to 1.0 so the software layer can never
        # attenuate output -- amixer hardware volume is the only control.
        self._neutralize_pygame()

    # ------------------------------------------------------------------ state

    def _load_state(self) -> int:
        """Read volume from the JSON file; fall back to DEFAULT_VOLUME.

        If the file is missing it is created with the default so that
        GET /audio/volume always has a real value to report.
        """
        try:
            with open(STATE_FILE, 'r') as f:
                data = json.load(f)
            vol = int(data.get('volume', DEFAULT_VOLUME))
            return max(0, min(100, vol))
        except FileNotFoundError:
            logger.info(f"{STATE_FILE} missing -- creating with default {DEFAULT_VOLUME}%")
            self._write_state(DEFAULT_VOLUME)
            return DEFAULT_VOLUME
        except Exception as e:
            logger.warning(f"Could not read {STATE_FILE} ({e}) -- using default {DEFAULT_VOLUME}%")
            return DEFAULT_VOLUME

    def _write_state(self, volume: int) -> bool:
        """Atomically write the JSON state file (temp file + rename).

        Atomic so a power cut mid-write cannot corrupt the file -- readers
        either see the old complete file or the new complete file.
        """
        payload = {
            'volume': int(volume),
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        try:
            os.makedirs(STATE_DIR, exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(dir=STATE_DIR, prefix='.audio_state_', suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(payload, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, STATE_FILE)
            except Exception:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
            return True
        except Exception as e:
            logger.error(f"Failed to write {STATE_FILE}: {e}")
            return False

    # --------------------------------------------------------------- hardware

    def _apply(self, volume: int) -> bool:
        """Apply volume to the USB audio hardware mixer via amixer."""
        if self._card is None:
            # Re-detect in case the USB card enumerated late.
            self._card = _detect_usb_audio_card()
        if self._card is None:
            logger.warning("No USB audio card detected -- cannot apply volume")
            return False

        # Try the previously-resolved control first, else probe the candidates.
        controls = (self._control,) if self._control else _SPEAKER_CONTROLS
        for control in controls:
            if control is None:
                continue
            try:
                result = subprocess.run(
                    ['amixer', '-c', str(self._card), 'sset', control, f'{volume}%'],
                    capture_output=True, timeout=3,
                )
                if result.returncode == 0:
                    self._control = control
                    logger.info(
                        f"Applied volume {volume}% via amixer "
                        f"(card {self._card}, control '{control}')"
                    )
                    return True
            except Exception as e:
                logger.debug(f"amixer '{control}' failed: {e}")
        logger.error(f"amixer could not apply volume on card {self._card}")
        return False

    def _neutralize_pygame(self) -> None:
        """Pin pygame's software music volume to 1.0 (best effort)."""
        try:
            import pygame
            if pygame.mixer.get_init():
                pygame.mixer.music.set_volume(1.0)
                logger.debug("pygame software volume pinned to 1.0")
        except Exception as e:
            logger.debug(f"pygame neutralize skipped: {e}")

    # ------------------------------------------------------------------- API

    def set_volume(self, volume: int) -> bool:
        """Set system volume (0-100). Applies to hardware AND persists to disk.

        This is the ONLY supported way to change volume. Returns True if both
        the hardware apply and the file write succeeded.
        """
        try:
            volume = max(0, min(100, int(volume)))
        except (TypeError, ValueError):
            logger.warning(f"Invalid volume value: {volume!r}")
            return False

        with self._lock:
            applied = self._apply(volume)
            persisted = self._write_state(volume)
            # Update in-memory value if either side succeeded so the API and
            # the file stay consistent with the user's intent.
            if applied or persisted:
                self._volume = volume
            return applied and persisted

    def get_volume(self) -> int:
        """Return the current authoritative volume (0-100)."""
        with self._lock:
            return self._volume

    def get_state(self) -> Dict[str, Any]:
        """Return full status for the API / diagnostics."""
        with self._lock:
            return {
                'volume': self._volume,
                'card': self._card,
                'control': self._control,
                'state_file': STATE_FILE,
            }


# Singleton -------------------------------------------------------------------

_volume_manager: Optional[VolumeManager] = None
_vm_lock = threading.Lock()


def get_volume_manager() -> VolumeManager:
    """Get the global VolumeManager singleton."""
    global _volume_manager
    with _vm_lock:
        if _volume_manager is None:
            _volume_manager = VolumeManager()
    return _volume_manager
