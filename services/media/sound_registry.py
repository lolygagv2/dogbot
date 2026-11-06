#!/usr/bin/env python3
"""
services/media/sound_registry.py - Factory sound registry with per-sound custom overrides

Factory sounds are read-only. A user may record a replacement for any factory
sound id; the replacement lives under the override root and plays instead of
the factory file whenever it exists. Reset deletes the override and the factory
sound is heard again — factory dirs are NEVER written to.

Groups (factory roots):
    wimz    -> /home/morgan/dogbot/VOICEMP3/wimz           (git-tracked system prompts)
    default -> /home/morgan/dogbot/VOICEMP3/talks/default  (shared, factory by convention)

Sound id = "<group>/<stem>" (stem = filename without extension).
Override file = <override_root>/<group>/<stem>.mp3
"""

import logging
import os
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger('SoundRegistry')

FACTORY_ROOTS: Dict[str, Path] = {
    'wimz': Path('/home/morgan/dogbot/VOICEMP3/wimz'),
    'default': Path('/home/morgan/dogbot/VOICEMP3/talks/default'),
}
OVERRIDE_ROOT = Path('/home/morgan/dogbot/VOICEMP3/talks/_custom')

FACTORY_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.aac'}
ACCEPTED_UPLOAD_FORMATS = {'mp3', 'wav', 'm4a', 'aac', 'ogg', 'opus', 'webm'}

# Re-scan factory dirs at most this often from list()/resolve() (cheap: a few listdirs)
_RESCAN_INTERVAL_S = 5.0


def _is_under(path: str, root: str) -> bool:
    """True if realpath `path` is inside realpath `root` (or equal)."""
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:
        return False


class SoundRegistry:
    """Discovers factory sound ids and manages per-id custom overrides."""

    def __init__(self, factory_roots: Optional[Dict[str, Path]] = None,
                 override_root: Optional[Path] = None):
        self._lock = threading.RLock()
        self.factory_roots: Dict[str, Path] = {
            g: Path(p) for g, p in (factory_roots or FACTORY_ROOTS).items()
        }
        self.override_root = Path(override_root or OVERRIDE_ROOT)
        self._override_root_real = os.path.realpath(str(self.override_root))

        # id -> factory path (str)
        self._factory: Dict[str, str] = {}
        # realpath(factory file) -> id  (for resolve_for_path)
        self._by_realpath: Dict[str, str] = {}
        self._last_scan = 0.0
        self._scan(force=True)
        logger.info(f"[SOUNDS] registry ready: {len(self._factory)} factory sounds, "
                    f"override_root={self.override_root}")

    # ------------------------------------------------------------------ scan

    def _scan(self, force: bool = False) -> None:
        import time
        now = time.monotonic()
        if not force and (now - self._last_scan) < _RESCAN_INTERVAL_S:
            return
        factory: Dict[str, str] = {}
        by_real: Dict[str, str] = {}
        for group, root in self.factory_roots.items():
            try:
                entries = sorted(os.listdir(root))
            except OSError:
                continue
            for name in entries:
                stem, ext = os.path.splitext(name)
                if ext.lower() not in FACTORY_EXTENSIONS or not stem:
                    continue
                full = os.path.join(str(root), name)
                if not os.path.isfile(full):
                    continue
                sid = f"{group}/{stem}"
                if sid in factory:
                    continue  # first extension wins (sorted order)
                factory[sid] = full
                try:
                    by_real[os.path.realpath(full)] = sid
                except OSError:
                    pass
        self._factory = factory
        self._by_realpath = by_real
        self._last_scan = now

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _split_id(sound_id: str) -> Optional[tuple]:
        if not isinstance(sound_id, str) or sound_id.count('/') != 1:
            return None
        group, stem = sound_id.split('/', 1)
        if not group or not stem or '..' in (group, stem) or '/' in stem or '\\' in stem \
                or stem.startswith('.') or os.sep in stem:
            return None
        return group, stem

    def _override_path(self, sound_id: str) -> Optional[Path]:
        parts = self._split_id(sound_id)
        if parts is None:
            return None
        group, stem = parts
        return self.override_root / group / f"{stem}.mp3"

    def _safe_override_path(self, sound_id: str) -> Optional[Path]:
        """Override path for a KNOWN id, guaranteed (realpath) to be under override_root."""
        if sound_id not in self._factory:
            return None
        p = self._override_path(sound_id)
        if p is None:
            return None
        if not _is_under(os.path.realpath(str(p)), self._override_root_real):
            logger.error(f"[SOUNDS] refusing override path outside root: {p}")
            return None
        return p

    # ------------------------------------------------------------------- API

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._scan()
            out = []
            for sid in sorted(self._factory):
                group, stem = sid.split('/', 1)
                ov = self._override_path(sid)
                has = bool(ov and ov.is_file())
                out.append({
                    'id': sid,
                    'group': group,
                    'stem': stem,
                    'has_override': has,
                    'factory_path': self._factory[sid],
                    'override_path': str(ov) if has else None,
                })
            return out

    def has_override(self, sound_id: str) -> bool:
        with self._lock:
            ov = self._safe_override_path(sound_id)
            return bool(ov and ov.is_file())

    def resolve(self, sound_id: str) -> Optional[str]:
        with self._lock:
            if sound_id not in self._factory:
                self._scan()
                if sound_id not in self._factory:
                    return None
            ov = self._safe_override_path(sound_id)
            if ov and ov.is_file():
                return str(ov)
            return self._factory[sound_id]

    def resolve_for_path(self, abs_path: str) -> str:
        """Audio hot path: map a factory file path to its override if one exists.
        Never raises; returns abs_path unchanged when not a known factory file."""
        try:
            real = os.path.realpath(abs_path)
            with self._lock:
                sid = self._by_realpath.get(real)
                if sid is None:
                    return abs_path
                ov = self._override_path(sid)
                if ov and _is_under(os.path.realpath(str(ov)), self._override_root_real) \
                        and ov.is_file():
                    return str(ov)
        except Exception:
            pass
        return abs_path

    def replace(self, sound_id: str, data: bytes, fmt: str) -> Dict[str, Any]:
        fmt = (fmt or '').lower().lstrip('.')
        with self._lock:
            self._scan()
            if sound_id not in self._factory:
                return {'success': False, 'error': 'unknown sound id'}
            if fmt not in ACCEPTED_UPLOAD_FORMATS:
                return {'success': False, 'error': f'unsupported format: {fmt}'}
            if not data:
                return {'success': False, 'error': 'empty audio data'}
            final = self._safe_override_path(sound_id)
            if final is None:
                return {'success': False, 'error': 'invalid override path'}

            try:
                final.parent.mkdir(parents=True, exist_ok=True)
                # Hard guard: directory we write into must be under override_root
                if not _is_under(os.path.realpath(str(final.parent)), self._override_root_real):
                    return {'success': False, 'error': 'override dir outside override root'}

                fd, tmp_out = tempfile.mkstemp(prefix=f".{final.stem}.", suffix='.mp3.tmp',
                                               dir=str(final.parent))
                os.close(fd)
                try:
                    if fmt == 'mp3':
                        with open(tmp_out, 'wb') as f:
                            f.write(data)
                    else:
                        with tempfile.NamedTemporaryFile(suffix=f'.{fmt}', delete=False) as tmp_in:
                            tmp_in.write(data)
                            tmp_in_path = tmp_in.name
                        try:
                            result = subprocess.run(
                                ['ffmpeg', '-y', '-i', tmp_in_path,
                                 '-codec:a', 'libmp3lame', '-b:a', '128k', tmp_out],
                                capture_output=True, timeout=30
                            )
                        except FileNotFoundError:
                            logger.error("[SOUNDS] ffmpeg not installed; cannot convert upload")
                            return {'success': False, 'error': 'ffmpeg not available'}
                        except subprocess.TimeoutExpired:
                            logger.error("[SOUNDS] ffmpeg conversion timed out")
                            return {'success': False, 'error': 'audio conversion timed out'}
                        finally:
                            try:
                                os.unlink(tmp_in_path)
                            except OSError:
                                pass
                        if result.returncode != 0:
                            err = result.stderr.decode(errors='replace')[-400:] if result.stderr else ''
                            logger.error(f"[SOUNDS] ffmpeg conversion failed: {err}")
                            return {'success': False, 'error': 'audio conversion failed'}
                        if not os.path.isfile(tmp_out) or os.path.getsize(tmp_out) == 0:
                            return {'success': False, 'error': 'audio conversion produced no output'}

                    os.replace(tmp_out, str(final))
                finally:
                    if os.path.exists(tmp_out):
                        try:
                            os.unlink(tmp_out)
                        except OSError:
                            pass

                size = final.stat().st_size
                logger.info(f"[SOUNDS] override saved: {sound_id} -> {final} ({size} bytes, fmt={fmt})")
                return {'success': True, 'id': sound_id, 'path': str(final), 'size_bytes': size}
            except Exception as e:
                logger.error(f"[SOUNDS] replace failed for {sound_id}: {e}")
                return {'success': False, 'error': str(e)}

    def reset(self, sound_id: str) -> Dict[str, Any]:
        with self._lock:
            if sound_id not in self._factory:
                self._scan()
                if sound_id not in self._factory:
                    return {'success': False, 'error': 'unknown sound id'}
            ov = self._safe_override_path(sound_id)
            if ov is None:
                return {'success': False, 'error': 'invalid override path'}
            removed = False
            try:
                if ov.is_file():
                    ov.unlink()
                    removed = True
                    logger.info(f"[SOUNDS] override removed: {sound_id}")
            except OSError as e:
                return {'success': False, 'error': str(e)}
            return {'success': True, 'id': sound_id, 'removed': removed,
                    'path': self._factory[sound_id]}

    def reset_all(self) -> Dict[str, Any]:
        with self._lock:
            removed = 0
            root = self.override_root
            if not root.is_dir():
                return {'success': True, 'removed': 0}
            try:
                for dirpath, _dirs, files in os.walk(str(root)):
                    for name in files:
                        full = os.path.join(dirpath, name)
                        if os.path.islink(full):
                            continue  # never follow links out of the override root
                        if not _is_under(os.path.realpath(full), self._override_root_real):
                            continue
                        try:
                            os.unlink(full)
                            removed += 1
                        except OSError as e:
                            logger.warning(f"[SOUNDS] could not remove {full}: {e}")
            except Exception as e:
                logger.error(f"[SOUNDS] reset_all failed: {e}")
                return {'success': False, 'error': str(e), 'removed': removed}
            logger.info(f"[SOUNDS] reset_all removed {removed} override(s)")
            return {'success': True, 'removed': removed}


# --------------------------------------------------------------- singleton

_registry: Optional[SoundRegistry] = None
_registry_lock = threading.Lock()


def get_sound_registry() -> SoundRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = SoundRegistry()
    return _registry
