"""Robot software version — single source of truth for sw_version.

Reads the VERSION file at the release root (repo root). NOT git describe:
OTA release directories are tarball unpacks with no .git. The value is read
once per process — a running robot keeps reporting the version it booted
with even after the updater flips the `current` symlink underneath it,
which is exactly what the app's update flow expects (the new version is
only reported once the new code is actually running).

Release tagging (OTA contract 2026-08-07): date-style strings like
2026.08.1 — compared by the app with plain string equality, never semver
ordering, and never reused.
"""

from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parent.parent / 'VERSION'

try:
    SW_VERSION = _VERSION_FILE.read_text().strip() or 'unknown'
except Exception:
    SW_VERSION = 'unknown'


def get_version() -> str:
    return SW_VERSION
