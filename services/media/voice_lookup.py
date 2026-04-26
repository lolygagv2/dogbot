"""
Voice file lookup - THE ONE canonical function for voice path resolution.

Every code path that plays a voice file MUST call resolve_voice_file().
No exceptions. No manual path construction. No fallback logic elsewhere.
"""
import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

EXTENSIONS = ('.wav', '.mp3', '.m4a', '.aac')
VOICE_ROOT = '/home/morgan/dogbot/VOICEMP3/talks'
DEFAULT_DIR = 'default'

# Generic tracker ids look like 'dog_0', 'dog_-1000', 'dog_42'.
# Real profile ids are timestamps like 'dog_1777167142852' (13+ digits).
_GENERIC_TRACKER_RE = re.compile(r'^dog_-?\d{1,5}$')


def is_real_dog_id(d: Optional[str]) -> bool:
    """Check if dog_id is a real profile ID vs generic tracker ID."""
    if not d:
        return False
    if _GENERIC_TRACKER_RE.match(d):
        return False
    return True


def _try_paths(dog_dir: str, command_id: str) -> Optional[str]:
    """Try to find voice file with any supported extension."""
    for ext in EXTENSIONS:
        p = os.path.join(VOICE_ROOT, dog_dir, f"{command_id}{ext}")
        if os.path.exists(p):
            return p
    return None


def resolve_voice_file(
    command_id: str,
    *,
    dog_id_override: Optional[str] = None,
    state=None,
) -> Optional[str]:
    """The ONE function that decides which voice file to play.

    Fallback chain:
      1. dog_id_override (filtered through is_real_dog_id)
      2. state.get_aruco_dog_within(seconds=5)
      3. state.get_session_dog_id() - from start_coach/start_mission
      4. state.get_current_dog_id() - from select_dog/reload_dogs
      5. default
    """
    if state is None:
        from core.state import get_state
        state = get_state()

    # Get each fallback value for logging
    aruco = state.get_aruco_dog_within(seconds=5)
    session = state.get_session_dog_id()
    current = state.get_current_dog_id()

    # Build candidate list (only real dog IDs)
    candidates = []
    if is_real_dog_id(dog_id_override):
        candidates.append(dog_id_override)
    if is_real_dog_id(aruco):
        candidates.append(aruco)
    if is_real_dog_id(session):
        candidates.append(session)
    if is_real_dog_id(current):
        candidates.append(current)

    # Try each candidate in priority order
    resolved = None
    path = None
    for d in candidates:
        p = _try_paths(d, command_id)
        if p:
            resolved = d
            path = p
            break

    # Fall back to default
    if path is None:
        path = _try_paths(DEFAULT_DIR, command_id)

    logger.info(
        f"[VOICE] command={command_id} "
        f"override={dog_id_override!r} aruco={aruco!r} "
        f"session={session!r} current={current!r} "
        f"resolved={resolved!r} -> {path or 'NONE'}"
    )
    return path


# === Utility functions (not for path resolution) ===

VOICEMP3_BASE = "/home/morgan/dogbot/VOICEMP3"


def get_songs_folder(dog_id: str = None) -> str:
    """Get folder for songs. Uses custom if exists and has files, else default."""
    songs_base = f"{VOICEMP3_BASE}/songs"

    if dog_id and is_real_dog_id(dog_id):
        custom_folder = f"{songs_base}/{dog_id}"
        if os.path.isdir(custom_folder):
            files = [f for f in os.listdir(custom_folder) if f.endswith('.mp3')]
            if files:
                return custom_folder

    return f"{songs_base}/default"


def save_custom_voice(dog_id: str, voice_type: str, audio_data: bytes) -> str:
    """Save a custom voice recording for a dog."""
    dog_folder = f"{VOICEMP3_BASE}/talks/{dog_id}"
    os.makedirs(dog_folder, exist_ok=True)

    file_path = f"{dog_folder}/{voice_type}.mp3"
    with open(file_path, 'wb') as f:
        f.write(audio_data)

    logger.info(f"[VOICE] Saved custom voice: {file_path}")
    return file_path


def restore_dog_to_defaults(dog_id: str) -> dict:
    """Delete all custom voice/song files for a dog."""
    import shutil

    deleted = {'talks': 0, 'songs': 0}

    talks_custom = f"{VOICEMP3_BASE}/talks/{dog_id}"
    if os.path.exists(talks_custom):
        files = os.listdir(talks_custom)
        deleted['talks'] = len(files)
        shutil.rmtree(talks_custom)
        logger.info(f"[VOICE] Removed {len(files)} custom talks for {dog_id}")

    songs_custom = f"{VOICEMP3_BASE}/songs/{dog_id}"
    if os.path.exists(songs_custom):
        files = os.listdir(songs_custom)
        deleted['songs'] = len(files)
        shutil.rmtree(songs_custom)
        logger.info(f"[VOICE] Removed {len(files)} custom songs for {dog_id}")

    return deleted


def list_voice_types() -> list:
    """Return list of valid voice types."""
    return ['sit', 'laydown', 'come', 'stay', 'no', 'good', 'treat', 'quiet', 'speak', 'spin', 'name']


def get_custom_voices(dog_id: str) -> list:
    """Get list of custom voice files for a dog."""
    if not dog_id:
        return []

    dog_folder = f"{VOICEMP3_BASE}/talks/{dog_id}"
    if not os.path.isdir(dog_folder):
        return []

    voices = []
    for f in os.listdir(dog_folder):
        for ext in EXTENSIONS:
            if f.endswith(ext):
                voice_type = f[:-len(ext)]
                voices.append(voice_type)
                break

    return voices
