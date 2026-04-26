"""
Voice file lookup with custom-first, default-fallback logic.

Consolidates all voice file resolution logic in one place.
"""
import os
import logging

logger = logging.getLogger(__name__)

VOICEMP3_BASE = "/home/morgan/dogbot/VOICEMP3"

# Supported audio extensions in priority order
# .wav first (app uploads WAV), .mp3 (legacy defaults), others as fallback
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.m4a', '.aac')

# Valid voice commands (matches app buttons)
VOICE_TYPES = ['sit', 'laydown', 'come', 'stay', 'no', 'good', 'treat', 'quiet', 'speak', 'spin', 'name']

def _is_real_dog_id(dog_id: str) -> bool:
    """Check if dog_id is a real profile ID vs generic tracker ID.

    Rejects generic auto-tracker ids like dog_0, dog_-1000, dog_5.
    Real profile ids from the app look like dog_1777167142852 (timestamp-based).
    """
    if not dog_id:
        return False
    if dog_id.startswith("dog_"):
        try:
            suffix = dog_id.split("_", 1)[1]
            val = int(suffix)
            # Generic tracker IDs are small integers or negative
            # Real profile IDs are timestamps (13+ digits, > 1e12)
            if val < 1000000000000:
                return False
        except (ValueError, IndexError):
            pass
    return True


# Alias map: voice_type -> actual default filename when {voice_type}.mp3 doesn't exist
VOICE_FILE_MAP = {
    'good': 'good_dog.mp3',
    'laydown': 'lie_down.mp3',
    'down': 'lie_down.mp3',  # Legacy alias
}


def get_voice_path(voice_type: str, dog_id: str = None) -> str | None:
    """
    Get path to voice file. Checks custom first, falls back to default.
    Tries multiple extensions: .wav (app uploads), .mp3 (legacy), .m4a, .aac.

    Args:
        voice_type: One of VOICE_TYPES (sit, down, come, stay, no, good, treat, quiet, etc.)
        dog_id: e.g. 'dog_1769441492377' or None for default only

    Returns:
        Path to audio file, or None if not found
    """
    if voice_type not in VOICE_TYPES:
        logger.warning(f"[VOICE] Unknown voice type: {voice_type} (valid: {VOICE_TYPES})")

    talks_base = f"{VOICEMP3_BASE}/talks"

    # 1. Try custom dog-specific file first (multiple extensions)
    if dog_id:
        for ext in AUDIO_EXTENSIONS:
            custom_path = f"{talks_base}/{dog_id}/{voice_type}{ext}"
            if os.path.exists(custom_path):
                logger.info(f"[VOICE] Found custom voice: {custom_path}")
                return custom_path
        logger.debug(f"[VOICE] Custom not found for {dog_id}/{voice_type}, checking default...")

    # 2. Fall back to default (multiple extensions)
    for ext in AUDIO_EXTENSIONS:
        default_path = f"{talks_base}/default/{voice_type}{ext}"
        if os.path.exists(default_path):
            logger.info(f"[VOICE] Using default: {default_path}")
            return default_path

    # 2b. Try filename alias (good -> good_dog, laydown -> lie_down)
    alias = VOICE_FILE_MAP.get(voice_type)
    if alias:
        alias_name = alias.rsplit('.', 1)[0]  # Strip extension
        for ext in AUDIO_EXTENSIONS:
            alias_path = f"{talks_base}/default/{alias_name}{ext}"
            if os.path.exists(alias_path):
                logger.info(f"[VOICE] Using alias: {voice_type} -> {alias_path}")
                return alias_path

    # 3. Not found
    logger.warning(f"[VOICE] File NOT FOUND: {voice_type} (dog={dog_id})")
    return None


def resolve_voice_file(command_id: str, dog_id_override: str = None) -> str | None:
    """
    Resolve voice file using C3.2 fallback chain with multi-extension support.

    This is the main entry point for autonomous voice triggers (coach rewards,
    Silent Guardian, Xbox controller buttons). Tries .wav first (app uploads WAV),
    then .mp3 (legacy defaults), then other formats.

    Fallback chain (highest priority wins):
    (a) dog_id_override if provided
    (b) ArUco-identified dog within last 5 seconds
    (c) Session dog_id (from start_coach/start_mission)
    (d) select_dog from app (persisted)
    (e) Default voice file

    Args:
        command_id: Voice command (sit, good, no, quiet, treat, come, etc.)
        dog_id_override: Explicit dog_id to use (skips fallback chain)

    Returns:
        Path to voice file, or None if not found
    """
    talks_base = f"{VOICEMP3_BASE}/talks"

    # Get dog_id from fallback chain
    dog_id = dog_id_override

    # Filter out generic tracker IDs like dog_0, dog_-1000
    if dog_id and not _is_real_dog_id(dog_id):
        logger.debug(f"[VOICE] Ignoring generic tracker ID: {dog_id}")
        dog_id = None

    if not dog_id:
        try:
            from core.state import get_state
            state = get_state()
            dog_id = state.get_active_dog_id(aruco_ttl=5.0)
        except Exception as e:
            logger.warning(f"[VOICE] Could not get active dog_id: {e}")

    # 1. Try per-dog custom file (multiple extensions)
    if dog_id:
        for ext in AUDIO_EXTENSIONS:
            path = f"{talks_base}/{dog_id}/{command_id}{ext}"
            if os.path.exists(path):
                logger.info(f"[VOICE] resolve: {command_id} dog={dog_id} -> {path}")
                return path
        logger.debug(f"[VOICE] resolve: {command_id} dog={dog_id} -> no custom file, trying default")

    # 2. Try default file (multiple extensions)
    for ext in AUDIO_EXTENSIONS:
        path = f"{talks_base}/default/{command_id}{ext}"
        if os.path.exists(path):
            logger.info(f"[VOICE] resolve: {command_id} dog={dog_id or 'none'} -> default {path}")
            return path

    # 3. Try alias (good -> good_dog, laydown -> lie_down)
    alias_base = VOICE_FILE_MAP.get(command_id)
    if alias_base:
        alias_name = alias_base.rsplit('.', 1)[0]  # Strip extension from alias
        for ext in AUDIO_EXTENSIONS:
            path = f"{talks_base}/default/{alias_name}{ext}"
            if os.path.exists(path):
                logger.info(f"[VOICE] resolve: {command_id} -> alias {path}")
                return path

    logger.warning(f"[VOICE] resolve: {command_id} dog={dog_id or 'none'} -> NOT FOUND")
    return None


def get_songs_folder(dog_id: str = None) -> str:
    """
    Get folder for songs. Uses custom if exists and has files, else default.
    """
    songs_base = f"{VOICEMP3_BASE}/songs"

    if dog_id:
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
    """
    Delete all custom voice/song files for a dog.
    After deletion, get_voice_path() will automatically use defaults.
    """
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
    return VOICE_TYPES.copy()


def get_custom_voices(dog_id: str) -> list:
    """Get list of custom voice files for a dog."""
    if not dog_id:
        return []

    dog_folder = f"{VOICEMP3_BASE}/talks/{dog_id}"
    if not os.path.isdir(dog_folder):
        return []

    voices = []
    for f in os.listdir(dog_folder):
        if f.endswith('.mp3'):
            voice_type = f[:-4]  # Remove .mp3
            voices.append(voice_type)

    return voices
