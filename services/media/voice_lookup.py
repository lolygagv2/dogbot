"""
Voice file lookup with custom-first, default-fallback logic.

Consolidates all voice file resolution logic in one place.
"""
import os
import logging

logger = logging.getLogger(__name__)

VOICEMP3_BASE = "/home/morgan/dogbot/VOICEMP3"

# Valid voice commands (matches app buttons)
VOICE_TYPES = ['sit', 'down', 'come', 'stay', 'no', 'good', 'treat', 'quiet', 'speak', 'spin', 'crosses']


def get_voice_path(voice_type: str, dog_id: str = None) -> str | None:
    """
    Get path to voice file. Checks custom first, falls back to default.

    Args:
        voice_type: One of VOICE_TYPES (sit, down, come, stay, no, good, treat, quiet, etc.)
        dog_id: e.g. 'dog_1769441492377' or None for default only

    Returns:
        Path to mp3 file, or None if not found
    """
    if voice_type not in VOICE_TYPES:
        logger.warning(f"[VOICE] Unknown voice type: {voice_type} (valid: {VOICE_TYPES})")

    talks_base = f"{VOICEMP3_BASE}/talks"

    # 1. Try custom dog-specific file first
    if dog_id:
        custom_path = f"{talks_base}/{dog_id}/{voice_type}.mp3"
        logger.info(f"[VOICE] Checking custom path: {custom_path}")
        if os.path.exists(custom_path):
            logger.info(f"[VOICE] Found custom voice: {custom_path}")
            return custom_path
        else:
            logger.info(f"[VOICE] Custom not found, checking default...")

    # 2. Fall back to default
    default_path = f"{talks_base}/default/{voice_type}.mp3"
    logger.info(f"[VOICE] Checking default path: {default_path}")
    if os.path.exists(default_path):
        logger.info(f"[VOICE] Using default: {default_path}")
        return default_path

    # 3. Not found
    logger.error(f"[VOICE] File NOT FOUND: {voice_type} (dog={dog_id}), tried: {default_path}")
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
