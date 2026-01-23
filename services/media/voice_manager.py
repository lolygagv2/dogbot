#!/usr/bin/env python3
"""
services/media/voice_manager.py - Custom Voice Command Storage

Manages custom voice recordings for dog-specific commands.
Voices are stored per dog and per command, with fallback to default audio.

Storage structure:
/home/morgan/dogbot/voices/{dog_id}/{command}.mp3
"""

import os
import base64
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Voice storage configuration
VOICES_DIR = Path("/home/morgan/dogbot/voices")
DEFAULT_VOICES_DIR = Path("/home/morgan/dogbot/VOICEMP3/talks")

# Standard commands that can have custom voices
STANDARD_COMMANDS = [
    "sit",
    "down",
    "stay",
    "come",
    "good_dog",
    "no",
    "quiet",
    "speak",
    "spin",
    "shake",
    "roll_over",
    "fetch",
    "drop_it",
    "leave_it",
    "heel",
    "wait",
    "free",
    "treat",
]


class VoiceManager:
    """
    Manages custom voice recordings for dog commands.

    Features:
    - Store custom voice files per dog per command
    - Retrieve voice file path (custom or default fallback)
    - List available custom voices for a dog
    - Delete custom voices
    """

    def __init__(self):
        self.logger = logging.getLogger('VoiceManager')
        self._lock = threading.Lock()

        # Ensure voices directory exists
        VOICES_DIR.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"VoiceManager initialized (voices_dir={VOICES_DIR})")

    def save_voice(self, dog_id: str, command: str, audio_data: bytes) -> Dict[str, Any]:
        """
        Save a custom voice recording for a dog command.

        Args:
            dog_id: Dog identifier (e.g., "1", "832", "bezik")
            command: Command name (e.g., "sit", "down", "good_dog")
            audio_data: Raw audio bytes (MP3 format expected)

        Returns:
            Dict with success status and file path
        """
        with self._lock:
            try:
                # Sanitize inputs
                dog_id = self._sanitize_filename(str(dog_id))
                command = self._sanitize_filename(command.lower())

                # Create dog directory if needed
                dog_dir = VOICES_DIR / dog_id
                dog_dir.mkdir(parents=True, exist_ok=True)

                # Save audio file
                filepath = dog_dir / f"{command}.mp3"
                with open(filepath, 'wb') as f:
                    f.write(audio_data)

                self.logger.info(f"Saved custom voice: {filepath} ({len(audio_data)} bytes)")

                return {
                    "success": True,
                    "filepath": str(filepath),
                    "dog_id": dog_id,
                    "command": command,
                    "size_bytes": len(audio_data)
                }

            except Exception as e:
                self.logger.error(f"Failed to save voice: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

    def save_voice_base64(self, dog_id: str, command: str, base64_data: str) -> Dict[str, Any]:
        """
        Save a custom voice recording from base64 encoded data.

        Args:
            dog_id: Dog identifier
            command: Command name
            base64_data: Base64 encoded audio data

        Returns:
            Dict with success status and file path
        """
        try:
            # Decode base64
            audio_data = base64.b64decode(base64_data)
            return self.save_voice(dog_id, command, audio_data)

        except Exception as e:
            self.logger.error(f"Failed to decode base64 audio: {e}")
            return {
                "success": False,
                "error": f"Invalid base64 data: {e}"
            }

    def get_voice_path(self, dog_id: str, command: str) -> Optional[str]:
        """
        Get the path to a voice file for a command.

        Priority:
        1. Custom voice for this dog
        2. Default voice from VOICEMP3/talks/

        Args:
            dog_id: Dog identifier
            command: Command name

        Returns:
            Path to audio file, or None if not found
        """
        command = command.lower()

        # Check for custom voice first
        custom_path = VOICES_DIR / self._sanitize_filename(str(dog_id)) / f"{command}.mp3"
        if custom_path.exists():
            self.logger.debug(f"Using custom voice: {custom_path}")
            return str(custom_path)

        # Fallback to default voice
        default_path = DEFAULT_VOICES_DIR / f"{command}.mp3"
        if default_path.exists():
            self.logger.debug(f"Using default voice: {default_path}")
            return str(default_path)

        # Try with underscore variations
        command_underscore = command.replace(" ", "_")
        default_path = DEFAULT_VOICES_DIR / f"{command_underscore}.mp3"
        if default_path.exists():
            return str(default_path)

        return None

    def has_custom_voice(self, dog_id: str, command: str) -> bool:
        """Check if a custom voice exists for a dog command."""
        custom_path = VOICES_DIR / self._sanitize_filename(str(dog_id)) / f"{command.lower()}.mp3"
        return custom_path.exists()

    def list_voices(self, dog_id: str) -> Dict[str, Any]:
        """
        List all voice commands and their custom status for a dog.

        Args:
            dog_id: Dog identifier

        Returns:
            Dict with voices status: {"sit": true, "down": false, ...}
        """
        dog_id = self._sanitize_filename(str(dog_id))
        dog_dir = VOICES_DIR / dog_id

        # Build voice status for all standard commands
        voices = {}
        for command in STANDARD_COMMANDS:
            custom_path = dog_dir / f"{command}.mp3"
            voices[command] = custom_path.exists()

        # Also include any extra custom voices not in standard list
        if dog_dir.exists():
            for filepath in dog_dir.glob("*.mp3"):
                command = filepath.stem
                if command not in voices:
                    voices[command] = True

        return {
            "dog_id": dog_id,
            "voices": voices,
            "custom_count": sum(1 for v in voices.values() if v),
            "total_commands": len(voices)
        }

    def delete_voice(self, dog_id: str, command: str) -> Dict[str, Any]:
        """
        Delete a custom voice recording.

        Args:
            dog_id: Dog identifier
            command: Command name

        Returns:
            Dict with success status
        """
        with self._lock:
            try:
                dog_id = self._sanitize_filename(str(dog_id))
                command = self._sanitize_filename(command.lower())

                filepath = VOICES_DIR / dog_id / f"{command}.mp3"

                if not filepath.exists():
                    return {
                        "success": False,
                        "error": "Voice file not found"
                    }

                filepath.unlink()
                self.logger.info(f"Deleted custom voice: {filepath}")

                return {
                    "success": True,
                    "filepath": str(filepath),
                    "dog_id": dog_id,
                    "command": command
                }

            except Exception as e:
                self.logger.error(f"Failed to delete voice: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }

    def get_all_dogs_voices(self) -> Dict[str, Any]:
        """
        Get voice status for all dogs.

        Returns:
            Dict mapping dog_id to their voice status
        """
        result = {}

        if VOICES_DIR.exists():
            for dog_dir in VOICES_DIR.iterdir():
                if dog_dir.is_dir():
                    dog_id = dog_dir.name
                    result[dog_id] = self.list_voices(dog_id)

        return {
            "dogs": result,
            "total_dogs": len(result)
        }

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize a string for use as a filename."""
        # Remove potentially dangerous characters
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")
        sanitized = "".join(c if c in safe_chars else "_" for c in name)
        return sanitized or "unknown"

    def get_status(self) -> Dict[str, Any]:
        """Get voice manager status."""
        dog_count = 0
        voice_count = 0

        if VOICES_DIR.exists():
            for dog_dir in VOICES_DIR.iterdir():
                if dog_dir.is_dir():
                    dog_count += 1
                    voice_count += len(list(dog_dir.glob("*.mp3")))

        return {
            "voices_dir": str(VOICES_DIR),
            "default_voices_dir": str(DEFAULT_VOICES_DIR),
            "dogs_with_custom_voices": dog_count,
            "total_custom_voices": voice_count,
            "standard_commands": STANDARD_COMMANDS
        }


# Global singleton instance
_voice_manager: Optional[VoiceManager] = None
_voice_lock = threading.Lock()


def get_voice_manager() -> VoiceManager:
    """Get the global VoiceManager instance."""
    global _voice_manager
    with _voice_lock:
        if _voice_manager is None:
            _voice_manager = VoiceManager()
    return _voice_manager
