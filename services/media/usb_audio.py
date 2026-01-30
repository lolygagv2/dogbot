#!/usr/bin/env python3
"""
USB Audio service for TreatBot
Uses pygame mixer for reliable audio playback from VOICEMP3 folder
"""

import os
import time
import logging
import threading
import subprocess
from typing import Optional, Dict, Any, List

def _detect_usb_audio_card() -> int:
    """Detect which card number the USB Audio Device is on"""
    try:
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'USB Audio' in line and 'card' in line:
                # Parse "card 2: Device [USB Audio Device]"
                card_num = int(line.split('card ')[1].split(':')[0])
                return card_num
    except Exception as e:
        logging.warning(f"Could not detect USB audio card: {e}")
    return 2  # Default fallback

# Detect USB audio card dynamically (can be 0, 1, or 2 depending on boot order)
USB_AUDIO_CARD = _detect_usb_audio_card()

# Set audio environment BEFORE importing pygame
os.environ['SDL_AUDIODRIVER'] = 'alsa'
os.environ['AUDIODEV'] = f'plughw:{USB_AUDIO_CARD},0'

import pygame

logger = logging.getLogger('USBAudio')
logger.info(f"USB Audio detected on card {USB_AUDIO_CARD}")

class USBAudioService:
    """USB Audio service using pygame mixer"""

    def __init__(self):
        self.logger = logging.getLogger('USBAudio')
        self.initialized = False
        self.base_path = "/home/morgan/dogbot/VOICEMP3"
        # Use RLock (reentrant lock) to allow nested locking from same thread
        # This fixes deadlock when play_next/play_previous call play_file
        self._lock = threading.RLock()

        # Track cycling state for music player
        self._playlist: List[str] = []
        self._current_index: int = 0  # Start at first song
        self._playlist_track: Optional[str] = None  # Currently selected playlist song
        self._music_playing: bool = False  # Music player state (separate from voice/sfx)
        self._is_looping: bool = False

        # General audio tracking (for any audio file)
        self._current_track: Optional[str] = None  # Any audio currently playing

        # Build initial playlist from songs folder
        self._build_playlist()

        # Set initial playlist track (but don't play)
        if self._playlist:
            self._playlist_track = self._playlist[0]

        try:
            # Initialize pygame mixer for audio playback (USB audio device)
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.initialized = True
            self.logger.info("USB Audio service initialized successfully (plughw:0,0)")
        except Exception as e:
            self.logger.error(f"USB Audio initialization failed: {e}")
            self.initialized = False

    def _build_playlist(self, dog_id: str = None):
        """Build playlist from songs folder, using dog-specific or default songs"""
        songs_base = os.path.join(self.base_path, "songs")
        default_path = os.path.join(songs_base, "default")

        playlist = []

        # Check for dog-specific songs first
        if dog_id:
            dog_path = os.path.join(songs_base, dog_id)
            if os.path.isdir(dog_path):
                dog_songs = sorted([
                    f for f in os.listdir(dog_path)
                    if f.lower().endswith(('.mp3', '.wav', '.ogg'))
                    and os.path.isfile(os.path.join(dog_path, f))
                ])
                if dog_songs:
                    playlist.extend([f"{dog_id}/{f}" for f in dog_songs])
                    self._playlist = playlist
                    self.logger.info(f"Built playlist with {len(playlist)} tracks for {dog_id}")
                    return

        # Fall back to default songs
        if os.path.isdir(default_path):
            playlist.extend(sorted([
                f"default/{f}" for f in os.listdir(default_path)
                if f.lower().endswith(('.mp3', '.wav', '.ogg'))
                and os.path.isfile(os.path.join(default_path, f))
            ]))

        self._playlist = playlist
        self.logger.info(f"Built playlist with {len(self._playlist)} tracks (default)")

    def play_file(self, filepath: str, loop: bool = False) -> Dict[str, Any]:
        """Play an audio file

        Args:
            filepath: Path to audio file
            loop: If True, loop the audio indefinitely until stopped
        """
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        with self._lock:
            try:
                # Handle relative paths by prepending base path
                if not filepath.startswith('/'):
                    full_path = os.path.join(self.base_path, filepath.lstrip('/'))
                elif filepath.startswith('/talks/') or filepath.startswith('/songs/') or filepath.startswith('/02/'):
                    # Map short paths to full paths
                    full_path = os.path.join(self.base_path, filepath[1:])

                    # For /talks/ and /songs/, check default/ subfolder first (consolidation)
                    if filepath.startswith('/talks/') and '/default/' not in filepath:
                        filename = os.path.basename(filepath)
                        default_path = os.path.join(self.base_path, 'talks', 'default', filename)
                        if os.path.exists(default_path):
                            full_path = default_path
                    elif filepath.startswith('/songs/') and '/default/' not in filepath:
                        filename = os.path.basename(filepath)
                        default_path = os.path.join(self.base_path, 'songs', 'default', filename)
                        if os.path.exists(default_path):
                            full_path = default_path
                else:
                    full_path = filepath

                # Check if file exists
                if not os.path.exists(full_path):
                    self.logger.error(f"Audio file not found: {full_path}")
                    return {"success": False, "error": f"File not found: {full_path}"}

                # Play the audio file
                pygame.mixer.music.load(full_path)
                # -1 = loop indefinitely, 0 = play once
                pygame.mixer.music.play(-1 if loop else 0)

                # Track current playing file
                self._current_track = os.path.basename(full_path)
                self._is_looping = loop

                # Update music player state if this is a playlist song
                if '/songs/' in full_path or full_path.startswith(os.path.join(self.base_path, 'songs')):
                    track_name = os.path.basename(full_path)
                    if track_name in self._playlist:
                        self._current_index = self._playlist.index(track_name)
                        self._playlist_track = track_name
                        self._music_playing = True

                loop_msg = " (looping)" if loop else ""
                # Log with call stack to trace where audio is triggered from
                import traceback
                caller_info = "".join(traceback.format_stack()[-4:-1]).strip().replace('\n', ' | ')
                self.logger.info(f"🔊 AUDIO PLAY{loop_msg}: {full_path} [source: {caller_info[:200]}...]")
                return {
                    "success": True,
                    "filepath": filepath,
                    "track": self._current_track,
                    "loop": loop,
                    "message": f"Playing {filepath}{loop_msg}"
                }

            except Exception as e:
                self.logger.error(f"Audio playback error: {e}")
                return {"success": False, "error": str(e)}

    def stop(self) -> Dict[str, Any]:
        """Stop audio playback"""
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        try:
            pygame.mixer.music.stop()
            # Keep _playlist_track so we know what song is selected
            self._is_looping = False
            self._music_playing = False
            self.logger.info("Audio playback stopped")
            return {"success": True, "message": "Audio stopped", "track": self._playlist_track}
        except Exception as e:
            self.logger.error(f"Audio stop error: {e}")
            return {"success": False, "error": str(e)}

    def pause(self) -> Dict[str, Any]:
        """Pause audio playback"""
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        try:
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
                self._music_playing = False
                self.logger.info("Audio playback paused")
                return {"success": True, "message": "Audio paused", "track": self._playlist_track}
            else:
                return {"success": False, "error": "No audio playing"}
        except Exception as e:
            self.logger.error(f"Audio pause error: {e}")
            return {"success": False, "error": str(e)}

    def resume(self) -> Dict[str, Any]:
        """Resume audio playback"""
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        try:
            pygame.mixer.music.unpause()
            self._music_playing = True
            self.logger.info("Audio playback resumed")
            return {"success": True, "message": "Audio resumed", "track": self._playlist_track}
        except Exception as e:
            self.logger.error(f"Audio resume error: {e}")
            return {"success": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get audio system status with current track info"""
        # Sync music player state with pygame's actual state
        # (detects when music finishes naturally)
        if self._music_playing and self.initialized:
            if not pygame.mixer.music.get_busy():
                self._music_playing = False
                self._is_looping = False

        return {
            "success": True,
            "status": {
                "initialized": self.initialized,
                "playing": self._music_playing,
                "base_path": self.base_path
            },
            "audio": {
                "playing": self._music_playing,
                "track": self._playlist_track,  # Show selected playlist track
                "looping": self._is_looping,
                "playlist_index": self._current_index,
                "playlist_length": len(self._playlist)
            }
        }

    @property
    def is_initialized(self) -> bool:
        """Check if audio is initialized"""
        return self.initialized

    @property
    def current_volume(self) -> int:
        """Get current volume (0-100)"""
        if self.initialized:
            return int(pygame.mixer.music.get_volume() * 100)
        return 0

    def is_busy(self) -> bool:
        """Check if any audio is currently playing"""
        if self.initialized:
            return pygame.mixer.music.get_busy()
        return False

    def toggle(self) -> Dict[str, Any]:
        """Toggle music playback - play current song if stopped, stop if playing"""
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        # Sync state with pygame before checking
        if self._music_playing and not pygame.mixer.music.get_busy():
            self._music_playing = False
            self._is_looping = False

        if self._music_playing:
            # Music playing - stop
            return self.stop()
        else:
            # Music not playing - play current song
            return self._play_current()

    def _play_current(self) -> Dict[str, Any]:
        """Play the current track in playlist"""
        if not self._playlist:
            return {"success": False, "error": "No tracks in playlist"}

        with self._lock:
            track = self._playlist[self._current_index]
            filepath = f"/songs/{track}"
            result = self.play_file(filepath, loop=False)
            result["track_index"] = self._current_index
            result["track_name"] = track
            return result

    def play_next(self) -> Dict[str, Any]:
        """Move to next track in playlist and auto-play"""
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        if not self._playlist:
            return {"success": False, "error": "No tracks in playlist"}

        with self._lock:
            # Move to next track (wrap around)
            self._current_index = (self._current_index + 1) % len(self._playlist)
            self._playlist_track = self._playlist[self._current_index]

            # Auto-play the new track
            filepath = f"/songs/{self._playlist_track}"
            result = self.play_file(filepath, loop=False)
            result["track_index"] = self._current_index
            result["track_name"] = self._playlist_track
            self.logger.info(f"Next track: {self._playlist_track} (auto-playing)")
            return result

    def play_previous(self) -> Dict[str, Any]:
        """Move to previous track in playlist and auto-play"""
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        if not self._playlist:
            return {"success": False, "error": "No tracks in playlist"}

        with self._lock:
            # Move to previous track (wrap around)
            self._current_index = (self._current_index - 1) % len(self._playlist)
            self._playlist_track = self._playlist[self._current_index]

            # Auto-play the new track
            filepath = f"/songs/{self._playlist_track}"
            result = self.play_file(filepath, loop=False)
            result["track_index"] = self._current_index
            result["track_name"] = self._playlist_track
            self.logger.info(f"Previous track: {self._playlist_track} (auto-playing)")
            return result

    def get_playlist(self) -> Dict[str, Any]:
        """Get current playlist"""
        return {
            "success": True,
            "playlist": self._playlist,
            "current_index": self._current_index,
            "current_track": self._playlist_track,
            "playing": self._music_playing
        }

    def refresh_playlist(self) -> Dict[str, Any]:
        """Refresh playlist from songs folder"""
        self._build_playlist()
        return {
            "success": True,
            "count": len(self._playlist),
            "playlist": self._playlist
        }

    def list_songs(self, dog_id: str = None) -> Dict[str, Any]:
        """List all songs with source info (default vs custom)"""
        self._build_playlist(dog_id)
        songs = []
        for track in self._playlist:
            if track.startswith("default/"):
                songs.append({"filename": track, "source": "default", "name": track[8:]})
            else:
                # Dog-specific: dog_XXXXX/filename.mp3
                parts = track.split("/", 1)
                songs.append({"filename": track, "source": parts[0], "name": parts[1] if len(parts) > 1 else track})
        return {
            "success": True,
            "songs": songs,
            "total": len(songs),
            "default_count": sum(1 for s in songs if s["source"] == "default"),
            "custom_count": sum(1 for s in songs if s["source"] != "default")
        }

    def play_command(self, command: str, dog_id: str = None, loop: bool = False) -> Dict[str, Any]:
        """
        Play a voice command, using custom voice if available.

        Priority:
        1. Custom voice for this dog (/VOICEMP3/talks/dog_{id}/{command}.mp3)
        2. Default voice (/VOICEMP3/talks/default/{command}.mp3)

        Args:
            command: Command name (e.g., "sit", "good_dog")
            dog_id: Dog identifier for custom voice lookup (optional)
            loop: If True, loop the audio

        Returns:
            Dict with success status and voice source info
        """
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        try:
            from services.media.voice_manager import get_voice_manager
            voice_manager = get_voice_manager()

            # Get voice path (custom or default)
            voice_path = None
            voice_source = "default"

            if dog_id:
                voice_path = voice_manager.get_voice_path(dog_id, command)
                if voice_path and "/dog_" in voice_path:
                    voice_source = "custom"

            # If no custom voice found, try default path
            if not voice_path:
                voice_path = f"/talks/default/{command}.mp3"
                voice_source = "default"

            # Play the voice
            result = self.play_file(voice_path, loop=loop)
            result["voice_source"] = voice_source
            result["command"] = command
            result["dog_id"] = dog_id

            if result.get("success"):
                self.logger.info(f"Playing {voice_source} voice for '{command}' (dog={dog_id})")

            return result

        except ImportError:
            # VoiceManager not available, fall back to default
            return self.play_file(f"/talks/default/{command}.mp3", loop=loop)
        except Exception as e:
            self.logger.error(f"Play command error: {e}")
            return {"success": False, "error": str(e)}

    def wait_for_completion(self, timeout: float = 5.0) -> bool:
        """
        Wait for current audio to finish playing.

        Args:
            timeout: Maximum time to wait in seconds (default 5s)

        Returns:
            True if audio completed, False if timed out
        """
        if not self.initialized:
            return True

        start_time = time.time()
        while pygame.mixer.music.get_busy():
            if time.time() - start_time > timeout:
                self.logger.warning(f"Audio wait timed out after {timeout}s")
                return False
            time.sleep(0.05)  # Check every 50ms
        return True

    def set_volume(self, volume: int) -> bool:
        """Set volume (0-100)"""
        if self.initialized:
            pygame.mixer.music.set_volume(volume / 100.0)
            return True
        return False

# Global instance
_usb_audio_service = None

def get_usb_audio_service() -> USBAudioService:
    """Get the global USB audio service instance"""
    global _usb_audio_service
    if _usb_audio_service is None:
        _usb_audio_service = USBAudioService()
    return _usb_audio_service


# =============================================================================
# AGC (Auto Gain Control) Management
# =============================================================================

def set_agc(enabled: bool) -> bool:
    """
    Enable or disable Auto Gain Control on USB microphone.

    AGC normalizes audio levels which is bad for bark detection (makes barks
    and ambient noise similar energy) but good for voice commands/dictation.

    Args:
        enabled: True to enable AGC, False to disable

    Returns:
        True if successful, False otherwise
    """
    state = 'on' if enabled else 'off'
    try:
        result = subprocess.run(
            ['amixer', '-c', str(USB_AUDIO_CARD), 'set', 'Auto Gain Control', state],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logging.info(f"AGC {'enabled' if enabled else 'disabled'} on card {USB_AUDIO_CARD}")
            return True
        else:
            logging.error(f"Failed to set AGC: {result.stderr}")
            return False
    except Exception as e:
        logging.error(f"AGC control error: {e}")
        return False


def get_agc_state() -> Optional[bool]:
    """
    Get current AGC state.

    Returns:
        True if AGC is on, False if off, None if error
    """
    try:
        result = subprocess.run(
            ['amixer', '-c', str(USB_AUDIO_CARD), 'get', 'Auto Gain Control'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # Parse output like "Mono: Playback [on]" or "[off]"
            if '[on]' in result.stdout:
                return True
            elif '[off]' in result.stdout:
                return False
        return None
    except Exception as e:
        logging.error(f"AGC state check error: {e}")
        return None


def set_mic_volume(volume: int) -> bool:
    """
    Set USB microphone capture volume.

    Args:
        volume: Volume level 0-100

    Returns:
        True if successful
    """
    try:
        result = subprocess.run(
            ['amixer', '-c', str(USB_AUDIO_CARD), 'set', 'Mic Capture Volume', f'{volume}%'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        logging.error(f"Mic volume error: {e}")
        return False