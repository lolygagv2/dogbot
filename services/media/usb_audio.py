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
        self._lock = threading.Lock()

        # Track cycling state
        self._playlist: List[str] = []
        self._current_index: int = -1
        self._current_track: Optional[str] = None
        self._is_looping: bool = False

        # Build initial playlist from songs folder
        self._build_playlist()

        try:
            # Initialize pygame mixer for audio playback (USB audio device)
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.initialized = True
            self.logger.info("USB Audio service initialized successfully (plughw:0,0)")
        except Exception as e:
            self.logger.error(f"USB Audio initialization failed: {e}")
            self.initialized = False

    def _build_playlist(self):
        """Build playlist from songs folder"""
        songs_path = os.path.join(self.base_path, "songs")
        if os.path.exists(songs_path):
            self._playlist = sorted([
                f for f in os.listdir(songs_path)
                if f.lower().endswith(('.mp3', '.wav', '.ogg'))
            ])
            self.logger.info(f"Built playlist with {len(self._playlist)} tracks")

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

                # Update playlist index if this is a song
                if '/songs/' in full_path or full_path.startswith(os.path.join(self.base_path, 'songs')):
                    track_name = os.path.basename(full_path)
                    if track_name in self._playlist:
                        self._current_index = self._playlist.index(track_name)

                loop_msg = " (looping)" if loop else ""
                self.logger.info(f"Playing audio{loop_msg}: {full_path}")
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
            self._current_track = None
            self._is_looping = False
            self.logger.info("Audio playback stopped")
            return {"success": True, "message": "Audio stopped"}
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
                self.logger.info("Audio playback paused")
                return {"success": True, "message": "Audio paused"}
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
            self.logger.info("Audio playback resumed")
            return {"success": True, "message": "Audio resumed"}
        except Exception as e:
            self.logger.error(f"Audio resume error: {e}")
            return {"success": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get audio system status with current track info"""
        is_playing = pygame.mixer.music.get_busy() if self.initialized else False
        return {
            "success": True,
            "status": {
                "initialized": self.initialized,
                "playing": is_playing,
                "base_path": self.base_path
            },
            "audio": {
                "playing": is_playing,
                "track": self._current_track if is_playing else None,
                "looping": self._is_looping if is_playing else False,
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
        """Check if audio is currently playing"""
        if self.initialized:
            return pygame.mixer.music.get_busy()
        return False

    def play_next(self) -> Dict[str, Any]:
        """Play next track in playlist"""
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        if not self._playlist:
            return {"success": False, "error": "No tracks in playlist"}

        with self._lock:
            # Move to next track (wrap around)
            self._current_index = (self._current_index + 1) % len(self._playlist)
            track = self._playlist[self._current_index]
            filepath = f"/songs/{track}"

            result = self.play_file(filepath, loop=False)
            result["track_index"] = self._current_index
            result["track_name"] = track
            return result

    def play_previous(self) -> Dict[str, Any]:
        """Play previous track in playlist"""
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        if not self._playlist:
            return {"success": False, "error": "No tracks in playlist"}

        with self._lock:
            # Move to previous track (wrap around)
            self._current_index = (self._current_index - 1) % len(self._playlist)
            track = self._playlist[self._current_index]
            filepath = f"/songs/{track}"

            result = self.play_file(filepath, loop=False)
            result["track_index"] = self._current_index
            result["track_name"] = track
            return result

    def get_playlist(self) -> Dict[str, Any]:
        """Get current playlist"""
        return {
            "success": True,
            "playlist": self._playlist,
            "current_index": self._current_index,
            "current_track": self._current_track
        }

    def refresh_playlist(self) -> Dict[str, Any]:
        """Refresh playlist from songs folder"""
        self._build_playlist()
        return {
            "success": True,
            "count": len(self._playlist),
            "playlist": self._playlist
        }

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