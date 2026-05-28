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

# Route pygame through PipeWire (via pipewire-pulse) for echo cancellation support.
# PipeWire's echo-cancel module needs both playback and capture to flow through it.
# Falls back to direct ALSA if PipeWire socket is unavailable (e.g. systemd service context).
_audio_backend = 'alsa'
if os.path.exists('/usr/bin/pipewire-pulse'):
    os.environ['SDL_AUDIODRIVER'] = 'pulseaudio'
    import pygame
    try:
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        pygame.mixer.quit()
        _audio_backend = 'pulseaudio'
    except Exception:
        os.environ['SDL_AUDIODRIVER'] = 'alsa'
        os.environ['AUDIODEV'] = f'plughw:{USB_AUDIO_CARD},0'
else:
    os.environ['SDL_AUDIODRIVER'] = 'alsa'
    os.environ['AUDIODEV'] = f'plughw:{USB_AUDIO_CARD},0'
    import pygame

logger = logging.getLogger('USBAudio')
logger.info(f"USB Audio detected on card {USB_AUDIO_CARD}, backend: {_audio_backend}")

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
        self._loading: bool = False  # Track when we're loading a file (prevents false state sync)
        self._last_play_time: float = 0  # Track when playback started

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
            self.logger.info(f"USB Audio service initialized successfully (driver={os.environ.get('SDL_AUDIODRIVER', 'default')})")
        except Exception as e:
            if os.environ.get('SDL_AUDIODRIVER') == 'pulseaudio':
                self.logger.warning(f"PulseAudio connection failed ({e}), falling back to ALSA")
                os.environ['SDL_AUDIODRIVER'] = 'alsa'
                os.environ['AUDIODEV'] = f'plughw:{USB_AUDIO_CARD},0'
                try:
                    pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                    self.initialized = True
                    self.logger.info(f"USB Audio service initialized successfully (ALSA fallback, plughw:{USB_AUDIO_CARD},0)")
                except Exception as e2:
                    self.logger.error(f"USB Audio initialization failed on ALSA fallback: {e2}")
                    self.initialized = False
            else:
                self.logger.error(f"USB Audio initialization failed: {e}")
                self.initialized = False

    def _build_playlist(self, dog_id: str = None):
        """Build playlist from songs folder, combining default and all dog-specific songs

        Combines default and all dog-specific songs so uploaded songs are always available.
        """
        songs_base = os.path.join(self.base_path, "songs")
        default_path = os.path.join(songs_base, "default")

        playlist = []

        # Always include default songs first
        if os.path.isdir(default_path):
            default_songs = sorted([
                f"default/{f}" for f in os.listdir(default_path)
                if f.lower().endswith(('.mp3', '.wav', '.ogg'))
                and os.path.isfile(os.path.join(default_path, f))
            ])
            playlist.extend(default_songs)

        # Add songs from ALL dog folders (not just the specified dog_id)
        # This ensures uploaded songs are always available
        if os.path.isdir(songs_base):
            for folder in sorted(os.listdir(songs_base)):
                if folder == 'default':
                    continue  # Already added above
                folder_path = os.path.join(songs_base, folder)
                if os.path.isdir(folder_path):
                    dog_songs = sorted([
                        f"{folder}/{f}" for f in os.listdir(folder_path)
                        if f.lower().endswith(('.mp3', '.wav', '.ogg'))
                        and os.path.isfile(os.path.join(folder_path, f))
                    ])
                    playlist.extend(dog_songs)

        self._playlist = playlist
        self.logger.info(f"Built playlist with {len(self._playlist)} tracks (default + {len(playlist) - len([p for p in playlist if p.startswith('default/')])} uploaded)")

    def _check_audio_health(self) -> bool:
        """Check if pygame.mixer is still functional"""
        try:
            # Try to get mixer state - this will fail if mixer died
            pygame.mixer.music.get_busy()
            pygame.mixer.music.get_volume()
            return True
        except Exception as e:
            self.logger.warning(f"Audio health check failed: {e}")
            return False

    def reinitialize(self) -> bool:
        """Reinitialize pygame.mixer after failure"""
        self.logger.info("Attempting to reinitialize audio system...")
        try:
            # Quit existing mixer if possible
            try:
                pygame.mixer.quit()
            except Exception:
                pass

            # Small delay for cleanup
            import time
            time.sleep(0.3)

            # Reinitialize
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.initialized = True
            self.logger.info("Audio system reinitialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Audio reinitialize failed: {e}")
            # Try ALSA fallback
            try:
                os.environ['SDL_AUDIODRIVER'] = 'alsa'
                os.environ['AUDIODEV'] = f'plughw:{USB_AUDIO_CARD},0'
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                self.initialized = True
                self.logger.info("Audio reinitialized via ALSA fallback")
                return True
            except Exception as e2:
                self.logger.error(f"Audio reinitialize ALSA fallback failed: {e2}")
                self.initialized = False
                return False

    def play_file(self, filepath: str, loop: bool = False) -> Dict[str, Any]:
        """Play an audio file

        Args:
            filepath: Path to audio file
            loop: If True, loop the audio indefinitely until stopped
        """
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        # Health check - auto-reinitialize if pygame.mixer died
        if not self._check_audio_health():
            self.logger.warning("Audio system unhealthy, attempting reinitialize...")
            if not self.reinitialize():
                return {"success": False, "error": "Audio system dead, reinitialize failed"}

        # Try to acquire lock with timeout to detect deadlocks
        lock_acquired = self._lock.acquire(timeout=2.0)
        if not lock_acquired:
            self.logger.error("AUDIO DEADLOCK DETECTED - lock held >2s, forcing reinitialize")
            try:
                pygame.mixer.quit()
            except:
                pass
            self.initialized = False
            if not self.reinitialize():
                return {"success": False, "error": "Audio deadlock recovery failed"}
            lock_acquired = self._lock.acquire(timeout=1.0)
            if not lock_acquired:
                return {"success": False, "error": "Audio lock still blocked after recovery"}

        try:
            # Set loading flag to prevent false state sync during load
            self._loading = True

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
                self._loading = False
                return {"success": False, "error": f"File not found: {full_path}"}

            # Play the audio file
            pygame.mixer.music.load(full_path)
            # -1 = loop indefinitely, 0 = play once
            pygame.mixer.music.play(-1 if loop else 0)

            # Track current playing file
            self._current_track = os.path.basename(full_path)
            self._is_looping = loop
            self._last_play_time = time.time()
            is_music = False

            # Update music player state if this is a playlist song
            if '/songs/' in full_path or full_path.startswith(os.path.join(self.base_path, 'songs')):
                track_name = os.path.basename(full_path)
                # Check if track (with or without prefix) is in playlist
                for i, pl_track in enumerate(self._playlist):
                    if pl_track.endswith(track_name):
                        self._current_index = i
                        self._playlist_track = pl_track
                        break
                self._music_playing = True
                is_music = True

            self._loading = False

            loop_msg = " (looping)" if loop else ""
            # Log with call stack to trace where audio is triggered from
            import traceback
            caller_info = "".join(traceback.format_stack()[-4:-1]).strip().replace('\n', ' | ')
            self.logger.debug(f"AUDIO PLAY{loop_msg}: {full_path} [source: {caller_info[:200]}...]")

            # Send audio state event to app (only for music, not voice clips)
            if is_music:
                self._send_audio_event('playing', self._playlist_track)

            # Suppress bark detection while speaker is active (echo prevention).
            # Music excluded — bark detector is already mode-gated and music
            # plays for minutes, so suppressing the full duration is unsafe.
            if not is_music:
                self._start_bark_suppression_monitor()

            return {
                "success": True,
                "filepath": filepath,
                "track": self._current_track,
                "loop": loop,
                "message": f"Playing {filepath}{loop_msg}"
            }

        except Exception as e:
            self.logger.error(f"Audio playback error: {e}")
            self._loading = False
            # Attempt recovery on playback failure
            if "mixer" in str(e).lower() or "audio" in str(e).lower():
                self.logger.warning("Audio mixer error, attempting reinitialize...")
                if self.reinitialize():
                    self.logger.info("Audio recovered, but playback request lost - retry manually")
            return {"success": False, "error": str(e)}
        finally:
            self._lock.release()

    def stop(self) -> Dict[str, Any]:
        """Stop audio playback"""
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        try:
            was_playing = self._music_playing
            pygame.mixer.music.stop()
            # Keep _playlist_track so we know what song is selected
            self._is_looping = False
            self._music_playing = False
            self._loading = False
            self.logger.info("Audio playback stopped")

            # Send audio state event to app
            if was_playing:
                self._send_audio_event('stopped', self._playlist_track)

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
                self._send_audio_event('paused', self._playlist_track)
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
            self._send_audio_event('playing', self._playlist_track)
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
        """Get current volume (0-100) from the VolumeManager source of truth."""
        try:
            from services.media.volume_manager import get_volume_manager
            return get_volume_manager().get_volume()
        except Exception:
            return 0

    def is_busy(self) -> bool:
        """Check if any audio is currently playing"""
        if self.initialized:
            return pygame.mixer.music.get_busy()
        return False

    def _send_audio_event(self, state: str, track: Optional[str] = None) -> None:
        """Send audio state event to app via relay client

        Args:
            state: 'playing', 'stopped', 'paused'
            track: Current track name (optional)
        """
        try:
            from services.cloud.relay_client import get_relay_client
            relay = get_relay_client()
            if relay and relay.connected:
                relay.send_event('audio_state', {
                    'state': state,
                    'track': track,
                    'playing': state == 'playing',
                    'playlist_index': self._current_index,
                    'playlist_length': len(self._playlist),
                })
                self.logger.debug(f"Sent audio_state event: {state}, track={track}")
        except Exception as e:
            # Don't let event sending failures affect audio playback
            self.logger.debug(f"Could not send audio_state event: {e}")

    def toggle(self) -> Dict[str, Any]:
        """Toggle music playback - play current song if stopped, stop if playing"""
        if not self.initialized:
            return {"success": False, "error": "Audio not initialized"}

        # Don't toggle while loading (prevents restart loops with large files)
        if self._loading:
            self.logger.warning("Toggle ignored - audio is loading")
            return {"success": False, "error": "Audio is loading, please wait"}

        # Check actual pygame state - more reliable than our flag
        is_busy = pygame.mixer.music.get_busy()

        # Grace period after play started (large files take time to start)
        time_since_play = time.time() - self._last_play_time
        if self._music_playing and time_since_play < 1.0:
            # Recently started playing - treat as playing even if pygame not busy yet
            self.logger.info("Toggle: stopping (recently started)")
            return self.stop()

        # Sync our state with pygame
        if self._music_playing and not is_busy:
            # Song finished naturally
            self._music_playing = False
            self._is_looping = False
            self._send_audio_event('stopped', self._playlist_track)

        if is_busy or self._music_playing:
            # Music playing - stop
            self.logger.info("Toggle: stopping playback")
            return self.stop()
        else:
            # Music not playing - play current song
            self.logger.info("Toggle: starting playback")
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
            from services.media.voice_lookup import resolve_voice_file

            # Use centralized voice lookup with full fallback chain:
            # dog_id_override > ArUco (5s TTL) > session dog > select_dog > default
            # Also handles multi-extension (.wav, .mp3, .m4a, .aac)
            voice_path = resolve_voice_file(command, dog_id_override=dog_id)

            if not voice_path:
                self.logger.warning(f"No voice file found for '{command}'")
                return {"success": False, "error": f"Voice file not found: {command}"}

            # Determine source for logging
            voice_source = "custom" if "/dog_" in voice_path else "default"

            # Play the voice
            result = self.play_file(voice_path, loop=loop)
            result["voice_source"] = voice_source
            result["command"] = command
            result["dog_id"] = dog_id

            if result.get("success"):
                self.logger.info(f"USBAudio: playing {voice_path} (cmd={command}, dog={dog_id})")

            return result

        except ImportError as e:
            self.logger.error(f"voice_lookup import failed: {e}")
            return {"success": False, "error": "Voice lookup unavailable"}
        except Exception as e:
            self.logger.error(f"Play command error: {e}")
            return {"success": False, "error": str(e)}

    def _start_bark_suppression_monitor(self):
        """Suppress bark detection while the speaker is active (echo prevention).

        Runs a daemon thread that re-extends the bark detector's suppression
        window every 200ms until pygame stops playing, then adds a 500ms tail
        to cover speaker reverb and mic AGC release.

        Fails silent if the bark detector isn't initialized yet — that just means
        nothing is listening for barks, so there's nothing to suppress.
        """
        def monitor():
            try:
                from services.perception.bark_detector import peek_bark_detector_service
                svc = peek_bark_detector_service()
                if svc is None:
                    return
                # Initial 1s window covers the gap before this thread loops
                svc.suppress_detection(1.0)
                while True:
                    try:
                        if not pygame.mixer.music.get_busy():
                            break
                    except Exception:
                        break
                    svc.suppress_detection(0.4)  # 200ms loop + 200ms slack
                    time.sleep(0.2)
                # Tail for speaker reverb / AGC release
                svc.suppress_detection(0.5)
            except Exception:
                pass

        threading.Thread(target=monitor, daemon=True, name="BarkSuppressMonitor").start()

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
        """Set volume (0-100).

        Delegates to VolumeManager (the single source of truth) so the change
        applies to the hardware mixer and persists across reboots. The pygame
        software volume is intentionally left pinned at 1.0 by VolumeManager.
        """
        try:
            from services.media.volume_manager import get_volume_manager
            return get_volume_manager().set_volume(volume)
        except Exception as e:
            self.logger.error(f"set_volume delegation failed: {e}")
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