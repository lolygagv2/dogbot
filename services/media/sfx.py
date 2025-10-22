#!/usr/bin/env python3
"""
Sound effect player using audio_controller
Handles celebration sounds and voice commands
"""

import threading
import time
import logging
from typing import Dict, Any, Optional, List

from core.bus import get_bus, publish_audio_event
from core.state import get_state
from core.hardware.audio_controller import AudioController


class SfxService:
    """
    Sound effect service using audio controller
    Plays celebration sounds and caches frequently used sounds
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.logger = logging.getLogger('SfxService')

        # Audio controller
        self.audio = None
        self.audio_initialized = False

        # Sound library (map names to DFPlayer track numbers)
        self.sound_library = {
            # Celebration sounds
            'good_dog': 'GOOD_BOY',      # DFPlayer command
            'excellent': 'GOOD_BOY',      # Using same for now
            'well_done': 'GOOD_BOY',
            'great_job': 'GOOD_BOY',

            # System sounds
            'startup': 'DOOR_SCAN',
            'shutdown': 'BUSY_SCAN',
            'dog_detected': 'HI_SCAN',
            'no_dog': 'BUSY_SCAN',

            # Training sounds
            'sit_reward': 'GOOD_BOY',
            'down_reward': 'GOOD_BOY',
            'stay_reward': 'GOOD_BOY',

            # Alert sounds
            'error': 'BUSY_SCAN',
            'warning': 'BUSY_SCAN',
            'emergency': 'BUSY_SCAN'
        }

        # Playing state
        self.currently_playing = None
        self.play_start_time = 0.0
        self.play_queue = []
        self._play_lock = threading.Lock()

        # Volume settings
        self.default_volume = 23
        self.current_volume = 23

    def initialize(self) -> bool:
        """Initialize audio controller"""
        try:
            self.audio = AudioController()

            if self.audio.is_initialized():
                self.audio_initialized = True
                self.logger.info("Audio system initialized")

                # Switch to DFPlayer mode
                self.audio.switch_to_dfplayer()

                # Set volume
                self.set_volume(self.default_volume)

                self.state.update_hardware(audio_initialized=True)
                return True
            else:
                self.logger.error("Audio controller initialization failed")
                return False

        except Exception as e:
            self.logger.error(f"Audio initialization error: {e}")
            return False

    def play_sound(self, sound_name: str, volume: Optional[int] = None,
                   interrupt: bool = False) -> bool:
        """
        Play a named sound

        Args:
            sound_name: Name of sound from library
            volume: Volume override (0-30)
            interrupt: Whether to interrupt current sound

        Returns:
            bool: True if sound started successfully
        """
        with self._play_lock:
            if not self.audio_initialized:
                self.logger.error("Audio not initialized")
                return False

            # Get sound command
            if sound_name not in self.sound_library:
                self.logger.warning(f"Unknown sound: {sound_name}")
                return False

            sound_command = self.sound_library[sound_name]

            # Check if already playing
            if self.currently_playing and not interrupt:
                self.logger.debug(f"Audio busy, queueing {sound_name}")
                self.play_queue.append((sound_name, volume))
                return True

            try:
                # Set volume if specified
                if volume is not None:
                    self.set_volume(volume)

                # Play sound
                success = self.audio.play_sound(sound_command)

                if success:
                    self.currently_playing = sound_name
                    self.play_start_time = time.time()

                    # Update state
                    self.state.update_hardware(audio_playing=True)

                    # Publish event
                    publish_audio_event('sound_started', {
                        'sound_name': sound_name,
                        'sound_command': sound_command,
                        'volume': self.current_volume,
                        'interrupt': interrupt,
                        'timestamp': time.time()
                    }, 'sfx_service')

                    self.logger.info(f"Playing sound: {sound_name}")

                    # Start monitoring thread to detect when sound finishes
                    threading.Thread(
                        target=self._monitor_playback,
                        args=(sound_name, time.time()),
                        daemon=True
                    ).start()

                    return True
                else:
                    self.logger.error(f"Failed to play sound: {sound_name}")
                    return False

            except Exception as e:
                self.logger.error(f"Sound playback error: {e}")
                return False

    def _monitor_playback(self, sound_name: str, start_time: float) -> None:
        """Monitor sound playback completion"""
        # Estimate sound duration (could be improved with actual duration tracking)
        estimated_duration = 2.0  # seconds, conservative estimate

        time.sleep(estimated_duration)

        with self._play_lock:
            # Check if this is still the current sound
            if self.currently_playing == sound_name and self.play_start_time == start_time:
                self._on_sound_finished(sound_name)

    def _on_sound_finished(self, sound_name: str) -> None:
        """Handle sound playback completion"""
        self.currently_playing = None
        self.play_start_time = 0.0

        # Update state
        self.state.update_hardware(audio_playing=False)

        # Publish event
        publish_audio_event('sound_finished', {
            'sound_name': sound_name,
            'timestamp': time.time()
        }, 'sfx_service')

        self.logger.debug(f"Sound finished: {sound_name}")

        # Play next in queue
        if self.play_queue:
            next_sound, next_volume = self.play_queue.pop(0)
            self.play_sound(next_sound, next_volume, interrupt=False)

    def stop_sound(self) -> bool:
        """Stop current sound playback"""
        with self._play_lock:
            if not self.currently_playing:
                return True

            try:
                # DFPlayer doesn't have explicit stop, but we can track it
                self.currently_playing = None
                self.play_start_time = 0.0

                # Update state
                self.state.update_hardware(audio_playing=False)

                # Clear queue
                self.play_queue.clear()

                publish_audio_event('sound_stopped', {}, 'sfx_service')
                self.logger.info("Sound playback stopped")
                return True

            except Exception as e:
                self.logger.error(f"Stop sound error: {e}")
                return False

    def set_volume(self, volume: int) -> bool:
        """Set audio volume (0-30)"""
        try:
            volume = max(0, min(30, volume))
            success = self.audio.set_volume(volume)

            if success:
                self.current_volume = volume
                self.logger.debug(f"Volume set to {volume}")

                publish_audio_event('volume_changed', {
                    'volume': volume
                }, 'sfx_service')

            return success

        except Exception as e:
            self.logger.error(f"Set volume error: {e}")
            return False

    def play_celebration(self, behavior: str = "good") -> bool:
        """Play celebration sound for specific behavior"""
        celebration_sounds = {
            'sit': 'sit_reward',
            'down': 'down_reward',
            'stay': 'stay_reward',
            'good': 'good_dog',
            'excellent': 'excellent'
        }

        sound_name = celebration_sounds.get(behavior, 'good_dog')
        return self.play_sound(sound_name)

    def play_system_sound(self, event: str) -> bool:
        """Play system event sound"""
        system_sounds = {
            'startup': 'startup',
            'shutdown': 'shutdown',
            'dog_detected': 'dog_detected',
            'no_dog': 'no_dog',
            'error': 'error',
            'warning': 'warning',
            'emergency': 'emergency'
        }

        sound_name = system_sounds.get(event, 'error')
        return self.play_sound(sound_name)

    def is_playing(self) -> bool:
        """Check if sound is currently playing"""
        return self.currently_playing is not None

    def get_queue_length(self) -> int:
        """Get number of sounds in queue"""
        return len(self.play_queue)

    def clear_queue(self) -> None:
        """Clear sound queue"""
        with self._play_lock:
            self.play_queue.clear()
            self.logger.debug("Sound queue cleared")

    def get_available_sounds(self) -> List[str]:
        """Get list of available sound names"""
        return list(self.sound_library.keys())

    def add_sound(self, name: str, command: str) -> None:
        """Add sound to library"""
        self.sound_library[name] = command
        self.logger.info(f"Added sound: {name} -> {command}")

    def get_status(self) -> Dict[str, Any]:
        """Get audio service status"""
        return {
            'initialized': self.audio_initialized,
            'currently_playing': self.currently_playing,
            'play_start_time': self.play_start_time,
            'time_playing': time.time() - self.play_start_time if self.currently_playing else 0,
            'queue_length': len(self.play_queue),
            'current_volume': self.current_volume,
            'available_sounds': len(self.sound_library),
            'audio_mode': 'dfplayer' if self.audio_initialized else 'unknown'
        }

    def cleanup(self) -> None:
        """Clean shutdown"""
        self.stop_sound()
        if self.audio:
            self.audio.cleanup()
        self.logger.info("Audio service cleaned up")


# Global SFX service instance
_sfx_instance = None
_sfx_lock = threading.Lock()

def get_sfx_service() -> SfxService:
    """Get the global SFX service instance (singleton)"""
    global _sfx_instance
    if _sfx_instance is None:
        with _sfx_lock:
            if _sfx_instance is None:
                _sfx_instance = SfxService()
    return _sfx_instance