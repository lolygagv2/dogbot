#!/usr/bin/env python3
"""
core/audio_controller.py - USB Audio controller for TreatBot
Simple USB audio playback using system commands
"""

import subprocess
import time
import os
import logging
from typing import Dict, Any, Optional

class AudioController:
    """USB Audio controller using system audio commands"""

    def __init__(self):
        self.logger = logging.getLogger('AudioController')
        self.initialized = False
        self.current_volume = 50
        self.audio_device = None

        # Initialize USB audio
        self._initialize_usb_audio()

    def _initialize_usb_audio(self):
        """Initialize USB audio system"""
        try:
            # Check if USB audio device is available
            result = subprocess.run(['aplay', '-l'], capture_output=True, text=True)

            if result.returncode == 0:
                # Look for USB audio devices
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'USB Audio' in line or 'usb' in line.lower():
                        # Extract card number
                        if 'card' in line:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if part == 'card' and i + 1 < len(parts):
                                    card_info = parts[i + 1].rstrip(':')
                                    self.audio_device = f"plughw:{card_info}"
                                    break

                if self.audio_device:
                    self.initialized = True
                    self.logger.info(f"USB audio device found: {self.audio_device}")
                else:
                    # Fallback to default device
                    self.audio_device = "default"
                    self.initialized = True
                    self.logger.info("Using default audio device")

                # Initialize USB audio levels on card 0 (USB Audio Device)
                # Speaker to 90% for good volume without distortion
                try:
                    subprocess.run(
                        ['amixer', '-c', '0', 'sset', 'Speaker', '90%'],
                        capture_output=True, timeout=2
                    )
                    self.current_volume = 90
                    self.logger.info("USB speaker set to 90%")
                except Exception as spk_err:
                    self.logger.warning(f"Could not set speaker volume: {spk_err}")

                # Microphone to 100% capture
                try:
                    subprocess.run(
                        ['amixer', '-c', '0', 'sset', 'Mic', '100%', 'cap'],
                        capture_output=True, timeout=2
                    )
                    self.logger.info("USB microphone set to 100% capture")
                except Exception as mic_err:
                    self.logger.warning(f"Could not set mic volume: {mic_err}")

                return True
            else:
                self.logger.warning("No audio devices found")
                return False
        except Exception as e:
            self.logger.error(f"USB audio initialization failed: {e}")
            return False

    def play_sound(self, sound_name: str) -> bool:
        """Play sound by name - placeholder for compatibility"""
        self.logger.info(f"Play sound requested: {sound_name}")
        # For now, just return True since we don't have actual audio files
        return True

    def play_file_by_path(self, filepath: str) -> bool:
        """Play specific audio file by path using aplay"""
        if not self.initialized:
            self.logger.error("Audio not initialized")
            return False

        if not os.path.exists(filepath):
            self.logger.warning(f"Audio file not found: {filepath}")
            return False

        try:
            # Use aplay to play the file
            cmd = ['aplay', '-D', self.audio_device, filepath]
            subprocess.run(cmd, capture_output=True, check=True)
            self.logger.info(f"Played audio file: {filepath}")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to play audio file {filepath}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Audio playback error: {e}")
            return False

    def set_volume(self, volume: int) -> bool:
        """Set system volume using amixer"""
        if not self.initialized:
            return False

        try:
            volume = max(0, min(100, volume))
            self.current_volume = volume

            # Set USB audio speaker volume (card 0)
            cmd = ['amixer', '-c', '0', 'sset', 'Speaker', f'{volume}%']
            result = subprocess.run(cmd, capture_output=True)

            if result.returncode == 0:
                self.logger.info(f"USB Speaker volume set to {volume}%")
                return True
            else:
                # Fallback to master volume
                cmd = ['amixer', 'sset', 'Master', f'{volume}%']
                result = subprocess.run(cmd, capture_output=True)
                if result.returncode == 0:
                    self.logger.info(f"Master volume set to {volume}%")
                    return True

            return False
        except Exception as e:
            self.logger.error(f"Volume control error: {e}")
            return False

    def play_pause_toggle(self) -> bool:
        """Toggle play/pause - not implemented for USB audio"""
        self.logger.info("Play/pause toggle not supported for USB audio")
        return True

    def is_initialized(self) -> bool:
        """Check if audio system is properly initialized"""
        return self.initialized

    def get_status(self) -> Dict[str, Any]:
        """Get audio system status"""
        return {
            'initialized': self.initialized,
            'audio_device': self.audio_device,
            'current_volume': self.current_volume,
            'type': 'usb_audio'
        }

    def cleanup(self) -> None:
        """Clean shutdown of audio system"""
        self.logger.info("Audio controller cleanup complete")
        self.initialized = False

# Test function
def test_audio():
    """Simple test function for audio controller"""
    print("Testing USB Audio Controller...")

    audio = AudioController()
    if audio.is_initialized():
        print("✅ Audio system initialized")
        print(f"Status: {audio.get_status()}")
    else:
        print("❌ Audio initialization failed")

if __name__ == "__main__":
    test_audio()