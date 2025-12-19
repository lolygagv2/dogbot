#!/usr/bin/env python3
"""
USB Audio service for TreatBot
Uses pygame mixer for reliable audio playback from VOICEMP3 folder
"""

import os
import time
import logging
import threading
from typing import Optional, Dict, Any

# Set audio environment BEFORE importing pygame
os.environ['SDL_AUDIODRIVER'] = 'alsa'
os.environ['AUDIODEV'] = 'plughw:2,0'

import pygame

logger = logging.getLogger('USBAudio')

class USBAudioService:
    """USB Audio service using pygame mixer"""

    def __init__(self):
        self.logger = logging.getLogger('USBAudio')
        self.initialized = False
        self.base_path = "/home/morgan/dogbot/VOICEMP3"
        self._lock = threading.Lock()

        try:
            # Initialize pygame mixer for audio playback (USB audio device)
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            self.initialized = True
            self.logger.info("USB Audio service initialized successfully (plughw:2,0)")
        except Exception as e:
            self.logger.error(f"USB Audio initialization failed: {e}")
            self.initialized = False

    def play_file(self, filepath: str) -> Dict[str, Any]:
        """Play an audio file"""
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
                pygame.mixer.music.play()

                self.logger.info(f"Playing audio: {full_path}")
                return {
                    "success": True,
                    "filepath": filepath,
                    "message": f"Playing {filepath}"
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
        """Get audio system status"""
        return {
            "success": True,
            "status": {
                "initialized": self.initialized,
                "playing": pygame.mixer.music.get_busy() if self.initialized else False,
                "base_path": self.base_path
            }
        }

# Global instance
_usb_audio_service = None

def get_usb_audio_service() -> USBAudioService:
    """Get the global USB audio service instance"""
    global _usb_audio_service
    if _usb_audio_service is None:
        _usb_audio_service = USBAudioService()
    return _usb_audio_service