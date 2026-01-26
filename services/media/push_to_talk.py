#!/usr/bin/env python3
"""
services/media/push_to_talk.py - Two-Way Audio Push-to-Talk Service

Handles:
1. Receiving audio from app (play through speaker)
2. Recording audio from microphone (send back to app)

Uses arecord/aplay for reliable USB audio device access.
"""

import os
import base64
import subprocess
import threading
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Temp file paths
TEMP_DIR = Path("/tmp/wimz_ptt")
INCOMING_AUDIO = TEMP_DIR / "incoming_audio"
OUTGOING_AUDIO = TEMP_DIR / "outgoing_audio"

# Recording settings
DEFAULT_DURATION = 5  # seconds
MAX_DURATION = 10  # seconds
SAMPLE_RATE = 16000  # 16kHz for speech
CHANNELS = 1  # mono

# Detect USB audio card dynamically
def _detect_usb_audio_card() -> int:
    """Detect which card number the USB Audio Device is on"""
    try:
        result = subprocess.run(['aplay', '-l'], capture_output=True, text=True, timeout=5)
        for line in result.stdout.split('\n'):
            if 'USB Audio' in line and 'card' in line:
                card_num = int(line.split('card ')[1].split(':')[0])
                return card_num
    except Exception as e:
        logging.warning(f"Could not detect USB audio card: {e}")
    return 2  # Default fallback


USB_AUDIO_CARD = _detect_usb_audio_card()


class PushToTalkService:
    """
    Two-way audio service for push-to-talk functionality.

    Features:
    - Play audio received from app (AAC, MP3, WAV, Opus)
    - Record audio from USB microphone
    - Compress and encode for transmission
    """

    def __init__(self):
        self.logger = logging.getLogger('PushToTalk')
        self._lock = threading.Lock()
        self._recording = False
        self._playing = False

        # Ensure temp directory exists
        TEMP_DIR.mkdir(parents=True, exist_ok=True)

        # Check hardware availability
        self._mic_available = self._check_microphone()
        self._speaker_available = self._check_speaker()

        self.logger.info(
            f"PushToTalkService initialized (card={USB_AUDIO_CARD}, "
            f"mic={self._mic_available}, speaker={self._speaker_available})"
        )

    def _check_microphone(self) -> bool:
        """Check if USB microphone is available"""
        try:
            result = subprocess.run(
                ['arecord', '-l'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'USB Audio' in result.stdout
        except Exception as e:
            self.logger.warning(f"Microphone check failed: {e}")
            return False

    def _check_speaker(self) -> bool:
        """Check if USB speaker is available"""
        try:
            result = subprocess.run(
                ['aplay', '-l'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return 'USB Audio' in result.stdout
        except Exception as e:
            self.logger.warning(f"Speaker check failed: {e}")
            return False

    def play_audio(self, audio_data: bytes, format: str = "aac") -> Dict[str, Any]:
        """
        Play audio received from app via USBAudio service (pygame).
        Routes through USBAudio to avoid device conflicts with aplay.

        Args:
            audio_data: Raw audio bytes
            format: Audio format (aac, mp3, wav, opus)

        Returns:
            Dict with success status
        """
        if self._playing:
            return {"success": False, "error": "Already playing audio"}

        with self._lock:
            self._playing = True
            try:
                ext = format.lower()
                if ext not in ('aac', 'mp3', 'wav', 'opus', 'm4a'):
                    ext = 'aac'

                input_file = INCOMING_AUDIO.with_suffix(f'.{ext}')

                # Write incoming audio to temp file
                with open(input_file, 'wb') as f:
                    f.write(audio_data)

                self.logger.info(f"PTT received: {len(audio_data)} bytes, format={format}")

                # Determine playable file path
                # pygame handles WAV and MP3 natively; convert others
                if ext in ('wav', 'mp3'):
                    play_file = input_file
                else:
                    # Convert AAC/opus/m4a to WAV for pygame compatibility
                    play_file = INCOMING_AUDIO.with_suffix('.wav')
                    result = subprocess.run(
                        ['ffmpeg', '-y', '-i', str(input_file),
                         '-ar', '44100', '-ac', '2', str(play_file)],
                        capture_output=True, timeout=10
                    )
                    if result.returncode != 0:
                        self.logger.error(f"Audio conversion failed: {result.stderr.decode()}")
                        return {"success": False, "error": "Audio conversion failed"}

                # Route through USBAudio service to avoid device conflicts
                # PTT has priority - stop any current playback first
                from services.media.usb_audio import get_usb_audio_service
                usb_audio = get_usb_audio_service()

                if usb_audio.is_busy():
                    usb_audio.stop()
                    self.logger.info("PTT: Stopped current audio for priority playback")

                result = usb_audio.play_file(str(play_file))

                if result.get('success'):
                    # Wait for PTT audio to finish playing
                    usb_audio.wait_for_completion(timeout=30)
                    self.logger.info("PTT playback completed")
                    return {
                        "success": True,
                        "message": "Audio played",
                        "size_bytes": len(audio_data),
                        "format": format
                    }
                else:
                    self.logger.error(f"PTT playback failed: {result.get('error')}")
                    return {"success": False, "error": result.get('error', 'Playback failed')}

            except subprocess.TimeoutExpired:
                self.logger.error("Audio conversion timed out")
                return {"success": False, "error": "Conversion timed out"}
            except Exception as e:
                self.logger.error(f"PTT play error: {e}")
                return {"success": False, "error": str(e)}
            finally:
                self._playing = False

    def play_audio_base64(self, base64_data: str, format: str = "aac") -> Dict[str, Any]:
        """
        Play base64 encoded audio from app.

        Args:
            base64_data: Base64 encoded audio
            format: Audio format

        Returns:
            Dict with success status
        """
        try:
            audio_data = base64.b64decode(base64_data)
            return self.play_audio(audio_data, format)
        except Exception as e:
            self.logger.error(f"Base64 decode error: {e}")
            return {"success": False, "error": f"Invalid base64 data: {e}"}

    def record_audio(self, duration: float = DEFAULT_DURATION, format: str = "aac") -> Dict[str, Any]:
        """
        Record audio from USB microphone.

        Args:
            duration: Recording duration in seconds (max 10s)
            format: Output format (aac, mp3, opus)

        Returns:
            Dict with base64 encoded audio data
        """
        if not self._mic_available:
            return {"success": False, "error": "Microphone not available"}

        if self._recording:
            return {"success": False, "error": "Already recording"}

        # Clamp duration
        duration = min(max(duration, 1), MAX_DURATION)

        with self._lock:
            self._recording = True
            try:
                wav_file = OUTGOING_AUDIO.with_suffix('.wav')
                output_file = OUTGOING_AUDIO.with_suffix(f'.{format}')

                # Pause bark detector if running
                bark_paused = self._pause_bark_detector()

                # Small delay to ensure mic is free
                if bark_paused:
                    time.sleep(0.3)

                # Record from USB microphone
                # Use plughw: (not hw:) for automatic sample rate conversion
                self.logger.info(f"Recording {duration}s from USB mic (card {USB_AUDIO_CARD})...")
                record_cmd = [
                    'arecord',
                    '-D', f'plughw:{USB_AUDIO_CARD},0',
                    '-f', 'S16_LE',
                    '-r', str(SAMPLE_RATE),
                    '-c', str(CHANNELS),
                    '-d', str(int(duration)),
                    str(wav_file)
                ]

                start_time = time.time()
                result = subprocess.run(
                    record_cmd,
                    capture_output=True,
                    timeout=duration + 5
                )

                # Re-enable bark detector
                if bark_paused:
                    self._resume_bark_detector()

                if result.returncode != 0:
                    self.logger.error(f"Recording failed: {result.stderr.decode()}")
                    return {"success": False, "error": "Recording failed - microphone busy"}

                actual_duration = time.time() - start_time
                self.logger.info(f"Recording completed ({actual_duration:.1f}s)")

                # Check file was created
                if not wav_file.exists() or wav_file.stat().st_size < 1000:
                    return {"success": False, "error": "Recording produced no data"}

                # Convert to requested format (with compression)
                if format == 'aac':
                    encode_cmd = [
                        'ffmpeg', '-y', '-i', str(wav_file),
                        '-c:a', 'aac', '-b:a', '64k',
                        '-ar', str(SAMPLE_RATE),
                        str(output_file)
                    ]
                elif format == 'opus':
                    encode_cmd = [
                        'ffmpeg', '-y', '-i', str(wav_file),
                        '-c:a', 'libopus', '-b:a', '48k',
                        '-ar', str(SAMPLE_RATE),
                        str(output_file)
                    ]
                elif format == 'mp3':
                    encode_cmd = [
                        'ffmpeg', '-y', '-i', str(wav_file),
                        '-c:a', 'libmp3lame', '-b:a', '64k',
                        '-ar', str(SAMPLE_RATE),
                        str(output_file)
                    ]
                else:
                    # Default to AAC
                    output_file = OUTGOING_AUDIO.with_suffix('.aac')
                    encode_cmd = [
                        'ffmpeg', '-y', '-i', str(wav_file),
                        '-c:a', 'aac', '-b:a', '64k',
                        '-ar', str(SAMPLE_RATE),
                        str(output_file)
                    ]

                result = subprocess.run(
                    encode_cmd,
                    capture_output=True,
                    timeout=30
                )

                if result.returncode != 0:
                    self.logger.error(f"Encoding failed: {result.stderr.decode()}")
                    return {"success": False, "error": "Audio encoding failed"}

                # Read and encode as base64
                with open(output_file, 'rb') as f:
                    audio_bytes = f.read()

                base64_data = base64.b64encode(audio_bytes).decode('utf-8')

                self.logger.info(
                    f"Audio encoded: {len(audio_bytes)} bytes, "
                    f"base64={len(base64_data)} chars"
                )

                return {
                    "success": True,
                    "data": base64_data,
                    "format": format,
                    "duration_ms": int(duration * 1000),
                    "size_bytes": len(audio_bytes)
                }

            except subprocess.TimeoutExpired:
                self.logger.error("Recording timed out")
                self._resume_bark_detector()
                return {"success": False, "error": "Recording timed out"}
            except Exception as e:
                self.logger.error(f"Record audio error: {e}")
                self._resume_bark_detector()
                return {"success": False, "error": str(e)}
            finally:
                self._recording = False

    def _pause_bark_detector(self) -> bool:
        """Pause bark detector to free microphone"""
        try:
            from services.perception.bark_detector import get_bark_detector_service
            bark_detector = get_bark_detector_service()
            if bark_detector and bark_detector.enabled:
                bark_detector.set_enabled(False)
                self.logger.debug("Bark detector paused")
                return True
        except Exception as e:
            self.logger.debug(f"Could not pause bark detector: {e}")
        return False

    def _resume_bark_detector(self):
        """Resume bark detector after recording"""
        try:
            from services.perception.bark_detector import get_bark_detector_service
            bark_detector = get_bark_detector_service()
            if bark_detector:
                bark_detector.set_enabled(True)
                self.logger.debug("Bark detector resumed")
        except Exception as e:
            self.logger.debug(f"Could not resume bark detector: {e}")

    def get_status(self) -> Dict[str, Any]:
        """Get push-to-talk service status"""
        return {
            "usb_audio_card": USB_AUDIO_CARD,
            "microphone_available": self._mic_available,
            "speaker_available": self._speaker_available,
            "recording": self._recording,
            "playing": self._playing,
            "sample_rate": SAMPLE_RATE,
            "max_duration": MAX_DURATION
        }

    def test_microphone(self, duration: float = 1.0) -> Dict[str, Any]:
        """Test microphone by recording a short clip"""
        result = self.record_audio(duration=duration, format='aac')
        if result.get('success'):
            return {
                "success": True,
                "message": f"Microphone test passed - recorded {result.get('size_bytes')} bytes",
                "size_bytes": result.get('size_bytes')
            }
        return result

    def test_speaker(self) -> Dict[str, Any]:
        """Test speaker by playing a tone"""
        try:
            # Generate a short beep using sox (if available) or ffmpeg
            beep_file = TEMP_DIR / "test_beep.wav"

            # Try sox first
            try:
                subprocess.run(
                    ['sox', '-n', str(beep_file), 'synth', '0.3', 'sine', '440'],
                    capture_output=True,
                    timeout=5
                )
            except FileNotFoundError:
                # Fall back to ffmpeg
                subprocess.run([
                    'ffmpeg', '-y', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=0.3',
                    str(beep_file)
                ], capture_output=True, timeout=5)

            # Play the beep
            result = subprocess.run(
                ['aplay', '-D', f'plughw:{USB_AUDIO_CARD},0', str(beep_file)],
                capture_output=True,
                timeout=5
            )

            if result.returncode == 0:
                return {"success": True, "message": "Speaker test passed"}
            else:
                return {"success": False, "error": f"Playback failed: {result.stderr.decode()}"}

        except Exception as e:
            return {"success": False, "error": str(e)}


# Global singleton instance
_ptt_service: Optional[PushToTalkService] = None
_ptt_lock = threading.Lock()


def get_push_to_talk_service() -> PushToTalkService:
    """Get the global PushToTalkService instance"""
    global _ptt_service
    with _ptt_lock:
        if _ptt_service is None:
            _ptt_service = PushToTalkService()
    return _ptt_service
