"""
Audio buffering for bark detection using arecord subprocess
Replaces PyAudio with arecord to avoid USB freezing issues
"""

import re
import subprocess
import tempfile
import threading
import queue
import logging
import time
import os
import wave
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def _find_usb_card_number() -> Optional[int]:
    """Return the ALSA card number of the USB audio device, or None."""
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'USB Audio' in line:
                m = re.match(r'card\s+(\d+)', line.strip())
                if m:
                    return int(m.group(1))
    except Exception as e:
        logger.warning(f"Error enumerating ALSA capture devices: {e}")
    return None


def _probe_device_has_signal(device: str, seconds: float = 0.4) -> bool:
    """Record a short probe and check whether any non-zero samples arrived."""
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        r = subprocess.run(
            ['arecord', '-D', device, '-f', 'S16_LE', '-r', '44100',
             '-c', '1', '-d', '1', tmp_path],
            capture_output=True, timeout=seconds + 2.0
        )
        if r.returncode != 0 or not os.path.exists(tmp_path):
            return False
        with wave.open(tmp_path, 'rb') as w:
            frames = w.readframes(w.getnframes())
        arr = np.frombuffer(frames, dtype=np.int16)
        return bool(np.any(arr != 0))
    except Exception as e:
        logger.warning(f"Probe of device '{device}' failed: {e}")
        return False
    finally:
        if os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def find_usb_audio_device() -> str:
    """
    Pick the best arecord device for bark detection.

    Prefers 'default' (PipeWire) so AEC stays in the capture path. If PipeWire's
    default source is dead (as seen on TB2 when WirePlumber hasn't adopted the
    USB device — arecord succeeds but every sample is zero), falls back to
    plughw:{card},0 against the USB card directly.
    """
    usb_card = _find_usb_card_number()

    if _probe_device_has_signal('default'):
        logger.info("Bark capture using 'default' (PipeWire healthy)")
        return 'default'

    if usb_card is not None:
        fallback = f'plughw:{usb_card},0'
        logger.warning(
            f"PipeWire 'default' source produced silence on probe — "
            f"falling back to {fallback}. AEC will be bypassed for bark detection."
        )
        return fallback

    logger.warning("No USB audio card found; using 'default' and hoping for the best")
    return 'default'


class BarkAudioBufferArecord:
    """
    Audio buffer that uses arecord subprocess instead of PyAudio
    Records 3-second chunks using arecord to avoid USB freezing
    """

    def __init__(self,
                 sample_rate: int = 44100,  # USB device native rate
                 chunk_duration: float = 3.0,  # Model expects 3 seconds
                 device: Optional[str] = None,  # Auto-detect if None
                 gain: float = 30.0):  # Amplification for quiet mic
        """
        Initialize audio buffer using arecord

        Args:
            sample_rate: Recording sample rate (44100 for USB device)
            chunk_duration: Duration of audio chunks in seconds (3.0 for model)
            device: ALSA device identifier (auto-detect if None)
            gain: Audio amplification factor
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.device = device if device else find_usb_audio_device()
        self.gain = gain

        # Threading
        self.is_recording = False
        self.record_thread = None
        self.audio_queue = queue.Queue(maxsize=5)

        # Statistics
        self.chunks_recorded = 0
        self.chunks_failed = 0
        self.silent_chunks = 0
        self._silent_warned = False

        logger.info(f"BarkAudioBufferArecord initialized: device={self.device}, {sample_rate}Hz, {chunk_duration}s chunks, gain={gain}x")

    def start(self):
        """Start recording audio"""
        if self.is_recording:
            logger.warning("Already recording")
            return

        self.is_recording = True
        self.record_thread = threading.Thread(
            target=self._record_loop,
            daemon=True,
            name="BarkRecorder"
        )
        self.record_thread.start()
        logger.info("Started audio recording with arecord")

    def stop(self):
        """Stop recording audio - drains queue before cleanup"""
        self.is_recording = False

        # Drain audio queue to release memory before killing arecord
        drained = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if drained > 0:
            logger.info(f"Drained {drained} audio chunks from queue")

        if self.record_thread:
            self.record_thread.join(timeout=5.0)

        logger.info(f"Stopped audio recording. Stats: {self.chunks_recorded} recorded, {self.chunks_failed} failed")

    def _record_loop(self):
        """Main recording loop using arecord subprocess"""
        logger.info("Recording loop started")

        while self.is_recording:
            try:
                # Record a chunk
                audio_data = self._record_chunk()

                if audio_data is not None:
                    # Watchdog: detect a silent mic / dead capture route.
                    # arecord can return success with all-zero samples when
                    # PipeWire's default source is routed to a Dummy sink.
                    if not np.any(audio_data):
                        self.silent_chunks += 1
                        if self.silent_chunks >= 10 and not self._silent_warned:
                            logger.warning(
                                f"Bark capture has produced {self.silent_chunks} all-zero chunks "
                                f"in a row from device={self.device}. Mic is dead or audio routing "
                                f"is broken (check 'wpctl status' / 'arecord -l'). "
                                f"Bark detection will not trigger until this is resolved."
                            )
                            self._silent_warned = True
                    else:
                        self.silent_chunks = 0
                        self._silent_warned = False

                    # Put in queue if not full
                    if not self.audio_queue.full():
                        self.audio_queue.put(audio_data)
                        self.chunks_recorded += 1

                        if self.chunks_recorded % 10 == 0:
                            logger.debug(f"Recorded {self.chunks_recorded} chunks")
                    else:
                        logger.debug("Audio queue full, dropping chunk")
                else:
                    self.chunks_failed += 1
                    if self.chunks_failed <= 3:  # Only log first few failures in detail
                        logger.warning(f"Recording failed (total failures: {self.chunks_failed}). Check device with: arecord -l")
                    else:
                        logger.debug(f"Recording failed (total failures: {self.chunks_failed})")
                    # Brief pause before retry
                    time.sleep(0.5)

            except Exception as e:
                logger.error(f"Error in recording loop: {e}")
                time.sleep(1.0)

        logger.info("Recording loop ended")

    def _record_chunk(self) -> Optional[np.ndarray]:
        """
        Record a single chunk using arecord subprocess

        Returns:
            Audio data as numpy array or None if failed
        """
        # Create temporary file
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        tmp_path = tmp.name
        tmp.close()

        # Build arecord command
        cmd = [
            'arecord',
            '-D', self.device,
            '-f', 'S16_LE',
            '-r', str(self.sample_rate),
            '-c', '1',  # Mono
            '-d', str(int(self.chunk_duration)),  # Must be integer
            tmp_path
        ]

        try:
            # Run arecord with timeout
            result = subprocess.run(
                cmd,
                timeout=self.chunk_duration + 0.5,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            if result.returncode == 0 and os.path.exists(tmp_path):
                # Read the WAV file
                with wave.open(tmp_path, 'rb') as w:
                    frames = w.readframes(w.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

                # Apply gain
                audio = audio * self.gain
                audio = np.clip(audio, -1.0, 1.0)

                os.unlink(tmp_path)
                return audio
            else:
                error_msg = result.stderr.decode().strip()
                if self.chunks_failed < 3:  # Log first few errors in detail
                    logger.warning(f"arecord failed with code {result.returncode}: {error_msg[:200]}")
                else:
                    logger.debug(f"arecord failed: {error_msg[:100]}")
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                return None

        except subprocess.TimeoutExpired:
            logger.warning(f"Recording timeout after {self.chunk_duration + 0.5}s")
            # Kill any stuck arecord process
            subprocess.run(['pkill', 'arecord'], capture_output=True)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None

        except Exception as e:
            logger.error(f"Recording error: {e}")
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            return None

    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get an audio chunk from the buffer

        Args:
            timeout: Maximum time to wait for audio

        Returns:
            Audio chunk as numpy array or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self):
        """Clear all pending audio chunks"""
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

    # Compatibility methods for drop-in replacement
    def start_recording(self):
        """Compatibility method"""
        return self.start()

    def stop_recording(self):
        """Compatibility method"""
        return self.stop()

    def __del__(self):
        """Cleanup on deletion"""
        if self.is_recording:
            self.stop()