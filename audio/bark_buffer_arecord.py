"""
Audio buffering for bark detection using arecord subprocess
Replaces PyAudio with arecord to avoid USB freezing issues
"""

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


def find_usb_audio_device():
    """
    Find the USB audio device card number
    Returns device string like 'hw:0,0' or 'default' if not found
    """
    try:
        result = subprocess.run(['arecord', '-l'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'USB Audio Device' in line or 'USB Audio' in line:
                # Extract card number from line like "card 0: Device [USB Audio Device]"
                if 'card ' in line:
                    card_num = line.split('card ')[1].split(':')[0]
                    device = f'hw:{card_num},0'
                    logger.info(f"Found USB audio device at {device}")
                    return device
    except Exception as e:
        logger.warning(f"Error finding USB audio device: {e}")

    logger.info("Using default audio device")
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
        """Stop recording audio"""
        self.is_recording = False

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