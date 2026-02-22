#!/usr/bin/env python3
"""
WebRTC Audio Track for WIM-Z
Captures from USB microphone and streams via WebRTC with mode-aware muting.

API Contract v1.3: Always-on robot microphone as persistent WebRTC audio track.
- idle/manual modes: Mic feeds audio frames normally
- silent_guardian/coach/mission modes: Audio track muted (silence)
- PTT playback: Temporarily muted to prevent echo
"""

import asyncio
import logging
import threading
import time
import queue
from typing import Optional

import numpy as np
import sounddevice as sd
from aiortc import MediaStreamTrack
from av import AudioFrame

from core.state import get_state, SystemMode


# Audio configuration matching WebRTC Opus codec expectations
SAMPLE_RATE = 48000  # Opus default
CHANNELS = 1  # Mono
SAMPLES_PER_FRAME = 960  # 20ms at 48kHz (standard WebRTC audio frame size)
FRAME_DURATION_MS = 20

# Modes where mic should be muted (AI modes need quiet for processing)
MUTED_MODES = {SystemMode.SILENT_GUARDIAN, SystemMode.COACH, SystemMode.MISSION}


class WIMZAudioTrack(MediaStreamTrack):
    """
    WebRTC audio track that captures from USB microphone.

    Features:
    - Always present in SDP offer (no renegotiation on mute)
    - Mode-aware muting (AI modes get silence)
    - PTT echo suppression (mutes during speaker playback)
    - Thread-safe operation
    """

    kind = "audio"

    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger('WIMZAudioTrack')
        self.state = get_state()

        # Audio capture state
        self._running = True
        self._muted = False
        self._ptt_muted = False  # Temporary mute during PTT playback
        self._ptt_mute_until = 0  # Timestamp when PTT mute expires

        # Audio buffer (thread-safe queue)
        self._audio_queue: queue.Queue = queue.Queue(maxsize=50)

        # Frame timing
        self._frame_count = 0
        self._start_time = time.time()
        self._last_frame_pts = 0

        # USB mic device detection
        self._device_index = self._find_usb_audio_device()

        # Audio capture stream
        self._stream: Optional[sd.InputStream] = None
        self._capture_thread: Optional[threading.Thread] = None

        # Start audio capture
        self._start_capture()

        # Subscribe to mode changes for auto-mute
        self.state.subscribe('mode_change', self._on_mode_change)

        # Set initial mute state based on current mode
        self._update_mute_for_mode(self.state.get_mode())

        self.logger.info(f"WIMZAudioTrack initialized (device={self._device_index}, muted={self._muted})")

    def _find_usb_audio_device(self) -> Optional[int]:
        """Find USB Audio Device index for input"""
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if 'USB' in dev.get('name', '') and dev.get('max_input_channels', 0) > 0:
                    self.logger.info(f"Found USB audio input: {dev['name']} (device {i})")
                    return i
        except Exception as e:
            self.logger.warning(f"Could not query audio devices: {e}")
        return None

    def _start_capture(self):
        """Start audio capture from USB microphone"""
        if self._device_index is None:
            self.logger.error("No USB audio device found - audio track will send silence")
            return

        try:
            # Create input stream
            self._stream = sd.InputStream(
                device=self._device_index,
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='int16',
                blocksize=SAMPLES_PER_FRAME,
                callback=self._audio_callback
            )
            self._stream.start()
            self.logger.info(f"Audio capture started: {SAMPLE_RATE}Hz, {CHANNELS}ch, {SAMPLES_PER_FRAME} samples/frame")
        except Exception as e:
            self.logger.error(f"Failed to start audio capture: {e}")
            self._stream = None

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback from sounddevice when audio frames are available"""
        if status:
            self.logger.debug(f"Audio callback status: {status}")

        if not self._running:
            return

        try:
            # Copy audio data (indata is temporary buffer)
            audio_data = indata.copy()

            # Try to enqueue, drop if full (prevents memory buildup)
            try:
                self._audio_queue.put_nowait(audio_data)
            except queue.Full:
                pass  # Drop oldest frames if queue is full

        except Exception as e:
            self.logger.error(f"Audio callback error: {e}")

    def _on_mode_change(self, data: dict):
        """Handle mode change events for auto-muting"""
        new_mode_str = data.get('new_mode', '')
        try:
            new_mode = SystemMode(new_mode_str)
            self._update_mute_for_mode(new_mode)
        except ValueError:
            pass

    def _update_mute_for_mode(self, mode: SystemMode):
        """Update mute state based on current mode"""
        should_mute = mode in MUTED_MODES
        if should_mute != self._muted:
            self._muted = should_mute
            self.logger.info(f"Audio track {'muted' if should_mute else 'unmuted'} for mode {mode.value}")

    def mute_for_ptt(self, duration_seconds: float):
        """
        Temporarily mute audio track during PTT playback to prevent echo.

        Args:
            duration_seconds: How long to mute (typically PTT clip duration + buffer)
        """
        self._ptt_muted = True
        self._ptt_mute_until = time.time() + duration_seconds
        self.logger.info(f"Audio track muted for PTT playback ({duration_seconds:.1f}s)")

    def _check_ptt_mute_expired(self):
        """Check if PTT mute has expired"""
        if self._ptt_muted and time.time() >= self._ptt_mute_until:
            self._ptt_muted = False
            self.logger.info("Audio track PTT mute expired, resuming")

    def is_muted(self) -> bool:
        """Check if audio track is currently muted (mode or PTT)"""
        self._check_ptt_mute_expired()
        return self._muted or self._ptt_muted

    def set_muted(self, muted: bool):
        """Manually set mute state (overrides mode-based muting)"""
        self._muted = muted
        self.logger.info(f"Audio track manually {'muted' if muted else 'unmuted'}")

    def pause_capture(self):
        """
        Pause audio capture to release USB mic for bark detection.
        Track continues sending silence via recv() (self._stream is None path).
        """
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                self.logger.warning(f"Error stopping audio stream for pause: {e}")
            self._stream = None
            # Drain the queue so stale audio isn't sent on resume
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break
            self.logger.info("Audio capture paused (mic released for bark detection)")
        else:
            self.logger.debug("Audio capture already paused")

    def resume_capture(self):
        """
        Resume audio capture after bark detection releases the mic.
        Re-opens the USB mic via sounddevice.
        """
        if self._stream is None and self._running:
            self._start_capture()
            if self._stream is not None:
                self.logger.info("Audio capture resumed (mic reclaimed from bark detection)")
            else:
                self.logger.warning("Audio capture resume failed - stream could not start")
        else:
            self.logger.debug("Audio capture already running")

    async def recv(self) -> AudioFrame:
        """
        Receive next audio frame for WebRTC transmission.

        Returns silence if muted (mode-based or PTT), otherwise returns
        captured audio from USB mic.
        """
        # Calculate PTS for this frame
        pts = self._frame_count * SAMPLES_PER_FRAME
        self._frame_count += 1

        # Check PTT mute expiration
        self._check_ptt_mute_expired()

        # Determine if we should send silence
        if self.is_muted() or self._stream is None:
            # Send silence
            audio_data = np.zeros((SAMPLES_PER_FRAME, CHANNELS), dtype='int16')
        else:
            # Get audio from queue
            try:
                audio_data = self._audio_queue.get_nowait()
            except queue.Empty:
                # No audio available, send silence
                audio_data = np.zeros((SAMPLES_PER_FRAME, CHANNELS), dtype='int16')

        # Create AudioFrame
        frame = AudioFrame.from_ndarray(audio_data.T, format='s16', layout='mono')
        frame.pts = pts
        frame.sample_rate = SAMPLE_RATE
        frame.time_base = f'1/{SAMPLE_RATE}'

        # Yield to event loop periodically to maintain timing
        await asyncio.sleep(FRAME_DURATION_MS / 1000.0)

        return frame

    def stop(self):
        """Stop audio capture and cleanup"""
        self.logger.info("Stopping WIMZAudioTrack")
        self._running = False

        # Stop audio stream
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:
                self.logger.warning(f"Error stopping audio stream: {e}")
            self._stream = None

        # Unsubscribe from mode changes
        try:
            self.state.unsubscribe('mode_change', self._on_mode_change)
        except Exception:
            pass

        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        self.logger.info("WIMZAudioTrack stopped")

    def get_stats(self) -> dict:
        """Get audio track statistics"""
        return {
            'frames_sent': self._frame_count,
            'muted': self._muted,
            'ptt_muted': self._ptt_muted,
            'device_index': self._device_index,
            'stream_active': self._stream is not None and self._stream.active,
            'queue_size': self._audio_queue.qsize(),
            'uptime': time.time() - self._start_time
        }


# Singleton instance for global access (used for PTT echo suppression)
_audio_track: Optional[WIMZAudioTrack] = None
_audio_track_lock = threading.Lock()


def get_audio_track() -> Optional[WIMZAudioTrack]:
    """Get the global audio track instance (if created)"""
    global _audio_track
    with _audio_track_lock:
        return _audio_track


def set_audio_track(track: Optional[WIMZAudioTrack]):
    """Set the global audio track instance"""
    global _audio_track
    with _audio_track_lock:
        _audio_track = track


def mute_audio_for_ptt(duration_seconds: float):
    """
    Helper function to mute audio track during PTT playback.
    Called by push_to_talk service when playing app audio.
    """
    track = get_audio_track()
    if track:
        track.mute_for_ptt(duration_seconds)
