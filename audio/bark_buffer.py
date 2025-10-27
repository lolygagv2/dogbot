"""
Audio buffering for real-time bark detection
Captures and stores audio chunks for classification
"""

import numpy as np
import pyaudio
import threading
import queue
import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

class BarkAudioBuffer:
    """
    Circular audio buffer that captures bark events
    """
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 chunk_duration: float = 3.0,
                 format: int = pyaudio.paInt16,
                 channels: int = 1):
        """
        Initialize audio buffer
        
        Args:
            sample_rate: Audio sample rate (Hz)
            chunk_duration: Duration of audio chunks (seconds)
            format: PyAudio format
            channels: Number of audio channels
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.format = format
        self.channels = channels
        
        # Calculate chunk size
        self.chunk_size = int(sample_rate * chunk_duration)
        self.frame_size = 1024  # Frames per buffer
        
        # Audio stream
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        # Buffer for storing audio
        self.audio_queue = queue.Queue(maxsize=10)
        self.record_thread = None
        
    def start(self):
        """Start recording audio"""
        if self.is_recording:
            logger.warning("Already recording")
            return
        
        try:
            # Open audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.frame_size
            )
            
            self.is_recording = True
            
            # Start recording thread
            self.record_thread = threading.Thread(target=self._record_loop, daemon=True)
            self.record_thread.start()
            
            logger.info(f"Started audio recording at {self.sample_rate}Hz")
            
        except Exception as e:
            logger.error(f"Failed to start audio recording: {e}")
            raise
    
    def stop(self):
        """Stop recording audio"""
        self.is_recording = False
        
        if self.record_thread:
            self.record_thread.join(timeout=2.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        logger.info("Stopped audio recording")
    
    def _record_loop(self):
        """Background thread that continuously records audio"""
        audio_buffer = deque(maxlen=self.chunk_size)
        
        while self.is_recording:
            try:
                # Read audio data
                data = self.stream.read(self.frame_size, exception_on_overflow=False)
                
                # Convert to numpy array
                audio_chunk = np.frombuffer(data, dtype=np.int16)
                
                # Add to buffer
                audio_buffer.extend(audio_chunk)
                
                # If buffer is full, add to queue for processing
                if len(audio_buffer) >= self.chunk_size:
                    audio_array = np.array(list(audio_buffer), dtype=np.float32)
                    # Normalize to [-1, 1]
                    audio_array = audio_array / 32768.0
                    
                    # Add to queue (non-blocking)
                    try:
                        self.audio_queue.put_nowait(audio_array)
                    except queue.Full:
                        # Drop oldest if queue is full
                        try:
                            self.audio_queue.get_nowait()
                            self.audio_queue.put_nowait(audio_array)
                        except:
                            pass
                            
            except Exception as e:
                logger.error(f"Error in record loop: {e}")
    
    def get_audio_chunk(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next audio chunk from buffer
        
        Args:
            timeout: Maximum time to wait for chunk
            
        Returns:
            Audio data as numpy array, or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.stop()
        self.audio.terminate()