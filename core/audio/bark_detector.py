"""
BarkDetector - Unified bark detection system for WIM-Z

Combines three stages:
- Stage 1: BarkGate (signal processing - no ML)
- Stage 2: Emotion classification (optional TFLite)
- Stage 3: BarkAnalytics (per-dog tracking)

This is the main interface used by all modes/missions.
"""

import time
import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from .bark_gate import BarkGate, BarkGateConfig
from .bark_analytics import BarkAnalytics

logger = logging.getLogger(__name__)


@dataclass
class BarkEvent:
    """Represents a detected bark event"""
    timestamp: float
    dog_id: Optional[str]
    dog_name: Optional[str]
    distance: str          # 'close', 'mid', 'far'
    peak_energy: float
    duration_ms: int
    emotion: Optional[str] = None
    emotion_confidence: float = 0.0


class BarkDetector:
    """
    Unified bark detection system.

    Combines signal processing (Stage 1), optional emotion classification
    (Stage 2), and per-dog analytics (Stage 3) into a single interface.

    Usage:
        detector = BarkDetector()
        detector.start()

        # In audio processing loop:
        event = detector.process_audio(audio_chunk, energy, dog_id="elsa")
        if event:
            print(f"Bark: {event.distance}, emotion={event.emotion}")

        # Get reports:
        summary = detector.get_dog_summary("elsa")
    """

    def __init__(self, config: Optional[BarkGateConfig] = None,
                 enable_emotion: bool = True):
        """
        Initialize bark detector.

        Args:
            config: Optional BarkGate configuration
            enable_emotion: Whether to run emotion classification (Stage 2)
        """
        # Stage 1: Signal processing
        self.gate = BarkGate(config)

        # Stage 2: Emotion classification (lazy loaded)
        self.enable_emotion = enable_emotion
        self._classifier = None
        self._classifier_loaded = False

        # Stage 3: Analytics
        self.analytics = BarkAnalytics()

        # Audio buffer for emotion classification
        self._audio_buffer: List[np.ndarray] = []
        self._buffer_duration_ms = 3000  # Keep 3 seconds of audio
        self._sample_rate = 22050  # Expected sample rate for classifier

        # State
        self.is_running = False

        logger.info(f"BarkDetector initialized (emotion={enable_emotion})")

    def start(self):
        """Start the detector"""
        self.is_running = True
        logger.info("BarkDetector started")

    def stop(self):
        """Stop the detector"""
        self.is_running = False
        logger.info("BarkDetector stopped")

    def _load_classifier(self):
        """Lazy load the emotion classifier"""
        if self._classifier_loaded:
            return

        try:
            from ai.bark_classifier import BarkClassifier
            self._classifier = BarkClassifier(
                model_path='/home/morgan/dogbot/ai/models/dog_bark_classifier.tflite',
                emotion_mapping_path='/home/morgan/dogbot/ai/models/emotion_mapping.json',
                sample_rate=22050,
                duration=3.0,
                n_mels=128
            )
            logger.info("Emotion classifier loaded")
        except Exception as e:
            logger.warning(f"Failed to load emotion classifier: {e}")
            self._classifier = None

        self._classifier_loaded = True

    def process_audio(self, audio_data: Optional[np.ndarray], energy: float,
                     dog_id: str = None, dog_name: str = None,
                     timestamp_ms: int = None) -> Optional[BarkEvent]:
        """
        Process audio and detect barks.

        Args:
            audio_data: Raw audio samples (for emotion classification)
            energy: Pre-computed audio energy (0.0-1.0+, typically RMS)
            dog_id: Dog identifier (for per-dog tracking)
            dog_name: Human-readable dog name
            timestamp_ms: Current timestamp in ms (auto-generated if not provided)

        Returns:
            BarkEvent if a bark was detected, None otherwise
        """
        if not self.is_running:
            return None

        if timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)

        # Buffer audio for emotion classification
        if audio_data is not None:
            self._audio_buffer.append(audio_data)
            # Keep only recent audio
            while len(self._audio_buffer) > 10:
                self._audio_buffer.pop(0)

        # Stage 1: Is this a bark?
        gate_result = self.gate.process_audio_chunk(energy, timestamp_ms)

        if not gate_result['is_bark']:
            return None

        # Bark detected! Build event
        event = BarkEvent(
            timestamp=time.time(),
            dog_id=dog_id,
            dog_name=dog_name,
            distance=gate_result['distance'],
            peak_energy=gate_result['peak'],
            duration_ms=gate_result['duration_ms']
        )

        # Stage 2: Emotion classification (optional)
        if self.enable_emotion and self._audio_buffer:
            emotion_result = self._classify_emotion()
            if emotion_result:
                event.emotion = emotion_result['emotion']
                event.emotion_confidence = emotion_result['confidence']

        # Stage 3: Record analytics
        if dog_id:
            self.analytics.record_bark(
                dog_id=dog_id,
                duration_ms=event.duration_ms,
                distance=event.distance,
                peak_energy=event.peak_energy,
                emotion=event.emotion,
                dog_name=dog_name or ""
            )

        logger.info(f"BarkEvent: {event.distance}, peak={event.peak_energy:.3f}, "
                   f"duration={event.duration_ms}ms, emotion={event.emotion}, "
                   f"dog={event.dog_name or event.dog_id or 'unknown'}")

        return event

    def _classify_emotion(self) -> Optional[Dict[str, Any]]:
        """
        Run emotion classification on buffered audio.

        Returns:
            Dict with 'emotion' and 'confidence', or None
        """
        if not self.enable_emotion:
            return None

        # Lazy load classifier
        self._load_classifier()

        if self._classifier is None:
            return None

        try:
            # Concatenate buffered audio
            if not self._audio_buffer:
                return None

            audio = np.concatenate(self._audio_buffer)

            # Ensure correct length for model (3 seconds at 22050 Hz)
            expected_len = int(3.0 * self._sample_rate)
            if len(audio) > expected_len:
                audio = audio[-expected_len:]  # Take most recent
            elif len(audio) < expected_len:
                # Pad with zeros
                audio = np.pad(audio, (0, expected_len - len(audio)))

            # Run classifier
            result = self._classifier.predict(audio, confidence_threshold=0.3)

            # Filter out 'notbark' - we already know it's a bark from Stage 1
            probs = result['all_probabilities']
            bark_emotions = {k: v for k, v in probs.items() if k != 'notbark'}

            if not bark_emotions:
                return None

            top_emotion = max(bark_emotions, key=bark_emotions.get)
            return {
                'emotion': top_emotion,
                'confidence': bark_emotions[top_emotion],
                'all_emotions': bark_emotions
            }

        except Exception as e:
            logger.error(f"Emotion classification error: {e}")
            return None

    def process_energy_only(self, energy: float, dog_id: str = None,
                           dog_name: str = None) -> Optional[BarkEvent]:
        """
        Process energy without audio data (no emotion classification).

        Useful for simple bark detection without the overhead of
        buffering audio for the classifier.

        Args:
            energy: Audio energy level
            dog_id: Dog identifier
            dog_name: Human-readable name

        Returns:
            BarkEvent if bark detected, None otherwise
        """
        return self.process_audio(
            audio_data=None,
            energy=energy,
            dog_id=dog_id,
            dog_name=dog_name
        )

    # Analytics methods (delegate to Stage 3)

    def get_dog_summary(self, dog_id: str) -> Optional[Dict[str, Any]]:
        """Get bark summary for specific dog"""
        return self.analytics.get_dog_summary(dog_id)

    def get_all_summaries(self) -> Dict[str, Dict[str, Any]]:
        """Get bark summaries for all dogs"""
        return self.analytics.get_all_summaries()

    def get_global_summary(self) -> Dict[str, Any]:
        """Get aggregated summary across all dogs"""
        return self.analytics.get_global_summary()

    def get_gate_stats(self) -> Dict[str, Any]:
        """Get Stage 1 detection statistics"""
        return self.gate.get_stats()

    # Configuration

    def update_thresholds(self, base: float = None, close: float = None,
                         mid: float = None, far: float = None):
        """Update bark gate thresholds"""
        self.gate.update_thresholds(base, close, mid, far)

    def reset_analytics(self, dog_id: str = None):
        """Reset analytics for one or all dogs"""
        if dog_id:
            self.analytics.reset_dog(dog_id)
        else:
            self.analytics.reset_all()


# Singleton instance
_bark_detector_instance: Optional[BarkDetector] = None


def get_bark_detector(config: Optional[BarkGateConfig] = None,
                     enable_emotion: bool = True) -> BarkDetector:
    """
    Get or create the singleton BarkDetector instance.

    Args:
        config: Optional configuration (only used on first call)
        enable_emotion: Whether to enable emotion classification

    Returns:
        BarkDetector singleton instance
    """
    global _bark_detector_instance

    if _bark_detector_instance is None:
        _bark_detector_instance = BarkDetector(config, enable_emotion)

    return _bark_detector_instance
