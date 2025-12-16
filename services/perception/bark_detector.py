"""
Bark detection service for TreatBot
Integrates bark emotion classifier with event-driven architecture
"""

import threading
import time
import logging
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

# Core components
from core.bus import get_bus, AudioEvent, RewardEvent, publish_audio_event, publish_reward_event
from core.state import get_state

# Audio components
from ai.bark_classifier import BarkClassifier
# Use arecord-based buffer to avoid USB freezing
from audio.bark_buffer_arecord import BarkAudioBufferArecord as BarkAudioBuffer

logger = logging.getLogger(__name__)


class BarkDetectorService:
    """
    Service that continuously monitors for dog barks and classifies emotions

    Features:
    - Real-time audio capture and buffering
    - Bark emotion classification
    - Reward triggering for specific emotions
    - Cooldown management
    - Event publishing
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize bark detector service

        Args:
            config: Bark detection configuration from robot_config.yaml
        """
        self.config = config
        self.bus = get_bus()
        self.state = get_state()

        # Service state
        self.enabled = config.get('enabled', True)
        self.is_running = False

        # Consumer-friendly settings
        self.bark_threshold = config.get('bark_threshold_db', 70)  # Adjustable volume threshold
        self.quiet_reward_time = config.get('quiet_reward_seconds', 30)  # Reward quiet periods
        self.last_bark_time = None
        self.consumer_mode = config.get('consumer_friendly', True)
        self.detection_thread = None

        # Initialize components
        self.classifier = None
        self.audio_buffer = None

        # Detection parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.55)
        self.reward_emotions = config.get('reward_emotions', ['alert', 'attention'])
        self.check_interval = config.get('check_interval', 0.5)
        self.cooldown_period = config.get('cooldown_period', 5.0)

        # Cooldown tracking
        self.last_reward_time = None
        self.last_detection_time = None

        # Statistics
        self.stats = {
            'total_barks': 0,
            'rewarded_barks': 0,
            'emotions_detected': {},
            'session_start': datetime.now()
        }

        logger.info(f"BarkDetectorService initialized (enabled={self.enabled})")

    def initialize(self) -> bool:
        """
        Initialize bark detection components

        Returns:
            True if initialization successful
        """
        if not self.enabled:
            logger.info("Bark detection disabled in config")
            return False

        try:
            # Initialize classifier
            model_path = self.config.get('model_path', '/home/morgan/dogbot/ai/models/dog_bark_classifier.tflite')
            emotion_mapping_path = self.config.get('emotion_mapping_path', '/home/morgan/dogbot/ai/models/emotion_mapping.json')

            self.classifier = BarkClassifier(
                model_path=model_path,
                emotion_mapping_path=emotion_mapping_path,
                sample_rate=self.config.get('sample_rate', 22050),
                duration=self.config.get('duration', 3.0),
                n_mels=self.config.get('n_mels', 128)
            )

            # Initialize audio buffer with correct parameters
            # Note: USB device records at 44100Hz, model expects 22050Hz
            self.audio_buffer = BarkAudioBuffer(
                sample_rate=44100,  # USB device native rate
                chunk_duration=self.config.get('duration', 3.0),
                gain=self.config.get('audio_gain', 30.0)  # Amplification for quiet mic
            )

            logger.info("Bark detection components initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize bark detection: {e}")
            self.enabled = False
            return False

    def start(self):
        """Start bark detection service"""
        if not self.enabled:
            logger.warning("Cannot start - bark detection disabled")
            return

        if self.is_running:
            logger.warning("Bark detection already running")
            return

        try:
            # Start audio capture
            self.audio_buffer.start()

            # Start detection thread
            self.is_running = True
            self.detection_thread = threading.Thread(
                target=self._detection_loop,
                daemon=True,
                name="BarkDetector"
            )
            self.detection_thread.start()

            # Publish service started event
            publish_audio_event('service_started', {
                'service': 'bark_detector',
                'timestamp': datetime.now().isoformat()
            })

            logger.info("Bark detection service started")

        except Exception as e:
            logger.error(f"Failed to start bark detection: {e}")
            self.is_running = False

    def stop(self):
        """Stop bark detection service"""
        if not self.is_running:
            return

        logger.info("Stopping bark detection service...")
        self.is_running = False

        # Stop audio capture
        if self.audio_buffer:
            self.audio_buffer.stop()

        # Wait for thread to finish
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)

        # Publish service stopped event
        publish_audio_event('service_stopped', {
            'service': 'bark_detector',
            'timestamp': datetime.now().isoformat()
        })

        logger.info("Bark detection service stopped")

    def _detection_loop(self):
        """Main detection loop running in background thread"""
        logger.info("Bark detection loop started")
        detection_count = 0

        while self.is_running:
            try:
                # Get audio chunk from buffer
                audio_chunk = self.audio_buffer.get_audio_chunk(timeout=self.check_interval)

                if audio_chunk is not None:
                    detection_count += 1
                    # Log every 10th chunk to avoid spam
                    if detection_count % 10 == 0:
                        logger.debug(f"Processing audio chunk #{detection_count}, shape: {audio_chunk.shape}")

                    # Check audio energy level (RMS is better than mean absolute)
                    audio_energy = np.sqrt(np.mean(audio_chunk**2))

                    # Log energy every 10th chunk to debug
                    if detection_count % 10 == 0:
                        logger.info(f"Audio energy check - RMS: {audio_energy:.6f}, Max: {np.max(np.abs(audio_chunk)):.6f}")

                    # Very low threshold since mic is quiet and we're amplifying by 30x
                    if audio_energy > 0.001:  # Process almost any audio
                        logger.info(f"Processing audio - Energy: {audio_energy:.4f}")

                        # Resample from 44100Hz to 22050Hz for the model
                        from scipy import signal
                        audio_resampled = signal.resample(
                            audio_chunk,
                            int(len(audio_chunk) * 22050 / 44100)
                        )

                        # Classify bark emotion
                        result = self.classifier.predict(
                            audio_resampled,
                            confidence_threshold=self.confidence_threshold
                        )

                        logger.info(f"Classification result: {result['emotion']} (conf: {result['confidence']:.2f}), all probs: {result['all_probabilities']}")

                        # Check if 'notbark' has high confidence - if so, this is NOT a bark
                        notbark_confidence = result['all_probabilities'].get('notbark', 0.0)

                        # Only consider it a bark if:
                        # 1. The top prediction is NOT 'notbark'
                        # 2. The confidence is above threshold
                        # 3. The 'notbark' confidence is below 0.5 (model thinks it's NOT "not a bark")
                        if (result['emotion'] != 'notbark' and
                            result['is_confident'] and
                            notbark_confidence < 0.5):
                            self._handle_bark_detected(result)
                        else:
                            logger.debug(f"Not a bark - emotion: {result['emotion']}, notbark conf: {notbark_confidence:.2f}")

                # Brief pause
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(1.0)

        logger.info("Bark detection loop ended")

    def _handle_bark_detected(self, result: Dict):
        """
        Handle detected bark

        Args:
            result: Classification result from bark classifier
        """
        emotion = result['emotion']
        confidence = result['confidence']

        # Update statistics
        self.stats['total_barks'] += 1
        self.stats['emotions_detected'][emotion] = \
            self.stats['emotions_detected'].get(emotion, 0) + 1

        # Log detection
        logger.info(f"Bark detected: {emotion} (confidence: {confidence:.2f})")

        # Publish bark detected event
        publish_audio_event('bark_detected', {
            'emotion': emotion,
            'confidence': confidence,
            'all_probabilities': result['all_probabilities'],
            'timestamp': datetime.now().isoformat()
        })

        # Check if this emotion triggers a reward
        if emotion in self.reward_emotions:
            self._check_and_trigger_reward(emotion, confidence)

        self.last_detection_time = datetime.now()

    def _check_and_trigger_reward(self, emotion: str, confidence: float):
        """
        Check cooldown and trigger reward if appropriate

        Args:
            emotion: Detected emotion
            confidence: Detection confidence
        """
        current_time = datetime.now()

        # Check cooldown
        if self.last_reward_time:
            time_since_reward = (current_time - self.last_reward_time).total_seconds()
            if time_since_reward < self.cooldown_period:
                remaining = self.cooldown_period - time_since_reward
                logger.debug(f"Cooldown active: {remaining:.1f}s remaining")
                return

        # Trigger reward
        logger.info(f"Triggering reward for {emotion} bark")

        # Update statistics
        self.stats['rewarded_barks'] += 1
        self.last_reward_time = current_time

        # Publish reward event
        publish_reward_event('trigger', {
            'source': 'bark_detection',
            'reason': f'{emotion}_bark',
            'confidence': confidence,
            'timestamp': current_time.isoformat()
        })

        # Also publish specific bark reward event
        publish_audio_event('bark_rewarded', {
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': current_time.isoformat()
        })

    def get_status(self) -> Dict:
        """
        Get service status and statistics

        Returns:
            Dictionary with service status
        """
        status = {
            'enabled': self.enabled,
            'running': self.is_running,
            'confidence_threshold': self.confidence_threshold,
            'reward_emotions': self.reward_emotions,
            'cooldown_period': self.cooldown_period,
            'statistics': self.stats.copy()
        }

        if self.last_detection_time:
            status['last_detection'] = self.last_detection_time.isoformat()

        if self.last_reward_time:
            status['last_reward'] = self.last_reward_time.isoformat()
            time_since = (datetime.now() - self.last_reward_time).total_seconds()
            status['cooldown_remaining'] = max(0, self.cooldown_period - time_since)

        return status

    def set_enabled(self, enabled: bool):
        """
        Enable or disable bark detection

        Args:
            enabled: True to enable, False to disable
        """
        if enabled and not self.enabled:
            self.enabled = True
            if self.initialize():
                self.start()
        elif not enabled and self.enabled:
            self.enabled = False
            self.stop()

    def set_confidence_threshold(self, threshold: float):
        """
        Update confidence threshold

        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to {self.confidence_threshold}")

    def set_reward_emotions(self, emotions: list):
        """
        Update list of emotions that trigger rewards

        Args:
            emotions: List of emotion names
        """
        self.reward_emotions = emotions
        logger.info(f"Reward emotions updated: {emotions}")

    def reset_statistics(self):
        """Reset detection statistics"""
        self.stats = {
            'total_barks': 0,
            'rewarded_barks': 0,
            'emotions_detected': {},
            'session_start': datetime.now()
        }
        logger.info("Statistics reset")


# Singleton instance
_bark_detector_instance = None


def get_bark_detector_service(config: Optional[Dict] = None) -> BarkDetectorService:
    """
    Get or create bark detector service instance

    Args:
        config: Optional configuration override

    Returns:
        BarkDetectorService singleton instance
    """
    global _bark_detector_instance

    if _bark_detector_instance is None:
        # Load config if not provided
        if config is None:
            import yaml
            with open('/home/morgan/dogbot/config/robot_config.yaml', 'r') as f:
                full_config = yaml.safe_load(f)
                config = full_config.get('bark_detection', {})

        _bark_detector_instance = BarkDetectorService(config)
        _bark_detector_instance.initialize()

    return _bark_detector_instance