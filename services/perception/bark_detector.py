"""
Bark detection service for TreatBot
Uses 3-stage bark detection: BarkGate (signal processing) -> Emotion -> Analytics

Updated to use new core/audio/bark_detector.py system which:
- Stage 1: BarkGate - Signal processing detection (works with AGC OFF)
- Stage 2: Emotion classification (spectral analysis, works with AGC OFF)
- Stage 3: Per-dog analytics
"""

import threading
import time
import logging
import numpy as np
from typing import Dict, Optional, Any
from datetime import datetime, timedelta

# Core components
from core.bus import get_bus, AudioEvent, RewardEvent, VisionEvent, publish_audio_event, publish_reward_event
from core.state import get_state

# New 3-stage bark detection system
from core.audio.bark_detector import BarkDetector, BarkEvent
from core.audio.bark_gate import BarkGateConfig

# Audio buffer for capturing audio
from audio.bark_buffer_arecord import BarkAudioBufferArecord as BarkAudioBuffer

# AGC control
from services.media.usb_audio import set_agc

logger = logging.getLogger(__name__)


class BarkDetectorService:
    """
    Service that continuously monitors for dog barks and classifies emotions

    Uses 3-stage detection:
    - Stage 1: BarkGate (signal processing, no ML) - determines IF it's a bark
    - Stage 2: Emotion classifier (TFLite) - determines WHAT KIND of bark
    - Stage 3: Analytics - per-dog tracking

    Features:
    - Real-time audio capture and buffering
    - Reliable bark detection via energy thresholds (AGC OFF)
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

        # Initialize components - using new 3-stage system
        self.bark_detector = None  # BarkDetector (3-stage)
        self.audio_buffer = None

        # Detection parameters
        self.confidence_threshold = config.get('confidence_threshold', 0.70)
        self.reward_emotions = config.get('reward_emotions', ['alert', 'attention'])
        self.check_interval = config.get('check_interval', 0.5)
        self.cooldown_period = config.get('cooldown_period', 5.0)

        # Cooldown tracking
        self.last_reward_time = None
        self.last_detection_time = None

        # Vision-audio fusion: track visible dogs for bark attribution
        self._visible_dogs = {}  # dog_id -> {time, name}
        self._visible_dogs_lock = threading.Lock()
        self._dog_visibility_timeout = 1.0  # 1 second visibility window

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
            # Initialize 3-stage bark detector with emotion classification enabled
            gate_config = BarkGateConfig(
                base_threshold=0.25,      # Minimum energy (calibrated for AGC OFF)
                thresh_close=0.50,        # Close/loud bark
                thresh_mid=0.35,          # Medium distance bark
                thresh_far=0.25,          # Far/quiet bark
                min_bark_duration_ms=30,  # Reject clicks
                max_bark_duration_ms=2000,# Cap duration
                grace_period_ms=100,      # Wait for bark "tail"
                bark_cooldown_ms=1000     # Cooldown between barks
            )

            self.bark_detector = BarkDetector(
                config=gate_config,
                enable_emotion=True  # Enable Stage 2 emotion classification
            )

            # Initialize audio buffer
            # Note: USB device records at 44100Hz, need 1.0s minimum for arecord
            self.audio_buffer = BarkAudioBuffer(
                sample_rate=44100,  # USB device native rate
                chunk_duration=1.0,  # 1 second chunks (arecord minimum)
                gain=self.config.get('audio_gain', 30.0)  # Amplification for quiet mic
            )

            logger.info("Bark detection components initialized (3-stage system)")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize bark detection: {e}")
            import traceback
            traceback.print_exc()
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
            # Disable AGC for reliable bark detection (raw energy levels)
            set_agc(False)
            logger.info("AGC disabled for bark detection")

            # Start the 3-stage bark detector
            self.bark_detector.start()

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

            # Subscribe to vision events for dog visibility tracking
            self.bus.subscribe(VisionEvent, self._on_vision_event)

            # Publish service started event
            publish_audio_event('service_started', {
                'service': 'bark_detector',
                'timestamp': datetime.now().isoformat()
            })

            logger.info("Bark detection service started (3-stage with AGC OFF)")

        except Exception as e:
            logger.error(f"Failed to start bark detection: {e}")
            self.is_running = False

    def stop(self):
        """Stop bark detection service"""
        if not self.is_running:
            return

        logger.info("Stopping bark detection service...")
        self.is_running = False

        # Stop the 3-stage bark detector
        if self.bark_detector:
            self.bark_detector.stop()

        # Stop audio capture
        if self.audio_buffer:
            self.audio_buffer.stop()

        # Wait for thread to finish
        if self.detection_thread:
            self.detection_thread.join(timeout=2.0)

        # Re-enable AGC when stopping bark detection
        set_agc(True)
        logger.info("AGC re-enabled")

        # Publish service stopped event
        publish_audio_event('service_stopped', {
            'service': 'bark_detector',
            'timestamp': datetime.now().isoformat()
        })

        logger.info("Bark detection service stopped")

    def _on_vision_event(self, event):
        """Track visible dogs from vision events for bark attribution"""
        if event.subtype == 'dog_detected':
            dog_id = event.data.get('dog_id')  # Format: "aruco_315" or "aruco_832"
            dog_name = event.data.get('dog_name')  # "Elsa" or "Bezik"
            if dog_id:
                with self._visible_dogs_lock:
                    self._visible_dogs[dog_id] = {
                        'time': time.time(),
                        'name': dog_name
                    }

    def _get_visible_dogs(self):
        """Get dogs visible within the timeout window"""
        now = time.time()
        with self._visible_dogs_lock:
            # Filter to dogs seen recently
            visible = {
                dog_id: info for dog_id, info in self._visible_dogs.items()
                if now - info['time'] < self._dog_visibility_timeout
            }
            return visible

    def _detection_loop(self):
        """Main detection loop running in background thread"""
        logger.info("Bark detection loop started (3-stage system)")
        detection_count = 0

        while self.is_running:
            try:
                # Get audio chunk from buffer
                audio_chunk = self.audio_buffer.get_audio_chunk(timeout=self.check_interval)

                if audio_chunk is not None:
                    detection_count += 1

                    # Calculate audio energy (RMS)
                    audio_energy = np.sqrt(np.mean(audio_chunk**2))

                    # Log energy periodically for debugging
                    if detection_count % 30 == 0:
                        logger.debug(f"Audio energy: {audio_energy:.4f}")

                    # Get visible dogs for bark attribution
                    visible_dogs = self._get_visible_dogs()
                    dog_id = None
                    dog_name = None

                    # Attribution: only attribute if exactly 1 dog visible
                    if len(visible_dogs) == 1:
                        dog_id, dog_info = next(iter(visible_dogs.items()))
                        dog_name = dog_info.get('name', 'unknown')

                    # Resample audio for emotion classifier (44100Hz -> 22050Hz)
                    from scipy import signal
                    audio_resampled = signal.resample(
                        audio_chunk,
                        int(len(audio_chunk) * 22050 / 44100)
                    )

                    # Process through 3-stage bark detector
                    # Stage 1: BarkGate determines IF it's a bark (energy thresholds)
                    # Stage 2: If bark, classify emotion
                    # Stage 3: Track per-dog analytics
                    bark_event = self.bark_detector.process_audio(
                        audio_data=audio_resampled,
                        energy=audio_energy,
                        dog_id=dog_id,
                        dog_name=dog_name
                    )

                    if bark_event:
                        # Bark detected! Build result dict for handler
                        loudness_db = 20 * np.log10(max(audio_energy, 1e-10))

                        result = {
                            'emotion': bark_event.emotion or 'unknown',
                            'confidence': bark_event.emotion_confidence,
                            'distance': bark_event.distance,
                            'peak_energy': bark_event.peak_energy,
                            'duration_ms': bark_event.duration_ms,
                            'all_probabilities': {}  # Not available from BarkEvent
                        }

                        self._handle_bark_detected(result, loudness_db, bark_event)

                # Brief pause
                time.sleep(0.05)

            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

        logger.info("Bark detection loop ended")

    def _handle_bark_detected(self, result: Dict, loudness_db: float = None, bark_event: BarkEvent = None):
        """
        Handle detected bark

        Args:
            result: Classification result dict with emotion, confidence, distance, etc.
            loudness_db: Loudness in decibels
            bark_event: Optional BarkEvent from 3-stage detector
        """
        emotion = result.get('emotion', 'unknown')
        confidence = result.get('confidence', 0.0)
        distance = result.get('distance', 'unknown')

        # Update statistics
        self.stats['total_barks'] += 1
        self.stats['emotions_detected'][emotion] = \
            self.stats['emotions_detected'].get(emotion, 0) + 1

        # Get dog info from bark_event if available, otherwise from visible dogs
        if bark_event and bark_event.dog_id:
            dog_id = bark_event.dog_id
            dog_name = bark_event.dog_name
        else:
            # Get currently visible dogs for bark attribution
            visible_dogs = self._get_visible_dogs()

            # Attribution logic: only attribute if exactly 1 dog visible
            if len(visible_dogs) == 1:
                dog_id, dog_info = next(iter(visible_dogs.items()))
                dog_name = dog_info.get('name', 'unknown')
            else:
                dog_id = None
                dog_name = None

        visible_dogs = self._get_visible_dogs()
        visible_ids = list(visible_dogs.keys())

        # Store bark in database
        try:
            from core.bark_store import get_bark_store
            bark_store = get_bark_store()
            bark_store.log_bark(
                emotion=emotion,
                confidence=confidence,
                loudness_db=loudness_db,
                dog_id=dog_id,
                dog_name=dog_name
            )
        except Exception as e:
            logger.warning(f"Failed to store bark: {e}")

        # Track bark frequency for mission triggers
        try:
            from core.bark_frequency_tracker import get_bark_frequency_tracker
            tracker = get_bark_frequency_tracker()
            freq_result = tracker.record_bark(dog_id or 'unknown')
            if freq_result['threshold_exceeded']:
                logger.info(f"Bark threshold exceeded for {dog_id or 'unknown'}: "
                           f"{freq_result['bark_count']}/{freq_result['threshold']} in {freq_result['window_seconds']}s")
        except Exception as e:
            logger.warning(f"Failed to track bark frequency: {e}")

        # Log detection with dog attribution
        if dog_name:
            logger.info(f"Bark detected: {dog_name} barked - {distance} {emotion} "
                       f"(conf: {confidence:.2f}, loudness: {loudness_db:.1f}dB)")
        else:
            logger.info(f"Bark detected: {distance} {emotion} "
                       f"(conf: {confidence:.2f}, loudness: {loudness_db:.1f}dB)")

        # Publish bark detected event with dog attribution and distance
        publish_audio_event('bark_detected', {
            'emotion': emotion,
            'confidence': confidence,
            'distance': distance,  # 'close', 'mid', 'far' from BarkGate
            'loudness_db': loudness_db,
            'peak_energy': result.get('peak_energy', 0.0),
            'duration_ms': result.get('duration_ms', 0),
            'all_probabilities': result.get('all_probabilities', {}),
            'timestamp': datetime.now().isoformat(),
            'visible_dogs': visible_ids,
            'dog_id': dog_id,
            'dog_name': dog_name
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