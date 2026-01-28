"""
TreatBot Bark Emotion Classifier
Detects and classifies dog bark emotions using TFLite model
"""

import numpy as np
import librosa
try:
    # Try tflite_runtime first (lighter weight)
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        # Fall back to tensorflow
        import tensorflow.lite as tflite
    except ImportError:
        # Last resort - full tensorflow
        import tensorflow as tf
        tflite = tf.lite
import json
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Singleton instance for BarkClassifier
_bark_classifier_instance = None


class BarkClassifier:
    """
    Real-time bark emotion classifier for TreatBot (SINGLETON)

    Emotions detected:
    - alert, attention, anxious, aggressive
    - scared, playful, notbark, other

    Uses singleton pattern to prevent loading the TFLite model multiple times.
    """

    def __new__(cls, *args, **kwargs):
        global _bark_classifier_instance
        if _bark_classifier_instance is None:
            _bark_classifier_instance = super().__new__(cls)
            _bark_classifier_instance._initialized = False
        return _bark_classifier_instance

    def __init__(self,
                 model_path: str = '/home/morgan/dogbot/ai/models/dog_bark_classifier.tflite',
                 emotion_mapping_path: str = '/home/morgan/dogbot/ai/models/emotion_mapping.json',
                 sample_rate: int = 22050,
                 duration: float = 3.0,
                 n_mels: int = 128):
        """
        Initialize bark classifier (singleton - only runs once)

        Args:
            model_path: Path to TFLite model
            emotion_mapping_path: Path to emotion mapping JSON
            sample_rate: Audio sample rate (Hz)
            duration: Audio clip duration (seconds)
            n_mels: Number of mel frequency bands
        """
        if self._initialized:
            logger.debug("BarkClassifier already initialized (singleton)")
            return

        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mels = n_mels

        # Load TFLite model
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            logger.info(f"Loaded bark classifier model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Get model input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Load emotion mapping
        try:
            with open(emotion_mapping_path, 'r') as f:
                self.emotion_mapping = json.load(f)
            self.id_to_emotion = {v: k for k, v in self.emotion_mapping.items()}
            logger.info(f"Loaded emotion mapping: {list(self.emotion_mapping.keys())}")
        except Exception as e:
            logger.error(f"Failed to load emotion mapping: {e}")
            raise

        self._initialized = True
        logger.info("BarkClassifier initialized (singleton)")
    
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Convert audio to normalized mel spectrogram
        
        Args:
            audio_data: Raw audio samples (1D numpy array)
            
        Returns:
            Preprocessed spectrogram ready for model (4D tensor)
        """
        # Ensure correct length
        target_length = int(self.sample_rate * self.duration)
        if len(audio_data) < target_length:
            audio_data = np.pad(audio_data, (0, target_length - len(audio_data)))
        else:
            audio_data = audio_data[:target_length]
        
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=self.sample_rate, 
            n_mels=self.n_mels
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Add batch and channel dimensions (B, H, W, C)
        mel_spec_norm = np.expand_dims(mel_spec_norm, axis=0)  # Batch
        mel_spec_norm = np.expand_dims(mel_spec_norm, axis=-1)  # Channel
        
        return mel_spec_norm.astype(np.float32)
    
    def predict(self, audio_data: np.ndarray, confidence_threshold: float = 0.5) -> Dict:
        """
        Predict bark emotion from audio data
        
        Args:
            audio_data: Raw audio samples
            confidence_threshold: Minimum confidence for valid prediction
            
        Returns:
            Dictionary with:
                - emotion: Predicted emotion string
                - confidence: Confidence score (0-1)
                - all_probabilities: Dict of all emotion probabilities
                - is_confident: Boolean if above threshold
        """
        # Preprocess audio
        input_data = self.preprocess_audio(audio_data)
        
        # Run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Get predictions
        predicted_id = np.argmax(output_data[0])
        confidence = float(output_data[0][predicted_id])
        emotion = self.id_to_emotion[predicted_id]
        
        # Get all probabilities
        all_probs = {
            self.id_to_emotion[i]: float(output_data[0][i])
            for i in range(len(output_data[0]))
        }
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'is_confident': confidence >= confidence_threshold
        }
    
    def predict_from_file(self, audio_path: str, confidence_threshold: float = 0.5) -> Dict:
        """
        Predict bark emotion from audio file
        
        Args:
            audio_path: Path to WAV file
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            Prediction dictionary
        """
        # Load audio file
        audio_data, _ = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        return self.predict(audio_data, confidence_threshold)