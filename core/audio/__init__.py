"""
Core audio processing module for WIM-Z bark detection.

Three-stage architecture:
- Stage 1: BarkGate - Signal processing bark detection (no ML)
- Stage 2: Emotion classification (optional, TFLite)
- Stage 3: BarkAnalytics - Per-dog session tracking
"""

from .bark_gate import BarkGate
from .bark_analytics import BarkSession, BarkAnalytics
from .bark_detector import BarkDetector, BarkEvent

__all__ = [
    'BarkGate',
    'BarkSession',
    'BarkAnalytics',
    'BarkDetector',
    'BarkEvent'
]
