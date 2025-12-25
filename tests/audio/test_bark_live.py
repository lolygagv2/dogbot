#!/usr/bin/env python3
"""
Interactive Bark Testing Tool for WIM-Z
Press Enter to start 20-second bark detection session
Captures all barks with metrics: emotion, confidence, loudness, dog attribution
"""

import sys
import os
import time
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.perception.bark_detector import BarkDetectorService
from core.bus import get_bus, AudioEvent


class BarkTestCollector:
    """Collects bark events during test session"""

    def __init__(self):
        self.barks = []
        self.bus = get_bus()
        self.bus.subscribe(AudioEvent, self._on_audio_event)

    def _on_audio_event(self, event):
        """Capture bark events"""
        if event.subtype == 'bark_detected':
            bark_data = {
                'time': datetime.now().strftime('%H:%M:%S.%f')[:-3],
                'emotion': event.data.get('emotion', 'unknown'),
                'confidence': event.data.get('confidence', 0.0),
                'loudness_db': event.data.get('loudness_db', 0.0),
                'dog_id': event.data.get('dog_id', None),
                'dog_name': event.data.get('dog_name', None),
            }
            self.barks.append(bark_data)
            print(f"  BARK #{len(self.barks)}: {bark_data['emotion']} "
                  f"(conf: {bark_data['confidence']:.2f}, "
                  f"loud: {bark_data['loudness_db']:.1f}dB)")

    def clear(self):
        """Clear collected barks"""
        self.barks = []

    def print_summary(self):
        """Print summary of collected barks"""
        print("\n" + "=" * 60)
        print("BARK DETECTION RESULTS")
        print("=" * 60)

        if not self.barks:
            print("No barks detected during test session.")
            return

        print(f"\nTotal barks detected: {len(self.barks)}\n")

        # Table header
        print(f"{'#':<3} {'Time':<12} {'Emotion':<12} {'Conf':<6} {'Loud(dB)':<10} {'Dog':<15}")
        print("-" * 60)

        # Table rows
        for i, bark in enumerate(self.barks, 1):
            dog = bark['dog_name'] or bark['dog_id'] or 'unknown'
            print(f"{i:<3} {bark['time']:<12} {bark['emotion']:<12} "
                  f"{bark['confidence']:.2f}  {bark['loudness_db']:>7.1f}   {dog:<15}")

        # Statistics
        print("\n" + "-" * 60)
        print("STATISTICS:")

        # Emotion breakdown
        emotions = {}
        for bark in self.barks:
            e = bark['emotion']
            emotions[e] = emotions.get(e, 0) + 1

        print(f"  Emotions: {emotions}")

        # Average confidence
        avg_conf = sum(b['confidence'] for b in self.barks) / len(self.barks)
        print(f"  Average confidence: {avg_conf:.2f}")

        # Average loudness
        avg_loud = sum(b['loudness_db'] for b in self.barks) / len(self.barks)
        print(f"  Average loudness: {avg_loud:.1f} dB")

        # Dog attribution
        dogs = {}
        for bark in self.barks:
            d = bark['dog_name'] or 'unknown'
            dogs[d] = dogs.get(d, 0) + 1
        print(f"  By dog: {dogs}")

        print("=" * 60)


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("WIM-Z BARK DETECTION TEST")
    print("=" * 60)

    # Initialize bark detector
    config = {
        'enabled': True,
        'confidence_threshold': 0.5,
        'reward_emotions': ['alert', 'attention'],
        'check_interval': 0.3,
        'cooldown_period': 1.0,  # Short for testing
        'audio_gain': 30.0,
    }

    print("\nInitializing bark detector...")
    detector = BarkDetectorService(config)

    if not detector.initialize():
        print("ERROR: Failed to initialize bark detector")
        print("Check that USB audio device is connected")
        return

    # Create collector
    collector = BarkTestCollector()

    print("\nBark detector ready!")
    print("\n" + "-" * 60)

    while True:
        input("\nPress ENTER to start 20-second bark detection (or Ctrl+C to quit)...")

        print("\n>>> LISTENING FOR BARKS (20 seconds) <<<")
        print("Make your dog bark now!\n")

        collector.clear()
        detector.start()

        # Listen for 20 seconds
        start_time = time.time()
        while time.time() - start_time < 20:
            remaining = 20 - (time.time() - start_time)
            print(f"\r  Time remaining: {remaining:.0f}s  ", end='', flush=True)
            time.sleep(0.5)

        detector.stop()
        print("\n\n>>> DETECTION COMPLETE <<<")

        # Print results
        collector.print_summary()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest ended by user.")
