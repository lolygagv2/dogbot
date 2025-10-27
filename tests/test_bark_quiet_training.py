#!/usr/bin/env python3
"""
WIM-Z Bark & Quiet Training Test
Simple practical fusion: Detect barking → Issue quiet command → Reward silence
"""

import sys
import os
import time
import threading
from datetime import datetime
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Audio components
from ai.bark_classifier import BarkClassifier
from audio.bark_buffer import BarkAudioBuffer
from services.media.sfx import SFXService

class BarkQuietTrainer:
    """
    Simple bark detection and quiet training system
    Main scenarios:
    1. Dog barks for attention/treats → Play "quiet" → Reward silence
    2. Log all events separately for reporting
    """

    def __init__(self):
        print("[INFO] Initializing WIM-Z Bark & Quiet Trainer")

        # Audio components
        self.bark_classifier = BarkClassifier()
        self.audio_buffer = BarkAudioBuffer(
            sample_rate=22050,
            chunk_duration=0.1,
            buffer_duration=3.0
        )

        # Sound effects
        try:
            self.sfx = SFXService({'sounds_directory': '/home/morgan/dogbot/sounds'})
        except:
            self.sfx = None
            print("[WARNING] SFX service not available")

        # Training state
        self.quiet_command_count = 0
        self.quiet_command_time = None
        self.last_bark_time = None
        self.silence_start_time = None
        self.rewarded_silence = False

        # Event log for reporting
        self.event_log = []

    def log_event(self, event_type: str, details: str):
        """Log event for reporting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {event_type}: {details}"
        self.event_log.append(entry)
        print(f"📝 {entry}")

    def detect_bark(self) -> Optional[str]:
        """Check for barking and return emotion type"""
        audio_data = self.audio_buffer.get_buffer()

        if audio_data is not None and len(audio_data) > 0:
            emotion, confidence = self.bark_classifier.classify(audio_data)

            if emotion and emotion != 'notbark' and confidence > 0.4:
                return emotion
        return None

    def play_quiet_command(self):
        """Play quiet command audio"""
        if self.sfx:
            self.sfx.play_sound('quiet_command')
        else:
            print("🔊 [QUIET COMMAND PLAYED]")

    def dispense_treat(self):
        """Dispense treat reward"""
        print("🍖 [TREAT DISPENSED]")
        self.log_event("REWARD", "Treat dispensed for good behavior")

    def run_training_loop(self, duration: int = 120):
        """
        Main training loop
        Args:
            duration: Test duration in seconds
        """
        print(f"\n=== WIM-Z BARK & QUIET TRAINING ===")
        print(f"Duration: {duration} seconds")
        print(f"Training Protocol:")
        print(f"  1. Detect barking")
        print(f"  2. Play 'quiet' command up to 3 times in 15 seconds")
        print(f"  3. Reward 20 seconds of silence")
        print("="*40 + "\n")

        # Start audio buffer
        self.audio_buffer.start()

        start_time = time.time()

        try:
            while (time.time() - start_time) < duration:
                # Check for barking
                bark_emotion = self.detect_bark()

                if bark_emotion:
                    self.last_bark_time = time.time()
                    self.silence_start_time = None
                    self.rewarded_silence = False

                    # Log bark event
                    self.log_event("BARK", f"Detected {bark_emotion} barking")

                    # Check if we should play quiet command
                    if self.quiet_command_time is None:
                        # Start new quiet training session
                        self.quiet_command_time = time.time()
                        self.quiet_command_count = 1
                        self.play_quiet_command()
                        self.log_event("COMMAND", "First 'quiet' command issued")

                    elif (time.time() - self.quiet_command_time) < 15:
                        # Within 15 second window
                        if self.quiet_command_count < 3:
                            self.quiet_command_count += 1
                            self.play_quiet_command()
                            self.log_event("COMMAND", f"'Quiet' command #{self.quiet_command_count} issued")
                    else:
                        # Reset if outside 15 second window
                        self.quiet_command_time = time.time()
                        self.quiet_command_count = 1
                        self.play_quiet_command()
                        self.log_event("COMMAND", "New quiet training session started")

                else:
                    # No barking detected
                    if self.last_bark_time and not self.rewarded_silence:
                        # Track silence duration
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()

                        silence_duration = time.time() - self.silence_start_time

                        # Reward 20 seconds of silence
                        if silence_duration >= 20:
                            self.dispense_treat()
                            self.rewarded_silence = True
                            self.log_event("SUCCESS", "Dog stayed quiet for 20 seconds!")

                            # Reset for next training
                            self.quiet_command_time = None
                            self.quiet_command_count = 0
                            self.last_bark_time = None

                        elif silence_duration > 0:
                            # Show progress
                            remaining = 20 - silence_duration
                            print(f"⏳ Silence timer: {remaining:.1f}s remaining for reward", end='\r')

                # Small delay
                time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted")
        finally:
            # Cleanup
            self.audio_buffer.stop()

            # Print summary report
            print(f"\n\n=== TRAINING SESSION REPORT ===")
            print(f"Duration: {int(time.time() - start_time)} seconds")
            print(f"\nEvent Log:")
            for event in self.event_log:
                print(f"  {event}")

            # Generate narrative report
            print(f"\n📊 Narrative Summary:")
            bark_events = [e for e in self.event_log if "BARK" in e]
            reward_events = [e for e in self.event_log if "REWARD" in e]

            if bark_events:
                print(f"The dog barked {len(bark_events)} times during the session.")
                print(f"Quiet commands were issued and the dog was successfully")
                print(f"rewarded {len(reward_events)} times for maintaining silence.")
            else:
                print("No barking detected during this session.")

def main():
    """Main test entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="WIM-Z Bark & Quiet Training")
    parser.add_argument('--duration', type=int, default=120, help='Training duration in seconds')
    args = parser.parse_args()

    trainer = BarkQuietTrainer()
    trainer.run_training_loop(duration=args.duration)

if __name__ == "__main__":
    main()