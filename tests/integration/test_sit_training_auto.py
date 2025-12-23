#!/usr/bin/env python3
"""
Automatic sit training mission test
Runs without requiring interactive input
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_mission_training import LiveMissionTrainer

def main():
    """Run automatic sit training test"""
    print("🤖 TreatBot Automatic Sit Training Test")
    print("=" * 50)

    # Create trainer
    trainer = LiveMissionTrainer()

    # Initialize systems
    if not trainer.initialize():
        print("❌ Failed to initialize trainer")
        return

    print("\n🚀 Starting sit_training mission automatically")
    print("Press Ctrl+C to stop\n")

    try:
        # Start the sit training mission
        trainer.start_mission("sit_training")

        # Let it run for 60 seconds
        start_time = time.time()
        while time.time() - start_time < 60:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    finally:
        trainer.cleanup()
        print("✅ Clean shutdown")

if __name__ == "__main__":
    main()