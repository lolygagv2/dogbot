#!/usr/bin/env python3
"""
WIM-Z Treat-on-Sit Autonomous Mode
Detects sitting behavior and automatically dispenses treats with proper cooldowns
Integrates ArUco dog identification for per-dog tracking
"""

import sys
import os
import time
import threading
from datetime import datetime
from typing import Dict, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Core imports
from core.bus import get_bus, VisionEvent, RewardEvent
from core.state import get_state, SystemMode
from core.ai_controller_3stage_fixed import AI3StageControllerFixed
from detect_aruco_id import detect_ids

# Services
from services.reward.dispenser import get_dispenser_service
from services.media.sfx import SFXService
from services.media.led import LEDService

# Camera
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

class TreatOnSitMode:
    """
    Autonomous mode that:
    1. Continuously monitors for dogs
    2. Detects sitting behavior
    3. Dispenses treats with per-dog cooldowns
    4. Tracks individual dogs via ArUco markers
    """

    def __init__(self, config: Dict = None):
        """Initialize treat-on-sit mode"""
        self.config = config or {}
        self.bus = get_bus()
        self.state = get_state()

        # AI controller
        self.ai_controller = AI3StageControllerFixed()

        # Services
        self.dispenser = get_dispenser_service()
        try:
            self.sfx = SFXService({'sounds_directory': '/home/morgan/dogbot/sounds'})
            self.led = LEDService({})
        except:
            self.sfx = None
            self.led = None

        # Camera
        self.camera = None
        self.camera_width = 1920
        self.camera_height = 1080

        # Training parameters
        self.sit_duration_required = self.config.get('sit_duration', 3.0)  # seconds
        self.treat_cooldown = self.config.get('cooldown', 30.0)  # seconds per dog
        self.daily_limit = self.config.get('daily_limit', 10)  # treats per dog per day

        # Dog tracking
        self.dog_profiles = {}  # dog_id -> profile
        self.sit_start_times = {}  # dog_id -> timestamp

        # Mode state
        self.running = False
        self.mode_thread = None

        # Statistics
        self.session_stats = {
            'sits_detected': 0,
            'treats_dispensed': 0,
            'dogs_seen': set()
        }

    def initialize_camera(self) -> bool:
        """Initialize camera"""
        if PICAMERA2_AVAILABLE:
            try:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": (self.camera_width, self.camera_height), "format": "BGR888"}
                )
                self.camera.configure(config)
                self.camera.start()
                print(f"[CAMERA] Initialized for treat-on-sit mode")
                return True
            except Exception as e:
                print(f"[ERROR] Camera init failed: {e}")
                return False
        return False

    def get_or_create_profile(self, dog_id: int) -> Dict:
        """Get or create dog profile"""
        if dog_id not in self.dog_profiles:
            self.dog_profiles[dog_id] = {
                'id': dog_id,
                'name': f"Dog #{dog_id}",
                'last_treat_time': None,
                'treats_today': 0,
                'total_sits': 0,
                'first_seen': datetime.now()
            }
        return self.dog_profiles[dog_id]

    def can_dispense_treat(self, dog_id: int) -> bool:
        """Check if dog can receive treat based on cooldown and daily limit"""
        profile = self.get_or_create_profile(dog_id)

        # Check daily limit
        if profile['treats_today'] >= self.daily_limit:
            return False

        # Check cooldown
        if profile['last_treat_time']:
            elapsed = (datetime.now() - profile['last_treat_time']).total_seconds()
            if elapsed < self.treat_cooldown:
                return False

        return True

    def dispense_treat_for_dog(self, dog_id: int):
        """Dispense treat and update dog profile"""
        profile = self.get_or_create_profile(dog_id)

        # Dispense treat
        print(f"🍖 Dispensing treat for {profile['name']}!")

        if self.dispenser:
            self.dispenser.dispense()

        if self.sfx:
            self.sfx.play_sound('good_dog')

        if self.led:
            self.led.celebrate()

        # Update profile
        profile['last_treat_time'] = datetime.now()
        profile['treats_today'] += 1
        profile['total_sits'] += 1

        # Update stats
        self.session_stats['treats_dispensed'] += 1

        # Publish event
        self.bus.publish(RewardEvent.TREAT_DISPENSED, {
            'dog_id': dog_id,
            'dog_name': profile['name'],
            'treats_today': profile['treats_today'],
            'source': 'treat_on_sit_mode'
        })

    def process_frame(self, frame):
        """Process single frame for sit detection"""
        # Detect ArUco markers for dog ID
        aruco_markers = detect_ids(frame)
        current_dogs = {}

        if aruco_markers:
            for marker_id, cx, cy in aruco_markers:
                current_dogs[marker_id] = (cx, cy)
                self.session_stats['dogs_seen'].add(marker_id)

        # Get AI detections
        detections = self.ai_controller.process_frame(frame)

        if not detections:
            # No dogs detected - reset sit timers
            self.sit_start_times.clear()
            return

        # Process each detection
        for det in detections:
            behavior = det.get('behavior')

            # Determine dog ID (use ArUco if available, otherwise default)
            dog_id = 0  # Default dog
            if current_dogs:
                # Use closest ArUco marker to detection bbox
                bbox = det.get('bbox', [])
                if len(bbox) >= 4:
                    det_center_x = (bbox[0] + bbox[2]) / 2
                    det_center_y = (bbox[1] + bbox[3]) / 2

                    # Find closest marker
                    min_dist = float('inf')
                    for marker_id, (mx, my) in current_dogs.items():
                        dist = ((det_center_x - mx)**2 + (det_center_y - my)**2)**0.5
                        if dist < min_dist:
                            min_dist = dist
                            dog_id = marker_id

            # Check for sitting behavior
            if behavior == 'sit':
                self.session_stats['sits_detected'] += 1

                # Track sit duration
                if dog_id not in self.sit_start_times:
                    self.sit_start_times[dog_id] = time.time()
                    profile = self.get_or_create_profile(dog_id)
                    print(f"🪑 {profile['name']} started sitting...")

                else:
                    # Check if held long enough
                    sit_duration = time.time() - self.sit_start_times[dog_id]

                    if sit_duration >= self.sit_duration_required:
                        # Good sit! Check if we can reward
                        if self.can_dispense_treat(dog_id):
                            self.dispense_treat_for_dog(dog_id)
                            # Reset sit timer after reward
                            del self.sit_start_times[dog_id]
                        else:
                            profile = self.get_or_create_profile(dog_id)
                            if dog_id in self.sit_start_times:
                                # Only show cooldown message once per sit
                                if sit_duration < self.sit_duration_required + 0.5:
                                    remaining = self.treat_cooldown - \
                                               (datetime.now() - profile['last_treat_time']).total_seconds()
                                    print(f"⏳ {profile['name']} in cooldown: {remaining:.0f}s")

            else:
                # Not sitting - reset timer for this dog
                if dog_id in self.sit_start_times:
                    profile = self.get_or_create_profile(dog_id)
                    print(f"❌ {profile['name']} stopped sitting")
                    del self.sit_start_times[dog_id]

    def run_mode(self):
        """Main mode execution loop"""
        print("\n=== WIM-Z TREAT-ON-SIT MODE ACTIVE ===")
        print(f"Sit duration required: {self.sit_duration_required}s")
        print(f"Cooldown between treats: {self.treat_cooldown}s")
        print(f"Daily limit per dog: {self.daily_limit} treats")
        print("="*40 + "\n")

        # Initialize camera
        if not self.initialize_camera():
            print("[ERROR] Cannot run without camera")
            return

        # Set system mode
        self.state.set_mode(SystemMode.AI_ACTIVE)

        try:
            while self.running:
                # Get frame
                if self.camera:
                    frame = self.camera.capture_array()
                    self.process_frame(frame)

                # Small delay to prevent CPU overload
                time.sleep(0.1)

        except Exception as e:
            print(f"[ERROR] Mode error: {e}")
        finally:
            self.cleanup()

    def start(self):
        """Start treat-on-sit mode"""
        if self.running:
            return

        self.running = True
        self.mode_thread = threading.Thread(target=self.run_mode, daemon=True)
        self.mode_thread.start()

        print("✅ Treat-on-sit mode started")

    def stop(self):
        """Stop treat-on-sit mode"""
        self.running = False

        if self.mode_thread:
            self.mode_thread.join(timeout=2.0)

        self.cleanup()
        self.print_summary()

    def cleanup(self):
        """Clean up resources"""
        if self.camera:
            self.camera.stop()
            self.camera = None

        # Reset system mode
        self.state.set_mode(SystemMode.IDLE)

    def print_summary(self):
        """Print session summary"""
        print("\n=== SESSION SUMMARY ===")
        print(f"Sits detected: {self.session_stats['sits_detected']}")
        print(f"Treats dispensed: {self.session_stats['treats_dispensed']}")
        print(f"Dogs seen: {len(self.session_stats['dogs_seen'])}")

        if self.dog_profiles:
            print("\nDog Profiles:")
            for dog_id, profile in self.dog_profiles.items():
                print(f"  {profile['name']}:")
                print(f"    Treats today: {profile['treats_today']}")
                print(f"    Total sits: {profile['total_sits']}")

def main():
    """Test treat-on-sit mode"""
    import signal

    mode = TreatOnSitMode({
        'sit_duration': 3.0,
        'cooldown': 15.0,  # Short for testing
        'daily_limit': 10
    })

    def signal_handler(sig, frame):
        print("\nShutting down...")
        mode.stop()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    mode.start()

    print("Mode active. Press Ctrl+C to exit.")

    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()