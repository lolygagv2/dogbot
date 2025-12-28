#!/usr/bin/env python3
"""
AI Field Demo Test - Clean output for reviewing behavior detection
Shows dog identity (ArUco), behaviors, and bark detection.

Uses new 3-stage bark detection:
- Stage 1: BarkGate (signal processing)
- Stage 2: Emotion classification (optional)
- Stage 3: Per-dog analytics

Includes "SPEAK" trick detection (sit + bark within 5 seconds)

Usage: DISPLAY=:0 python3 tests/test_ai_field_demo.py
"""

import sys
import os
import time
import signal
import threading
from datetime import datetime
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_controller_3stage_fixed import AI3StageControllerFixed
import cv2
import numpy as np

# Camera setup
try:
    from picamera2 import Picamera2
    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False

# New bark detection system
try:
    from core.audio.bark_detector import BarkDetector, BarkEvent
    from audio.bark_buffer_arecord import BarkAudioBufferArecord as BarkAudioBuffer
    from services.media.usb_audio import set_agc
    BARK_DETECTION_AVAILABLE = True
except ImportError as e:
    print(f"Bark detection not available: {e}")
    BARK_DETECTION_AVAILABLE = False

# ArUco configuration - CORRECT settings for dog collars
ARUCO_DICT = cv2.aruco.DICT_4X4_1000
DOG_MARKERS = {
    315: "Elsa",
    832: "Bezik"
}

# Speak trick settings
SPEAK_WINDOW = 5.0  # Seconds after sit to detect bark for "speak"


def timestamp():
    """Return formatted timestamp for logging"""
    return datetime.now().strftime("%H:%M:%S")


def detect_aruco(frame, aruco_dict, aruco_params):
    """Detect ArUco markers and return dog identifications"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    detected_dogs = []
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in DOG_MARKERS:
                # Get marker center
                corner = corners[i][0]
                cx = int(np.mean(corner[:, 0]))
                cy = int(np.mean(corner[:, 1]))
                detected_dogs.append({
                    'id': marker_id,
                    'name': DOG_MARKERS[marker_id],
                    'center': (cx, cy)
                })
    return detected_dogs


class BarkDetectionThread:
    """Background thread for bark detection using new 3-stage system"""

    def __init__(self):
        self.running = False
        self.thread = None
        self.detector = None
        self.audio_buffer = None
        self.bark_events = []  # List of BarkEvent
        self.lock = threading.Lock()
        self.current_dog_id = None
        self.current_dog_name = None

    def start(self):
        """Start bark detection in background"""
        if not BARK_DETECTION_AVAILABLE:
            print("Bark detection not available")
            return False

        try:
            # Initialize new 3-stage detector
            self.detector = BarkDetector(enable_emotion=True)
            self.detector.start()

            # Initialize audio buffer
            # Note: arecord needs at least 1.0s chunks to work reliably
            self.audio_buffer = BarkAudioBuffer(
                sample_rate=44100,
                chunk_duration=1.0,  # 1 second chunks (arecord minimum)
                gain=30.0
            )
            self.audio_buffer.start()

            # Start detection thread
            self.running = True
            self.thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.thread.start()
            return True

        except Exception as e:
            print(f"Failed to start bark detection: {e}")
            import traceback
            traceback.print_exc()
            return False

    def stop(self):
        """Stop bark detection"""
        self.running = False
        if self.detector:
            self.detector.stop()
        if self.audio_buffer:
            self.audio_buffer.stop()
        if self.thread:
            self.thread.join(timeout=2.0)

    def set_current_dog(self, dog_id: str, dog_name: str):
        """Update current dog for bark attribution"""
        self.current_dog_id = dog_id
        self.current_dog_name = dog_name

    def _detection_loop(self):
        """Background detection loop using BarkGate"""
        while self.running:
            try:
                audio_chunk = self.audio_buffer.get_audio_chunk(timeout=0.5)

                if audio_chunk is not None:
                    # Calculate energy (RMS)
                    audio_energy = np.sqrt(np.mean(audio_chunk**2))

                    # Process through 3-stage detector
                    event = self.detector.process_energy_only(
                        energy=audio_energy,
                        dog_id=self.current_dog_id,
                        dog_name=self.current_dog_name
                    )

                    if event:
                        with self.lock:
                            self.bark_events.append(event)
                        print(f"  [BARK!] {event.distance} | peak={event.peak_energy:.3f} | "
                              f"duration={event.duration_ms}ms")

                time.sleep(0.05)

            except Exception as e:
                time.sleep(0.5)

    def get_recent_barks(self, since_time: float):
        """Get barks since a specific time (exclusive to avoid duplicates)"""
        with self.lock:
            # Use > not >= to avoid returning same bark twice
            return [b for b in self.bark_events if b.timestamp > since_time]

    def get_bark_count(self):
        """Get total bark count"""
        with self.lock:
            return len(self.bark_events)

    def get_analytics(self):
        """Get bark analytics from detector"""
        if self.detector:
            return self.detector.get_all_summaries()
        return {}


def main():
    """Run field demo test with clean CLI output"""

    if not PICAMERA_AVAILABLE:
        print("ERROR: Picamera2 required")
        return 1

    print("=" * 50)
    print("AI FIELD DEMO TEST (v2 - 3-Stage Bark Detection)")
    print("=" * 50)
    print(f"ArUco: DICT_4X4_1000")
    print(f"Dogs: Elsa=315, Bezik=832")
    print(f"Bark detection: {'YES (3-stage)' if BARK_DETECTION_AVAILABLE else 'NO'}")
    print("-" * 50)

    # Initialize AI controller
    print("Initializing AI...")
    ai = AI3StageControllerFixed()
    if not ai.initialize():
        print("ERROR: AI controller failed")
        return 1
    print("AI ready")

    # Initialize camera
    print("Initializing camera...")
    camera = Picamera2()
    config = camera.create_preview_configuration(
        main={"size": (640, 640), "format": "RGB888"}
    )
    camera.configure(config)
    camera.start()
    time.sleep(1.0)
    print("Camera ready")

    # Initialize ArUco detector (OpenCV 4.7+ API)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    aruco_params = cv2.aruco.DetectorParameters()

    # Initialize bark detection
    bark_detector = None
    if BARK_DETECTION_AVAILABLE:
        print("Initializing bark detection (3-stage)...")
        # Disable AGC for proper bark detection (raw energy levels)
        set_agc(False)
        print("AGC disabled for bark detection")
        bark_detector = BarkDetectionThread()
        if bark_detector.start():
            print("Bark detection ready")
        else:
            print("Bark detection failed - continuing without")
            bark_detector = None

    print("-" * 50)
    print("STARTING - Press Ctrl+C to stop")
    print("-" * 50)
    print()

    # State tracking
    running = True
    current_dog = None
    current_dog_id = None
    last_behavior = None
    last_behavior_time = 0
    behavior_cooldown = 2.0  # Seconds before same behavior can be logged again

    # Speak trick tracking
    last_sit_time = None  # When dog last sat (for speak detection)
    last_bark_logged = 0  # Prevent duplicate bark logs

    # Per-dog behavior tallies
    behavior_counts = defaultdict(lambda: defaultdict(int))
    bark_counts = defaultdict(int)  # Per-dog bark counts
    speak_counts = defaultdict(int)  # Per-dog speak trick counts
    dogs_seen = set()
    start_time = time.time()

    # Signal handler
    def signal_handler(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while running:
            # Capture frame
            frame = camera.capture_array()
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Detect ArUco markers
            detected_dogs = detect_aruco(frame, aruco_dict, aruco_params)

            # Update current dog if ArUco detected
            if detected_dogs:
                dog = detected_dogs[0]  # Use first detected dog
                if dog['name'] != current_dog:
                    current_dog = dog['name']
                    current_dog_id = str(dog['id'])
                    dogs_seen.add(current_dog)
                    print(f"[{timestamp()}] DOG: {current_dog} (ArUco {dog['id']})")

                    # Update bark detector with current dog
                    if bark_detector:
                        bark_detector.set_current_dog(current_dog_id, current_dog)

            # Run AI detection
            detections, poses, behaviors = ai.process_frame(frame)

            # Process behaviors
            if behaviors and current_dog:
                for bhv in behaviors:
                    behavior_name = bhv.behavior
                    conf = bhv.confidence
                    now = time.time()

                    # Only log if behavior changed OR cooldown expired
                    is_new_behavior = behavior_name != last_behavior
                    cooldown_expired = (now - last_behavior_time) > behavior_cooldown

                    if is_new_behavior or cooldown_expired:
                        if conf >= 0.7:  # Minimum confidence threshold
                            print(f"[{timestamp()}] {behavior_name.upper()} ({conf:.2f})")
                            behavior_counts[current_dog][behavior_name] += 1
                            last_behavior = behavior_name
                            last_behavior_time = now

                            # Track sit time for speak detection
                            if behavior_name == 'sit':
                                last_sit_time = now

            # Check for barks (and speak trick)
            if bark_detector and current_dog:
                now = time.time()

                # Get any new barks
                barks = bark_detector.get_recent_barks(last_bark_logged)
                for bark in barks:
                    bark_time = bark.timestamp
                    distance = bark.distance

                    # Log the bark
                    print(f"[{timestamp()}] BARK: {distance} (peak={bark.peak_energy:.2f})")
                    bark_counts[current_dog] += 1
                    last_bark_logged = bark_time

                    # Check for SPEAK trick (bark within SPEAK_WINDOW of a sit)
                    if last_sit_time and (bark_time - last_sit_time) <= SPEAK_WINDOW:
                        print(f"[{timestamp()}] >>> SPEAK TRICK! (sat + barked)")
                        speak_counts[current_dog] += 1
                        last_sit_time = None  # Reset to prevent double-counting

            # Small delay to prevent CPU overload
            time.sleep(0.05)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Session summary
        duration = time.time() - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)

        print()
        print("=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        print(f"Duration: {minutes}m {seconds}s")
        print(f"Dogs seen: {', '.join(dogs_seen) if dogs_seen else 'None'}")
        print()

        # Behavior tallies
        total_behaviors = 0
        total_barks = sum(bark_counts.values())
        total_speaks = sum(speak_counts.values())

        if behavior_counts or bark_counts:
            print("BEHAVIOR TALLY:")
            all_dogs = set(behavior_counts.keys()) | set(bark_counts.keys())
            for dog_name in sorted(all_dogs):
                counts = behavior_counts.get(dog_name, {})
                print(f"  {dog_name}:")
                for behavior in ['sit', 'stand', 'lie', 'cross', 'spin']:
                    count = counts.get(behavior, 0)
                    total_behaviors += count
                    if count > 0:
                        print(f"    {behavior}: {count}")
                # Add bark and speak counts
                if bark_counts.get(dog_name, 0) > 0:
                    print(f"    bark: {bark_counts[dog_name]}")
                if speak_counts.get(dog_name, 0) > 0:
                    print(f"    SPEAK: {speak_counts[dog_name]}")
            print()
            print(f"TOTAL BEHAVIORS: {total_behaviors}")
            if total_barks > 0:
                print(f"TOTAL BARKS: {total_barks}")
            if total_speaks > 0:
                print(f"TOTAL SPEAK TRICKS: {total_speaks}")
        else:
            print("No behaviors detected")

        # Show bark analytics if available
        if bark_detector:
            analytics = bark_detector.get_analytics()
            if analytics:
                print()
                print("BARK ANALYTICS:")
                for dog_id, summary in analytics.items():
                    print(f"  {summary.get('dog_name', dog_id)}:")
                    print(f"    Total barks: {summary['total_barks']}")
                    print(f"    By distance: {summary['by_distance']}")

        print("=" * 50)

        # Cleanup
        if bark_detector:
            bark_detector.stop()
            # Re-enable AGC on exit
            set_agc(True)
            print("AGC re-enabled")
        camera.stop()
        camera.close()
        ai.cleanup()
        print("Cleanup complete")

    return 0


if __name__ == "__main__":
    exit(main())
