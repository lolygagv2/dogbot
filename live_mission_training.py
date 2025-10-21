#!/usr/bin/env python3
"""
REAL Mission Training - Connects Mission API to Live AI Detection
This actually does stuff with real hardware and real dogs!
"""

import sys
import os
import time
import numpy as np
import cv2
import threading
import queue
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the working systems
from core.ai_controller_3stage_fixed import AI3StageControllerFixed
from missions import MissionController

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False

try:
    from servo_control_module import ServoController
    SERVO_AVAILABLE = True
except ImportError:
    SERVO_AVAILABLE = False

try:
    from core.hardware.audio_controller import AudioController
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: AudioController not available")

class LiveMissionTrainer:
    """Real-time mission training with live AI detection"""

    def __init__(self):
        # AI and camera
        self.ai = AI3StageControllerFixed()
        self.camera = None

        # Mission controller
        self.mission = None
        self.current_mission_name = None

        # Hardware
        self.servo_controller = None
        if SERVO_AVAILABLE:
            try:
                self.servo_controller = ServoController()
                self.servo_controller.initialize()
                print("✅ Servo controller ready")
            except Exception as e:
                print(f"❌ Servo failed: {e}")

        # Audio controller (DFPlayer Pro)
        self.audio_controller = None
        if AUDIO_AVAILABLE:
            try:
                self.audio_controller = AudioController()
                print("✅ DFPlayer Pro audio ready")
            except Exception as e:
                print(f"❌ Audio failed: {e}")

        # Real-time state
        self.current_pose = None
        self.pose_confidence = 0.0
        self.pose_start_time = None
        self.last_reward_time = 0

        # Threading
        self.running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.detection_queue = queue.Queue(maxsize=5)

        # Mission tracking
        self.mission_stats = {
            'detections': 0,
            'rewards_given': 0,
            'conditions_met': 0,
            'start_time': None
        }

    def initialize(self):
        """Initialize all systems"""
        print("🎯 Initializing Live Mission Trainer")
        print("=" * 50)

        # Initialize AI
        if not self.ai.initialize():
            print("❌ AI initialization failed")
            return False
        print("✅ AI system ready")

        # Initialize camera
        if not self._init_camera():
            print("❌ Camera initialization failed")
            return False
        print("✅ Camera ready")

        # Create directories
        Path("data").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

        print("✅ Live Mission Trainer ready!")
        return True

    def _init_camera(self):
        """Initialize camera"""
        if not PICAMERA2_AVAILABLE:
            return False

        try:
            self.camera = Picamera2()
            config = self.camera.create_still_configuration(
                main={"size": (1920, 1080)}
            )
            self.camera.configure(config)
            self.camera.start()
            time.sleep(1)
            return True
        except Exception as e:
            print(f"Camera error: {e}")
            return False

    def start_mission(self, mission_name: str):
        """Start a training mission"""
        print(f"\n🚀 Starting Mission: {mission_name}")
        print("=" * 40)

        # Stop any existing mission
        if self.mission and self.mission.is_active:
            self.mission.end(success=False)

        # Create new mission
        self.mission = MissionController(mission_name)
        self.current_mission_name = mission_name

        # Start mission
        mission_id = self.mission.start()
        self.mission_stats['start_time'] = time.time()

        print(f"✅ Mission '{mission_name}' started (ID: {mission_id})")

        # Start processing threads
        self.running = True

        # Camera capture thread
        capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        capture_thread.start()

        # AI processing thread
        ai_thread = threading.Thread(target=self._ai_loop, daemon=True)
        ai_thread.start()

        # Mission logic thread
        mission_thread = threading.Thread(target=self._mission_loop, daemon=True)
        mission_thread.start()

        return mission_id

    def _capture_loop(self):
        """Camera capture loop"""
        while self.running and self.camera:
            try:
                frame = self.camera.capture_array()
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    pass  # Skip if queue full

                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Capture error: {e}")
                time.sleep(0.1)

    def _ai_loop(self):
        """AI processing loop"""
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1.0)

                # Process with AI
                detections, poses, behaviors = self.ai.process_frame(frame)

                # Extract pose information
                current_pose = None
                pose_confidence = 0.0

                if behaviors and len(behaviors) > 0:
                    behavior = behaviors[0]
                    current_pose = behavior.behavior
                    pose_confidence = behavior.confidence

                # Send to mission logic
                detection_data = {
                    'timestamp': time.time(),
                    'pose': current_pose,
                    'confidence': pose_confidence,
                    'detections_count': len(detections),
                    'frame_shape': frame.shape
                }

                try:
                    self.detection_queue.put_nowait(detection_data)
                except queue.Full:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                print(f"AI processing error: {e}")
                time.sleep(0.1)

    def _mission_loop(self):
        """Mission logic loop - the brain with training protocol!"""
        # Training state machine
        self.training_state = "waiting_for_dog"  # waiting_for_dog, attention, command, compliance, reward, cooldown
        self.state_start_time = time.time()
        self.command_given_time = None
        self.name_call_count = 0
        self.command_repeat_count = 0

        while self.running and self.mission:
            try:
                # Get latest detection
                detection = self.detection_queue.get(timeout=1.0)

                # Update mission with current pose
                if detection['pose']:
                    self.mission.set_current_pose(detection['pose'], detection['confidence'])
                    self._update_pose_state(detection)

                # Run training protocol state machine
                self._run_training_protocol(detection)

            except queue.Empty:
                # Even without new detections, check timeouts
                self._check_protocol_timeouts()
                continue
            except Exception as e:
                print(f"Mission logic error: {e}")
                time.sleep(0.1)

    def _update_pose_state(self, detection):
        """Update pose tracking state"""
        pose = detection['pose']
        confidence = detection['confidence']
        timestamp = detection['timestamp']

        # Track pose changes
        if pose != self.current_pose:
            if self.current_pose and self.pose_start_time:
                duration = timestamp - self.pose_start_time
                print(f"📝 Pose '{self.current_pose}' held for {duration:.1f}s")

            self.current_pose = pose
            self.pose_confidence = confidence
            self.pose_start_time = timestamp

            if pose:
                print(f"🎯 New pose detected: {pose} (confidence: {confidence:.2f})")
                self.mission_stats['detections'] += 1

    def _run_training_protocol(self, detection):
        """Run the training protocol state machine"""
        if not self.mission or not self.mission.config:
            return

        config = self.mission.config
        protocol = config.get('training_protocol', {})

        current_time = time.time()

        # State machine for training protocol
        if self.training_state == "waiting_for_dog":
            # Wait for any dog detection
            if detection['detections_count'] > 0:
                print("🐕 Dog detected! Starting training protocol...")
                self._start_attention_phase()

        elif self.training_state == "attention":
            # Get dog's attention by calling name
            self._handle_attention_phase(protocol, current_time)

        elif self.training_state == "command":
            # Give command and wait for compliance
            self._handle_command_phase(protocol, current_time)

        elif self.training_state == "compliance":
            # Monitor for pose compliance
            self._handle_compliance_phase(protocol, detection, current_time)

        elif self.training_state == "reward":
            # Give reward
            self._trigger_reward()
            self._start_cooldown_phase()

        elif self.training_state == "cooldown":
            # Wait before next training cycle
            self._handle_cooldown_phase(config, current_time)

    def _start_attention_phase(self):
        """Start the attention phase - call dog's name"""
        self.training_state = "attention"
        self.state_start_time = time.time()
        self.name_call_count = 0

        config = self.mission.config
        protocol = config.get('training_protocol', {})
        attention = protocol.get('attention_phase', {})

        # Call dog's name
        dog_name_audio = attention.get('dog_name_audio', '/talks/0003.mp3')
        print(f"📢 Calling dog: {dog_name_audio}")
        self._play_audio(dog_name_audio)
        self.name_call_count = 1

    def _handle_attention_phase(self, protocol, current_time):
        """Handle attention phase - repeat name if needed"""
        attention = protocol.get('attention_phase', {})
        repeat_delay = attention.get('repeat_delay', 10.0)
        max_repeats = attention.get('max_repeats', 2)

        # Check if we should repeat the name
        if (current_time - self.state_start_time >= repeat_delay and
            self.name_call_count < max_repeats):

            dog_name_audio = attention.get('dog_name_audio', '/talks/0003.mp3')
            print(f"📢 Repeating dog name: {dog_name_audio}")
            self._play_audio(dog_name_audio)
            self.name_call_count += 1
            self.state_start_time = current_time  # Reset timer

        # Move to command phase after getting attention or max repeats
        if self.name_call_count >= max_repeats:
            self._start_command_phase()

    def _start_command_phase(self):
        """Start the command phase - give sit command"""
        self.training_state = "command"
        self.state_start_time = time.time()
        self.command_repeat_count = 0

        config = self.mission.config
        protocol = config.get('training_protocol', {})
        command = protocol.get('command_phase', {})

        # Give command
        command_audio = command.get('command_audio', '/talks/0015.mp3')
        print(f"🎯 Giving command: {command_audio}")
        self._play_audio(command_audio)
        self.command_given_time = time.time()
        self.command_repeat_count = 1

        # Wait a bit then move to compliance monitoring
        wait_time = command.get('wait_after_command', 2.0)
        self.compliance_start_time = time.time() + wait_time

    def _handle_command_phase(self, protocol, current_time):
        """Handle command phase - wait then move to compliance"""
        command = protocol.get('command_phase', {})

        # Check if we should move to compliance monitoring
        if hasattr(self, 'compliance_start_time') and current_time >= self.compliance_start_time:
            self._start_compliance_phase()
            return

        # Check if we should repeat the command
        repeat_delay = command.get('repeat_command_delay', 8.0)
        max_repeats = command.get('max_command_repeats', 3)

        if (current_time - self.state_start_time >= repeat_delay and
            self.command_repeat_count < max_repeats):

            command_audio = command.get('command_audio', '/talks/0015.mp3')
            print(f"🎯 Repeating command: {command_audio}")
            self._play_audio(command_audio)
            self.command_repeat_count += 1
            self.state_start_time = current_time

    def _start_compliance_phase(self):
        """Start monitoring for pose compliance"""
        self.training_state = "compliance"
        self.state_start_time = time.time()
        print("👁️ Monitoring for compliance...")

    def _handle_compliance_phase(self, protocol, detection, current_time):
        """Handle compliance monitoring"""
        compliance = protocol.get('compliance_phase', {})
        target_pose = compliance.get('target_pose', 'sit')
        duration_required = compliance.get('duration_required', 3.0)
        confidence_threshold = compliance.get('confidence_threshold', 0.6)
        timeout = compliance.get('compliance_timeout', 15.0)

        # Check for timeout
        if current_time - self.state_start_time >= timeout:
            print("⏰ Compliance timeout - restarting protocol")
            self._restart_protocol()
            return

        # Check if dog is in correct pose
        if (self.current_pose == target_pose and
            self.pose_confidence >= confidence_threshold and
            self.pose_start_time and
            current_time - self.pose_start_time >= duration_required):

            print(f"✅ Compliance achieved! {target_pose} for {duration_required}s")
            self.training_state = "reward"

    def _start_cooldown_phase(self):
        """Start cooldown phase"""
        self.training_state = "cooldown"
        self.state_start_time = time.time()
        print("😴 Cooldown phase started...")

    def _handle_cooldown_phase(self, config, current_time):
        """Handle cooldown phase"""
        cooldown = config.get('cooldown_between_rewards', 15)

        if current_time - self.state_start_time >= cooldown:
            print("🔄 Cooldown complete - ready for next training cycle")
            self._restart_protocol()

    def _restart_protocol(self):
        """Restart the training protocol"""
        self.training_state = "waiting_for_dog"
        self.state_start_time = time.time()
        self.name_call_count = 0
        self.command_repeat_count = 0

    def _check_protocol_timeouts(self):
        """Check for timeouts in current state"""
        current_time = time.time()

        # Add timeout logic for states that might get stuck
        if self.training_state in ["attention", "command"] and current_time - self.state_start_time > 30:
            print("⏰ State timeout - restarting protocol")
            self._restart_protocol()

    def _trigger_reward(self):
        """Trigger reward actions - REAL HARDWARE!"""
        if not self.mission:
            return

        config = self.mission.config
        reward_config = config.get('reward', {})

        print(f"\n🎉 REWARD TRIGGERED! Pose: {self.current_pose}")
        print("-" * 30)

        # Dispense treat (if hardware available)
        treat_success = False
        if reward_config.get('treat', False):
            treat_success = self._dispense_treat()

        # Play audio (if hardware available)
        audio_success = False
        audio_file = reward_config.get('audio')
        if audio_file:
            audio_success = self._play_audio(audio_file)

        # Light pattern (if hardware available)
        lights_success = False
        light_pattern = reward_config.get('lights')
        if light_pattern:
            lights_success = self._activate_lights(light_pattern)

        # Log the reward
        self.mission.reward(
            treat=reward_config.get('treat', False),
            audio=audio_file,
            lights=light_pattern
        )

        # Update tracking
        self.last_reward_time = time.time()
        self.mission_stats['rewards_given'] += 1
        self.mission_stats['conditions_met'] += 1

        # Reset pose timer
        self.pose_start_time = time.time()

        print(f"✅ Reward complete - Total rewards: {self.mission_stats['rewards_given']}")

    def _dispense_treat(self):
        """Actually dispense a treat using carousel rotation"""
        try:
            if self.servo_controller:
                print("🍖 Dispensing treat...")
                # Rotate carousel to dispense one treat (advance to next position)
                success = self.servo_controller.rotate_carousel(1)  # Rotate 1 position (60°)
                if success:
                    print("✅ Treat dispensed!")
                    return True
                else:
                    print("⚠️ Carousel rotation failed")
                    return False
            else:
                print("🍖 [MOCK] Treat dispensed!")
                return True
        except Exception as e:
            print(f"❌ Treat dispense failed: {e}")
            return False

    def _play_audio(self, audio_file):
        """Play audio file using DFPlayer Pro with AT commands"""
        try:
            print(f"🔊 Playing audio: {audio_file}")

            if self.audio_controller:
                # Use the DFPlayer Pro with proper AT command format
                # The audio_controller handles the AT+PLAYFILE command
                success = self.audio_controller.play_file_by_path(audio_file)

                if success:
                    print(f"✅ Audio playing: {audio_file}")
                    # Give audio time to play
                    time.sleep(2.0)  # Adjust based on your audio files
                else:
                    print(f"⚠️ Audio playback failed for: {audio_file}")

                return success
            else:
                # Mock mode when no hardware available
                print(f"🔊 [MOCK] Would play: {audio_file}")
                time.sleep(2.0)  # Simulate audio duration
                return True

        except Exception as e:
            print(f"❌ Audio failed: {e}")
            return False

    def _activate_lights(self, pattern):
        """Activate light pattern"""
        try:
            # TODO: Connect to your LED controller
            print(f"💡 Lights activated: {pattern}")
            return True
        except Exception as e:
            print(f"❌ Lights failed: {e}")
            return False

    def get_mission_status(self):
        """Get real-time mission status"""
        if not self.mission:
            return None

        duration = time.time() - self.mission_stats['start_time'] if self.mission_stats['start_time'] else 0

        return {
            'mission_name': self.current_mission_name,
            'duration': duration,
            'current_pose': self.current_pose,
            'pose_confidence': self.pose_confidence,
            'detections': self.mission_stats['detections'],
            'rewards_given': self.mission_stats['rewards_given'],
            'conditions_met': self.mission_stats['conditions_met'],
            'is_active': self.mission.is_active if self.mission else False,
            'training_state': getattr(self, 'training_state', 'unknown'),
            'name_calls': getattr(self, 'name_call_count', 0),
            'command_repeats': getattr(self, 'command_repeat_count', 0)
        }

    def stop_mission(self):
        """Stop current mission"""
        if self.mission and self.mission.is_active:
            summary = self.mission.end(success=True)
            print(f"\n🏁 Mission Complete!")
            print(f"   Duration: {summary['duration_seconds']:.1f}s")
            print(f"   Events: {summary['total_events']}")
            print(f"   Rewards: {self.mission_stats['rewards_given']}")

        self.running = False
        return summary if self.mission else None

    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.camera:
            self.camera.stop()
        if self.audio_controller:
            self.audio_controller.cleanup()
        self.ai.cleanup()

def main():
    """Interactive mission training"""
    print("🤖 TreatBot Live Mission Training")
    print("=" * 50)

    trainer = LiveMissionTrainer()

    if not trainer.initialize():
        print("❌ Failed to initialize trainer")
        return

    try:
        print("\n📋 Available Missions:")
        print("1. sit_training - Dog sits for 3 seconds")
        print("2. quiet_training - Dog stays quiet")
        print("3. custom - Enter mission name")

        choice = input("\nSelect mission (1-3): ").strip()

        if choice == "1":
            mission_name = "sit_training"
        elif choice == "2":
            mission_name = "quiet_training"
        elif choice == "3":
            mission_name = input("Enter mission name: ").strip()
        else:
            mission_name = "sit_training"

        print(f"\n🚀 Starting live training with mission: {mission_name}")

        # Start the mission
        trainer.start_mission(mission_name)

        print("\n🎮 Controls:")
        print("  ENTER - Show status")
        print("  'q' - Quit")
        print("\n🐕 Ready for dog training! Watch for pose detection...")

        # Interactive loop
        while trainer.running:
            user_input = input().strip()

            if user_input.lower() == 'q':
                break
            else:
                # Show status
                status = trainer.get_mission_status()
                if status:
                    print(f"\n📊 Mission Status:")
                    print(f"   Mission: {status['mission_name']}")
                    print(f"   Duration: {status['duration']:.1f}s")
                    print(f"   Training State: {status['training_state'].upper()}")
                    print(f"   Current Pose: {status['current_pose']} ({status['pose_confidence']:.2f})")
                    print(f"   Name Calls: {status['name_calls']}")
                    print(f"   Command Repeats: {status['command_repeats']}")
                    print(f"   Detections: {status['detections']}")
                    print(f"   Rewards Given: {status['rewards_given']}")
                    print(f"   Conditions Met: {status['conditions_met']}")

        # Stop mission
        trainer.stop_mission()

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()