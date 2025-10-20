#!/usr/bin/env python3
"""
Interactive Training Mode for DogBot
Live camera view with voice commands and manual confirmation
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import queue
import time
import speech_recognition as sr
import logging
from datetime import datetime
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.treat_dispenser_robot import TreatDispenserRobot, RobotMode
from core.hardware.led_controller import LEDMode

class InteractiveTrainingGUI:
    """Interactive training interface with live camera and voice commands"""

    def __init__(self, robot: TreatDispenserRobot):
        self.robot = robot
        self.root = tk.Tk()
        self.root.title("🐕 DogBot Interactive Training")
        self.root.geometry("1200x800")

        # Voice recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.listening = False

        # Camera and detection
        self.current_frame = None
        self.detection_overlay = True
        self.latest_detection = None

        # Training state
        self.training_active = False
        self.waiting_for_confirmation = False
        self.current_command = None
        self.detected_behavior = None

        # UI elements
        self.video_label = None
        self.status_label = None
        self.detection_text = None
        self.command_history = []

        self.setup_gui()
        self.setup_voice_recognition()

        # Start camera feed
        self.start_camera_feed()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger('InteractiveTraining')

    def setup_gui(self):
        """Setup the GUI layout"""
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Left side - Camera feed
        camera_frame = tk.Frame(main_frame, relief='ridge', bd=2)
        camera_frame.pack(side='left', fill='both', expand=True, padx=(0,10))

        # Camera title
        tk.Label(camera_frame, text="🎥 Live Camera Feed",
                font=('Arial', 14, 'bold')).pack(pady=5)

        # Video display
        self.video_label = tk.Label(camera_frame, bg='black')
        self.video_label.pack(padx=10, pady=10)

        # Camera controls
        cam_controls = tk.Frame(camera_frame)
        cam_controls.pack(pady=5)

        tk.Button(cam_controls, text="📹 Center Camera",
                 command=self.center_camera).pack(side='left', padx=5)
        tk.Button(cam_controls, text="👀 Toggle Detection Overlay",
                 command=self.toggle_overlay).pack(side='left', padx=5)

        # Servo controls
        servo_frame = tk.Frame(camera_frame)
        servo_frame.pack(pady=10, fill='x')

        # Pan control
        tk.Label(servo_frame, text="Pan (Left/Right)").pack()
        self.pan_scale = tk.Scale(servo_frame, from_=-90, to=90,
                                 orient='horizontal', command=self.update_pan)
        self.pan_scale.set(0)
        self.pan_scale.pack(fill='x', padx=20)

        # Tilt control
        tk.Label(servo_frame, text="Tilt (Up/Down)").pack()
        self.tilt_scale = tk.Scale(servo_frame, from_=-45, to=45,
                                  orient='horizontal', command=self.update_tilt)
        self.tilt_scale.set(0)
        self.tilt_scale.pack(fill='x', padx=20)

        # Right side - Training controls
        control_frame = tk.Frame(main_frame, relief='ridge', bd=2)
        control_frame.pack(side='right', fill='y', padx=(10,0))

        # Training title
        tk.Label(control_frame, text="🎯 Training Controls",
                font=('Arial', 14, 'bold')).pack(pady=10)

        # Status display
        status_frame = tk.Frame(control_frame, relief='sunken', bd=1)
        status_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(status_frame, text="System Status:", font=('Arial', 10, 'bold')).pack()
        self.status_label = tk.Label(status_frame, text="🟢 Ready",
                                    font=('Arial', 12), fg='green')
        self.status_label.pack(pady=5)

        # Voice control
        voice_frame = tk.Frame(control_frame, relief='ridge', bd=1)
        voice_frame.pack(fill='x', padx=10, pady=10)

        tk.Label(voice_frame, text="🎤 Voice Commands",
                font=('Arial', 12, 'bold')).pack(pady=5)

        self.voice_button = tk.Button(voice_frame, text="🎤 Start Listening",
                                     command=self.toggle_voice_listening,
                                     bg='lightgreen', font=('Arial', 10, 'bold'))
        self.voice_button.pack(pady=5)

        # Current detection info
        detection_frame = tk.Frame(control_frame, relief='ridge', bd=1)
        detection_frame.pack(fill='both', expand=True, padx=10, pady=5)

        tk.Label(detection_frame, text="🔍 Current Detection",
                font=('Arial', 12, 'bold')).pack(pady=5)

        self.detection_text = tk.Text(detection_frame, height=8, width=30,
                                     font=('Courier', 10))
        self.detection_text.pack(fill='both', expand=True, padx=5, pady=5)

        # Manual training controls
        manual_frame = tk.Frame(control_frame, relief='ridge', bd=1)
        manual_frame.pack(fill='x', padx=10, pady=5)

        tk.Label(manual_frame, text="🎮 Manual Training",
                font=('Arial', 12, 'bold')).pack(pady=5)

        # Quick command buttons
        quick_commands = ['sit', 'stay', 'lie_down', 'good_dog']
        for cmd in quick_commands:
            tk.Button(manual_frame, text=f"🔊 {cmd.replace('_', ' ').title()}",
                     command=lambda c=cmd: self.manual_command(c),
                     width=15).pack(pady=2)

        # Confirmation buttons (initially hidden)
        self.confirm_frame = tk.Frame(control_frame, relief='ridge', bd=2, bg='yellow')

        tk.Label(self.confirm_frame, text="❓ Confirm Detection",
                font=('Arial', 12, 'bold'), bg='yellow').pack(pady=5)

        self.confirm_text = tk.Label(self.confirm_frame, text="",
                                    font=('Arial', 10), bg='yellow')
        self.confirm_text.pack(pady=5)

        confirm_buttons = tk.Frame(self.confirm_frame, bg='yellow')
        confirm_buttons.pack(pady=5)

        tk.Button(confirm_buttons, text="✅ Correct",
                 command=self.confirm_behavior, bg='green', fg='white',
                 font=('Arial', 10, 'bold')).pack(side='left', padx=5)
        tk.Button(confirm_buttons, text="❌ Wrong",
                 command=self.reject_behavior, bg='red', fg='white',
                 font=('Arial', 10, 'bold')).pack(side='left', padx=5)

        # Command history
        history_frame = tk.Frame(control_frame, relief='ridge', bd=1)
        history_frame.pack(fill='x', padx=10, pady=(5,10))

        tk.Label(history_frame, text="📋 Recent Commands",
                font=('Arial', 10, 'bold')).pack()

        self.history_text = tk.Text(history_frame, height=4, width=30,
                                   font=('Courier', 8))
        self.history_text.pack(fill='x', padx=5, pady=5)

    def setup_voice_recognition(self):
        """Setup voice recognition system"""
        try:
            # Adjust for ambient noise
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            self.logger.info("Voice recognition initialized")
        except Exception as e:
            self.logger.error(f"Voice recognition setup failed: {e}")

    def start_camera_feed(self):
        """Start the camera feed thread"""
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def camera_loop(self):
        """Main camera processing loop"""
        while True:
            try:
                if self.robot.vision and hasattr(self.robot.vision, 'get_latest_frame'):
                    frame = self.robot.vision.get_latest_frame()
                    if frame is not None:
                        self.current_frame = frame
                        self.update_video_display(frame)

                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                self.logger.error(f"Camera loop error: {e}")
                time.sleep(1)

    def update_video_display(self, frame):
        """Update the video display with detection overlays"""
        if frame is None:
            return

        display_frame = frame.copy()

        # Add detection overlay if enabled
        if self.detection_overlay and self.latest_detection:
            self.draw_detection_overlay(display_frame)

        # Add status text overlay
        status_text = f"Mode: {self.robot.current_mode.value} | "
        if self.listening:
            status_text += "🎤 LISTENING"
        elif self.waiting_for_confirmation:
            status_text += "❓ WAITING FOR CONFIRMATION"
        else:
            status_text += "🟢 Ready"

        cv2.putText(display_frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert to RGB and resize
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        display_frame = cv2.resize(display_frame, (640, 480))

        # Convert to PhotoImage
        image = Image.fromarray(display_frame)
        photo = ImageTk.PhotoImage(image=image)

        # Update display (thread-safe)
        self.root.after(0, self._update_video_label, photo)

    def _update_video_label(self, photo):
        """Thread-safe video label update"""
        if self.video_label:
            self.video_label.configure(image=photo)
            self.video_label.image = photo

    def draw_detection_overlay(self, frame):
        """Draw detection information on frame"""
        # This would draw bounding boxes, keypoints, etc.
        # Based on what's currently detected
        detection_info = "Dog detected!" if self.latest_detection else "Searching..."
        cv2.putText(frame, detection_info, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    def toggle_voice_listening(self):
        """Toggle voice recognition on/off"""
        if not self.listening:
            self.start_voice_listening()
        else:
            self.stop_voice_listening()

    def start_voice_listening(self):
        """Start listening for voice commands"""
        self.listening = True
        self.voice_button.config(text="🔴 Stop Listening", bg='red')
        self.status_label.config(text="🎤 Listening...", fg='blue')

        # Start voice recognition thread
        voice_thread = threading.Thread(target=self.voice_recognition_loop, daemon=True)
        voice_thread.start()

    def stop_voice_listening(self):
        """Stop voice recognition"""
        self.listening = False
        self.voice_button.config(text="🎤 Start Listening", bg='lightgreen')
        self.status_label.config(text="🟢 Ready", fg='green')

    def voice_recognition_loop(self):
        """Main voice recognition loop"""
        while self.listening:
            try:
                with self.microphone as source:
                    # Listen for command with timeout
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)

                try:
                    # Recognize speech
                    command = self.recognizer.recognize_google(audio).lower()
                    self.logger.info(f"Voice command recognized: {command}")

                    # Process command
                    self.root.after(0, self.process_voice_command, command)

                except sr.UnknownValueError:
                    pass  # No speech detected
                except sr.RequestError as e:
                    self.logger.error(f"Speech recognition error: {e}")

            except sr.WaitTimeoutError:
                pass  # Normal timeout, continue listening
            except Exception as e:
                self.logger.error(f"Voice recognition loop error: {e}")
                time.sleep(1)

    def process_voice_command(self, command):
        """Process a recognized voice command"""
        self.add_to_history(f"Voice: {command}")

        # Parse commands
        if any(word in command for word in ['sit', 'stay', 'lie', 'down', 'come']):
            # Extract behavior command
            if 'sit' in command:
                self.initiate_training_sequence('sit')
            elif 'stay' in command:
                self.initiate_training_sequence('stay')
            elif 'lie' in command or 'down' in command:
                self.initiate_training_sequence('lie_down')

        elif any(word in command for word in ['good', 'treat', 'reward']):
            # Immediate reward
            self.give_treat_and_praise()

        elif 'center' in command or 'reset' in command:
            self.center_camera()

    def initiate_training_sequence(self, behavior_command):
        """Start a training sequence for a specific behavior"""
        self.current_command = behavior_command
        self.status_label.config(text=f"🎯 Training: {behavior_command}", fg='orange')

        # Play the command audio
        if self.robot.hardware.get('audio'):
            self.robot.hardware['audio'].play_sound(behavior_command.upper())

        # Wait a moment, then check what the dog is doing
        self.root.after(3000, self.analyze_current_behavior)  # 3 second delay

    def analyze_current_behavior(self):
        """Analyze what the dog is currently doing"""
        # This would integrate with the behavior analyzer
        # For now, simulate detection

        detected = "sitting"  # This would come from actual detection
        confidence = 0.85

        self.detected_behavior = detected
        self.show_confirmation_dialog(detected, confidence)

    def show_confirmation_dialog(self, behavior, confidence):
        """Show confirmation dialog for detected behavior"""
        self.waiting_for_confirmation = True
        self.confirm_text.config(text=f"Detected: {behavior}\nConfidence: {confidence:.1%}\nIs this correct?")
        self.confirm_frame.pack(fill='x', padx=10, pady=5)

        # Update detection text
        self.update_detection_display(behavior, confidence)

    def confirm_behavior(self):
        """User confirmed the detection is correct"""
        self.hide_confirmation_dialog()
        self.give_treat_and_praise()
        self.add_to_history(f"✅ Correct: {self.detected_behavior}")

    def reject_behavior(self):
        """User said the detection is wrong"""
        self.hide_confirmation_dialog()
        self.add_to_history(f"❌ Wrong detection: {self.detected_behavior}")

        # Play correction sound
        if self.robot.hardware.get('audio'):
            self.robot.hardware['audio'].play_sound('NO')

    def hide_confirmation_dialog(self):
        """Hide the confirmation dialog"""
        self.waiting_for_confirmation = False
        self.confirm_frame.pack_forget()
        self.status_label.config(text="🟢 Ready", fg='green')

    def give_treat_and_praise(self):
        """Give treat and positive audio feedback"""
        try:
            # Play good dog sound
            if self.robot.hardware.get('audio'):
                self.robot.hardware['audio'].play_sound('GOOD_DOG')

            # Dispense treat
            if hasattr(self.robot, 'reward_system'):
                self.robot.reward_system.manual_dispense("manual_training")

            # Set celebration LEDs
            if self.robot.hardware.get('leds'):
                self.robot.hardware['leds'].set_mode(LEDMode.TREAT_LAUNCHING)

            self.add_to_history("🎉 Treat dispensed!")

        except Exception as e:
            self.logger.error(f"Treat dispensing error: {e}")

    def manual_command(self, command):
        """Manually trigger a training command"""
        self.add_to_history(f"Manual: {command}")

        if command == 'good_dog':
            self.give_treat_and_praise()
        else:
            self.initiate_training_sequence(command)

    def center_camera(self):
        """Center the camera servos"""
        self.pan_scale.set(0)
        self.tilt_scale.set(0)
        if self.robot.hardware.get('servos'):
            self.robot.hardware['servos'].center_camera()

    def update_pan(self, value):
        """Update pan servo position"""
        angle = float(value)
        if self.robot.hardware.get('servos'):
            try:
                # Assuming servo controller has set_camera_pan method
                if hasattr(self.robot.hardware['servos'], 'set_camera_pan'):
                    self.robot.hardware['servos'].set_camera_pan(angle)
            except Exception as e:
                self.logger.error(f"Pan update error: {e}")

    def update_tilt(self, value):
        """Update tilt servo position"""
        angle = float(value)
        if self.robot.hardware.get('servos'):
            try:
                # Assuming servo controller has set_camera_tilt method
                if hasattr(self.robot.hardware['servos'], 'set_camera_tilt'):
                    self.robot.hardware['servos'].set_camera_tilt(angle)
            except Exception as e:
                self.logger.error(f"Tilt update error: {e}")

    def toggle_overlay(self):
        """Toggle detection overlay on/off"""
        self.detection_overlay = not self.detection_overlay

    def update_detection_display(self, behavior, confidence):
        """Update the detection information display"""
        info = f"Timestamp: {datetime.now().strftime('%H:%M:%S')}\n"
        info += f"Behavior: {behavior}\n"
        info += f"Confidence: {confidence:.1%}\n"
        info += f"Command: {self.current_command or 'None'}\n"

        self.detection_text.delete(1.0, tk.END)
        self.detection_text.insert(1.0, info)

    def add_to_history(self, message):
        """Add message to command history"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] {message}\n"

        self.command_history.append(entry)
        if len(self.command_history) > 10:  # Keep last 10 entries
            self.command_history.pop(0)

        # Update history display
        self.history_text.delete(1.0, tk.END)
        self.history_text.insert(1.0, ''.join(self.command_history[-5:]))  # Show last 5

    def run(self):
        """Start the interactive training GUI"""
        # Set robot to tracking mode for detection
        self.robot.set_mode(RobotMode.TRACKING)

        # Start GUI
        self.root.mainloop()

    def cleanup(self):
        """Clean up resources"""
        self.listening = False

def main():
    """Main function to run interactive training"""
    try:
        # Initialize robot
        print("🔧 Initializing robot systems...")
        robot = TreatDispenserRobot()

        if not robot.initialization_successful:
            print("❌ Robot initialization failed!")
            return 1

        print("🎯 Starting Interactive Training Mode...")

        # Create and run GUI
        gui = InteractiveTrainingGUI(robot)
        gui.run()

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training error: {e}")
        logging.error(f"Training error: {e}")
    finally:
        if 'robot' in locals():
            robot.cleanup()
        if 'gui' in locals():
            gui.cleanup()

if __name__ == "__main__":
    main()