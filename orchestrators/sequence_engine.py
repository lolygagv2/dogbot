#!/usr/bin/env python3
"""
Sequence engine for executing celebration sequences
Coordinates LED, audio, treat timing based on YAML files
"""

import threading
import time
import logging
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path

from core.bus import get_bus, publish_system_event
from core.state import get_state
from services.media.sfx import get_sfx_service
from services.media.led import get_led_service
from services.reward.dispenser import get_dispenser_service
from services.motion.pan_tilt import get_pantilt_service


class SequenceEngine:
    """
    Executes celebration and system sequences
    Parses YAML sequence files and coordinates multiple services
    """

    def __init__(self):
        self.bus = get_bus()
        self.state = get_state()
        self.sfx = get_sfx_service()
        self.led = get_led_service()
        self.dispenser = get_dispenser_service()
        self.pantilt = get_pantilt_service()
        self.logger = logging.getLogger('SequenceEngine')

        # Sequence state
        self.active_sequences = {}  # sequence_id -> thread
        self.sequence_counter = 0
        self._lock = threading.Lock()

        # Built-in sequences (fallbacks if YAML files missing)
        self.builtin_sequences = {
            'celebrate': self._builtin_celebrate_sequence,
            'startup': self._builtin_startup_sequence,
            'shutdown': self._builtin_shutdown_sequence,
            'error': self._builtin_error_sequence
        }

    def list_sequences(self) -> List[str]:
        """List available sequences"""
        sequences = list(self.builtin_sequences.keys())

        # Check for YAML sequence files
        sequence_dir = Path('/home/morgan/dogbot/configs/sequences')
        if sequence_dir.exists():
            for yaml_file in sequence_dir.glob('*.yaml'):
                sequence_name = yaml_file.stem
                if sequence_name not in sequences:
                    sequences.append(sequence_name)

        return sequences

    def load_sequence(self, sequence_path: str) -> Optional[Dict[str, Any]]:
        """Load sequence from YAML file"""
        try:
            with open(sequence_path, 'r') as file:
                sequence = yaml.safe_load(file)
                self.logger.info(f"Loaded sequence: {sequence_path}")
                return sequence

        except FileNotFoundError:
            self.logger.warning(f"Sequence file not found: {sequence_path}")
            return None
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parse error in {sequence_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error loading sequence {sequence_path}: {e}")
            return None

    def execute_sequence(self, sequence_name: str, context: Dict[str, Any] = None,
                        interrupt: bool = False) -> Optional[str]:
        """
        Execute a sequence by name

        Args:
            sequence_name: Name of sequence to execute
            context: Context data for sequence (dog_id, behavior, etc.)
            interrupt: Whether to interrupt other sequences

        Returns:
            str: Sequence ID if started successfully
        """
        with self._lock:
            # Generate sequence ID
            self.sequence_counter += 1
            sequence_id = f"{sequence_name}_{self.sequence_counter}"

            context = context or {}

            # Try to load from file first
            sequence_path = f"/home/morgan/dogbot/configs/sequences/{sequence_name}.yaml"
            sequence_data = self.load_sequence(sequence_path)

            if sequence_data:
                # Execute YAML sequence
                thread = threading.Thread(
                    target=self._execute_yaml_sequence,
                    args=(sequence_id, sequence_data, context),
                    daemon=True,
                    name=f"Sequence_{sequence_name}"
                )
            elif sequence_name in self.builtin_sequences:
                # Execute built-in sequence
                thread = threading.Thread(
                    target=self._execute_builtin_sequence,
                    args=(sequence_id, sequence_name, context),
                    daemon=True,
                    name=f"Builtin_{sequence_name}"
                )
            else:
                self.logger.error(f"Unknown sequence: {sequence_name}")
                return None

            # Stop conflicting sequences if interrupt requested
            if interrupt:
                self._stop_all_sequences()

            # Start sequence
            self.active_sequences[sequence_id] = thread
            thread.start()

            # Publish event
            publish_system_event('sequence_started', {
                'sequence_id': sequence_id,
                'sequence_name': sequence_name,
                'context': context,
                'interrupt': interrupt
            }, 'sequence_engine')

            self.logger.info(f"Started sequence: {sequence_name} ({sequence_id})")
            return sequence_id

    def _execute_yaml_sequence(self, sequence_id: str, sequence_data: Dict[str, Any],
                              context: Dict[str, Any]) -> None:
        """Execute sequence from YAML data"""
        try:
            sequence_name = sequence_data.get('name', 'unknown')
            steps = sequence_data.get('steps', [])

            self.logger.info(f"Executing YAML sequence: {sequence_name}")

            for i, step in enumerate(steps):
                if sequence_id not in self.active_sequences:
                    break  # Sequence was stopped

                step_type = step.get('type', 'action')

                if step_type == 'action':
                    self._execute_action_step(step, context)
                elif step_type == 'parallel':
                    self._execute_parallel_step(step, context)
                elif step_type == 'wait':
                    self._execute_wait_step(step)
                else:
                    self.logger.warning(f"Unknown step type: {step_type}")

        except Exception as e:
            self.logger.error(f"YAML sequence execution error: {e}")

        finally:
            self._sequence_finished(sequence_id, sequence_name)

    def _execute_builtin_sequence(self, sequence_id: str, sequence_name: str,
                                 context: Dict[str, Any]) -> None:
        """Execute built-in sequence"""
        try:
            sequence_func = self.builtin_sequences[sequence_name]
            sequence_func(context)

        except Exception as e:
            self.logger.error(f"Built-in sequence execution error: {e}")

        finally:
            self._sequence_finished(sequence_id, sequence_name)

    def _execute_action_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Execute a single action step"""
        service = step.get('service')
        command = step.get('command')
        params = step.get('params', {})

        # Substitute context variables in params
        params = self._substitute_context(params, context)

        if service == 'audio':
            self._execute_audio_command(command, params)
        elif service == 'led':
            self._execute_led_command(command, params)
        elif service == 'treat':
            self._execute_treat_command(command, params)
        elif service == 'motion':
            self._execute_motion_command(command, params)
        elif service == 'photo':
            self._execute_photo_command(command, params, context)
        else:
            self.logger.warning(f"Unknown service: {service}")

    def _execute_parallel_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Execute multiple actions in parallel"""
        actions = step.get('actions', [])
        threads = []

        for action in actions:
            thread = threading.Thread(
                target=self._execute_action_step,
                args=(action, context),
                daemon=True
            )
            threads.append(thread)
            thread.start()

        # Wait for all parallel actions to complete
        for thread in threads:
            thread.join()

    def _execute_wait_step(self, step: Dict[str, Any]) -> None:
        """Execute wait step"""
        duration = step.get('duration', 1.0)
        time.sleep(duration)

    def _execute_audio_command(self, command: str, params: Dict[str, Any]) -> None:
        """Execute audio command"""
        if command == 'play':
            sound = params.get('sound', 'good_dog')
            volume = params.get('volume')
            self.sfx.play_sound(sound, volume)
        elif command == 'stop':
            self.sfx.stop_sound()
        elif command == 'volume':
            volume = params.get('volume', 75)
            self.sfx.set_volume(volume)

    def _execute_led_command(self, command: str, params: Dict[str, Any]) -> None:
        """Execute LED command"""
        if command == 'pattern':
            pattern = params.get('pattern', 'rainbow')
            duration = params.get('duration')
            color = params.get('color')
            self.led.set_pattern(pattern, duration, color)
        elif command == 'off':
            self.led.set_pattern('off')
        elif command == 'celebration':
            duration = params.get('duration', 5.0)
            self.led.celebration_sequence(duration)

    def _execute_treat_command(self, command: str, params: Dict[str, Any]) -> None:
        """Execute treat command"""
        if command == 'dispense':
            count = params.get('count', 1)
            dog_id = params.get('dog_id')
            reason = params.get('reason', 'sequence')

            if count == 1:
                self.dispenser.dispense_treat(dog_id, reason)
            else:
                self.dispenser.dispense_multiple(count, dog_id, reason)

    def _execute_motion_command(self, command: str, params: Dict[str, Any]) -> None:
        """Execute motion command"""
        # Check if pantilt service exists (might be disabled for Xbox controller)
        if not self.pantilt:
            self.logger.debug("PanTilt service disabled, skipping motion command")
            return

        if command == 'stop':
            self.pantilt.set_tracking_enabled(False)
        elif command == 'center':
            self.pantilt.center_camera()
        elif command == 'track':
            self.pantilt.set_tracking_enabled(True)

    def _execute_photo_command(self, command: str, params: Dict[str, Any],
                                context: Dict[str, Any]) -> None:
        """Execute photo capture command"""
        if command == 'capture':
            try:
                from core.vision.camera_manager import get_camera_manager
                camera = get_camera_manager()

                reason = params.get('reason', 'sequence')
                dog_id = context.get('dog_id', 'unknown')
                dog_name = context.get('dog_name', '')

                # Generate filename with context
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                prefix = f"{reason}_{dog_name or dog_id}_{timestamp}"

                # Capture photo
                filename = camera.capture_photo(
                    prefix=prefix,
                    directory="captures"
                )

                self.logger.info(f"Photo captured: {filename}")

            except ImportError:
                self.logger.warning("Camera manager not available for photo capture")
            except Exception as e:
                self.logger.error(f"Photo capture failed: {e}")

    def _substitute_context(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute context variables in parameters"""
        substituted = {}

        for key, value in params.items():
            if isinstance(value, str) and value.startswith('$'):
                # Context variable substitution
                var_name = value[1:]  # Remove $
                substituted[key] = context.get(var_name, value)
            else:
                substituted[key] = value

        return substituted

    def _sequence_finished(self, sequence_id: str, sequence_name: str) -> None:
        """Handle sequence completion"""
        with self._lock:
            if sequence_id in self.active_sequences:
                del self.active_sequences[sequence_id]

        publish_system_event('sequence_finished', {
            'sequence_id': sequence_id,
            'sequence_name': sequence_name
        }, 'sequence_engine')

        self.logger.info(f"Sequence finished: {sequence_name} ({sequence_id})")

    def stop_sequence(self, sequence_id: str) -> bool:
        """Stop a specific sequence"""
        with self._lock:
            if sequence_id in self.active_sequences:
                # Remove from active sequences (thread will detect and stop)
                del self.active_sequences[sequence_id]
                self.logger.info(f"Stopped sequence: {sequence_id}")
                return True
            return False

    def _stop_all_sequences(self) -> None:
        """Stop all active sequences"""
        with self._lock:
            sequence_ids = list(self.active_sequences.keys())
            for sequence_id in sequence_ids:
                del self.active_sequences[sequence_id]

            if sequence_ids:
                self.logger.info(f"Stopped {len(sequence_ids)} sequences")

    def stop_all_sequences(self) -> int:
        """Stop all active sequences (public method)"""
        with self._lock:
            count = len(self.active_sequences)
            self._stop_all_sequences()
            return count

    # Built-in sequences
    def _builtin_celebrate_sequence(self, context: Dict[str, Any]) -> None:
        """Built-in celebration sequence"""
        dog_id = context.get('dog_id')
        behavior = context.get('behavior', 'good')

        # Stop motion
        self.pantilt.set_tracking_enabled(False)

        # Start LED celebration and audio in parallel
        led_thread = threading.Thread(target=lambda: self.led.celebration_sequence(3.0))
        audio_thread = threading.Thread(target=lambda: self.sfx.play_celebration(behavior))

        led_thread.start()
        audio_thread.start()

        # Wait a moment
        time.sleep(0.5)

        # Dispense treat
        self.dispenser.dispense_treat(dog_id, 'celebration', behavior, context.get('confidence', 0.0))

        # Wait for celebration to finish
        led_thread.join()
        audio_thread.join()

        # Wait a bit more
        time.sleep(2.0)

        # Resume tracking
        self.pantilt.set_tracking_enabled(True)

    def _builtin_startup_sequence(self, context: Dict[str, Any]) -> None:
        """Built-in startup sequence"""
        # Play startup sound
        self.sfx.play_system_sound('startup')

        # LED startup pattern
        self.led.set_pattern('searching', 3.0)

        # Center camera (only if pantilt service is available)
        if self.pantilt:
            self.pantilt.center_camera()
        else:
            self.logger.debug("PanTilt disabled, skipping camera center")

        time.sleep(1.0)

        # Set to idle
        self.led.set_pattern('idle')

    def _builtin_shutdown_sequence(self, context: Dict[str, Any]) -> None:
        """Built-in shutdown sequence"""
        # Play shutdown sound
        self.sfx.play_system_sound('shutdown')

        # LED shutdown pattern
        self.led.set_pattern('error', 2.0)

        # Center camera (only if pantilt service is available)
        if self.pantilt:
            self.pantilt.center_camera()

        # Turn off LEDs
        time.sleep(2.0)
        self.led.set_pattern('off')

    def _builtin_error_sequence(self, context: Dict[str, Any]) -> None:
        """Built-in error sequence"""
        # Play error sound
        self.sfx.play_system_sound('error')

        # LED error pattern
        self.led.set_pattern('error', 5.0)

    def get_status(self) -> Dict[str, Any]:
        """Get sequence engine status"""
        with self._lock:
            return {
                'active_sequences': len(self.active_sequences),
                'sequence_ids': list(self.active_sequences.keys()),
                'sequence_counter': self.sequence_counter,
                'builtin_sequences': list(self.builtin_sequences.keys())
            }

    def cleanup(self) -> None:
        """Clean shutdown"""
        self.stop_all_sequences()
        self.logger.info("Sequence engine cleaned up")


# Global sequence engine instance
_sequence_instance = None
_sequence_lock = threading.Lock()

def get_sequence_engine() -> SequenceEngine:
    """Get the global sequence engine instance (singleton)"""
    global _sequence_instance
    if _sequence_instance is None:
        with _sequence_lock:
            if _sequence_instance is None:
                _sequence_instance = SequenceEngine()
    return _sequence_instance