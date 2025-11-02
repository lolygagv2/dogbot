#!/usr/bin/env python3
"""
Xbox Controller Service for WIM-Z
Integrates working xbox_hybrid_controller.py with event-driven architecture
"""

import subprocess
import os
import time
import threading
import logging
from typing import Optional, Dict, Any
import signal

# Core imports
from core.bus import get_bus, publish_system_event
from core.state import get_state, SystemMode

logger = logging.getLogger(__name__)


class XboxControllerService:
    """
    Xbox Controller Service for WIM-Z
    Manages the working xbox_hybrid_controller.py process and integrates with event system

    Features:
    - Auto-detection and reconnection
    - Manual input timeout (120 seconds)
    - Event bus integration
    - Mode management
    """

    def __init__(self):
        """Initialize Xbox controller service"""
        self.bus = get_bus()
        self.state = get_state()

        # Service state
        self.running = False
        self.monitor_thread = None
        self.controller_process = None
        self.last_device_activity = 0

        # Controller state tracking
        self.is_connected = False
        self.last_manual_input_time = 0.0
        self.manual_timeout = 120.0  # 2 minutes

        # Device paths
        self.device_path = '/dev/input/js0'
        self.controller_script = '/home/morgan/dogbot/xbox_hybrid_controller.py'

        logger.info("Xbox Controller Service initialized")

    def _check_controller_connection(self) -> bool:
        """Check if Xbox controller device exists"""
        return os.path.exists(self.device_path)

    def start(self) -> bool:
        """Start Xbox controller service"""
        try:
            self.running = True

            # Start monitoring thread
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="XboxControllerMonitor"
            )
            self.monitor_thread.start()

            logger.info("Xbox Controller Service started")
            return True

        except Exception as e:
            logger.error(f"Failed to start Xbox controller service: {e}")
            return False

    def stop(self):
        """Stop Xbox controller service"""
        logger.info("Stopping Xbox Controller Service...")

        self.running = False

        # Stop controller process
        self._stop_controller_process()

        # Wait for monitor thread
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

        logger.info("Xbox Controller Service stopped")

    def _monitor_loop(self):
        """Monitor Xbox controller connection and manage process"""
        logger.info("Xbox controller monitor started")

        while self.running:
            try:
                # Check controller connection
                controller_available = self._check_controller_connection()

                if controller_available and not self.is_connected:
                    self._on_controller_connected()
                elif not controller_available and self.is_connected:
                    self._on_controller_disconnected()

                # Monitor controller process if connected
                if self.is_connected:
                    self._monitor_controller_process()

                # Monitor device activity for manual input detection
                if self.is_connected:
                    self._check_device_activity()

                # Check manual input timeout
                self._check_manual_timeout()

                time.sleep(2.0)  # Check every 2 seconds

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")
                time.sleep(5.0)

        logger.info("Xbox controller monitor stopped")

    def _on_controller_connected(self):
        """Handle controller connection"""
        self.is_connected = True
        logger.info("Xbox controller connected")

        # Start controller process
        self._start_controller_process()

        # Publish connection event
        publish_system_event('controller_connected', {
            'device': 'xbox_controller',
            'path': self.device_path
        }, source='xbox_service')

    def _on_controller_disconnected(self):
        """Handle controller disconnection"""
        self.is_connected = False
        logger.info("Xbox controller disconnected")

        # Stop controller process
        self._stop_controller_process()

        # Publish disconnection event
        publish_system_event('controller_disconnected', {
            'device': 'xbox_controller'
        }, source='xbox_service')

    def _start_controller_process(self):
        """Start the Xbox hybrid controller process"""
        try:
            # Start the working xbox_hybrid_controller.py
            self.controller_process = subprocess.Popen(
                ['python3', self.controller_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )

            logger.info(f"Xbox controller process started (PID: {self.controller_process.pid})")

        except Exception as e:
            logger.error(f"Failed to start controller process: {e}")

    def _stop_controller_process(self):
        """Stop the Xbox controller process"""
        if self.controller_process:
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(self.controller_process.pid), signal.SIGTERM)
                self.controller_process.wait(timeout=5.0)
                logger.info("Xbox controller process stopped")
            except Exception as e:
                logger.error(f"Error stopping controller process: {e}")
                try:
                    # Force kill if needed
                    os.killpg(os.getpgid(self.controller_process.pid), signal.SIGKILL)
                except:
                    pass
            finally:
                self.controller_process = None

    def _monitor_controller_process(self):
        """Monitor if controller process is still running"""
        if self.controller_process:
            if self.controller_process.poll() is not None:
                # Process has terminated
                logger.warning("Xbox controller process terminated unexpectedly")
                self.controller_process = None

                # Try to restart if controller is still connected
                if self._check_controller_connection():
                    logger.info("Attempting to restart controller process...")
                    time.sleep(2.0)  # Brief delay before restart
                    self._start_controller_process()

    def _check_manual_timeout(self):
        """Check for manual input timeout and auto-switch to autonomous"""
        if self.last_manual_input_time == 0:
            return  # No manual input yet

        current_time = time.time()
        time_since_input = current_time - self.last_manual_input_time

        # Check if in manual mode and timeout exceeded
        current_mode = self.state.get_mode()
        if (current_mode == SystemMode.MANUAL and
            time_since_input > self.manual_timeout):

            logger.info(f"Manual timeout ({self.manual_timeout}s), switching to autonomous")

            # Switch to appropriate autonomous mode
            if self.is_connected:
                self.state.set_mode(SystemMode.VIGILANT, "Manual timeout - controller connected")
            else:
                self.state.set_mode(SystemMode.IDLE, "Manual timeout - no controller")

            # Publish timeout event
            publish_system_event('manual_timeout', {
                'timeout_seconds': self.manual_timeout,
                'last_input_time': self.last_manual_input_time
            }, source='xbox_service')

    def _check_device_activity(self):
        """Monitor device file for controller activity"""
        try:
            if os.path.exists(self.device_path):
                device_stat = os.stat(self.device_path)
                current_activity = device_stat.st_mtime  # Use mtime instead of atime

                if current_activity > self.last_device_activity:
                    # Device has been accessed (controller input detected)
                    self.on_manual_input()
                    self.last_device_activity = current_activity
                    logger.debug("Xbox controller activity detected")
        except Exception as e:
            logger.debug(f"Device activity check error: {e}")

    def on_manual_input(self):
        """Called when manual input is detected (external API)"""
        self.last_manual_input_time = time.time()

        # Switch to manual mode if not already
        current_mode = self.state.get_mode()
        if current_mode != SystemMode.MANUAL:
            self.state.set_mode(SystemMode.MANUAL, "Xbox controller input")
            logger.info("Switched to MANUAL mode due to controller input")

        # Publish manual input event
        publish_system_event('manual_input_detected', {
            'timestamp': self.last_manual_input_time,
            'source': 'xbox_controller'
        }, source='xbox_service')

    def get_status(self) -> Dict[str, Any]:
        """Get controller service status"""
        return {
            'service_running': self.running,
            'controller_connected': self.is_connected,
            'process_running': self.controller_process is not None and self.controller_process.poll() is None,
            'last_manual_input_time': self.last_manual_input_time,
            'manual_timeout': self.manual_timeout,
            'device_path': self.device_path,
            'controller_script': self.controller_script
        }

    def cleanup(self):
        """Clean up resources"""
        self.stop()


# Global service instance
_xbox_service_instance = None
_xbox_service_lock = threading.Lock()

def get_xbox_service() -> XboxControllerService:
    """Get the global Xbox controller service instance (singleton)"""
    global _xbox_service_instance
    if _xbox_service_instance is None:
        with _xbox_service_lock:
            if _xbox_service_instance is None:
                _xbox_service_instance = XboxControllerService()
    return _xbox_service_instance