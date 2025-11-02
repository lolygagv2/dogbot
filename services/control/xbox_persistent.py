#!/usr/bin/env python3
"""
Xbox controller handler with persistent connection management
Maintains connection while reading input
"""

import subprocess
import time
import threading
import pygame
import logging
from typing import Optional, Callable
import os
import signal

logger = logging.getLogger(__name__)

class XboxControllerPersistent:
    def __init__(self, mac_address: str = "AC:8E:BD:4A:0F:97"):
        self.mac = mac_address
        self.controller = None
        self.running = False
        self.connected = False
        self.connection_thread = None
        self.read_thread = None
        self.callback = None

        # Initialize pygame
        pygame.init()
        pygame.joystick.init()

    def start(self, callback: Optional[Callable] = None):
        """Start the controller service"""
        self.callback = callback
        self.running = True

        # Start connection manager thread
        self.connection_thread = threading.Thread(target=self._connection_manager)
        self.connection_thread.daemon = True
        self.connection_thread.start()

        # Start input reader thread
        self.read_thread = threading.Thread(target=self._input_reader)
        self.read_thread.daemon = True
        self.read_thread.start()

        logger.info("Xbox controller service started")

    def stop(self):
        """Stop the controller service"""
        self.running = False
        if self.connection_thread:
            self.connection_thread.join(timeout=2)
        if self.read_thread:
            self.read_thread.join(timeout=2)
        pygame.quit()
        logger.info("Xbox controller service stopped")

    def _connection_manager(self):
        """Thread to manage controller connection"""
        while self.running:
            try:
                # Check if connected
                result = subprocess.run(
                    f"bluetoothctl info {self.mac} | grep 'Connected: yes'",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=2
                )

                if result.returncode != 0:
                    # Not connected, try to connect
                    logger.info("Controller disconnected, attempting reconnection...")

                    # Use timeout command to prevent hanging
                    subprocess.run(
                        f"timeout 3 bluetoothctl connect {self.mac}",
                        shell=True,
                        capture_output=True,
                        timeout=4
                    )

                    time.sleep(2)

                    # Check if connection succeeded
                    result = subprocess.run(
                        f"bluetoothctl info {self.mac} | grep 'Connected: yes'",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=2
                    )

                    if result.returncode == 0:
                        logger.info("Controller reconnected successfully")
                        self.connected = True
                        self._init_pygame_controller()
                else:
                    if not self.connected:
                        self.connected = True
                        logger.info("Controller is connected")
                        self._init_pygame_controller()

            except Exception as e:
                logger.error(f"Connection manager error: {e}")

            # Check every 5 seconds
            time.sleep(5)

    def _init_pygame_controller(self):
        """Initialize pygame controller after connection"""
        try:
            # Re-scan for controllers
            pygame.joystick.quit()
            pygame.joystick.init()

            if pygame.joystick.get_count() > 0:
                self.controller = pygame.joystick.Joystick(0)
                self.controller.init()
                logger.info(f"Pygame controller initialized: {self.controller.get_name()}")
            else:
                logger.warning("No pygame controllers detected despite connection")

        except Exception as e:
            logger.error(f"Failed to initialize pygame controller: {e}")

    def _input_reader(self):
        """Thread to read controller input"""
        clock = pygame.time.Clock()

        while self.running:
            try:
                if self.connected and self.controller:
                    # Process events
                    for event in pygame.event.get():
                        if event.type == pygame.JOYBUTTONDOWN:
                            logger.debug(f"Button {event.button} pressed")
                            if self.callback:
                                self.callback('button', event.button, 1)

                        elif event.type == pygame.JOYBUTTONUP:
                            logger.debug(f"Button {event.button} released")
                            if self.callback:
                                self.callback('button', event.button, 0)

                        elif event.type == pygame.JOYAXISMOTION:
                            if abs(event.value) > 0.1:  # Deadzone
                                logger.debug(f"Axis {event.axis}: {event.value:.2f}")
                                if self.callback:
                                    self.callback('axis', event.axis, event.value)

                clock.tick(30)

            except pygame.error as e:
                logger.warning(f"Pygame error (controller may have disconnected): {e}")
                self.connected = False
                self.controller = None
                time.sleep(1)

            except Exception as e:
                logger.error(f"Input reader error: {e}")
                time.sleep(1)


def test_callback(input_type, index, value):
    """Test callback to print controller input"""
    if input_type == 'button':
        state = "pressed" if value == 1 else "released"
        print(f"Button {index} {state}")

        # Map common buttons
        button_names = {
            0: "A", 1: "B", 2: "X", 3: "Y",
            4: "LB", 5: "RB", 6: "Back", 7: "Start"
        }
        if index in button_names:
            print(f"  -> {button_names[index]} button {state}")

    elif input_type == 'axis':
        axis_names = {
            0: "Left X", 1: "Left Y",
            3: "Right X", 4: "Right Y",
            2: "Left Trigger", 5: "Right Trigger"
        }
        name = axis_names.get(index, f"Axis {index}")
        print(f"{name}: {value:.2f}")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("Xbox Controller Persistent Service")
    print("="*40)
    print("This will maintain connection and read input")
    print("Press Ctrl+C to exit\n")

    # Create controller service
    controller = XboxControllerPersistent()

    # Handle Ctrl+C
    def signal_handler(sig, frame):
        print("\n\nShutting down...")
        controller.stop()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start the service
    controller.start(callback=test_callback)

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass

    controller.stop()