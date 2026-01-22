#!/usr/bin/env python3
"""
services/network/wifi_provisioning.py - WiFi provisioning orchestrator

Main orchestration flow for WiFi provisioning:
1. Check for existing WiFi connections
2. Try to connect to known networks
3. If no connection, start AP mode with captive portal
4. When credentials saved, reboot to connect
"""

import os
import sys
import time
import asyncio
import logging
import signal
import threading
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.network.wifi_manager import WiFiManager
from services.network.captive_portal import CaptivePortal

logger = logging.getLogger(__name__)


class WiFiProvisioningService:
    """WiFi provisioning orchestrator"""

    HOTSPOT_PASSWORD = "wimzsetup"
    CONNECTION_TIMEOUT = 30  # seconds to wait for known WiFi

    def __init__(self):
        self.wifi_manager = WiFiManager()
        self.captive_portal: Optional[CaptivePortal] = None
        self.led_controller = None
        self._running = False
        self._credentials_saved = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    def _init_led_controller(self):
        """Initialize LED controller for status indication"""
        try:
            from core.hardware.led_controller import LEDController, LEDMode
            from config.settings import Colors
            self.led_controller = LEDController()
            self.LEDMode = LEDMode
            self.Colors = Colors
            logger.info("LED controller initialized")
            return True
        except Exception as e:
            logger.warning(f"Could not initialize LED controller: {e}")
            self.led_controller = None
            return False

    def _set_led_searching(self):
        """Set LED pattern for searching/connecting"""
        if self.led_controller:
            try:
                self.led_controller.set_mode(self.LEDMode.SEARCHING)
            except Exception as e:
                logger.warning(f"LED error: {e}")

    def _set_led_ap_mode(self):
        """Set LED pattern for AP mode (pulsing blue)"""
        if self.led_controller:
            try:
                # Pulse blue for AP mode
                self.led_controller.start_animation(
                    self.led_controller.pulse_color,
                    self.Colors.BLUE,
                    20,
                    0.05
                )
            except Exception as e:
                logger.warning(f"LED error: {e}")

    def _set_led_connected(self):
        """Set LED pattern for connected (solid green)"""
        if self.led_controller:
            try:
                self.led_controller.stop_animation()
                self.led_controller.set_solid_color(self.Colors.GREEN)
            except Exception as e:
                logger.warning(f"LED error: {e}")

    def _set_led_error(self):
        """Set LED pattern for error"""
        if self.led_controller:
            try:
                self.led_controller.set_mode(self.LEDMode.ERROR)
            except Exception as e:
                logger.warning(f"LED error: {e}")

    def _cleanup_led(self):
        """Cleanup LED controller"""
        if self.led_controller:
            try:
                self.led_controller.cleanup()
            except Exception as e:
                logger.warning(f"LED cleanup error: {e}")

    def _generate_hotspot_ssid(self) -> str:
        """Generate hotspot SSID using device serial"""
        serial = self.wifi_manager.get_device_serial()
        return f"WIMZ-{serial}"

    def _on_credentials_saved(self, ssid: str):
        """Callback when WiFi credentials are saved"""
        logger.info(f"Credentials saved for {ssid}, preparing to reboot...")
        self._credentials_saved = True

    def _reboot_system(self):
        """Reboot the system"""
        logger.info("Rebooting system...")
        time.sleep(2)
        os.system("sudo reboot")

    def run(self) -> bool:
        """
        Main provisioning flow.
        Returns True if WiFi is connected, False otherwise.
        """
        self._running = True
        logger.info("=" * 50)
        logger.info("WIM-Z WiFi Provisioning Service Starting")
        logger.info("=" * 50)

        # Initialize LED controller
        self._init_led_controller()

        try:
            # Step 1: Set LED to searching pattern
            self._set_led_searching()

            # Step 2: Check if already connected
            if self.wifi_manager.is_connected():
                status = self.wifi_manager.get_connection_status()
                logger.info(f"Already connected to: {status['ssid']} ({status['ip_address']})")
                self._set_led_connected()
                return True

            # Step 3: Check for saved connections
            saved_connections = self.wifi_manager.get_saved_connections()
            logger.info(f"Found {len(saved_connections)} saved WiFi connections")

            # Step 4: Try to connect to known networks
            if saved_connections:
                logger.info(f"Attempting to connect to known networks (timeout: {self.CONNECTION_TIMEOUT}s)...")
                if self.wifi_manager.try_connect_known(timeout=self.CONNECTION_TIMEOUT):
                    status = self.wifi_manager.get_connection_status()
                    logger.info(f"Connected to: {status['ssid']} ({status['ip_address']})")
                    self._set_led_connected()
                    return True
                logger.info("Could not connect to any known networks")

            # Step 5: No connection - start AP mode
            logger.info("Starting WiFi provisioning AP mode...")
            self._start_ap_mode()

            return False

        except Exception as e:
            logger.error(f"Provisioning error: {e}")
            self._set_led_error()
            return False

        finally:
            self._cleanup_led()

    def _start_ap_mode(self):
        """Start AP mode with captive portal"""
        # Generate SSID
        ssid = self._generate_hotspot_ssid()
        password = self.HOTSPOT_PASSWORD

        logger.info(f"Starting hotspot: {ssid}")
        logger.info(f"Hotspot password: {password}")
        logger.info(f"Portal address: http://192.168.4.1")

        # Set LED to AP mode pattern
        self._set_led_ap_mode()

        # Start hotspot
        if not self.wifi_manager.start_hotspot(ssid, password):
            logger.error("Failed to start hotspot")
            self._set_led_error()
            return

        # Create and start captive portal
        self.captive_portal = CaptivePortal(
            self.wifi_manager,
            on_credentials_saved=self._on_credentials_saved
        )

        logger.info("Captive portal starting on port 80...")
        logger.info("=" * 50)
        logger.info(f"Connect to WiFi: {ssid}")
        logger.info(f"Password: {password}")
        logger.info(f"Then open: http://192.168.4.1")
        logger.info("=" * 50)

        # Run captive portal (blocking)
        try:
            self.captive_portal.run(host="0.0.0.0", port=80)
        except Exception as e:
            logger.error(f"Captive portal error: {e}")
        finally:
            # Cleanup hotspot if still running
            self.wifi_manager.stop_hotspot()

    async def run_async(self) -> bool:
        """
        Async version of main provisioning flow.
        """
        self._running = True
        logger.info("=" * 50)
        logger.info("WIM-Z WiFi Provisioning Service Starting (async)")
        logger.info("=" * 50)

        # Initialize LED controller
        self._init_led_controller()

        try:
            # Set LED to searching pattern
            self._set_led_searching()

            # Check if already connected
            if self.wifi_manager.is_connected():
                status = self.wifi_manager.get_connection_status()
                logger.info(f"Already connected to: {status['ssid']} ({status['ip_address']})")
                self._set_led_connected()
                return True

            # Check for saved connections
            saved_connections = self.wifi_manager.get_saved_connections()
            logger.info(f"Found {len(saved_connections)} saved WiFi connections")

            # Try to connect to known networks
            if saved_connections:
                logger.info(f"Attempting to connect (timeout: {self.CONNECTION_TIMEOUT}s)...")

                # Run blocking connect in thread pool
                loop = asyncio.get_event_loop()
                connected = await loop.run_in_executor(
                    None,
                    self.wifi_manager.try_connect_known,
                    self.CONNECTION_TIMEOUT
                )

                if connected:
                    status = self.wifi_manager.get_connection_status()
                    logger.info(f"Connected to: {status['ssid']} ({status['ip_address']})")
                    self._set_led_connected()
                    return True

                logger.info("Could not connect to any known networks")

            # No connection - start AP mode
            logger.info("Starting WiFi provisioning AP mode...")
            await self._start_ap_mode_async()

            return False

        except Exception as e:
            logger.error(f"Provisioning error: {e}")
            self._set_led_error()
            return False

        finally:
            self._cleanup_led()

    async def _start_ap_mode_async(self):
        """Start AP mode with captive portal (async)"""
        # Generate SSID
        ssid = self._generate_hotspot_ssid()
        password = self.HOTSPOT_PASSWORD

        logger.info(f"Starting hotspot: {ssid}")

        # Set LED to AP mode pattern
        self._set_led_ap_mode()

        # Start hotspot
        loop = asyncio.get_event_loop()
        started = await loop.run_in_executor(
            None,
            self.wifi_manager.start_hotspot,
            ssid,
            password
        )

        if not started:
            logger.error("Failed to start hotspot")
            self._set_led_error()
            return

        # Create and start captive portal
        self.captive_portal = CaptivePortal(
            self.wifi_manager,
            on_credentials_saved=self._on_credentials_saved
        )

        logger.info("Captive portal starting on port 80...")
        logger.info("=" * 50)
        logger.info(f"Connect to WiFi: {ssid}")
        logger.info(f"Password: {password}")
        logger.info(f"Then open: http://192.168.4.1")
        logger.info("=" * 50)

        try:
            await self.captive_portal.start_async(host="0.0.0.0", port=80)
        except Exception as e:
            logger.error(f"Captive portal error: {e}")
        finally:
            # Cleanup hotspot
            await loop.run_in_executor(None, self.wifi_manager.stop_hotspot)


# Test function
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    service = WiFiProvisioningService()
    result = service.run()
    sys.exit(0 if result else 1)
