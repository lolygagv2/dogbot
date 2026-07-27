#!/usr/bin/env python3
"""
services/network/wifi_provisioning.py - WiFi provisioning orchestrator

Bounded boot job:
1. Boot → try saved WiFi (15s timeout)
2. If no WiFi → start credential AP with captive portal (5 min timeout)
3. If credentials received → connect to WiFi → relay mode
4. Either way, EXIT. Runtime AP fallback (WiFi loss, app-commanded local
   mode) is owned solely by main_treatbot's WiFi monitor, which defers
   while this service is active. This process must never park holding
   hardware: it runs as root, and a lingering LedService here claims
   GPIO25 + writes the NeoPixel SPI strip in parallel with treatbot's.
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
    """WiFi provisioning orchestrator — bounded boot job, exits when done"""

    CONNECTION_TIMEOUT = 15  # seconds to wait for known WiFi
    CREDENTIAL_AP_TIMEOUT = 300  # 5 minutes for credential AP, then exit

    def __init__(self):
        self.wifi_manager = WiFiManager()
        self.captive_portal: Optional[CaptivePortal] = None
        self._led_service = None
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
        if self.captive_portal:
            self.captive_portal.stop()

    def _init_led_controller(self):
        """LEDs are intentionally NOT initialized here.

        This service runs as root in its own process; a LedService here is a
        SECOND writer on the NeoPixel SPI strip and an exclusive claim on
        GPIO25 that blocks treatbot's mood LED (the in-module "singleton"
        never spanned processes). treatbot owns all LEDs; the _set_led_*
        helpers below are no-ops while _led_service is None.
        """
        self._led_service = None
        return False

    def _set_led_searching(self):
        """Set LED pattern for searching/connecting"""
        if self._led_service:
            try:
                self._led_service.set_pattern('searching')
            except Exception as e:
                logger.warning(f"LED error: {e}")

    def _set_led_ap_mode(self):
        """Set LED pattern for AP mode (pulsing blue)"""
        if self._led_service:
            try:
                self._led_service.set_pattern('pulse_blue')
            except Exception as e:
                logger.warning(f"LED error: {e}")

    def _set_led_connected(self):
        """Set LED pattern for connected (solid green)"""
        if self._led_service:
            try:
                self._led_service.set_pattern('solid_green')
            except Exception as e:
                logger.warning(f"LED error: {e}")

    def _set_led_error(self):
        """Set LED pattern for error"""
        if self._led_service:
            try:
                self._led_service.set_pattern('error')
            except Exception as e:
                logger.warning(f"LED error: {e}")

    def _cleanup_led(self):
        """Release LED to idle (singleton persists for main system)"""
        # Don't cleanup the singleton - just set to idle so main system can take over
        if self._led_service:
            try:
                self._led_service.set_pattern('idle')
                logger.info("LED set to idle (singleton persists)")
            except Exception as e:
                logger.warning(f"LED release error: {e}")
        self._led_service = None

    def _generate_hotspot_ssid(self) -> str:
        """The single fleet-wide AP SSID (shared with every runtime AP)."""
        return self.wifi_manager.get_ap_ssid()

    def _on_credentials_saved(self, ssid: str):
        """Callback when WiFi credentials are saved"""
        logger.info(f"Credentials saved for {ssid}, preparing to reboot...")
        self._credentials_saved = True

    def _reboot_system(self):
        """Reboot the system"""
        logger.info("Rebooting system...")
        time.sleep(2)
        os.system("sudo reboot")

    # ── Main provisioning flow ───────────────────────────────────────

    def run(self) -> bool:
        """
        Main provisioning flow (bounded — always returns, process then exits).
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

            # Step 5: No connection — start credential AP with 5-min timeout
            logger.info("Starting WiFi provisioning AP mode...")
            credentials_received = self._start_credential_ap_with_timeout()

            if credentials_received:
                # Credentials were saved — reboot to reconnect
                return True

            # Step 6: Credential window closed with no credentials. Exit —
            # treatbot's WiFi monitor owns all runtime AP fallback and will
            # raise the same SSID once this service is no longer active.
            logger.info("[LOCAL] Credential window closed — exiting; "
                        "treatbot WiFi monitor owns AP fallback from here")
            return False

        except Exception as e:
            logger.error(f"Provisioning error: {e}")
            self._set_led_error()
            return False

        finally:
            self._cleanup_led()

    def _start_credential_ap_with_timeout(self) -> bool:
        """Start credential AP + captive portal with a 5-minute timeout.

        Returns True if credentials were saved, False if timed out.
        """
        ssid = self._generate_hotspot_ssid()
        password = self.wifi_manager.AP_PASSWORD
        portal_ip = self.wifi_manager.HOTSPOT_IP

        logger.info(f"[LOCAL] Credential AP started — waiting {self.CREDENTIAL_AP_TIMEOUT // 60} min for WiFi setup")

        # Set LED to AP mode pattern
        self._set_led_ap_mode()

        # Start hotspot (hostapd + dnsmasq with DNS hijack)
        if not self.wifi_manager.start_hotspot(ssid, password):
            logger.error("Failed to start credential hotspot")
            self._set_led_error()
            return False

        # Create captive portal
        self.captive_portal = CaptivePortal(
            self.wifi_manager,
            on_credentials_saved=self._on_credentials_saved
        )

        logger.info("=" * 50)
        logger.info(f"Connect to WiFi: {ssid}")
        logger.info(f"Password: {password}")
        logger.info(f"Then open: http://{portal_ip}")
        logger.info("=" * 50)

        # Run captive portal in a background thread so we can timeout
        portal_thread = threading.Thread(
            target=self.captive_portal.run,
            kwargs={"host": "0.0.0.0", "port": 80},
            daemon=True,
            name="captive-portal"
        )
        portal_thread.start()

        # Wait for credentials or timeout
        deadline = time.time() + self.CREDENTIAL_AP_TIMEOUT
        while time.time() < deadline and self._running:
            if self._credentials_saved:
                logger.info("Credentials received during AP mode")
                return True
            time.sleep(1)

        if not self._running:
            # Service is shutting down
            self.captive_portal.stop()
            self.wifi_manager.stop_hotspot()
            return False

        # Timeout — shut down credential AP completely
        logger.info("[LOCAL] Credential AP timeout — no credentials received, shutting down")
        self.captive_portal.stop()
        time.sleep(1)  # Let uvicorn shut down
        self.wifi_manager.stop_hotspot()
        time.sleep(1)  # Let interface settle

        return False

    # ── Async variants (for future use) ─────────────────────────────

    async def run_async(self) -> bool:
        """Async version of main provisioning flow."""
        self._running = True
        logger.info("=" * 50)
        logger.info("WIM-Z WiFi Provisioning Service Starting (async)")
        logger.info("=" * 50)

        # Initialize LED controller
        self._init_led_controller()

        try:
            self._set_led_searching()

            # Check if already connected
            if self.wifi_manager.is_connected():
                status = self.wifi_manager.get_connection_status()
                logger.info(f"Already connected to: {status['ssid']} ({status['ip_address']})")
                self._set_led_connected()
                return True

            # Try saved connections
            saved_connections = self.wifi_manager.get_saved_connections()
            logger.info(f"Found {len(saved_connections)} saved WiFi connections")

            if saved_connections:
                logger.info(f"Attempting to connect (timeout: {self.CONNECTION_TIMEOUT}s)...")
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

            # No connection — run the credential AP window, then exit
            logger.info("Starting WiFi provisioning AP mode...")
            loop = asyncio.get_event_loop()
            credentials_received = await loop.run_in_executor(
                None,
                self._start_credential_ap_with_timeout
            )

            if credentials_received:
                return True

            logger.info("[LOCAL] Credential window closed — exiting; "
                        "treatbot WiFi monitor owns AP fallback from here")
            return False

        except Exception as e:
            logger.error(f"Provisioning error: {e}")
            self._set_led_error()
            return False

        finally:
            self._cleanup_led()


# Test function
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    service = WiFiProvisioningService()
    result = service.run()
    sys.exit(0 if result else 1)
