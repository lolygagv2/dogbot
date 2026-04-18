#!/usr/bin/env python3
"""
services/network/wifi_provisioning.py - WiFi provisioning orchestrator

Two-phase AP system:
1. Boot → try saved WiFi (15s timeout)
2. If no WiFi → start credential AP with captive portal (5 min timeout)
3. If credentials received → connect to WiFi → relay mode
4. If 5 min timeout → shut down credential AP → start WIMZ-Demo AP
5. WIMZ-Demo runs permanently until power off or WiFi configured via API
"""

import os
import sys
import time
import asyncio
import logging
import signal
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.network.wifi_manager import WiFiManager
from services.network.captive_portal import CaptivePortal

logger = logging.getLogger(__name__)

# iOS/Android captive portal suppression: return "Success" so the OS
# thinks the network has internet and doesn't auto-disconnect or nag.
IOS_SUCCESS_HTML = b"<HTML><HEAD><TITLE>Success</TITLE></HEAD><BODY>Success</BODY></HTML>"


class _iOSSuccessHandler(BaseHTTPRequestHandler):
    """Tiny HTTP handler that tells iOS/Android 'this network has internet'."""

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(IOS_SUCCESS_HTML)))
        self.end_headers()
        self.wfile.write(IOS_SUCCESS_HTML)

    def log_message(self, format, *args):
        pass  # Suppress request logging


class WiFiProvisioningService:
    """WiFi provisioning orchestrator with two-phase AP system"""

    HOTSPOT_PASSWORD = "wimzsetup"
    CONNECTION_TIMEOUT = 15  # seconds to wait for known WiFi
    CREDENTIAL_AP_TIMEOUT = 300  # 5 minutes for credential AP before switching to demo

    DEMO_SSID_PREFIX = "WIMZ-Demo"
    DEMO_PASSWORD = "wimzdemo"

    def __init__(self):
        self.wifi_manager = WiFiManager()
        self.captive_portal: Optional[CaptivePortal] = None
        self._led_service = None
        self._running = False
        self._credentials_saved = False
        self._ios_server: Optional[HTTPServer] = None
        self._ios_thread: Optional[threading.Thread] = None
        self._in_demo_mode = False
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

    def _handle_signal(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False
        self._stop_ios_server()
        if self.captive_portal:
            self.captive_portal.stop()

    def _init_led_controller(self):
        """Initialize LED service (uses singleton to prevent conflicts with main system)"""
        try:
            # Use singleton LedService instead of creating separate LEDController
            # This prevents SPI bus conflicts when both wifi_provisioning and treatbot run
            from services.media.led import get_led_service
            self._led_service = get_led_service()
            if not self._led_service.led_initialized:
                self._led_service.initialize()
            logger.info("LED service initialized (singleton - no conflicts)")
            return True
        except Exception as e:
            logger.warning(f"Could not initialize LED service: {e}")
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
        """Generate hotspot SSID using device serial"""
        serial = self.wifi_manager.get_device_serial()
        return f"WIMZ-{serial}"

    def _generate_demo_ssid(self) -> str:
        """Generate per-unit demo SSID so TB1/TB2 don't collide."""
        serial = self.wifi_manager.get_device_serial()
        return f"{self.DEMO_SSID_PREFIX}-{serial}"

    def _on_credentials_saved(self, ssid: str):
        """Callback when WiFi credentials are saved"""
        logger.info(f"Credentials saved for {ssid}, preparing to reboot...")
        self._credentials_saved = True

    def _reboot_system(self):
        """Reboot the system"""
        logger.info("Rebooting system...")
        time.sleep(2)
        os.system("sudo reboot")

    # ── iOS captive portal suppression server ────────────────────────

    def _start_ios_server(self):
        """Start a tiny HTTP server on port 80 that tells iOS the network has internet."""
        try:
            self._ios_server = HTTPServer(("0.0.0.0", 80), _iOSSuccessHandler)
            self._ios_thread = threading.Thread(
                target=self._ios_server.serve_forever,
                daemon=True,
                name="ios-success-http"
            )
            self._ios_thread.start()
            logger.info("[LOCAL] iOS captive portal suppression server started on port 80")
        except Exception as e:
            logger.warning(f"[LOCAL] Could not start iOS suppression server on port 80: {e}")

    def _stop_ios_server(self):
        """Stop the iOS suppression HTTP server."""
        if self._ios_server:
            try:
                self._ios_server.shutdown()
                logger.info("[LOCAL] iOS suppression server stopped")
            except Exception:
                pass
            self._ios_server = None
            self._ios_thread = None

    # ── Main provisioning flow ───────────────────────────────────────

    def run(self) -> bool:
        """
        Main provisioning flow.
        Returns True if WiFi is connected, False if running in demo AP mode.
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

            # Step 6: Credential AP timed out — switch to WIMZ-Demo
            self._start_demo_mode()

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
        password = self.HOTSPOT_PASSWORD
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

    def _start_demo_mode(self):
        """Start WIMZ-Demo AP for direct robot control without internet.

        This AP stays up permanently until the robot is powered off
        or WiFi is configured via the /system/wifi/connect API endpoint.
        """
        demo_ssid = self._generate_demo_ssid()
        logger.info(f"[LOCAL] Starting {demo_ssid} AP...")

        # Start clean AP (no DNS hijack, no captive portal)
        if not self.wifi_manager.start_demo_hotspot(
            ssid=demo_ssid,
            password=self.DEMO_PASSWORD
        ):
            logger.error(f"[LOCAL] Failed to start {demo_ssid} AP")
            self._set_led_error()
            return

        # Start iOS captive portal suppression on port 80
        self._start_ios_server()

        logger.info(f"[LOCAL] {demo_ssid} AP started at {self.wifi_manager.HOTSPOT_IP}:8000")
        self._in_demo_mode = True

        # Release LED controller so treatbot's LED service can claim GPIO25 + NeoPixels
        self._cleanup_led()
        self._led_service = None

        # Block here — keep the service alive so hostapd/dnsmasq/ios-server persist
        # The treatbot service runs independently and serves the API at :8000
        logger.info("[LOCAL] Treatbot available at http://192.168.4.1:8000")
        logger.info("[LOCAL] Waiting for WiFi configuration via app or power cycle...")

        while self._running:
            time.sleep(5)

    @property
    def in_demo_mode(self) -> bool:
        return self._in_demo_mode

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

            # No connection — start credential AP with timeout, then demo mode
            logger.info("Starting WiFi provisioning AP mode...")
            loop = asyncio.get_event_loop()
            credentials_received = await loop.run_in_executor(
                None,
                self._start_credential_ap_with_timeout
            )

            if credentials_received:
                return True

            # Start demo mode (blocking in executor)
            await loop.run_in_executor(None, self._start_demo_mode)
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
