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

    DEMO_SSID = "WIMZ-Demo"
    DEMO_PASSWORD = "wimzdemo"

    # State file: if this exists, we were in demo AP mode and should resume it
    DEMO_STATE_FILE = "/tmp/wimz-demo-ap-active"

    def __init__(self):
        self.wifi_manager = WiFiManager()
        self.captive_portal: Optional[CaptivePortal] = None
        self.led_controller = None
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
            # Step 0: Check if we were previously in demo AP mode
            # If the state file exists, skip WiFi and go straight to demo AP
            if os.path.exists(self.DEMO_STATE_FILE):
                logger.info("[LOCAL] Demo AP state file found — resuming WIMZ-Demo AP mode")
                self._start_demo_mode()
                return False

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
        logger.info("[LOCAL] Starting WIMZ-Demo AP...")

        # Start clean AP (no DNS hijack, no captive portal)
        if not self.wifi_manager.start_demo_hotspot(
            ssid=self.DEMO_SSID,
            password=self.DEMO_PASSWORD
        ):
            logger.error("[LOCAL] Failed to start WIMZ-Demo AP")
            self._set_led_error()
            return

        # Start iOS captive portal suppression on port 80
        self._start_ios_server()

        logger.info(f"[LOCAL] WIMZ-Demo AP started at {self.wifi_manager.HOTSPOT_IP}:8000")
        self._in_demo_mode = True

        # Write state file so we resume AP mode if the service restarts
        try:
            with open(self.DEMO_STATE_FILE, 'w') as f:
                f.write("demo_ap_active")
        except Exception as e:
            logger.warning(f"Could not write demo state file: {e}")

        # Release LED controller so treatbot's LED service can claim GPIO25 + NeoPixels
        self._cleanup_led()
        self.led_controller = None

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
