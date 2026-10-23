#!/usr/bin/env python3
"""
services/network/captive_portal.py - Captive portal web server for WiFi provisioning

FastAPI server running on port 80 that provides:
- WiFi setup page
- Network scanning endpoint
- Credential saving endpoint
"""

import os
import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from .wifi_manager import WiFiManager

logger = logging.getLogger(__name__)

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"

# Diagnostic log: an in-memory ring buffer of the last N log records from the
# services.network.* loggers (wifi_manager, captive_portal, provisioning).
# Exposed read-only at GET /log so the setup page can show the user WHY a
# connect attempt failed (raw nmcli output etc.). Survives AP restarts because
# it lives in the main process, not the browser.
PORTAL_LOG_MAX = 200


class _PortalLogBuffer(logging.Handler):
    """Collects log records into a bounded deque for the /log endpoint."""

    def __init__(self, maxlen: int = PORTAL_LOG_MAX):
        super().__init__(level=logging.INFO)
        self.records: deque = deque(maxlen=maxlen)
        self._seq = 0

    def emit(self, record: logging.LogRecord):
        try:
            self._seq += 1
            self.records.append({
                "seq": self._seq,
                "ts": record.created,
                "time": time.strftime("%H:%M:%S", time.localtime(record.created)),
                "level": record.levelname,
                "source": record.name.rsplit(".", 1)[-1],
                "message": self.format(record) if self.formatter else record.getMessage(),
            })
        except Exception:
            self.handleError(record)

    def since(self, seq: int = 0):
        return [r for r in self.records if r["seq"] > seq]


_portal_log = _PortalLogBuffer()
_portal_log.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger("services.network").addHandler(_portal_log)


class WiFiCredentials(BaseModel):
    ssid: str
    password: str


class CaptivePortal:
    """Captive portal web server for WiFi provisioning"""

    def __init__(self, wifi_manager: WiFiManager, on_credentials_saved: callable = None):
        self.wifi_manager = wifi_manager
        self.on_credentials_saved = on_credentials_saved
        self.app = self._create_app()
        self._server = None
        self._credentials_saved = False

    def _create_app(self) -> FastAPI:
        """Create FastAPI application"""
        app = FastAPI(
            title="WIM-Z WiFi Setup",
            description="WiFi provisioning portal for WIM-Z robot"
        )

        @app.get("/", response_class=HTMLResponse)
        async def setup_page(request: Request):
            """Serve the WiFi setup page"""
            template_path = TEMPLATE_DIR / "setup.html"
            if not template_path.exists():
                logger.error(f"Template not found: {template_path}")
                return HTMLResponse(
                    content=self._get_fallback_html(),
                    status_code=200
                )

            with open(template_path, 'r') as f:
                html = f.read()
            return HTMLResponse(content=html, status_code=200)

        @app.get("/scan")
        async def scan_networks():
            """Scan and return available WiFi networks"""
            try:
                networks = self.wifi_manager.scan_networks()
                return JSONResponse(content={
                    "success": True,
                    "networks": networks
                })
            except Exception as e:
                logger.error(f"Scan error: {e}")
                return JSONResponse(content={
                    "success": False,
                    "error": str(e),
                    "networks": []
                })

        @app.get("/saved")
        async def get_saved_networks():
            """Return list of saved WiFi connections"""
            try:
                saved = self.wifi_manager.get_saved_connections()
                # Filter out the hotspot connection
                saved = [c for c in saved if c['name'] != self.wifi_manager.HOTSPOT_CONNECTION_NAME]
                return JSONResponse(content={
                    "success": True,
                    "connections": saved
                })
            except Exception as e:
                logger.error(f"Saved networks error: {e}")
                return JSONResponse(content={
                    "success": False,
                    "error": str(e),
                    "connections": []
                })

        @app.post("/forget")
        async def forget_network(credentials: WiFiCredentials):
            """Forget a saved WiFi connection by SSID"""
            if not credentials.ssid:
                raise HTTPException(status_code=400, detail="SSID is required")
            try:
                success, message = self.wifi_manager.forget_connection(credentials.ssid)
                return JSONResponse(content={
                    "success": success,
                    "message": message
                })
            except Exception as e:
                logger.error(f"Forget error: {e}")
                return JSONResponse(content={
                    "success": False,
                    "message": f"Error: {str(e)}"
                })

        @app.post("/connect")
        async def connect_wifi(credentials: WiFiCredentials):
            """Accept WiFi credentials and start the connect in the background.

            Answers immediately. The connect itself tears the setup AP down,
            which drops the phone that sent this request — an inline result
            could never reach it (the page then said "Connection failed" even
            when the password was right). Outcome: GET /connect/status.
            """
            logger.info(f"Received credentials for: '{credentials.ssid}' "
                        f"(password length {len(credentials.password or '')})")

            if not credentials.ssid:
                raise HTTPException(status_code=400, detail="SSID is required")

            try:
                started, job = self.wifi_manager.connect_in_background(
                    credentials.ssid, credentials.password,
                    on_success=self._on_connect_success)
            except Exception as e:
                logger.error(f"Connection error: {e}")
                return JSONResponse(status_code=500, content={
                    "success": False, "message": f"Error: {str(e)}"})

            if not started:
                return JSONResponse(status_code=409, content={
                    "success": False, "pending": True, "ssid": job.get("ssid"),
                    "message": f"Already connecting to '{job.get('ssid')}' — please wait."})

            ap_ssid = self._ap_ssid()
            return JSONResponse(content={
                "success": True,
                "pending": True,
                "ssid": credentials.ssid,
                "ap_ssid": ap_ssid,
                "message": (
                    f"Credentials received — WIM-Z is joining '{credentials.ssid}' now. "
                    f"This setup network will disappear for about a minute. "
                    f"If '{ap_ssid}' comes back, the password was rejected: rejoin it and try again. "
                    f"Otherwise WIM-Z is online — rejoin your home WiFi and open the WIM-Z app."),
            })

        @app.get("/connect/status")
        async def connect_status():
            """Outcome of the last credential connect (see POST /connect)."""
            job = None
            getter = getattr(self.wifi_manager, "get_connect_job", None)
            if getter:
                try:
                    job = getter()
                except Exception as e:
                    logger.error(f"connect status error: {e}")
            return JSONResponse(content={
                "success": True,
                "job": job or {"state": "idle"},
                "robot_time": time.strftime("%H:%M:%S"),
            })

        @app.get("/status")
        async def get_status():
            """Get current WiFi connection status"""
            try:
                status = self.wifi_manager.get_connection_status()
                return JSONResponse(content={
                    "success": True,
                    "status": status
                })
            except Exception as e:
                logger.error(f"Status error: {e}")
                return JSONResponse(content={
                    "success": False,
                    "error": str(e)
                })

        @app.get("/log")
        async def get_portal_log(since: int = 0):
            """Diagnostic log (read-only) — last network/provisioning log lines.
            ?since=<seq> returns only records newer than that sequence number."""
            entries = _portal_log.since(since)
            return JSONResponse(content={
                "success": True,
                "entries": entries,
                "latest_seq": entries[-1]["seq"] if entries else since,
                "robot_time": time.strftime("%H:%M:%S"),
            })

        @app.get("/reboot")
        async def trigger_reboot():
            """Trigger system reboot"""
            logger.info("Reboot requested")
            asyncio.create_task(self._do_reboot())
            return JSONResponse(content={
                "success": True,
                "message": "Rebooting in 3 seconds..."
            })

        # Captive portal detection endpoints
        # These help mobile devices detect the captive portal
        @app.get("/generate_204")
        async def android_captive_check():
            """Android captive portal detection"""
            return RedirectResponse(url="/", status_code=302)

        @app.get("/hotspot-detect.html")
        async def apple_captive_check():
            """Apple captive portal detection"""
            return RedirectResponse(url="/", status_code=302)

        @app.get("/connecttest.txt")
        async def windows_captive_check():
            """Windows captive portal detection"""
            return RedirectResponse(url="/", status_code=302)

        @app.get("/ncsi.txt")
        async def windows_ncsi_check():
            """Windows NCSI check"""
            return RedirectResponse(url="/", status_code=302)

        @app.get("/favicon.ico")
        async def favicon():
            """Return empty favicon"""
            return HTMLResponse(content="", status_code=204)

        return app

    def _ap_ssid(self) -> str:
        try:
            return self.wifi_manager.get_ap_ssid()
        except Exception:
            return "WIMZ"

    def _on_connect_success(self, ssid: str, result: dict):
        """Worker-thread callback from WiFiManager.connect_in_background."""
        self._credentials_saved = True
        if self.on_credentials_saved:
            self.on_credentials_saved(ssid)

    async def _restart_ap_mode(self, failed_ssid: str):
        """Restart AP mode after a failed connection attempt"""
        await asyncio.sleep(2)  # Give the response time to reach the client
        logger.info(f"Restarting AP mode after failed connection to {failed_ssid}")

        # The single fleet-wide AP SSID/password
        ap_ssid = self.wifi_manager.get_ap_ssid()
        ap_password = self.wifi_manager.AP_PASSWORD

        # Restart the hotspot
        success = self.wifi_manager.start_hotspot(ap_ssid, ap_password)
        if success:
            logger.info(f"AP mode restarted: {ap_ssid}")
        else:
            logger.error("Failed to restart AP mode!")

    async def _do_reboot(self):
        """Perform system reboot after delay"""
        await asyncio.sleep(3)
        logger.info("Rebooting system...")
        os.system("sudo reboot")

    def _get_fallback_html(self) -> str:
        """Fallback HTML if template is missing"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>WIM-Z WiFi Setup</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; max-width: 400px; margin: 40px auto; padding: 20px; }
        h1 { color: #333; }
        .error { color: red; padding: 10px; background: #fee; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>WIM-Z WiFi Setup</h1>
    <div class="error">
        <p>Setup page template is missing.</p>
        <p>Please reinstall the WiFi provisioning service.</p>
    </div>
</body>
</html>
"""

    def run(self, host: str = "0.0.0.0", port: int = 80):
        """Run the captive portal server (blocking)"""
        logger.info(f"Starting captive portal on {host}:{port}")
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="warning"
        )
        self._server = uvicorn.Server(config)
        self._server.run()

    def stop(self):
        """Stop the captive portal server (call from another thread)"""
        if self._server:
            logger.info("Stopping captive portal server...")
            self._server.should_exit = True

    async def start_async(self, host: str = "0.0.0.0", port: int = 80):
        """Start the captive portal server (async)"""
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="warning"
        )
        self._server = uvicorn.Server(config)
        await self._server.serve()

    async def stop_async(self):
        """Stop the captive portal server"""
        if self._server:
            self._server.should_exit = True

    @property
    def credentials_saved(self) -> bool:
        """Check if credentials have been saved"""
        return self._credentials_saved


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    wifi = WiFiManager()
    portal = CaptivePortal(wifi)
    portal.run(port=8080)  # Use 8080 for testing
