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

        @app.post("/connect")
        async def connect_wifi(credentials: WiFiCredentials):
            """Save WiFi credentials and trigger connection"""
            logger.info(f"Received credentials for: {credentials.ssid}")

            if not credentials.ssid:
                raise HTTPException(status_code=400, detail="SSID is required")

            try:
                # Save credentials (this stops the hotspot)
                success = self.wifi_manager.save_credentials(
                    credentials.ssid,
                    credentials.password
                )

                if success:
                    self._credentials_saved = True
                    if self.on_credentials_saved:
                        # Schedule callback
                        asyncio.create_task(self._trigger_callback(credentials.ssid))

                    return JSONResponse(content={
                        "success": True,
                        "message": f"Connected to {credentials.ssid}! Rebooting..."
                    })
                else:
                    return JSONResponse(content={
                        "success": False,
                        "message": "Could not connect. Please check password and try again."
                    })

            except Exception as e:
                logger.error(f"Connection error: {e}")
                return JSONResponse(content={
                    "success": False,
                    "message": str(e)
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

    async def _trigger_callback(self, ssid: str):
        """Trigger the credentials saved callback after a delay"""
        await asyncio.sleep(2)
        if self.on_credentials_saved:
            self.on_credentials_saved(ssid)

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
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="warning"
        )

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
