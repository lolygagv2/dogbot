#!/usr/bin/env python3
"""
wifi_provision_main.py - WiFi provisioning service entry point

This script is called by systemd wifi-provision.service before main_treatbot.py starts.
It ensures WiFi connectivity is available before the main application launches.

Flow:
1. Check for existing WiFi connection
2. Try to connect to saved networks
3. If no connection, start AP mode for provisioning
4. Exit with code 0 when WiFi is connected (allows treatbot.service to start)
"""

import os
import sys
import logging
from pathlib import Path

# Ensure we're in the right directory
os.chdir(Path(__file__).parent)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/home/morgan/dogbot/logs/wifi_provision.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for WiFi provisioning"""
    logger.info("=" * 60)
    logger.info("WIM-Z WiFi Provisioning Service")
    logger.info("=" * 60)

    try:
        from services.network.wifi_provisioning import WiFiProvisioningService

        service = WiFiProvisioningService()
        result = service.run()

        if result:
            logger.info("WiFi connected successfully - treatbot.service can start")
            return 0
        else:
            # AP mode is running, this is not an error
            # The service will keep running until credentials are saved
            logger.info("Running in AP provisioning mode")
            return 0

    except KeyboardInterrupt:
        logger.info("Provisioning interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Provisioning failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
