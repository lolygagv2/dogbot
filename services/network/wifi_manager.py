#!/usr/bin/env python3
"""
services/network/wifi_manager.py - NetworkManager wrapper using nmcli

Provides WiFi management functions for:
- Connecting to known networks
- Scanning for available networks
- Creating AP hotspot mode
- Saving new WiFi credentials
"""

import subprocess
import re
import time
import logging
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)


class WiFiManager:
    """NetworkManager wrapper for WiFi operations"""

    DEFAULT_INTERFACE = "wlan0"
    HOTSPOT_CONNECTION_NAME = "WIMZ-Hotspot"
    HOTSPOT_IP = "192.168.4.1"

    def __init__(self, interface: str = None):
        self.interface = interface or self.DEFAULT_INTERFACE

    def _run_nmcli(self, args: List[str], timeout: int = 30) -> Tuple[bool, str]:
        """Run nmcli command and return (success, output)"""
        cmd = ["nmcli"] + args
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            output = result.stdout + result.stderr
            success = result.returncode == 0
            if not success:
                logger.warning(f"nmcli command failed: {' '.join(cmd)}")
                logger.warning(f"Output: {output}")
            return success, output
        except subprocess.TimeoutExpired:
            logger.error(f"nmcli command timed out: {' '.join(cmd)}")
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"nmcli command error: {e}")
            return False, str(e)

    def get_device_serial(self) -> str:
        """Get last 4 characters of device serial from /proc/cpuinfo"""
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if line.startswith('Serial'):
                        serial = line.strip().split(':')[-1].strip()
                        return serial[-4:].upper()
        except Exception as e:
            logger.warning(f"Could not read serial: {e}")

        # Fallback: use part of MAC address
        try:
            success, output = self._run_nmcli([
                "-t", "-f", "GENERAL.HWADDR",
                "device", "show", self.interface
            ])
            if success and output:
                mac = output.strip().split(':')[-1].replace(':', '')
                return mac[-4:].upper()
        except:
            pass

        return "0000"

    def get_saved_connections(self) -> List[Dict]:
        """Get list of saved WiFi connections"""
        connections = []
        success, output = self._run_nmcli([
            "-t", "-f", "NAME,TYPE,DEVICE",
            "connection", "show"
        ])

        if not success:
            return connections

        for line in output.strip().split('\n'):
            if not line:
                continue
            parts = line.split(':')
            if len(parts) >= 2 and parts[1] == '802-11-wireless':
                connections.append({
                    'name': parts[0],
                    'type': 'wifi',
                    'device': parts[2] if len(parts) > 2 else ''
                })

        return connections

    def get_connection_status(self) -> Dict:
        """Get current WiFi connection status"""
        status = {
            'connected': False,
            'ssid': None,
            'ip_address': None,
            'signal': None,
            'state': 'disconnected'
        }

        # Check device state
        success, output = self._run_nmcli([
            "-t", "-f", "GENERAL.STATE,GENERAL.CONNECTION,IP4.ADDRESS",
            "device", "show", self.interface
        ])

        if not success:
            return status

        for line in output.strip().split('\n'):
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            if 'STATE' in key and '100' in value:
                status['connected'] = True
                status['state'] = 'connected'
            elif 'CONNECTION' in key and value and value != '--':
                status['ssid'] = value
            elif 'IP4.ADDRESS' in key and value:
                # Extract IP from CIDR notation
                status['ip_address'] = value.split('/')[0]

        # Get signal strength if connected
        if status['connected']:
            success, output = self._run_nmcli([
                "-t", "-f", "IN-USE,SIGNAL,SSID",
                "device", "wifi", "list", "ifname", self.interface
            ])
            if success:
                for line in output.strip().split('\n'):
                    if line.startswith('*:'):
                        parts = line.split(':')
                        if len(parts) >= 2:
                            try:
                                status['signal'] = int(parts[1])
                            except ValueError:
                                pass

        return status

    def is_connected(self) -> bool:
        """Check if WiFi is connected to any network"""
        return self.get_connection_status()['connected']

    def try_connect_known(self, timeout: int = 30) -> bool:
        """Try to connect to any known WiFi network"""
        logger.info("Attempting to connect to known WiFi networks...")

        # First check if already connected
        if self.is_connected():
            logger.info("Already connected to WiFi")
            return True

        # Get saved connections
        saved = self.get_saved_connections()
        if not saved:
            logger.info("No saved WiFi connections found")
            return False

        logger.info(f"Found {len(saved)} saved WiFi connections")

        # Try to bring up the device and auto-connect
        self._run_nmcli(["device", "connect", self.interface])

        # Wait for connection
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_connected():
                status = self.get_connection_status()
                logger.info(f"Connected to {status['ssid']} ({status['ip_address']})")
                return True
            time.sleep(2)

        logger.warning(f"Failed to connect to any known network within {timeout}s")
        return False

    def scan_networks(self) -> List[Dict]:
        """Scan for available WiFi networks"""
        networks = []

        # Rescan first
        self._run_nmcli(["device", "wifi", "rescan", "ifname", self.interface], timeout=10)
        time.sleep(2)  # Give time for scan to complete

        # Get scan results
        success, output = self._run_nmcli([
            "-t", "-f", "SSID,SIGNAL,SECURITY,BSSID",
            "device", "wifi", "list", "ifname", self.interface
        ])

        if not success:
            logger.error("Failed to scan networks")
            return networks

        seen_ssids = set()
        for line in output.strip().split('\n'):
            if not line:
                continue
            parts = line.split(':')
            if len(parts) >= 3:
                ssid = parts[0]
                if not ssid or ssid in seen_ssids:
                    continue
                seen_ssids.add(ssid)

                try:
                    signal = int(parts[1])
                except ValueError:
                    signal = 0

                security = parts[2] if parts[2] else 'Open'

                networks.append({
                    'ssid': ssid,
                    'signal': signal,
                    'security': security,
                    'secured': security != 'Open' and security != ''
                })

        # Sort by signal strength
        networks.sort(key=lambda x: x['signal'], reverse=True)
        return networks

    def start_hotspot(self, ssid: str, password: str) -> bool:
        """Start WiFi hotspot (AP mode)"""
        logger.info(f"Starting hotspot: {ssid}")

        # First, delete any existing hotspot connection
        self._run_nmcli(["connection", "delete", self.HOTSPOT_CONNECTION_NAME])
        time.sleep(1)

        # Stop any existing wifi connections on the interface
        self._run_nmcli(["device", "disconnect", self.interface])
        time.sleep(1)

        # Create hotspot
        success, output = self._run_nmcli([
            "device", "wifi", "hotspot",
            "ifname", self.interface,
            "con-name", self.HOTSPOT_CONNECTION_NAME,
            "ssid", ssid,
            "password", password
        ], timeout=30)

        if not success:
            logger.error(f"Failed to create hotspot: {output}")
            return False

        # Configure IP address for captive portal
        time.sleep(2)
        success, output = self._run_nmcli([
            "connection", "modify", self.HOTSPOT_CONNECTION_NAME,
            "ipv4.addresses", f"{self.HOTSPOT_IP}/24",
            "ipv4.method", "shared"
        ])

        if not success:
            logger.warning(f"Could not configure hotspot IP: {output}")

        # Reactivate with new settings
        self._run_nmcli(["connection", "up", self.HOTSPOT_CONNECTION_NAME])
        time.sleep(2)

        logger.info(f"Hotspot started: {ssid} @ {self.HOTSPOT_IP}")
        return True

    def stop_hotspot(self) -> bool:
        """Stop WiFi hotspot and return to client mode"""
        logger.info("Stopping hotspot...")

        # Bring down the hotspot connection
        self._run_nmcli(["connection", "down", self.HOTSPOT_CONNECTION_NAME])
        time.sleep(1)

        # Delete the hotspot connection
        self._run_nmcli(["connection", "delete", self.HOTSPOT_CONNECTION_NAME])
        time.sleep(1)

        logger.info("Hotspot stopped")
        return True

    def save_credentials(self, ssid: str, password: str) -> bool:
        """Save WiFi credentials and test connection"""
        logger.info(f"Saving credentials for: {ssid}")

        # Stop hotspot if running
        self.stop_hotspot()
        time.sleep(2)

        # Delete any existing connection with same name
        self._run_nmcli(["connection", "delete", ssid])
        time.sleep(1)

        # Create new connection
        success, output = self._run_nmcli([
            "device", "wifi", "connect", ssid,
            "password", password,
            "ifname", self.interface
        ], timeout=30)

        if not success:
            logger.error(f"Failed to save credentials: {output}")
            return False

        # Wait for connection
        time.sleep(5)

        if self.is_connected():
            status = self.get_connection_status()
            logger.info(f"Successfully connected to {ssid} ({status['ip_address']})")
            return True
        else:
            logger.warning(f"Credentials saved but connection failed")
            return False

    def get_hotspot_clients(self) -> List[Dict]:
        """Get list of clients connected to hotspot"""
        clients = []
        try:
            # Check ARP table for connected clients
            result = subprocess.run(
                ["arp", "-n"],
                capture_output=True,
                text=True,
                timeout=5
            )
            for line in result.stdout.strip().split('\n')[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 3 and parts[0].startswith('192.168.4.'):
                    clients.append({
                        'ip': parts[0],
                        'mac': parts[2] if len(parts) > 2 else 'unknown'
                    })
        except Exception as e:
            logger.warning(f"Could not get hotspot clients: {e}")

        return clients


# Test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    wifi = WiFiManager()

    print(f"Device serial: {wifi.get_device_serial()}")
    print(f"Connection status: {wifi.get_connection_status()}")
    print(f"Saved connections: {wifi.get_saved_connections()}")

    print("\nScanning for networks...")
    networks = wifi.scan_networks()
    for net in networks[:5]:
        print(f"  {net['ssid']}: {net['signal']}% ({net['security']})")
