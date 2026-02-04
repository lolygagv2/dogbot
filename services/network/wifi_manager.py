#!/usr/bin/env python3
"""
services/network/wifi_manager.py - NetworkManager wrapper using nmcli

Provides WiFi management functions for:
- Connecting to known networks
- Scanning for available networks
- Creating AP hotspot mode (via hostapd + dnsmasq)
- Saving new WiFi credentials
"""

import os
import subprocess
import re
import time
import signal
import logging
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

# hostapd + dnsmasq config paths for AP mode
HOSTAPD_CONF = "/tmp/wimz-hostapd.conf"
DNSMASQ_CONF = "/tmp/wimz-dnsmasq.conf"
HOSTAPD_PID = "/tmp/wimz-hostapd.pid"
DNSMASQ_PID = "/tmp/wimz-dnsmasq.pid"


class WiFiManager:
    """NetworkManager wrapper for WiFi operations"""

    DEFAULT_INTERFACE = "wlan0"
    HOTSPOT_CONNECTION_NAME = "WIMZ-Hotspot"
    HOTSPOT_IP = "192.168.4.1"

    def __init__(self, interface: str = None):
        self.interface = interface or self.DEFAULT_INTERFACE
        self._cached_networks = []  # Cache scan results before AP mode
        self._in_ap_mode = False

    def _run_cmd(self, cmd: List[str], timeout: int = 30) -> Tuple[bool, str]:
        """Run a shell command and return (success, output)"""
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
                logger.warning(f"Command failed: {' '.join(cmd)}")
                logger.warning(f"Output: {output}")
            return success, output
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {' '.join(cmd)}")
            return False, "Command timed out"
        except Exception as e:
            logger.error(f"Command error: {e}")
            return False, str(e)

    def _run_nmcli(self, args: List[str], timeout: int = 30) -> Tuple[bool, str]:
        """Run nmcli command and return (success, output)"""
        return self._run_cmd(["nmcli"] + args, timeout=timeout)

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
        # Cap nmcli timeout to avoid blocking longer than our overall timeout
        nmcli_timeout = min(timeout, 15)
        success, output = self._run_nmcli(
            ["device", "connect", self.interface],
            timeout=nmcli_timeout
        )

        # If nmcli says no network found, fail immediately - no point polling
        if not success and "could not be found" in output.lower():
            logger.info("No known WiFi networks in range - skipping wait")
            return False

        # Poll for connection for remaining time
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
        """Scan for available WiFi networks (returns cached results in AP mode)"""
        if self._in_ap_mode:
            logger.info(f"In AP mode - returning {len(self._cached_networks)} cached networks")
            return self._cached_networks
        return self._do_scan()

    def _do_scan(self) -> List[Dict]:
        """Actually perform WiFi scan via nmcli"""
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
        """Start WiFi hotspot using hostapd + dnsmasq (bypasses NM's broken AP mode)"""
        logger.info(f"Starting hotspot: {ssid}")

        # Cache scan results BEFORE we take over the interface
        # (can't scan while in AP mode)
        logger.info("Caching WiFi scan results before AP mode...")
        self._cached_networks = self._do_scan()
        logger.info(f"Cached {len(self._cached_networks)} networks")

        # Step 1: Tell NetworkManager to stop managing wlan0
        self._run_nmcli(["device", "disconnect", self.interface])
        self._run_nmcli(["device", "set", self.interface, "managed", "no"])
        time.sleep(0.5)

        # Step 2: Configure interface with static IP (needs sudo if not root)
        self._run_cmd(["sudo", "ip", "addr", "flush", "dev", self.interface])
        self._run_cmd(["sudo", "ip", "addr", "add", f"{self.HOTSPOT_IP}/24", "dev", self.interface])
        self._run_cmd(["sudo", "ip", "link", "set", self.interface, "up"])
        time.sleep(0.5)

        # Step 3: Write hostapd config
        hostapd_config = f"""interface={self.interface}
driver=nl80211
ssid={ssid}
hw_mode=g
channel=6
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase={password}
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
"""
        try:
            with open(HOSTAPD_CONF, 'w') as f:
                f.write(hostapd_config)
            logger.info("hostapd config written")
        except Exception as e:
            logger.error(f"Failed to write hostapd config: {e}")
            return False

        # Step 4: Write dnsmasq config (DHCP + DNS hijack for captive portal)
        dnsmasq_config = f"""interface={self.interface}
bind-interfaces
dhcp-range=192.168.4.10,192.168.4.100,255.255.255.0,24h
address=/#/{self.HOTSPOT_IP}
"""
        try:
            with open(DNSMASQ_CONF, 'w') as f:
                f.write(dnsmasq_config)
            logger.info("dnsmasq config written")
        except Exception as e:
            logger.error(f"Failed to write dnsmasq config: {e}")
            return False

        # Step 5: Start hostapd (needs sudo if not root)
        success, output = self._run_cmd([
            "sudo", "hostapd", "-B", "-P", HOSTAPD_PID, HOSTAPD_CONF
        ], timeout=10)

        if not success:
            logger.error(f"Failed to start hostapd: {output}")
            self._cleanup_ap()
            return False

        logger.info("hostapd started")

        # Step 6: Start dnsmasq for DHCP + DNS
        # Kill any existing dnsmasq on this interface first
        self._run_cmd(["pkill", "-f", f"dnsmasq.*{DNSMASQ_CONF}"])
        time.sleep(0.3)

        success, output = self._run_cmd([
            "sudo", "dnsmasq", "-C", DNSMASQ_CONF, "--pid-file=" + DNSMASQ_PID
        ], timeout=10)

        if not success:
            logger.error(f"Failed to start dnsmasq: {output}")
            self._cleanup_ap()
            return False

        logger.info("dnsmasq started (DHCP + DNS)")
        logger.info(f"Hotspot started: {ssid} @ {self.HOTSPOT_IP}")
        self._in_ap_mode = True
        return True

    def _cleanup_ap(self):
        """Kill hostapd and dnsmasq, clean up config files"""
        # Stop hostapd and dnsmasq via their PID files
        for pidfile in [HOSTAPD_PID, DNSMASQ_PID]:
            try:
                with open(pidfile, 'r') as f:
                    pid = f.read().strip()
                # Use sudo kill since these run as root
                self._run_cmd(["sudo", "kill", pid])
                logger.info(f"Stopped process {pid} ({pidfile})")
            except (FileNotFoundError, ValueError):
                pass

        # Fallback: pkill
        self._run_cmd(["sudo", "pkill", "-f", f"hostapd.*{HOSTAPD_CONF}"])
        self._run_cmd(["sudo", "pkill", "-f", f"dnsmasq.*{DNSMASQ_CONF}"])
        time.sleep(0.5)

        # Clean up config files
        for f in [HOSTAPD_CONF, DNSMASQ_CONF, HOSTAPD_PID, DNSMASQ_PID]:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass

    def stop_hotspot(self) -> bool:
        """Stop WiFi hotspot and return to client mode"""
        logger.info("Stopping hotspot...")

        # Stop hostapd and dnsmasq
        self._cleanup_ap()

        # Flush interface IP
        self._run_cmd(["sudo", "ip", "addr", "flush", "dev", self.interface])

        # Return interface to NetworkManager control
        self._run_nmcli(["device", "set", self.interface, "managed", "yes"])
        time.sleep(1)

        logger.info("Hotspot stopped")
        self._in_ap_mode = False
        return True

    def save_credentials(self, ssid: str, password: str) -> Dict:
        """
        Save WiFi credentials and test connection.
        Returns dict with: success, message, has_internet, should_restart_ap
        """
        logger.info(f"Saving credentials for: {ssid}")
        result = {
            "success": False,
            "message": "",
            "has_internet": False,
            "should_restart_ap": False
        }

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
            logger.error(f"Failed to connect: {output}")
            # Check if it's an auth failure
            if "secrets were required" in output.lower() or "no secrets" in output.lower():
                result["message"] = "Wrong password. Please try again."
            elif "not found" in output.lower():
                result["message"] = "Network not found. It may be out of range."
            else:
                result["message"] = "Connection failed. Please try again."
            result["should_restart_ap"] = True
            return result

        # Wait for connection
        time.sleep(3)

        if not self.is_connected():
            logger.warning("Connection failed after credentials accepted")
            result["message"] = "Could not connect. Password may be wrong."
            result["should_restart_ap"] = True
            return result

        status = self.get_connection_status()
        logger.info(f"Connected to {ssid} ({status['ip_address']})")

        # Test internet connectivity
        has_internet = self.check_internet()
        result["has_internet"] = has_internet

        if has_internet:
            result["success"] = True
            result["message"] = f"Connected to {ssid} with internet access!"
        else:
            result["success"] = True  # WiFi connected, just no internet
            result["message"] = f"Connected to {ssid} but no internet detected. Proceeding anyway."

        return result

    def check_internet(self, timeout: int = 5) -> bool:
        """Check if we have internet connectivity"""
        # Try to ping a reliable host
        success, _ = self._run_cmd(
            ["ping", "-c", "1", "-W", str(timeout), "8.8.8.8"],
            timeout=timeout + 2
        )
        if success:
            logger.info("Internet connectivity confirmed")
            return True

        # Fallback: try DNS resolution
        success, _ = self._run_cmd(
            ["host", "-W", str(timeout), "google.com"],
            timeout=timeout + 2
        )
        if success:
            logger.info("DNS working, internet likely available")
            return True

        logger.warning("No internet connectivity detected")
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
