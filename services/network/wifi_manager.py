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
import threading
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger(__name__)

# Thread-safe singleton pattern to prevent race conditions in wireless driver
_wifi_manager_instance: Optional['WiFiManager'] = None
_wifi_manager_lock = threading.Lock()


def get_wifi_manager() -> 'WiFiManager':
    """Get singleton WiFiManager instance (thread-safe).

    Multiple threads creating separate WiFiManager instances and calling
    nmcli/pgrep concurrently causes kernel wireless driver deadlock.
    Use this factory function to ensure all code shares one instance.
    """
    global _wifi_manager_instance
    with _wifi_manager_lock:
        if _wifi_manager_instance is None:
            _wifi_manager_instance = WiFiManager()
        return _wifi_manager_instance

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
        self._wifi_driver: Optional[str] = None  # cached for netdev recovery
        # Serialize all wireless operations to prevent kernel driver deadlock
        self._op_lock = threading.RLock()

    def is_ap_mode(self) -> bool:
        """Detect if the interface is currently in AP mode by checking for hostapd process."""
        with self._op_lock:
            if self._in_ap_mode:
                return True
            # Check if hostapd is running with our config
            success, output = self._run_cmd(["pgrep", "-f", f"hostapd.*{HOSTAPD_CONF}"])
            return success

    def has_associated_stations(self) -> bool:
        """True if any STA is associated to our AP, via `iw station dump` (L2).

        Unlike get_hotspot_clients() (which parses the ARP table and only sees a
        client *after* it has done DHCP/ARP), this reflects association the
        instant the phone connects. That closes the race where the WiFi monitor
        tore the AP down in the same second a phone associated — before it ever
        got an IP, so the ARP table was still empty.

        Only meaningful while in AP mode: on a station interface `iw station
        dump` lists the upstream router, so callers MUST gate this on AP mode.
        """
        success, output = self._run_cmd(
            ["iw", "dev", self.interface, "station", "dump"], timeout=5
        )
        if not success:
            return False
        return any(line.strip().startswith("Station") for line in output.splitlines())

    def get_active_ap_ssid(self) -> Optional[str]:
        """Return SSID of the currently running AP by parsing the hostapd config."""
        try:
            with open(HOSTAPD_CONF, 'r') as f:
                for line in f:
                    if line.startswith('ssid='):
                        return line.split('=', 1)[1].strip()
        except Exception:
            pass
        return None

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

    def forget_connection(self, ssid: str) -> Tuple[bool, str]:
        """Delete a saved WiFi connection by SSID"""
        logger.info(f"Forgetting saved connection: {ssid}")
        success, output = self._run_nmcli(["connection", "delete", ssid])
        if success:
            logger.info(f"Deleted saved connection: {ssid}")
            return True, f"Forgot '{ssid}'"
        else:
            logger.warning(f"Failed to delete connection '{ssid}': {output}")
            return False, f"Could not forget '{ssid}'"

    def get_connection_status(self) -> Dict:
        """Get current WiFi connection status"""
        with self._op_lock:
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

    def try_connect_known(self, timeout: int = 15) -> bool:
        """Try to connect to any known WiFi network"""
        with self._op_lock:
            logger.info("Attempting to connect to known WiFi networks...")

            # First check if already connected (RLock allows nested acquisition)
            if self.is_connected():
                logger.info("Already connected to WiFi")
                return True

            # Get saved connections
            saved = self.get_saved_connections()
            if not saved:
                logger.info("No saved WiFi connections found")
                return False

            logger.info(f"Found {len(saved)} saved WiFi connections")

            nmcli_timeout = min(timeout, 10)

            # Reactivate a saved profile *by name* first. Coming out of AP mode,
            # wlan0 has no active profile, so a bare `nmcli device connect wlan0`
            # fails with "A 'wireless' setting is required if no AP path was
            # given" (NM has nothing to auto-pick) — the exact error that left
            # the robot stranded for ~40s during the demo. Bringing a saved
            # connection up by name avoids it and is what actually recovers WiFi.
            for conn in saved:
                name = conn.get('name')
                if not name:
                    continue
                ok, _out = self._run_nmcli(
                    ["connection", "up", name, "ifname", self.interface],
                    timeout=nmcli_timeout
                )
                if ok and self.is_connected():
                    status = self.get_connection_status()
                    logger.info(f"Connected to {status['ssid']} ({status['ip_address']}) via '{name}'")
                    return True

            # Fall back to letting NM auto-pick an in-range autoconnect profile.
            success, output = self._run_nmcli(
                ["device", "connect", self.interface],
                timeout=nmcli_timeout
            )

            if success:
                status = self.get_connection_status()
                logger.info(f"Connected to {status['ssid']} ({status['ip_address']})")
                return True

            # If nmcli says no network found, fail immediately - no point polling
            if "could not be found" in output.lower() or "timed out" in output.lower():
                logger.info("No known WiFi networks in range - skipping wait")
                return False

            # Poll for connection for remaining time (nmcli may have started a bg connect)
            start_time = time.time()
            remaining = timeout - nmcli_timeout
            while time.time() - start_time < remaining:
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

    def _wait_for_ssid_in_scan(self, ssid: str, timeout: int = 30) -> bool:
        """Poll NM scan results until target SSID appears (after AP→client switch)"""
        logger.info(f"Waiting for '{ssid}' to appear in scan results (timeout={timeout}s)...")
        start = time.time()
        attempt = 0
        while time.time() - start < timeout:
            attempt += 1
            # Trigger a rescan
            self._run_nmcli(
                ["device", "wifi", "rescan", "ifname", self.interface],
                timeout=10
            )
            time.sleep(3)
            # Check scan results for target SSID
            success, output = self._run_nmcli([
                "-t", "-f", "SSID",
                "device", "wifi", "list", "ifname", self.interface
            ])
            if success:
                ssids = [line.strip() for line in output.strip().split('\n') if line.strip()]
                if ssid in ssids:
                    elapsed = time.time() - start
                    logger.info(f"Found '{ssid}' in scan results after {elapsed:.1f}s (attempt {attempt})")
                    return True
                # Log what we ARE seeing on first and last attempts for debugging
                if attempt == 1 or time.time() - start > timeout - 5:
                    logger.info(f"Scan attempt {attempt}: {len(ssids)} networks visible, target '{ssid}' not found")
                    if attempt == 1 and ssids:
                        logger.info(f"  Sample SSIDs: {ssids[:5]}")
                else:
                    logger.debug(f"Scan attempt {attempt}: {len(ssids)} networks, '{ssid}' not yet found")
            else:
                logger.warning(f"Scan attempt {attempt}: nmcli wifi list failed (NM may not be ready)")

        logger.warning(f"SSID '{ssid}' not found after {timeout}s ({attempt} attempts)")
        return False

    # ── AP interface resilience + cross-user file handling ───────────

    def _sudo_rm(self, path: str):
        """Remove a file that may be root-owned.

        wifi-provision.service runs as root and creates /tmp/wimz-*.conf and
        .pid files owned by root. The treatbot app runs as 'morgan', so a plain
        os.remove()/open('w') on those files raises EPERM and the AP can never
        be rebuilt. Removing via sudo lets either user manage the AP.
        """
        try:
            self._run_cmd(["sudo", "rm", "-f", path])
        except Exception as e:
            logger.debug(f"_sudo_rm({path}) failed: {e}")

    def _cache_wifi_driver(self):
        """Remember the kernel module backing wlanX while it still exists, so
        _ensure_interface_ready() can reload the right driver if the netdev is
        torn down by a failed hostapd AP-mode init."""
        if self._wifi_driver:
            return
        try:
            link = os.path.realpath(
                f"/sys/class/net/{self.interface}/device/driver/module"
            )
            mod = os.path.basename(link)
            if mod and mod != "module":
                self._wifi_driver = mod
                logger.info(f"WiFi driver module: {mod}")
        except Exception as e:
            logger.debug(f"Could not detect WiFi driver: {e}")

    def _interface_present(self) -> bool:
        """True if the wlan netdev currently exists in the kernel."""
        return os.path.exists(f"/sys/class/net/{self.interface}")

    def _wait_for_interface(self, timeout: int) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._interface_present():
                logger.info(f"{self.interface} present again")
                time.sleep(1)  # let the netdev settle
                return True
            time.sleep(1)
        return False

    def _reclaim_for_ap(self) -> bool:
        """Put a freshly-recovered interface back into AP-ready (unmanaged) state."""
        self._run_nmcli(["device", "disconnect", self.interface])
        self._run_nmcli(["device", "set", self.interface, "managed", "no"])
        time.sleep(0.5)
        return True

    def _ensure_interface_ready(self, timeout: int = 15) -> bool:
        """Make sure wlanX exists before AP setup; recover it if it vanished.

        A failed hostapd nl80211 AP-mode init can leave the brcmfmac/rtw88
        netdev torn down ('Could not read interface flags: No such device'),
        after which every later command fails and the AP can't be rebuilt.
        Bring the interface back so AP bring-up always starts from a live
        device.
        """
        if self._interface_present():
            return True

        logger.warning(f"{self.interface} missing — attempting recovery")

        # 1) Let NetworkManager re-probe / power the radio back on
        self._run_nmcli(["radio", "wifi", "on"])
        self._run_nmcli(["device", "set", self.interface, "managed", "yes"])
        if self._wait_for_interface(timeout):
            return self._reclaim_for_ap()

        # 2) Last resort: reload the WiFi driver module
        if self._wifi_driver:
            logger.warning(f"Reloading WiFi driver {self._wifi_driver}")
            self._run_cmd(["sudo", "modprobe", "-r", self._wifi_driver])
            time.sleep(2)
            self._run_cmd(["sudo", "modprobe", self._wifi_driver])
            if self._wait_for_interface(timeout):
                return self._reclaim_for_ap()

        logger.error(f"{self.interface} could not be recovered")
        return False

    def _start_hostapd_with_retry(self, attempts: int = 3) -> bool:
        """Configure the interface IP and start hostapd, retrying on failure.

        AP-mode nl80211 init can fail intermittently if hostapd grabs the
        interface mid-transition. Each retry re-settles (and recovers) the
        interface before trying again. Assumes HOSTAPD_CONF is already written.
        """
        for attempt in range(1, attempts + 1):
            if not self._ensure_interface_ready():
                logger.error("Interface unavailable — cannot start hostapd")
                return False

            # Static AP IP on the interface
            self._run_cmd(["sudo", "ip", "addr", "flush", "dev", self.interface])
            self._run_cmd(["sudo", "ip", "addr", "add", f"{self.HOTSPOT_IP}/24", "dev", self.interface])
            self._run_cmd(["sudo", "ip", "link", "set", self.interface, "up"])
            time.sleep(1.5)  # let the driver settle before AP-mode init

            success, output = self._run_cmd(
                ["sudo", "hostapd", "-B", "-P", HOSTAPD_PID, HOSTAPD_CONF],
                timeout=15
            )
            if success:
                logger.info(f"hostapd started (attempt {attempt}/{attempts})")
                return True

            first = output.strip().splitlines()[0] if output.strip() else "unknown error"
            logger.warning(f"hostapd attempt {attempt}/{attempts} failed: {first}")

            # Clean up the half-started hostapd + cycle the interface before retry
            self._run_cmd(["sudo", "pkill", "-f", f"hostapd.*{HOSTAPD_CONF}"])
            time.sleep(0.5)
            self._run_cmd(["sudo", "ip", "link", "set", self.interface, "down"])
            time.sleep(1)
            self._run_cmd(["sudo", "ip", "link", "set", self.interface, "up"])
            time.sleep(2)

        logger.error(f"hostapd failed to start after {attempts} attempts")
        return False

    def start_demo_hotspot(self, ssid: str = "WIMZ-Demo", password: str = "wimzdemo") -> bool:
        """Start a clean WiFi AP for direct robot control (no captive portal, no DNS hijack).

        Unlike start_hotspot(), this creates a plain WiFi network:
        - No DNS redirect (address=/#/...)
        - DHCP range limited to 192.168.4.10-50
        - Designed for the app to connect directly to treatbot at 192.168.4.1:8000
        """
        with self._op_lock:
            logger.info(f"[LOCAL] Starting demo hotspot: {ssid}")

            # Cache scan results before taking over the interface
            logger.info("Caching WiFi scan results before demo AP mode...")
            self._cached_networks = self._do_scan()
            logger.info(f"Cached {len(self._cached_networks)} networks")

            # Recover the interface if a prior AP attempt tore it down, then
            # release it from NetworkManager for AP mode.
            self._cache_wifi_driver()
            self._ensure_interface_ready()
            self._run_nmcli(["device", "disconnect", self.interface])
            self._run_nmcli(["device", "set", self.interface, "managed", "no"])
            time.sleep(0.5)

            # Write hostapd config (clear any root-owned stale copy first so a
            # morgan-run caller can overwrite a file created by the root service)
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
            self._sudo_rm(HOSTAPD_CONF)
            try:
                with open(HOSTAPD_CONF, 'w') as f:
                    f.write(hostapd_config)
            except Exception as e:
                logger.error(f"Failed to write hostapd config: {e}")
                return False

            # Write dnsmasq config — DHCP only, NO DNS hijack
            dnsmasq_config = f"""interface={self.interface}
bind-interfaces
dhcp-range=192.168.4.10,192.168.4.50,255.255.255.0,24h
"""
            self._sudo_rm(DNSMASQ_CONF)
            try:
                with open(DNSMASQ_CONF, 'w') as f:
                    f.write(dnsmasq_config)
            except Exception as e:
                logger.error(f"Failed to write dnsmasq config: {e}")
                return False

            # Configure interface + start hostapd (with retry/recovery)
            if not self._start_hostapd_with_retry():
                logger.error("Failed to start hostapd for demo AP")
                self._cleanup_ap()
                return False

            # Start dnsmasq for DHCP only
            self._run_cmd(["sudo", "pkill", "-f", f"dnsmasq.*{DNSMASQ_CONF}"])
            time.sleep(0.3)

            success, output = self._run_cmd([
                "sudo", "dnsmasq", "-C", DNSMASQ_CONF, "--pid-file=" + DNSMASQ_PID
            ], timeout=10)

            if not success:
                logger.error(f"Failed to start dnsmasq: {output}")
                self._cleanup_ap()
                return False

            logger.info(f"[LOCAL] Demo hotspot started: {ssid} @ {self.HOTSPOT_IP}")
            self._in_ap_mode = True
            return True

    def start_hotspot(self, ssid: str, password: str) -> bool:
        """Start WiFi hotspot using hostapd + dnsmasq (bypasses NM's broken AP mode)"""
        with self._op_lock:
            logger.info(f"Starting hotspot: {ssid}")

            # Cache scan results BEFORE we take over the interface
            # (can't scan while in AP mode)
            logger.info("Caching WiFi scan results before AP mode...")
            self._cached_networks = self._do_scan()
            logger.info(f"Cached {len(self._cached_networks)} networks")

            # Recover the interface if a prior AP attempt tore it down, then
            # release it from NetworkManager for AP mode.
            self._cache_wifi_driver()
            self._ensure_interface_ready()
            self._run_nmcli(["device", "disconnect", self.interface])
            self._run_nmcli(["device", "set", self.interface, "managed", "no"])
            time.sleep(0.5)

            # Write hostapd config (clear any root-owned stale copy first)
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
            self._sudo_rm(HOSTAPD_CONF)
            try:
                with open(HOSTAPD_CONF, 'w') as f:
                    f.write(hostapd_config)
                logger.info("hostapd config written")
            except Exception as e:
                logger.error(f"Failed to write hostapd config: {e}")
                return False

            # Write dnsmasq config (DHCP + DNS hijack for captive portal)
            dnsmasq_config = f"""interface={self.interface}
bind-interfaces
dhcp-range=192.168.4.10,192.168.4.100,255.255.255.0,24h
address=/#/{self.HOTSPOT_IP}
"""
            self._sudo_rm(DNSMASQ_CONF)
            try:
                with open(DNSMASQ_CONF, 'w') as f:
                    f.write(dnsmasq_config)
                logger.info("dnsmasq config written")
            except Exception as e:
                logger.error(f"Failed to write dnsmasq config: {e}")
                return False

            # Configure interface + start hostapd (with retry/recovery)
            if not self._start_hostapd_with_retry():
                logger.error("Failed to start hostapd")
                self._cleanup_ap()
                return False

            logger.info("hostapd started")

            # Start dnsmasq for DHCP + DNS — kill any existing first
            self._run_cmd(["sudo", "pkill", "-f", f"dnsmasq.*{DNSMASQ_CONF}"])
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

        # Clean up config/PID files. These may be root-owned (created by the
        # root-run wifi-provision.service), so remove via sudo — a plain
        # os.remove() as 'morgan' would raise EPERM and leave stale configs.
        for path in [HOSTAPD_CONF, DNSMASQ_CONF, HOSTAPD_PID, DNSMASQ_PID]:
            self._sudo_rm(path)

    def _wait_for_nm_ready(self, timeout: int = 20) -> bool:
        """Wait for NetworkManager to show wlan0 in a scannable state (disconnected)"""
        logger.info(f"Waiting for NetworkManager to be ready (timeout={timeout}s)...")
        start = time.time()
        while time.time() - start < timeout:
            success, output = self._run_nmcli([
                "-t", "-f", "DEVICE,STATE",
                "device", "status"
            ])
            if success:
                for line in output.strip().split('\n'):
                    parts = line.split(':')
                    if len(parts) >= 2 and parts[0] == self.interface:
                        state = parts[1].strip()
                        if state in ("disconnected", "connected"):
                            elapsed = time.time() - start
                            logger.info(f"NM ready: wlan0 state={state} after {elapsed:.1f}s")
                            return True
                        logger.debug(f"NM device state: {state} (waiting for disconnected/connected)")
            time.sleep(1)

        logger.warning(f"NM not ready after {timeout}s")
        return False

    def stop_hotspot(self) -> bool:
        """Stop WiFi hotspot and return to client mode"""
        with self._op_lock:
            logger.info("Stopping hotspot...")

            # Stop hostapd and dnsmasq
            self._cleanup_ap()

            # Flush interface IP
            self._run_cmd(["sudo", "ip", "addr", "flush", "dev", self.interface])

            # Cycle interface down/up to force driver out of AP mode at kernel level
            # Without this, the wireless driver stays in AP mode and NM scans fail
            logger.info("Cycling interface to force station mode...")
            self._run_cmd(["sudo", "ip", "link", "set", self.interface, "down"])
            time.sleep(0.5)
            self._run_cmd(["sudo", "ip", "link", "set", self.interface, "up"])
            time.sleep(1)

            # Return interface to NetworkManager control
            self._run_nmcli(["device", "set", self.interface, "managed", "yes"])

            # Wait for NM to actually reclaim the interface and be ready to scan
            self._wait_for_nm_ready(timeout=20)

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

        # Stop hotspot if running (this now cycles the interface and waits for NM)
        self.stop_hotspot()

        # Wait for target SSID to appear in scan results
        ssid_found = self._wait_for_ssid_in_scan(ssid, timeout=30)
        if not ssid_found:
            logger.warning(f"SSID '{ssid}' not in scan results — attempting connect anyway")

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

        # Pin dual-band SSIDs to 5 GHz. The rtw88 dongle silently loses inbound
        # LAN traffic while it band-flaps between 2.4/5 GHz BSSIDs of the same
        # SSID — outbound (relay/internet) still works, so the failure is
        # invisible from the app side. Single-band networks are left alone.
        if self._ssid_has_5ghz(ssid):
            ok, out = self._run_nmcli([
                "connection", "modify", ssid,
                "802-11-wireless.band", "a"
            ])
            if ok:
                logger.info(f"Pinned '{ssid}' to 5 GHz (dual-band SSID)")
            else:
                logger.warning(f"Could not pin '{ssid}' to 5 GHz: {out}")

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

    def _ssid_has_5ghz(self, ssid: str) -> bool:
        """Return True if any visible BSSID for this SSID is on 5 GHz."""
        self._run_nmcli(["device", "wifi", "rescan", "ifname", self.interface], timeout=10)
        time.sleep(3)
        success, output = self._run_nmcli([
            "-t", "-f", "SSID,FREQ",
            "device", "wifi", "list", "ifname", self.interface, "--rescan", "no"
        ])
        if not success:
            return False
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            idx = line.rfind(':')
            if idx < 0:
                continue
            scanned_ssid = line[:idx].replace('\\:', ':')
            try:
                freq_mhz = int(line[idx + 1:].strip().split()[0])
            except (ValueError, IndexError):
                continue
            if scanned_ssid == ssid and 4900 <= freq_mhz <= 5900:
                return True
        return False

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
