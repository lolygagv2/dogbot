"""AP-state regression tests — treatbot2 relay-stranding bug, 2026-09-04.

A second local_mode command rebuilt a live AP, failed on dnsmasq, and
_cleanup_ap() tore the AP down WITHOUT clearing _in_ap_mode. is_ap_mode()
short-circuited on that flag, so for the rest of the boot the robot believed
it was hosting an AP while sitting on home WiFi. The relay reconnect loop
deferred silently forever and the robot dropped off the cloud until a power
cycle, with nothing in the journal.

Run: env_new/bin/python -m pytest tests/network/test_wifi_ap_flag_recovery.py -v
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.network.wifi_manager import WiFiManager  # noqa: E402


def _quiet_manager():
    """WiFiManager with every shell-out stubbed — no hardware is touched."""
    wifi = WiFiManager()
    wifi._run_cmd = lambda *a, **k: (True, "")
    wifi._run_nmcli = lambda *a, **k: (True, "")
    wifi._sudo_rm = lambda *a, **k: True
    return wifi


def test_cleanup_ap_clears_both_flags():
    """_cleanup_ap() is the common exit of every failed bring-up."""
    wifi = _quiet_manager()
    wifi._in_ap_mode = True
    wifi.ap_deliberate = True

    wifi._cleanup_ap()

    assert wifi._in_ap_mode is False
    assert wifi.ap_deliberate is False


def test_is_ap_mode_reconciles_stale_flag():
    """Flag says AP, hostapd says otherwise -> reality wins, flag is cleared."""
    wifi = _quiet_manager()
    wifi._in_ap_mode = True

    with patch.object(wifi, "_hostapd_running", return_value=False):
        assert wifi.is_ap_mode() is False

    assert wifi._in_ap_mode is False, "stale flag must not survive the check"


def test_is_ap_mode_true_while_hostapd_lives():
    wifi = _quiet_manager()
    with patch.object(wifi, "_hostapd_running", return_value=True):
        assert wifi.is_ap_mode() is True


def test_failed_second_bringup_does_not_poison_state():
    """Replays the 17:13 sequence: AP up, second command fails on dnsmasq."""
    wifi = _quiet_manager()

    # Command #1 succeeded.
    wifi._in_ap_mode = True
    wifi.ap_deliberate = True

    # Command #2 re-enters and dies on dnsmasq -> _cleanup_ap() -> return False.
    wifi._cleanup_ap()

    # hostapd is now dead. The robot must NOT claim to be an AP.
    with patch.object(wifi, "_hostapd_running", return_value=False):
        assert wifi.is_ap_mode() is False
    assert wifi.ap_deliberate is False


def test_reconnect_guard_defers_to_a_real_route():
    """Station-mode association beats the AP flag: dial the relay anyway."""
    from services.cloud.relay_client import RelayClient

    class FakeWifi:
        def is_ap_mode(self):
            return True      # even if this is somehow still wrong...

        def is_connected(self):
            return True      # ...we demonstrably have a cloud route

    with patch("services.network.wifi_manager.get_wifi_manager",
               return_value=FakeWifi()):
        result = asyncio.run(RelayClient._is_serving_local_ap(object()))

    assert result is False, "must not strand the robot when WiFi is up"


def test_reconnect_guard_still_holds_for_a_genuine_ap():
    from services.cloud.relay_client import RelayClient

    class FakeWifi:
        def is_ap_mode(self):
            return True

        def is_connected(self):
            return False     # serving an AP, no upstream

    with patch("services.network.wifi_manager.get_wifi_manager",
               return_value=FakeWifi()):
        result = asyncio.run(RelayClient._is_serving_local_ap(object()))

    assert result is True


if __name__ == "__main__":
    # pytest isn't installed in env_new — run standalone.
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {name}: {exc.__class__.__name__}: {exc}")
    print(f"\n{'FAILED' if failures else 'ALL PASS'} ({failures} failure(s))")
    sys.exit(1 if failures else 0)
